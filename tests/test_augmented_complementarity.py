"""Regional dual calculus and deterministic continuation at accepted endpoints."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from interface_extension import regional_power
from pilot.augmented_constraints import RegionalAugmentation, regional_inner
from pilot.plasticity import _fischer_burmeister


def points():
    return SimpleNamespace(
        xy_all=torch.arange(18, dtype=torch.float64).reshape(9, 2),
        n_pde=2, n_wall=3,
        interior_weights=torch.tensor([.3, .7], dtype=torch.float64),
        wall_weights=torch.tensor([.2, .3, .5], dtype=torch.float64))


def point_measure():
    return torch.tensor([.24, .56, .02, .03, .05, .025, .025, .025, .025],
                        dtype=torch.float64)


def test_pairing_preserves_regions_component_mean_and_repeated_interior():
    pts = points()
    left = (pts.xy_all - 5) / 7
    right = (pts.xy_all + 2).cos().requires_grad_()
    expected = torch.einsum("n,nc,nc->", point_measure(), left, right) / 2
    torch.testing.assert_close(regional_inner(left, right, pts), expected)
    gradient, = torch.autograd.grad(regional_inner(left, right, pts), right)
    torch.testing.assert_close(gradient, point_measure()[:, None] * left / 2)
    torch.testing.assert_close(regional_inner(right, right, pts), regional_power(right, pts))

    repeated = deepcopy(pts)
    repeated.n_pde = 4
    repeated.xy_all = torch.cat((pts.xy_all[:2].repeat(2, 1), pts.xy_all[2:]))
    repeated.interior_weights = pts.interior_weights.repeat(2) / 2
    repeat = lambda x: torch.cat((x[:2].repeat(2, 1), x[2:]))
    torch.testing.assert_close(regional_inner(repeat(left), repeat(right), repeated), expected)


def test_fb_gradient_and_pointwise_dual_update_have_no_extra_measure_factor():
    pts = points()
    augmentation = RegionalAugmentation(pts, weight=3.7, evaluations_per_update=1000, components=2)
    a = (pts.xy_all / 21 - .3).requires_grad_()
    b = (.4 * pts.xy_all.cos()).requires_grad_()
    residual = _fischer_burmeister(a, b, .01)
    augmentation.update(residual, evaluations=17)
    torch.testing.assert_close(augmentation.dual, 7.4 * residual.detach())
    assert not augmentation.dual.requires_grad and augmentation.dual.grad_fn is None
    assert augmentation.dual_updates == 1 and augmentation.next_update_evaluation == 1017

    loss = 3.7 * regional_power(residual, pts) + augmentation.linear_term(residual)
    gradients = torch.autograd.grad(loss, (a, b))
    cotangent = point_measure()[:, None] * (7.4 * residual.detach() + augmentation.dual) / 2
    magnitude = torch.sqrt(a.detach().square() + b.detach().square() + .01**2)
    for gradient, coordinate in zip(gradients, (a, b)):
        torch.testing.assert_close(gradient, cotangent * (coordinate.detach() / magnitude - 1))
    da, db = pts.xy_all.sin(), pts.xy_all.cos()
    def value(offset):
        r = _fischer_burmeister(a.detach() + offset * da, b.detach() + offset * db, .01)
        return 3.7 * regional_power(r, pts) + augmentation.linear_term(r)
    torch.testing.assert_close((value(1e-6) - value(-1e-6)) / 2e-6,
                               (gradients[0] * da + gradients[1] * db).sum(),
                               atol=1e-8, rtol=1e-8)


def test_dual_checkpoint_is_a_detached_snapshot_and_restores_next_update(tmp_path):
    pts = points()
    original = RegionalAugmentation(pts, 5., 1000, 2)
    original.update(pts.xy_all / 13, 1027)
    snapshot = original.state_dict()
    original.update(-pts.xy_all / 13, 2000)
    torch.testing.assert_close(snapshot["dual"], 10 * pts.xy_all / 13)
    checkpoint = tmp_path / "dual.pt"
    torch.save(snapshot, checkpoint)
    restored = RegionalAugmentation(pts, 5., 1000, 2)
    restored.load_state_dict(torch.load(checkpoint, weights_only=False))
    assert restored.dual_updates == 1 and restored.next_update_evaluation == 2027
    restored.update(-pts.xy_all / 13, 2000)
    torch.testing.assert_close(restored.dual, original.dual)
    assert restored.state_dict()["next_update_evaluation"] == original.next_update_evaluation
    assert not restored.dual.requires_grad


class ToyIncrement(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor([.4, -.25], dtype=torch.float64))

    def coefficients(self):
        return self.value

    centrosymmetric = False


class ToyHistory(nn.Module):
    def __init__(self):
        super().__init__()
        self.increments = nn.ModuleList()

    @property
    def n_steps(self):
        return len(self.increments)

    def append_increment(self, *, copy_previous=False):
        self.increments.append(ToyIncrement())


def toy_runner(monkeypatch, *, interval=12, stationary=False, constitutive_interval=None):
    """Exercise the real checkpoint/block loop with a two-variable objective."""
    import run_orientation_extension as runner
    pts = points()
    monkeypatch.setattr(runner, "geometry", lambda *args, **kwargs: (
        pts, torch.eye(4, dtype=torch.float64), torch.zeros(4), SimpleNamespace(R=3.)))
    monkeypatch.setattr(runner, "make_causal_model", lambda *args: ToyHistory())
    def initial(*args):
        history = ToyHistory()
        history.append_increment()
        return None, None, history
    monkeypatch.setattr(runner, "load_model", initial)
    monkeypatch.setattr(runner, "prefix_cache", lambda *args: {})
    def residuals(current, *args, projected_coefficients=None):
        assert projected_coefficients is None
        a = current.value[None].expand(9, -1)
        b = .3 + .02 * pts.xy_all.cos()
        return {}, dict(complementarity=_fischer_burmeister(a, b, .01),
                        constitutive=torch.cat((a - .12, 2 * a + .03), -1),
                        mechanics=current.value - current.value.new_tensor([-.3, .1]))
    monkeypatch.setattr(runner, "current_residuals", residuals)
    def objective(r, pts, sf, weights, *args):
        loss = r["mechanics"].square().sum() + weights["complementarity"] * regional_power(r["complementarity"], pts)
        if "constitutive" in weights:
            loss = loss + weights["constitutive"] * regional_power(r["constitutive"], pts)
        return loss
    monkeypatch.setattr(runner, "objective", objective)
    monkeypatch.setattr(runner, "wall_response", lambda *args: {})
    monkeypatch.setattr(runner, "export_case", lambda *args: None)
    step = runner.budgeted_lbfgs_step
    observed_states = []
    def small_block(optimizer, closure, **kwargs):
        observed_states.append(bool(optimizer.state))
        if stationary:
            loss = closure()
            optimizer.state[next(iter(optimizer.param_groups[0]["params"]))]["marker"] = 1
            return loss
        return step(optimizer, closure, remaining_evals=kwargs["remaining_evals"], chunk_max_eval=3)
    monkeypatch.setattr(runner, "budgeted_lbfgs_step", small_block)
    settings = dict(plastic_initialization="preceding_trained_increment", initialization_run="unused",
                    load_steps=1, weights=dict(complementarity=3.7), physics={},
                    lbfgs_evaluations_per_step=90)
    if interval is not None:
        settings["complementarity_augmentation"] = dict(evaluations_per_update=interval)
    if constitutive_interval is not None:
        settings["constitutive_augmentation"] = dict(evaluations_per_update=constitutive_interval)
        settings["weights"]["constitutive"] = 2.3
    case = dict(name="toy", joint_normal_deg=0.)
    return runner, case, settings, observed_states


@pytest.mark.parametrize("after_update", [False, True])
@pytest.mark.parametrize("constitutive_interval", [None, 12, 19])
def test_runner_resume_matches_uninterrupted_primal_dual_and_schedule(
        tmp_path, monkeypatch, after_update, constitutive_interval):
    runner, case, settings, _ = toy_runner(monkeypatch, constitutive_interval=constitutive_interval)
    whole = tmp_path / "whole"
    runner.run_case(case, settings, whole, 1, "cpu")
    expected = torch.load(whole / "model.pt", weights_only=False)
    assert expected["complementarity_augmentation"]["dual_updates"] > 0
    save = runner.save
    def interrupt(path, payload):
        save(path, payload)
        updates = payload["complementarity_augmentation"]["dual_updates"]
        if payload["evaluations"] > 0 and not payload["completed_steps"] and (updates > 0) == after_update:
            if after_update:
                assert not payload["optimizer"]["state"]
            raise InterruptedError("Accepted endpoint saved")
    monkeypatch.setattr(runner, "save", interrupt)
    resumed = tmp_path / "resumed"
    with pytest.raises(InterruptedError):
        runner.run_case(case, settings, resumed, 1, "cpu")
    monkeypatch.setattr(runner, "save", save)
    runner.run_case(case, settings, resumed, 1, "cpu")
    actual = torch.load(resumed / "model.pt", weights_only=False)
    assert actual["evaluations"] == expected["evaluations"]
    for name in expected["model"]:
        torch.testing.assert_close(actual["model"][name], expected["model"][name], atol=0, rtol=0)
    for key in ("complementarity_augmentation", "constitutive_augmentation"):
        if key in settings:
            for name, value in expected[key].items():
                torch.testing.assert_close(actual[key][name], value, atol=0, rtol=0)
    for name in ("objective", "base_objective", "complementarity_rms", "dual_updates"):
        assert actual["rows"][-1][name] == expected["rows"][-1][name]
    if constitutive_interval is not None:
        for name in ("constitutive_rms", "constitutive_dual_updates"):
            assert actual["rows"][-1][name] == expected["rows"][-1][name]


def test_four_component_stress_constraint_has_one_component_mean():
    pts = points()
    augmentation = RegionalAugmentation(pts, 2.3, 1000, 4)
    residual = torch.cat((pts.xy_all / 9, pts.xy_all.sin()), -1).requires_grad_()
    augmentation.update(residual, 32)
    torch.testing.assert_close(augmentation.dual, 4.6 * residual.detach())
    loss = 2.3 * regional_power(residual, pts) + augmentation.linear_term(residual)
    gradient, = torch.autograd.grad(loss, residual)
    torch.testing.assert_close(gradient, point_measure()[:, None] * 9.2 * residual.detach() / 4)


def trust_runner(monkeypatch, evaluations):
    """Keep the real neural/trust solver and checkpoint loop on a small system."""
    import pilot.orientation_tangent as tangent
    runner, case, settings, _ = toy_runner(monkeypatch, constitutive_interval=12)
    pts = points()

    class ToyResidual:
        def __init__(self, model, *args, **kwargs):
            self.current = model.increments[-1]
            self.duals = {}

        def __call__(self, values):
            a = values[None].expand(9, -1)
            residuals = dict(complementarity=_fischer_burmeister(a, .3+.02*pts.xy_all.cos(), .01),
                             constitutive=torch.cat((a-.12, 2*a+.03), -1))
            blocks = [values-values.new_tensor([-.3, .1])]
            for name, value in residuals.items():
                weight = settings["weights"][name]
                shifted = value+self.duals[name]/(2*weight)
                blocks.append((shifted*(weight*point_measure()[:, None]/value.shape[-1]).sqrt()).flatten())
            return torch.cat(blocks)

    monkeypatch.setattr(tangent, "OrientationResidual", ToyResidual)
    settings["trust_region"] = dict(warmup_evaluations=0, evaluations_per_solve=evaluations,
                                    linear_iterations=20, gradient_reduction=.1)
    return runner, case, settings


def assert_trust_state_equal(actual, expected):
    assert actual["last_result"] == expected["last_result"]
    torch.testing.assert_close({key: value for key, value in actual.items() if key != "last_result"},
                               {key: value for key, value in expected.items() if key != "last_result"},
                               atol=0, rtol=0)


@pytest.mark.parametrize("evaluations,polish", [(1, 0), (20, 0), (1, 100)])
def test_trust_causal_checkpoint_updates_dual_only_after_primal_reduction(tmp_path, monkeypatch, evaluations, polish):
    runner, case, settings = trust_runner(monkeypatch, evaluations)
    settings["trust_region"]["polish_evaluations"] = polish
    complete = runner.run_case(case, settings, tmp_path, 1, "cpu")
    saved = torch.load(tmp_path/"model.pt", weights_only=False)
    assert complete == saved["completed_steps"] == 0
    assert saved["model_steps"] == 1
    assert not saved["rows"][-1]["completed"]
    result = saved["trust_region_state"]["last_result"]
    converged = evaluations == 20 or bool(polish)
    assert result["status"] == int(converged)
    assert bool(saved["trust_region_state"]["inner"]) != converged
    current = ToyIncrement()
    current.value.data.copy_(saved["model"]["increments.0.value"])
    _, residuals = runner.current_residuals(current)
    for name in ("constitutive", "complementarity"):
        augmentation = saved[name+"_augmentation"]
        assert augmentation["dual_updates"] == int(converged)
        expected = 2*settings["weights"][name]*residuals[name]*int(converged)
        torch.testing.assert_close(augmentation["dual"], expected)
    runner.run_case(case, settings, tmp_path, 1, "cpu", trust_blocks=0)
    restored = torch.load(tmp_path/"model.pt", weights_only=False)
    torch.testing.assert_close(restored["model"], saved["model"], atol=0, rtol=0)
    assert_trust_state_equal(restored["trust_region_state"], saved["trust_region_state"])
    for name in ("constitutive", "complementarity"):
        torch.testing.assert_close(restored[name+"_augmentation"], saved[name+"_augmentation"], atol=0, rtol=0)


def test_trust_resume_and_assessed_acceptance_preserve_actual_increment(tmp_path, monkeypatch):
    runner, case, settings = trust_runner(monkeypatch, 20)
    runner.run_case(case, settings, tmp_path/"whole", 1, "cpu", trust_blocks=2)
    runner.run_case(case, settings, tmp_path/"resumed", 1, "cpu", trust_blocks=1)
    runner.run_case(case, settings, tmp_path/"resumed", 1, "cpu", trust_blocks=1)
    whole = torch.load(tmp_path/"whole/model.pt", weights_only=False)
    resumed = torch.load(tmp_path/"resumed/model.pt", weights_only=False)
    for name in ("model", "constitutive_augmentation", "complementarity_augmentation"):
        torch.testing.assert_close(whole[name], resumed[name], atol=0, rtol=0)
    assert_trust_state_equal(whole["trust_region_state"], resumed["trust_region_state"])
    completed = runner.run_case(case, settings, tmp_path/"resumed", 1, "cpu", accept_step=1)
    accepted = torch.load(tmp_path/"resumed/model.pt", weights_only=False)
    assert completed == accepted["completed_steps"] == 1
    torch.testing.assert_close(accepted["model"], resumed["model"], atol=0, rtol=0)
    with pytest.raises(ValueError, match="pending"):
        runner.run_case(case, settings, tmp_path/"resumed", 1, "cpu", accept_step=1)


def test_trust_resume_can_change_resource_limits_without_resetting_inner_state(tmp_path, monkeypatch):
    runner, case, settings = trust_runner(monkeypatch, 1)
    runner.run_case(case, settings, tmp_path, 1, "cpu")
    before = torch.load(tmp_path/"model.pt", weights_only=False)
    # Earlier checkpoints predate the per-row resource fields.
    for row in before["rows"]:
        row.pop("trust_linear_iterations")
        row.pop("trust_evaluation_limit")
        row.pop("primal_polish_limit")
    torch.save(before, tmp_path/"model.pt")
    settings["trust_region"].update(linear_iterations=60, evaluations_per_solve=2, polish_evaluations=0)
    runner.run_case(case, settings, tmp_path, 1, "cpu", trust_blocks=0)
    after = torch.load(tmp_path/"model.pt", weights_only=False)
    torch.testing.assert_close(after["model"], before["model"], atol=0, rtol=0)
    assert_trust_state_equal(after["trust_region_state"], before["trust_region_state"])
    assert after["rows"][-1]["trust_linear_iterations"] == 60
    assert after["rows"][-1]["trust_evaluation_limit"] == 2
    runner.run_case(case, settings, tmp_path, 1, "cpu")
    resumed = torch.load(tmp_path/"model.pt", weights_only=False)
    assert resumed["evaluations"] == before["evaluations"]+2
    assert resumed["trust_region_state"]["inner"]["gradient_tolerance"] == before["trust_region_state"]["inner"]["gradient_tolerance"]
    settings["weights"]["constitutive"] *= 2
    with pytest.raises(ValueError, match="Changed experiment inputs"):
        runner.run_case(case, settings, tmp_path, 1, "cpu")


def test_stationary_inner_block_updates_dual_and_clears_history(tmp_path, monkeypatch):
    runner, case, settings, states = toy_runner(monkeypatch, interval=1000, stationary=True)
    settings["lbfgs_evaluations_per_step"] = 29
    runner.run_case(case, settings, tmp_path, 1, "cpu")
    saved = torch.load(tmp_path / "model.pt", weights_only=False)
    assert saved["evaluations"] == 3
    assert saved["complementarity_augmentation"]["dual_updates"] == 3
    assert saved["complementarity_augmentation"]["next_update_evaluation"] == 1003
    assert states == [False, False, False]


def test_absent_setting_keeps_original_objective_and_stationary_stop(tmp_path, monkeypatch):
    runner, case, settings, states = toy_runner(monkeypatch, interval=None, stationary=True)
    runner.run_case(case, settings, tmp_path, 1, "cpu")
    saved = torch.load(tmp_path / "model.pt", weights_only=False)
    assert saved["evaluations"] == 1 and states == [False]
    assert "complementarity_augmentation" not in saved
    assert not {"base_objective", "complementarity_rms", "dual_updates"}.intersection(saved["rows"][-1])
