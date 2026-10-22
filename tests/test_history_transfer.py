"""Check coordinate and load scaling in reusable history migration."""

from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from extract_unloading_results import sample_fields
from pilot.causal_mixed import IncrementFields
from pilot.returned_orientation import ReturnedOrientationResidual
from pilot.orientation_tangent import NeuralCoordinates
from pilot.spectral_mixed import CircularGridEvaluation, make_causal_model
from refine_crossed_history import warm_start
from recompute_orientation_history import initialize_continuation, source_total_predictor
from refine_returned_head import branch_margin
from run_orientation_extension import run_return_case
from test_returned_orientation import fixture


def test_transfer_preserves_network_field_under_domain_and_load_refinement():
    original, saved, _ = fixture(5)
    settings = deepcopy(saved["settings"])
    settings.update(radius=4., load_steps=24)
    settings["projection"]["radial_order"] = 8
    migrated = make_causal_model(settings, saved["c"], saved["case"])
    warm_start(migrated, original.increments[-1])
    points = saved["c"].new_tensor([[1.1, .2], [-1.5, .4], [3.2, -.1]])
    before = IncrementFields.forward(original.increments[-1], points)[0]
    after = IncrementFields.forward(migrated.increments[-1], points)[0]
    torch.testing.assert_close(after * 2, before, rtol=1e-12, atol=1e-14)
    assert migrated.increments[-1].radial_order == 8
    assert migrated.increments[-1].input_scale == 4.
    before_background = original.increments[-1].elastic_background(points)
    after_background = migrated.increments[-1].elastic_background(points)
    for expected, actual in zip(before_background, after_background):
        torch.testing.assert_close(actual * 2, expected, rtol=1e-12, atol=1e-14)


def test_field_export_rejects_an_unsupported_return_query():
    model, saved, params = fixture(5)
    model.return_insitu.copy_(saved["c"].new_tensor([2., 0., 0., 0.]))
    with pytest.raises(ValueError, match="rounded apex"):
        sample_fields(dict(params=params, s0=saved["s0"]), model,
                      saved["c"].new_tensor([[1.2, .1]]).numpy())


def test_ring_history_matches_point_queries():
    model, saved, _ = fixture(5)
    angle = torch.arange(64, dtype=torch.float64) * (2 * torch.pi / 64)
    points = 1.15 * torch.stack((angle.cos(), angle.sin()), -1)
    grid = CircularGridEvaluation(model.increments[0], (points[None],))
    direct = model.returned_history(points)
    fast = model.returned_history(points, grid=grid)
    for expected, actual in zip(direct, fast):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_unsupported_transferred_guess_is_damped_without_rescaling_elastic_load():
    model, saved, params = fixture(5)
    current = model.increments[-1]
    current.requires_grad_(True)
    with torch.no_grad():
        current.kinematic_head.weight[:2].mul_(-100000.)
        background = current.project(torch.cat(current.elastic_background(current.projection_nodes), -1))
        current.background_subtraction.copy_(torch.view_as_real(background))
    problem = ReturnedOrientationResidual(model, saved, params)
    coefficients = current.coefficients().detach().clone()
    fields, _ = problem.fields(coefficients, equilibrium=False)
    assert not torch.isfinite(fields["stress_total"]).all()
    points = saved["c"].new_tensor([[1.1, .2], [-1.5, .4]])
    elastic = current.elastic_background(points)[0].detach().clone()
    correction = (current(points)[0]-elastic).detach().clone()
    factor = problem.scale_initial_correction(current)
    assert 0 <= factor < 1
    torch.testing.assert_close(current.coefficients()[..., :2], factor*coefficients[..., :2])
    torch.testing.assert_close(current.elastic_background(points)[0], elastic, rtol=0, atol=0)
    torch.testing.assert_close(current(points)[0]-elastic, factor*correction)
    assert torch.isfinite(problem(current.coefficients())).all()


@pytest.mark.parametrize("subdivisions", [1, 2])
def test_source_total_predictor_preserves_prefix_and_matches_physical_displacement(subdivisions):
    source, saved, _ = fixture(5)
    settings = deepcopy(saved["settings"])
    settings["load_steps"] *= subdivisions
    settings["projection"]["radial_order"] += 2
    model = make_causal_model(settings, saved["c"], saved["case"])
    for increment in source.increments:
        warm_start(model, increment)
    with torch.no_grad():
        model.increments[-2].kinematic_head.weight[:2].add_(.1)
    prefix = [{key: value.clone() for key, value in inc.state_dict().items()}
              for inc in model.increments[:-1]]
    source_total_predictor(model, source)
    for expected, inc in zip(prefix, model.increments[:-1]):
        for key, value in inc.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    points = saved["c"].new_tensor([[1.1, .2], [-1.5, .4], [2.7, -.1]])
    whole, remainder = divmod(model.n_steps, subdivisions)
    expected = sum(inc(points)[0] for inc in source.increments[:whole])
    if remainder:
        expected = expected + remainder/subdivisions * source.increments[whole](points)[0]
    actual = sum(inc(points)[0] for inc in model.increments)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    refined_settings = deepcopy(settings)
    refined_settings["projection"]["radial_order"] += 2
    refined_settings["load_steps"] *= 2
    refined = make_causal_model(refined_settings, saved["c"], saved["case"])
    warm_start(refined, model.increments[-1])
    before = model.increments[-1]
    after = refined.increments[-1]
    expected_offset = before.evaluate(torch.view_as_complex(before.background_subtraction), points)
    actual_offset = after.evaluate(torch.view_as_complex(after.background_subtraction), points)
    torch.testing.assert_close(2 * actual_offset, expected_offset, rtol=1e-10, atol=1e-12)


def test_return_margin_is_affine_in_the_displacement_head_with_fixed_trunk():
    model, saved, params = fixture(5)
    problem = ReturnedOrientationResidual(model, saved, params)
    coordinates = NeuralCoordinates(model.increments[-1], "displacement_head")
    zero = coordinates.zero()
    margin = lambda delta: branch_margin(problem, coordinates(delta))
    matrix = torch.func.jacfwd(margin)(zero)
    delta = torch.linspace(-.001, .002, len(zero), dtype=zero.dtype)
    torch.testing.assert_close(margin(delta), margin(zero)+matrix@delta, rtol=1e-10, atol=1e-12)
    extra = problem.pts.xy_all[[0, 3, 9]]
    direct = branch_margin(problem, coordinates(delta), extra)
    torch.testing.assert_close(direct, margin(delta)[[0, 3, 9]], rtol=1e-10, atol=1e-12)
    trunk = {key: value.clone() for key, value in model.increments[-1].kinematic_trunk.state_dict().items()}
    coordinates.apply(delta)
    for key, value in model.increments[-1].kinematic_trunk.state_dict().items():
        torch.testing.assert_close(value, trunk[key], rtol=0, atol=0)


def test_continuation_preserves_recorded_feasibility_recovery(tmp_path):
    model, saved, _ = fixture(5)
    saved["settings"].update(equilibrium_form="virtual_work", trust_region=dict(
        evaluations_per_solve=2, linear_iterations=5, polish_evaluations=0, gradient_reduction=.1))
    restarts = [dict(step=5, evaluations=12, elapsed_s=1., objective=.1)]
    heads = [dict(step=5, evaluations=7, elapsed_s=2., objective=.01)]
    enrichment = dict(step=5, evaluations=12,
        previous_test_space=dict(radial_order=6, angular_modes=6),
        test_space=dict(radial_order=8, angular_modes=8))
    saved["settings"]["work_test_space"] = enrichment["test_space"]
    saved.update(rows=[], evaluations=0, step_elapsed_s=0.,
        initialization_checkpoint=str(tmp_path / "model.pt"),
        trust_region_state=dict(solves=0, inner={}, last_result={}),
        predictor_restarts=restarts, constrained_head_solves=heads,
        work_test_enrichment=enrichment)
    torch.save(saved, tmp_path / "model.pt")
    run_return_case(saved["case"], saved["settings"], tmp_path, 5, "cpu", trust_blocks=0,
                    accept_step=None, initialize_checkpoint=None)
    restored = torch.load(tmp_path / "model.pt", weights_only=False)
    assert restored["predictor_restarts"] == restarts
    assert restored["constrained_head_solves"] == heads
    assert restored["work_test_enrichment"] == enrichment
    assert restored["settings"]["work_test_space"] == enrichment["test_space"]
    assert restored["rows"][-1]["step"] == 5


def test_restart_preserves_prefix_and_trial_and_discards_later_acceptance(tmp_path):
    model, saved, _ = fixture(5)
    saved.update(completed_steps=5, rows=[dict(step=k, completed=True) for k in range(1, 6)],
                 constrained_head_solves=[dict(step=3), dict(step=5)])
    source = tmp_path / "source.pt"
    torch.save(saved, source)
    output = tmp_path / "continued"
    initialize_continuation(source, output, 4)
    resumed = torch.load(output / "model.pt", weights_only=False)
    assert resumed["model_steps"] == 4 and resumed["completed_steps"] == 3
    assert [r["step"] for r in resumed["rows"]] == [1, 2, 3]
    assert resumed["constrained_head_solves"] == [dict(step=3)]
    assert resumed["initialization_checkpoint"] == str(source.resolve())
    assert resumed["trust_region_state"]["solves"] == 0
    for key, value in resumed["model"].items():
        torch.testing.assert_close(value, saved["model"][key], rtol=0, atol=0)
    assert not any(key.startswith("increments.4.") for key in resumed["model"])
    assert torch.load(source, weights_only=False)["completed_steps"] == 5


def test_weak_assessment_requires_explicit_pending_selection(tmp_path):
    from assess_history_resolution import assess
    _, saved, _ = fixture(5)
    checkpoint = tmp_path / "pending.pt"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="explicitly include"):
        assess(checkpoint, [5], 6, 6, 8, 24)
    result = assess(checkpoint, [5], 6, 6, 8, 24, include_pending=True)
    assert result["pending_step"] == 5 and result["completed_steps"] == 4
    assert result["rows"][0]["relative_residual_to_source"] > 0
    assert torch.load(checkpoint, weights_only=False)["completed_steps"] == 4
