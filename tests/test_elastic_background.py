"""The accepted circular prefix is the physical history used by plastic steps."""

import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code"))
from interface_extension import geometry
from pilot import plasticity as P
from pilot.causal_mixed import current_residuals, prefix_cache
from pilot.spectral_mixed import freeze_history, load_orientation_model, make_causal_model


def setup():
    settings = json.loads((ROOT/"configs/orientation_spectral.json").read_text())
    settings["hidden"] = [8, 8]
    settings["projection"].update(radial_order=8, angular_modes=6, angular_samples=32)
    settings["quadrature"] = dict(radial=8, angular=32, boundary=128)
    case = settings["cases"][0]
    points, c, s0, sf = geometry(case, settings, "cpu")
    params = P.WeakPlaneParameters(**settings["physics"], joint_normal_deg=case["joint_normal_deg"])
    return settings, case, points, c, s0, sf, params


def test_exact_prefix_satisfies_the_equations_and_enters_the_next_cache():
    settings, case, points, c, s0, sf, params = setup()
    model = make_causal_model(settings, c, case)
    for step in range(1, 5):
        freeze_history(model)
        cache = prefix_cache(model, points.xy_all, c, s0, params)
        current = model.append_increment(copy_previous=False)
        h, r = current_residuals(current, cache, points, c, s0, sf, params, step)
        for name in ("equilibrium", "wall", "interface", "constitutive", "gauge"):
            torch.testing.assert_close(r[name], torch.zeros_like(r[name]), atol=2e-12, rtol=0)
        assert torch.max(h["yield_values"]) < 0
        assert torch.count_nonzero(h["plastic_strain"]) == 0
    freeze_history(model)
    cache = prefix_cache(model, points.xy_all, c, s0, params)
    for name in ("displacement", "stress_increment", "strain", "plastic_strain"):
        torch.testing.assert_close(cache[name], h[name])
    current = model.append_increment(copy_previous=False)
    next_h, _ = current_residuals(current, cache, points, c, s0, sf, params, 5)
    assert torch.max(next_h["yield_values"]) > 0


def test_partial_history_diagnostics_keep_load_fractions_and_exclude_pending_step(tmp_path):
    from summarize_extensions import exterior_yield_rows, orientation_history_residuals

    settings, case, _, c, s0, _, _ = setup()
    model = make_causal_model(settings, c, case)
    for _ in range(3):
        model.append_increment(copy_previous=False)
    with torch.no_grad():
        model.increments[-1].stress_head.weight.fill_(5.)
    checkpoint = tmp_path/"model.pt"
    torch.save(dict(settings=settings, case=case, c=c, s0=s0, model_steps=3,
                    completed_steps=2, model=model.state_dict()), checkpoint)

    with pytest.raises(RuntimeError, match="Incomplete orientation history"):
        orientation_history_residuals(checkpoint, "cpu")
    rows = orientation_history_residuals(checkpoint, "cpu", require_complete=False)
    assert [(row["sampling"], row["step"]) for row in rows] == [
        (sampling, step) for sampling in ("training", "independent") for step in (1, 2)]
    for row in rows:
        # Exact elastic increments unload by 1/12, including at partial checkpoints.
        for key in ("equilibrium_rms", "constitutive_rms", "wall_traction_rms_mpa"):
            assert row[key] < 2e-11
    assert [row["step"] for row in exterior_yield_rows(checkpoint, "cpu")] == [1, 2]


def test_parity_is_part_of_values_derivatives_and_parameter_gradients():
    settings, case, _, c, _, _, _ = setup()
    model = make_causal_model(settings, c, case)
    field = model.append_increment(copy_previous=False)
    with torch.no_grad():
        field.kinematic_head.weight.normal_(std=.1)
        field.stress_head.weight.normal_(std=.1)
    xy = c.new_tensor([[1., 0.], [1.4, .7], [-.8, 2.], [0., -3.]])
    values, derivatives = field.spatial_fields(xy)
    opposite, opposite_derivatives = field.spatial_fields(-xy)
    for i, sign in enumerate((-1., 1., 1.)):
        torch.testing.assert_close(opposite[i], sign*values[i], atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(opposite_derivatives[i], -sign*derivatives[i], atol=2e-12, rtol=2e-12)
    automatic, d_automatic = torch.func.jvp(field, (xy,), (torch.ones_like(xy),))
    for actual, expected in zip(derivatives, d_automatic):
        torch.testing.assert_close(actual.sum(0), expected, atol=2e-12, rtol=2e-12)
    sum(v.square().sum() for v in (*values, *derivatives)).backward()
    assert field.kinematic_trunk[0].weight.grad.norm() > 0


@pytest.mark.parametrize("stress_correction", [False, True])
def test_checkpoint_restores_elastic_background_and_physical_history(tmp_path, stress_correction):
    settings, case, _, c, s0, _, _ = setup()
    settings["projection"]["stress_coefficient_correction"] = stress_correction
    model = make_causal_model(settings, c, case)
    model.append_increment(copy_previous=False)
    with torch.no_grad():
        model.increments[0].kinematic_head.weight.normal_(std=.1)
        if stress_correction:
            model.increments[0].stress_spectral_correction.normal_(std=.01)
    freeze_history(model)
    checkpoint = tmp_path/"model.pt"
    torch.save(dict(settings=settings, case=case, c=c, s0=s0, model_steps=1,
                    model=model.state_dict()), checkpoint)
    bundle, _, restored = load_orientation_model(checkpoint)
    xy = c.new_tensor([[1., 0.], [-1.2, .3], [.5, 2.6]])
    for expected, actual in zip(model(xy), restored(xy)):
        torch.testing.assert_close(actual, expected)
    from publication_model import load_model
    public_checkpoint = tmp_path/"public.pt"
    torch.save(dict(spec=bundle["spec"], c=c, s0=s0, load_step=1,
                    model=model.state_dict()), public_checkpoint)
    _, _, public_model = load_model(tmp_path, public_checkpoint)
    for expected, actual in zip(model(xy), public_model(xy)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("stress_correction", [False, True])
def test_total_initializer_is_converted_to_a_background_correction(stress_correction):
    settings, case, _, c, _, _, _ = setup()
    settings["projection"].update(radial_order=32, angular_modes=64, angular_samples=256)
    settings["projection"]["stress_coefficient_correction"] = stress_correction
    initial_settings = {key: value for key, value in settings.items() if key != "elastic_prefix"}
    initial = make_causal_model(initial_settings, c, case).append_increment(copy_previous=False)
    with torch.no_grad():
        initial.kinematic_head.weight.normal_(std=.1)
        initial.stress_head.weight.normal_(std=.1)
        if stress_correction:
            initial.stress_spectral_correction.normal_(std=.001)
    corrected = make_causal_model(settings, c, case).append_increment(copy_previous=False)
    corrected.initialize_from_total(initial)
    xy = c.new_tensor([[1., 0.], [-1.2, .3], [.5, 2.6]])
    for actual, expected in zip(corrected(xy), initial(xy)):
        # The analytic background has a small finite-projection truncation error.
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


def test_cached_background_preserves_current_residuals_and_parameter_gradients():
    settings, case, points, c, s0, sf, params = setup()
    model = make_causal_model(settings, c, case)
    cache = prefix_cache(model, points.xy_all, c, s0, params)
    current = model.append_increment(copy_previous=False)
    with torch.no_grad():
        current.kinematic_head.weight.normal_(std=.1)
        current.stress_head.weight.normal_(std=.1)
    _, direct = current_residuals(current, cache, points, c, s0, sf, params, 1)
    direct_grad = torch.autograd.grad(sum(v.square().sum() for v in direct.values()),
                                     tuple(current.parameters()))
    with torch.no_grad():
        cache["increment_background"] = current.elastic_background.spatial_fields(points.xy_all)
    _, cached = current_residuals(current, cache, points, c, s0, sf, params, 1)
    for key in direct:
        torch.testing.assert_close(cached[key], direct[key])
    cached_grad = torch.autograd.grad(sum(v.square().sum() for v in cached.values()),
                                     tuple(current.parameters()))
    for actual, expected in zip(cached_grad, direct_grad):
        torch.testing.assert_close(actual, expected)
