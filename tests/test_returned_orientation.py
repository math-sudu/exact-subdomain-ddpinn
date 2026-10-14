"""Exercise the condensed physics through actual neural and history interfaces."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"code"))
from interface_extension import geometry
from pilot import plasticity as P
from pilot.mixed_plasticity import field_history
from pilot.orientation_tangent import NeuralCoordinates, column_scale, linear_operator, neural_primal_step
from pilot.returned_orientation import ReturnedOrientationResidual, active_set_line_probe
from pilot.spectral_mixed import CircularGridEvaluation, freeze_history, load_orientation_model, make_causal_model


def fixture(steps=5):
    torch.manual_seed(124)
    settings = dict(hidden=[8, 8], radius=3., load_steps=12, representation="spectral",
        elastic_prefix="circular", constitutive_update="joint_return", physics={},
        projection=dict(radial_order=6, angular_modes=6, angular_samples=24, centrosymmetric=True),
        quadrature=dict(radial=8, angular=24, boundary=48, stagger_angles=True, radial_power=1),
        weights=dict(equilibrium=3., wall=3., interface=5., constitutive=6.,
                     complementarity=100., gauge=1., virtual_work=8*np.pi))
    case = dict(name="bedding_oblique", eh_over_ev=2.625, bedding_dip_deg=45., opening_axes=[1., 1.],
                opening_angle_deg=0., joint_normal_deg=0.)
    _, c, s0, _ = geometry(case, settings, "cpu")
    model = make_causal_model(settings, c, case)
    for _ in range(steps):
        model.append_increment(copy_previous=False)
    with torch.no_grad():
        model.increments[-1].kinematic_head.weight.normal_(std=.001)
    freeze_history(model)
    saved = dict(settings=settings, case=case, c=c, s0=s0,
        completed_steps=steps-1, model_steps=steps, model=model.state_dict())
    return model, saved, P.WeakPlaneParameters(joint_normal_deg=0.)


def test_grid_hessian_matches_arbitrary_point_autodiff():
    model, _, _ = fixture()
    inc = model.increments[-1]
    radius = torch.tensor([1.01, 1.3, 2.6], dtype=torch.float64)
    angle = (torch.arange(24, dtype=radius.dtype)[None]+torch.tensor([[.2], [.5], [.7]]))*2*torch.pi/24
    xy = radius[:, None, None]*torch.stack((angle.cos(), angle.sin()), -1)
    grid = CircularGridEvaluation(inc, (xy,))
    coef = inc.coefficients()[..., :2]
    _, gradient, hessian = grid(coef, second=True)
    flat = xy.reshape(-1, 2)
    _, expected_gradient = inc._spatial_expansion(coef, flat)
    expected_hessian = torch.stack([torch.func.jvp(lambda points: inc._spatial_expansion(coef, points)[1],
        (flat,), (torch.eye(2, dtype=flat.dtype)[axis].expand_as(flat),))[1] for axis in range(2)])
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-10, atol=1e-13)
    torch.testing.assert_close(hessian, expected_hessian, rtol=1e-10, atol=1e-13)


def test_returned_history_matches_grid_physics_and_plastic_prefix(tmp_path):
    # Step six consumes actual step-five plasticity and its spatial derivative.
    model, saved, params = fixture(6)
    p = ReturnedOrientationResidual(model, saved, params)
    h, r = p.fields(p.current.coefficients())
    queried = field_history(model, p.pts.xy_all, p.c, p.s0, params)
    for key in ("displacement", "stress_total", "strain", "plastic_strain", "multiplier_increment"):
        torch.testing.assert_close(h[key], queried[key][:, -1], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(r["equilibrium"], queried["equilibrium"][:p.pts.n_pde, -1],
                               rtol=1e-9, atol=1e-11)
    assert queried["plastic_strain"][:, 4].abs().max() > 0
    assert queried["plastic_strain"][:, :4].abs().max() == 0
    torch.testing.assert_close(queried["constitutive"], torch.zeros_like(queried["constitutive"]),
                               rtol=0., atol=1e-13)
    checkpoint = tmp_path/"model.pt"
    torch.save(saved, checkpoint)
    bundle, _, restored = load_orientation_model(checkpoint)
    for actual, expected in zip(restored(p.pts.xy_all), model(p.pts.xy_all)):
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)
    from publication_model import load_model
    publication = tmp_path/"public.pt"
    torch.save(dict(spec=bundle["spec"], c=p.c, s0=p.s0, load_step=6, model=model.state_dict()), publication)
    _, _, public = load_model(tmp_path, publication)
    for actual, expected in zip(public(p.pts.xy_all), model(p.pts.xy_all)):
        torch.testing.assert_close(actual, expected, atol=0., rtol=0.)


@pytest.mark.parametrize("equilibrium_form", ["strong", "virtual_work"])
def test_returned_neural_tangent_and_actual_parameter_update(equilibrium_form):
    model, saved, params = fixture()
    saved["settings"]["equilibrium_form"] = equilibrium_form
    p = ReturnedOrientationResidual(model, saved, params)
    coordinates = NeuralCoordinates(p.current, "displacement")
    assert len(coordinates.indices["increment.kinematic_head.bias"]) == 0
    assert len(coordinates.indices["increment.kinematic_head.weight"]) == 16
    point = coordinates.zero()
    direction = torch.randn_like(point)
    direction /= direction.norm()
    function = lambda value: p(coordinates(value))
    operator, value, _ = linear_operator(function, point, torch.ones_like(point))
    finite = (function(point+1e-6*direction)-function(point-1e-6*direction))/(2e-6)
    np.testing.assert_allclose(operator@direction.numpy(), finite.detach().numpy(), atol=2e-8, rtol=2e-5)
    covector = np.random.default_rng(9).normal(size=len(value))
    np.testing.assert_allclose(covector@(operator@direction.numpy()),
                               direction.numpy()@operator.rmatvec(covector), rtol=1e-11)
    before = float(value.square().sum())
    result = neural_primal_step(p, dict(linear_iterations=30, evaluations_per_solve=8,
        polish_evaluations=100, gradient_reduction=.1), {})
    assert result["evaluations"] > 1
    assert float(p(p.current.coefficients()).detach().square().sum()) < before
    h, _ = p.fields(p.current.coefficients())
    actual = model(p.pts.xy_all)
    torch.testing.assert_close(actual[1][:, -1]+p.s0, h["stress_total"], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("completed_onset", [False, True])
def test_runner_preserves_pending_state_resume_and_explicit_acceptance(tmp_path, completed_onset):
    from run_orientation_extension import run_case

    _, saved, _ = fixture()
    settings = dict(saved["settings"], cases=[saved["case"]], trust_region=dict(linear_iterations=20,
        evaluations_per_solve=3, polish_evaluations=30, gradient_reduction=.1))
    mixed_settings = {key: value for key, value in settings.items() if key != "constitutive_update"}
    mixed = make_causal_model(mixed_settings, saved["c"], saved["case"])
    for _ in range(5):
        mixed.append_increment(copy_previous=False)
    # A real initialized displacement avoids the zero-head parameter nullspace.
    mixed.increments[-1].kinematic_head.load_state_dict(
        {key.split("increments.4.kinematic_head.", 1)[1]: value
         for key, value in saved["model"].items() if key.startswith("increments.4.kinematic_head.")})
    source = tmp_path/"mixed.pt"
    source_completed = 5 if completed_onset else 4
    torch.save(dict(saved, completed_steps=source_completed,
                    settings=mixed_settings, model=mixed.state_dict()), source)
    destination = tmp_path/"returned"
    assert run_case(saved["case"], settings, destination, 5, "cpu", trust_blocks=0,
                    initialize_checkpoint=source) == 4
    _, initialized, initialized_model = load_orientation_model(destination/"model.pt")
    assert initialized["completed_steps"] == 4 and initialized["evaluations"] == 0
    for key, value in mixed.state_dict().items():
        torch.testing.assert_close(initialized_model.state_dict()[key], value, atol=0., rtol=0.)
    with pytest.raises(ValueError, match="assessed pending"):
        run_case(saved["case"], settings, destination, 5, "cpu", trust_blocks=0, accept_step=5)
    assert run_case(saved["case"], settings, destination, 5, "cpu", trust_blocks=1,
                    initialize_checkpoint=source) == 4
    _, first, first_model = load_orientation_model(destination/"model.pt")
    assert first["completed_steps"] == 4 and first["trust_region_state"]["solves"] == 1
    expected = {key: value.clone() for key, value in first_model.state_dict().items()}
    settings["cases"] = settings["cases"]+[dict(saved["case"], name="another_orientation")]
    assert run_case(saved["case"], settings, destination, 5, "cpu", trust_blocks=0) == 4
    _, resumed, restored = load_orientation_model(destination/"model.pt")
    assert resumed["evaluations"] == first["evaluations"]
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0., rtol=0.)
    settings["equilibrium_form"] = "virtual_work"
    assert run_case(saved["case"], settings, destination, 5, "cpu", trust_blocks=0) == 4
    _, changed, restored = load_orientation_model(destination/"model.pt")
    assert changed["trust_region_state"] == dict(solves=0, inner={}, last_result={})
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0., rtol=0.)
    with pytest.raises(ValueError, match="assessed pending"):
        run_case(saved["case"], settings, destination, 5, "cpu", trust_blocks=0, accept_step=5)
    assert run_case(saved["case"], settings, destination, 5, "cpu", trust_blocks=1) == 4
    assert run_case(saved["case"], settings, destination, 5, "cpu", trust_blocks=0,
                    accept_step=5) == 5
    _, accepted, _ = load_orientation_model(destination/"model.pt")
    assert accepted["completed_steps"] == 5
    assert torch.load(source, weights_only=False)["completed_steps"] == source_completed


def test_matrix_yield_is_reported_without_changing_pending_fields(tmp_path):
    from run_orientation_extension import run_case

    _, saved, _ = fixture()
    settings = dict(saved["settings"], physics=dict(matrix_cohesion_mpa=.01),
        trust_region=dict(linear_iterations=20, evaluations_per_solve=3,
                          polish_evaluations=0, gradient_reduction=.1))
    saved.update(settings=settings, rows=[], evaluations=1, step_elapsed_s=0.,
        trust_region_state=dict(solves=1, inner={}, last_result={}),
        initialization_checkpoint=str(tmp_path/"mixed.pt"))
    checkpoint = tmp_path/"model.pt"
    torch.save(saved, checkpoint)
    assert run_case(saved["case"], settings, tmp_path, 5, "cpu", trust_blocks=0) == 4
    pending = torch.load(checkpoint, weights_only=False)
    assert pending["rows"][-1]["maximum_matrix_yield_mpa"] > 0
    assert pending["completed_steps"] == 4
    torch.testing.assert_close(pending["model"], saved["model"], atol=0., rtol=0.)


def test_mixed_onset_conversion_rejects_a_plastic_prefix(tmp_path):
    from run_orientation_extension import run_case

    _, saved, _ = fixture(6)
    settings = dict(saved["settings"], trust_region={})
    mixed_settings = {key: value for key, value in settings.items() if key != "constitutive_update"}
    mixed = make_causal_model(mixed_settings, saved["c"], saved["case"])
    for _ in range(6):
        mixed.append_increment(copy_previous=False)
    with torch.no_grad():
        mixed.increments[4].kinematic_head.weight.normal_(std=.001)
    source = tmp_path/"mixed.pt"
    torch.save(dict(saved, completed_steps=6, settings=mixed_settings,
                    model=mixed.state_dict()), source)
    with pytest.raises(ValueError, match="exact elastic prefix"):
        run_case(saved["case"], settings, tmp_path/"returned", 6, "cpu", trust_blocks=0,
                 initialize_checkpoint=source)


def test_active_switch_probe_uses_actual_neural_residual_and_preserves_fields():
    model, saved, params = fixture()
    p = ReturnedOrientationResidual(model, saved, params)
    coefficients = p.current.coefficients().clone()
    probe = active_set_line_probe(p)
    assert probe["directional_slope"] < 0
    np.testing.assert_allclose(probe["line"][0]["objective"], float(p(coefficients).square().sum()), rtol=1e-12)
    assert len(probe["nearest_crossings"]) == 2
    assert probe["line"][0]["changed_active"] == []
    torch.testing.assert_close(p.current.coefficients(), coefficients, atol=0., rtol=0.)


def test_weak_neural_residual_is_continuous_across_loading_switch():
    model, saved, params = fixture()
    saved["settings"]["equilibrium_form"] = "virtual_work"
    p = ReturnedOrientationResidual(model, saved, params)
    coordinates = NeuralCoordinates(p.current, "displacement")
    point = coordinates.zero()
    function = lambda delta: p(coordinates(delta))
    value, reverse = torch.func.vjp(function, point)
    direction = -reverse(value)[0].detach()
    direction /= direction.norm()
    def interior_yield(delta):
        h, _ = p.fields(coordinates(delta), equilibrium=False)
        return P.joint_yield(h["trial_stress"], params)[:p.pts.n_pde]

    initial, derivative = torch.func.jvp(interior_yield, (point,), (direction,))
    crossing = -initial/derivative
    crossing = torch.where(crossing > 0, crossing, torch.full_like(crossing, torch.inf))
    index = int(crossing.argmin())

    @torch.no_grad()
    def trial(length):
        h, _ = p.fields(coordinates(length*direction), equilibrium=False)
        return float(P.joint_yield(h["trial_stress"], params)[index])

    left, right = 0., 2*float(crossing[index].detach())
    assert trial(left)*trial(right) < 0
    for _ in range(45):
        middle = (left+right)/2
        if trial(middle)*trial(left) > 0:
            left = middle
        else:
            right = middle
    root = (left+right)/2
    differences, strong = [], []
    with torch.no_grad():
        for width in (root*1e-3, root*1e-6):
            minus, plus = [(root+sign*width)*direction for sign in (-1, 1)]
            assert trial(root-width)*trial(root+width) < 0
            differences.append(float((function(plus)-function(minus)).norm()))
            residuals = [p.fields(coordinates(delta))[1]["equilibrium"] for delta in (minus, plus)]
            strong.append(float((residuals[1]-residuals[0]).norm()))
    assert differences[1] < .01*differences[0]
    assert strong[1] > .5*strong[0]
    assert "equilibrium" not in p.blocks(coordinates(point))
    assert {"virtual_work", "wall", "interface"} <= p.blocks(coordinates(point)).keys()


def test_neural_column_scaling_keeps_exact_null_directions_finite():
    point = torch.zeros(2, dtype=torch.float64)
    value, reverse = torch.func.vjp(lambda x: torch.stack((2*x[0], x[1]*0)), point)
    scale = column_scale(reverse, value, point)
    torch.testing.assert_close(scale, point.new_tensor([.5, 1.]))
