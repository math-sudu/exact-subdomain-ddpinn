"""The projection layer resolves both field values and spatial derivatives."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from pilot.spectral_mixed import CircularGridEvaluation, SpectralIncrementFields, make_causal_model
from pilot.causal_mixed import IncrementFields
from interface_extension import geometry


def field():
    return SpectralIncrementFields((8, 8), 3., 12, torch.tensor(.3, dtype=torch.float64),
                                   radial_order=8, angular_modes=6, angular_samples=32).double()


def formula(xy):
    radius = xy.norm(dim=-1)
    angle = torch.atan2(xy[:, 1], xy[:, 0])
    return (radius.square()*(3*angle).cos() + .2*radius*(2*angle).sin())[:, None]


def test_projection_reproduces_polynomial_modes_and_derivatives():
    layer = field()
    coefficients = layer.project(formula(layer.projection_nodes))
    xy = torch.tensor([[1.02, .1], [-1.2, .8], [-2.8, -.2], [.3, -2.1]], dtype=torch.float64)
    tangent = torch.tensor([[.4, -.1], [.2, .3], [-.5, .2], [1., 0.]], dtype=xy.dtype)
    value, derivative = torch.func.jvp(lambda x: layer.evaluate(coefficients, x), (xy,), (tangent,))
    expected, expected_derivative = torch.func.jvp(formula, (xy,), (tangent,))
    torch.testing.assert_close(value, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(derivative, expected_derivative, atol=1e-11, rtol=1e-11)


def test_parameter_gradients_pass_through_the_projection_and_derivative():
    layer = field()
    with torch.no_grad():
        layer.kinematic_head.weight.normal_(std=.1)
    xy = torch.tensor([[1.1, .4], [-1.3, .7]], dtype=torch.float64)
    value, derivative = torch.func.jvp(layer, (xy,), (torch.ones_like(xy),))
    loss = value[0].square().sum()+derivative[0].square().sum()
    loss.backward()
    assert layer.kinematic_head.weight.grad.norm() > 0
    assert layer.kinematic_trunk[0].weight.grad.norm() > 0


def test_frozen_projection_keeps_coordinate_derivatives_and_invalidates_on_load():
    layer = field()
    with torch.no_grad():
        layer.kinematic_head.weight.normal_(std=.1)
    layer.requires_grad_(False)
    xy = torch.tensor([[1.1, .4], [-1.3, .7]], dtype=torch.float64)
    first = torch.func.jvp(layer, (xy,), (torch.ones_like(xy),))
    layer.cache_projection()
    cached = torch.func.jvp(layer, (xy,), (torch.ones_like(xy),))
    for expected, actual in zip(first[1], cached[1]):
        torch.testing.assert_close(actual, expected)
    layer.load_state_dict(layer.state_dict())
    assert layer._coefficient_cache is None


def test_basis_derivatives_and_parameter_gradients_match_spatial_ad():
    layer = field()
    with torch.no_grad():
        layer.kinematic_head.weight.normal_(std=.1)
        layer.stress_head.weight.normal_(std=.1)
    xy = torch.tensor([[1., 0.], [-1.3, .7], [0., -3.]], dtype=torch.float64)
    direct, direct_derivative = layer.spatial_fields(xy)
    automatic, automatic_derivative = IncrementFields.spatial_fields(layer, xy)
    for actual, expected in zip((*direct, *direct_derivative), (*automatic, *automatic_derivative)):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
    params = tuple(layer.parameters())
    loss = lambda values, derivatives: sum(v.square().sum() for v in (*values, *derivatives))
    direct_gradient = torch.autograd.grad(loss(direct, direct_derivative), params)
    automatic_gradient = torch.autograd.grad(loss(automatic, automatic_derivative), params)
    for actual, expected in zip(direct_gradient, automatic_gradient):
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("evaluation", [False, True])
@pytest.mark.parametrize("stress_correction", [False, True])
def test_circular_grid_matches_spatial_ad_and_parameter_gradients(evaluation, stress_correction):
    settings = dict(hidden=[8, 8], radius=3., load_steps=12, representation="spectral",
                    elastic_prefix="circular",
                    projection=dict(radial_order=6, angular_modes=8, angular_samples=32,
                                    centrosymmetric=True, stress_coefficient_correction=stress_correction),
                    quadrature=dict(radial=8, angular=32, boundary=64,
                                    stagger_angles=True, radial_power=1))
    case = dict(eh_over_ev=2.625, bedding_dip_deg=45., opening_axes=[1., 1.],
                opening_angle_deg=0.)
    pts, c, _, _ = geometry(case, settings, "cpu", evaluation=evaluation)
    layer = make_causal_model(settings, c, case).make_increment()
    with torch.no_grad():
        layer.kinematic_head.weight.normal_(std=.1)
        layer.stress_head.weight.normal_(std=.1)
        if stress_correction:
            layer.stress_spectral_correction.normal_(std=.01)
    n0, n1 = pts.n_pde, pts.n_pde+pts.n_wall
    nt = settings["quadrature"]["angular"]*(2 if evaluation else 1)
    grid = CircularGridEvaluation(layer, (pts.xy_all[:n0].reshape(-1, nt, 2),
                                          pts.xy_all[n0:n1][None], pts.xy_all[n1:][None]))
    actual = layer.spatial_fields(pts.xy_all, grid=grid, coefficients=layer.coefficients())
    expected = IncrementFields.spatial_fields(layer, pts.xy_all)
    for a, e in zip((*actual[0], *actual[1]), (*expected[0], *expected[1])):
        torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-12)
    parameters = tuple(layer.parameters())
    loss = lambda result: sum(v.square().sum() for group in result for v in group)
    actual_grad = torch.autograd.grad(loss(actual), parameters)
    expected_grad = torch.autograd.grad(loss(expected), parameters)
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-12)


def test_stress_correction_starts_at_same_field_and_retains_only_physical_modes():
    original = field()
    with torch.no_grad():
        original.stress_head.weight.normal_(std=.1)
    corrected = SpectralIncrementFields((8, 8), 3., 12, original.multiplier_scale,
        radial_order=8, angular_modes=6, angular_samples=32,
        centrosymmetric=True, stress_coefficient_correction=True).double()
    original.centrosymmetric = True
    state = corrected.state_dict()
    state.update(original.state_dict())
    corrected.load_state_dict(state)
    xy = torch.tensor([[1.1, .2], [-1.5, .7], [.4, -2.6]], dtype=torch.float64)
    for actual, expected in zip(corrected.spatial_fields(xy)[0], original.spatial_fields(xy)[0]):
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    with torch.no_grad():
        corrected.stress_spectral_correction[2, 2, 1, 1] = .7
        corrected.stress_spectral_correction[1, 0, 0, 0] = .2
        corrected.stress_spectral_correction[:, 3, :, :] = 5.  # Disallowed odd stress mode.
        corrected.stress_spectral_correction[:, 0, :, 1] = 9.  # Imaginary mean has no field.
    actual = corrected(xy)
    before = original(xy)
    xi = xy.norm(dim=-1)-2
    angle = torch.atan2(xy[:, 1], xy[:, 0])
    expected = torch.zeros_like(actual[1])
    expected[:, 0] = .2*xi
    expected[:, 1] = -1.4*(2*xi.square()-1)*(2*angle).sin()
    torch.testing.assert_close(actual[1]-before[1], expected)
    torch.testing.assert_close(corrected(-xy)[1], actual[1])
    for channel in (0, 2):
        torch.testing.assert_close(actual[channel], before[channel], rtol=0., atol=0.)


def test_circular_grid_retains_the_full_spectral_band():
    layer = field()
    radius = torch.tensor([1., 1.23, 3.], dtype=torch.float64)
    angle = 2*torch.pi*(torch.arange(32, dtype=radius.dtype)[None]
                       +torch.tensor([.17, .91, .5])[:, None])/32
    xy = radius[:, None, None]*torch.stack((angle.cos(), angle.sin()), -1)
    coefficients = torch.randn((9, 7, 8), dtype=torch.complex128, requires_grad=True)
    grid = CircularGridEvaluation(layer, (xy,))
    actual = grid(coefficients)
    expected = layer._spatial_expansion(coefficients, xy.flatten(0, 1))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-11)
    cotangents = [torch.randn_like(value) for value in actual]
    for result in (actual, expected):
        loss = sum((value*weight).sum() for value, weight in zip(result, cotangents))
        gradient, = torch.autograd.grad(loss, coefficients)
        if result is actual:
            actual_gradient = gradient
        else:
            torch.testing.assert_close(actual_gradient, gradient, rtol=1e-10, atol=1e-11)
