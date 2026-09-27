"""The projection layer resolves both field values and spatial derivatives."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from pilot.spectral_mixed import SpectralIncrementFields
from pilot.causal_mixed import IncrementFields


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
