"""Operator conventions and geometry used by the paired interface study."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from interface_extension import geometry, metric_value, opening_boundary


def case():
    return dict(name="ellipse", eh_over_ev=2.625, bedding_dip_deg=45.,
                opening_axes=[1.4, 1/1.4], opening_angle_deg=30.)


def settings():
    return dict(radius=3., quadrature=dict(radial=16, angular=64, boundary=256))


def test_compliance_matches_existing_energy_and_gradient():
    _, _, _, sf = geometry(case(), settings(), "cpu")
    torch.manual_seed(4)
    residual = torch.randn(sf.n, 2, dtype=torch.float64, requires_grad=True)
    value = metric_value(sf, residual, "compliance")
    expected = sf.dual_metric(residual.T) / (4 * np.pi * sf.R**2)
    torch.testing.assert_close(value, expected, rtol=1e-13, atol=1e-14)
    a = torch.autograd.grad(value, residual, retain_graph=True)[0]
    b = torch.autograd.grad(expected, residual)[0]
    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-14)


def test_uniform_metric_is_projected_component_mean_square():
    _, _, _, sf = geometry(case(), settings(), "cpu")
    theta = torch.arange(sf.n, dtype=torch.float64) * 2 * torch.pi / sf.n
    r = torch.stack((1 + 2 * theta.cos(), 3 * (7 * theta).sin()), -1)
    torch.testing.assert_close(metric_value(sf, r, "uniform"), r.square().mean())
    torch.testing.assert_close(metric_value(sf, r, "wavelength"),
                               torch.tensor(0.5 + 1 + 2.25/7, dtype=torch.float64))


def test_rotated_ellipse_area_and_boundary_normals():
    pts, _, s0, _ = geometry(case(), settings(), "cpu")
    assert abs(pts.area - np.pi * (9 - 1)) < 1e-9
    angle = np.linspace(0, 2*np.pi, 117, endpoint=False)
    xy, normal, _ = opening_boundary(angle, [1.4, 1/1.4], 30.)
    eps = 1e-5
    plus = opening_boundary(angle+eps, [1.4, 1/1.4], 30.)[0]
    minus = opening_boundary(angle-eps, [1.4, 1/1.4], 30.)[0]
    assert np.max(abs(np.sum(normal*(plus-minus)/(2*eps), axis=1))) < 1e-8
    assert np.all(np.sum(xy*normal, axis=1) > 0)
    np.testing.assert_allclose(pts.wall_weights.sum().item(), 1)
    np.testing.assert_allclose(pts.interior_weights.sum().item(), 1)
    assert s0.shape == (4,)


def test_staggered_rings_preserve_area_and_centroid():
    elliptical = dict(case(), quadrature=dict(radial=32, angular=128, boundary=512, stagger_angles=True))
    pts, _, _, _ = geometry(elliptical, settings(), "cpu")
    xy = pts.xy_all[:pts.n_pde].numpy()
    weights = pts.interior_weights.numpy()
    np.testing.assert_allclose(pts.area, np.pi*8, rtol=1e-13)
    np.testing.assert_allclose((xy*weights[:, None]).sum(0), [0, 0], atol=1e-14)
    angle = np.arctan2(xy[:, 1], xy[:, 0]).reshape(32, 128)
    assert np.ptp(angle[:, 0]) > .02


def test_graded_rings_integrate_area_and_resolve_near_wall():
    circular = dict(case(), opening_axes=[1., 1.],
                    quadrature=dict(radial=32, angular=256, boundary=1024,
                                    stagger_angles=True, radial_power=2))
    pts, _, _, _ = geometry(circular, settings(), "cpu")
    radius = pts.xy_all[:pts.n_pde].norm(dim=-1).numpy()
    weights = pts.interior_weights.numpy()
    np.testing.assert_allclose(pts.area, 8*np.pi, rtol=1e-13)
    np.testing.assert_allclose(np.dot(weights, radius**2), 5., rtol=1e-13)
    assert np.count_nonzero(radius < 1.05) > 1000
