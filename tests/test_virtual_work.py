"""Virtual work uses actual PINN fields and differentiates both physical routes."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"code"))
from interface_extension import geometry
from pilot import config as C
from pilot.virtual_work import VirtualWorkMetric


def setup():
    settings = json.loads((ROOT/"configs/orientation_spectral.json").read_text())
    settings["projection"].update(radial_order=4, angular_modes=6, angular_samples=32)
    settings["quadrature"].update(radial=8, angular=32, boundary=64)
    case = settings["cases"][0]
    pts, c, _, _ = geometry(case, settings, "cpu")
    material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
    metric = VirtualWorkMetric(c, C.nondim_lambda(material), settings, pts)
    return metric, pts


def plastic(xy):
    radius = xy.norm(dim=-1)
    angle = torch.atan2(xy[:, 1], xy[:, 0])
    return torch.stack((radius*torch.cos(2*angle), radius.square()*torch.sin(4*angle),
                        .3*torch.cos(2*angle), .4*radius*torch.sin(2*angle)), -1)


def test_staggered_load_matches_unshifted_work_including_plastic_zz():
    metric, pts = setup()
    actual = metric.source_load(plastic(pts.xy_all[:pts.n_pde]))
    op = metric.operator
    xy = (op.r[:, None, None]*np.stack((np.cos(op.theta), np.sin(op.theta)), -1)[None])
    source = plastic(torch.tensor(xy.reshape(-1, 2))).numpy().reshape(metric.nr, metric.nt, 1, 4)
    expected = op.source_load(source)[:op.ndof, 0].reshape(len(op.modes), op.block)[op.modes % 2 != 0]
    torch.testing.assert_close(actual, torch.tensor(expected.ravel()), atol=2e-12, rtol=2e-12)


def test_energy_equals_work_of_represented_error_and_differentiates_both_routes():
    metric, pts = setup()
    generator = torch.Generator().manual_seed(71)
    raw = torch.randn((5, 7, 2, 2), generator=generator, dtype=torch.float64)
    raw.requires_grad_()
    source = plastic(pts.xy_all[:pts.n_pde]).requires_grad_()
    coefficients = torch.view_as_complex(raw)
    energy = metric(coefficients, source)
    residual = metric.matrix@metric.correction_vector(coefficients)-metric.source_load(source)
    error = torch.linalg.solve(metric.matrix, residual)
    expected = torch.vdot(error, metric.matrix@error).real/metric.area
    torch.testing.assert_close(energy, expected, atol=1e-9, rtol=1e-12)
    gradients = torch.autograd.grad(energy, (raw, source))
    for parameter, gradient in zip((raw, source), gradients):
        direction = torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
        direction /= direction.norm()
        h = 1e-4
        if parameter is raw:
            plus = metric(torch.view_as_complex(raw+h*direction), source)
            minus = metric(torch.view_as_complex(raw-h*direction), source)
        else:
            plus = metric(coefficients, source+h*direction)
            minus = metric(coefficients, source-h*direction)
        torch.testing.assert_close((plus-minus)/(2*h), (gradient*direction).sum(),
                                   atol=1e-6, rtol=2e-6)
        assert gradient.norm() > 0


def test_exact_elastic_history_has_zero_work_residual():
    metric, pts = setup()
    coefficients = torch.zeros((5, 7, 2), dtype=torch.complex128)
    source = torch.zeros((pts.n_pde, 4), dtype=torch.float64)
    assert metric(coefficients, source).item() == 0
