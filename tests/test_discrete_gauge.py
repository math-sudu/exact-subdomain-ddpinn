"""Complex endpoint convention and work identity on the training quadrature."""

from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from pilot.discrete_gauge import DiscreteGaugeInfluence
from test_virtual_work import setup, plastic


def test_endpoint_functional_matches_direct_physical_displacement():
    metric, _ = setup()
    angle = .37
    direction = np.array([np.cos(angle), np.sin(angle)])
    influence = DiscreteGaugeInfluence(metric, np.stack((-direction, direction)))
    generator = torch.Generator().manual_seed(431)
    raw = torch.randn((5, 7, 2, 2), generator=generator, dtype=torch.float64)
    coefficients = torch.view_as_complex(raw)
    coefficients[:, ::2] = 0
    vector = metric.correction_vector(coefficients)
    # Evaluate the actual positive-mode displacement representation at the wall.
    modes = np.arange(7)
    wall = (-1.) ** np.arange(5)
    endpoint_u = 2 * np.real(np.einsum("pm,j,jmc->pc",
        np.exp(1j * np.array([angle + np.pi, angle])[:, None] * modes),
        wall, coefficients.numpy()))
    expected = -(endpoint_u[1] - endpoint_u[0]) @ direction
    np.testing.assert_allclose(torch.vdot(influence.functional, vector).real, expected, atol=2e-12)
    torch.testing.assert_close(metric.matrix.mH @ influence.adjoint, influence.functional,
                               atol=2e-11, rtol=2e-11)


def test_adjoint_density_and_residual_recover_endpoint_work():
    metric, points = setup()
    xy = points.xy_all[:points.n_pde]
    source = plastic(xy)
    load = metric.source_load(source)
    op = metric.operator
    angles = torch.atan2(xy[:, 1], xy[:, 0]).reshape(metric.nr, metric.nt).numpy()
    generator = torch.Generator().manual_seed(73)
    coefficients = torch.view_as_complex(torch.randn((5, 7, 2, 2),
        generator=generator, dtype=torch.float64))
    vector = metric.correction_vector(coefficients)
    residual = metric.matrix @ vector - load
    for angle in (0., np.pi / 2, .71):
        direction = np.array([np.cos(angle), np.sin(angle)])
        influence = DiscreteGaugeInfluence(metric, np.stack((-direction, direction)))
        stress = influence.stress_on_rings(op.r, angles)
        work = np.einsum("rtc,rtc,r->", stress,
            source.numpy().reshape(metric.nr, metric.nt, 4), op.w / metric.nt)
        expected = torch.vdot(influence.adjoint, load).real
        np.testing.assert_allclose(work, expected, atol=2e-11, rtol=2e-11)
        endpoint = torch.vdot(influence.functional, vector).real
        torch.testing.assert_close(endpoint, expected + torch.vdot(influence.adjoint, residual).real,
                                   atol=2e-11, rtol=2e-11)
