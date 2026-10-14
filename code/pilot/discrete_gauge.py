"""Endpoint adjoint in the actual exterior-coupled displacement test space."""

import numpy as np
import torch


class DiscreteGaugeInfluence:
    """Use the work metric's H and coefficient convention without changing u."""

    def __init__(self, metric, endpoints):
        endpoints = np.asarray(endpoints, dtype=float)
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
            raise ValueError("The gauge needs two finite Cartesian endpoints")
        if not np.allclose(np.linalg.norm(endpoints, axis=1), 1., rtol=1e-12, atol=1e-12):
            raise ValueError("The discrete gauge endpoints must lie on the unit cavity")
        chord = endpoints[1] - endpoints[0]
        if np.linalg.norm(chord) == 0:
            raise ValueError("The gauge endpoints must differ")
        direction = chord / np.linalg.norm(chord)
        angles = np.arctan2(endpoints[:, 1], endpoints[:, 0])
        self.operator = metric.operator
        self.modes = metric.modes.detach().cpu().numpy()
        wall_basis = (-1.) ** np.arange(self.operator.j + 1)
        phase = np.exp(1j * self.modes * angles[0]) - np.exp(1j * self.modes * angles[1])
        # D_c = Re(g^H d), with both positive and negative Fourier modes in d.
        functional = phase[:, None, None] * wall_basis[None, :, None] * direction
        self.functional = torch.as_tensor(functional.conj().reshape(-1), device=metric.matrix.device)
        self.adjoint = torch.cholesky_solve(self.functional[:, None], metric.factor)[:, 0]
        self.coefficients = self.adjoint.detach().cpu().numpy().reshape(len(self.modes), self.operator.j + 1, 2)

    def stress_on_rings(self, radius, angles):
        """Return (xx, yy, zz, xy) stress on uniform or staggered circular rings."""
        radius = np.asarray(radius)
        angles = np.broadcast_to(angles, (len(radius), np.shape(angles)[-1]))
        op = self.operator
        coordinate = 2 * (radius - 1) / (op.rg - 1) - 1
        basis = np.polynomial.chebyshev.chebvander(coordinate, op.j)
        derivative = np.polynomial.chebyshev.chebder(np.eye(op.j + 1), axis=0)
        dbasis = np.polynomial.chebyshev.chebval(coordinate, derivative).T * 2 / (op.rg - 1)
        strain = np.zeros((*angles.shape, 4), dtype=complex)
        for shift in (-1, 1):
            radial = (dbasis[:, None] - shift * self.modes[None, :, None]
                      * basis[:, None] / radius[:, None, None]) / 2
            values = np.einsum("rmj,mjc->rmc", radial, self.coefficients)
            terms = np.zeros((*values.shape[:-1], 4), dtype=complex)
            terms[..., 0] = values[..., 0]
            terms[..., 1] = -1j * shift * values[..., 1]
            terms[..., 3] = -1j * shift * values[..., 0] + values[..., 1]
            phase = np.exp(1j * angles[..., None] * (self.modes + shift))
            strain += np.einsum("rtm,rmc->rtc", phase, terms)
        return strain.real @ op.c.T
