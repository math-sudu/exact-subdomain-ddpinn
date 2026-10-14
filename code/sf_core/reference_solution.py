"""Closed-form reference solution: circular tunnel in an infinite
transversely isotropic medium under uniform in-situ stress, traction-free
excavation wall.

The perturbation field (excavation-induced part, sigma = sigma0 + sigma~) is
a decaying exterior Lekhnitskii field for the circle r = a.  For uniform
in-situ stress the boundary forcing has finite harmonic content in the mapped
plane, so the exact solution lies in the span of the zeta^{-m} potentials and
the least-squares collocation solve below recovers it to machine precision —
this is a closed form evaluated numerically, not a truncated approximation
(asserted: wall residual at held-out angles and trailing-coefficient decay).

Error metrics for the pilot are computed against THIS field.  Never against a
truncated numerical solution: truncation carries a modeling bias that does not
vanish under refinement and fakes plateaus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .exterior_field import ExteriorField
from .material import TIMaterial

__all__ = ["HoleProblem", "solve_circular_hole"]


@dataclass
class HoleProblem:
    mat: TIMaterial
    a: float                 # excavation radius
    sx: float                # in-situ sigma_xx, tension positive
    sy: float
    txy: float
    field: ExteriorField     # perturbation field (decaying)
    wall_residual_rel: float # held-out wall-BC residual / load scale
    coeff_tail_rel: float    # max |c_{k,m>=2}| / max |c_{k,1}|

    def released_traction(self, th: np.ndarray) -> np.ndarray:
        """f = -sigma0 . rhat on the wall: the traction the perturbation
        field must carry there (excavation release), 2 x n."""
        c, s = np.cos(th), np.sin(th)
        return -np.stack([self.sx * c + self.txy * s,
                          self.txy * c + self.sy * s])

    def total_hoop_wall(self, th: np.ndarray) -> np.ndarray:
        """Total tangential stress on the excavation wall."""
        x, y = self.a * np.cos(th), self.a * np.sin(th)
        sxx, syy, sxy = self.field.stresses(x, y)
        sxx = sxx + self.sx
        syy = syy + self.sy
        sxy = sxy + self.txy
        tx, ty = -np.sin(th), np.cos(th)
        return sxx * tx * tx + 2.0 * sxy * tx * ty + syy * ty * ty


def solve_circular_hole(mat: TIMaterial, a: float, sx: float, sy: float,
                        txy: float, n_modes: int = 8, n_col: int = 128
                        ) -> HoleProblem:
    """Least-squares collocation on the wall traction condition
    sigma~ . rhat = -sigma0 . rhat  at r = a (rhat pointing away from the
    tunnel; equivalent to zero total traction on the excavation face)."""
    th = 2.0 * np.pi * (np.arange(n_col) + 0.31) / n_col   # offset: no axis lock
    c, s = np.cos(th), np.sin(th)
    rhs = -np.stack([sx * c + txy * s, txy * c + sy * s]).ravel()

    def traction_of(coeffs) -> np.ndarray:
        f = ExteriorField(mat, a, coeffs)
        tx, ty = f.traction(a * c, a * s, c, s)
        return np.concatenate([tx, ty])

    cols = []
    for k in (0, 1):
        for m in range(1, n_modes + 1):
            for unit in (1.0, 1.0j):
                cf = [np.zeros(n_modes, complex), np.zeros(n_modes, complex)]
                cf[k][m - 1] = unit
                cols.append(traction_of((cf[0], cf[1])))
    A = np.stack(cols, axis=1)
    x, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    coeffs = (np.zeros(n_modes, complex), np.zeros(n_modes, complex))
    i = 0
    for k in (0, 1):
        for m in range(1, n_modes + 1):
            coeffs[k][m - 1] = x[i] + 1j * x[i + 1]
            i += 2
    field = ExteriorField(mat, a, coeffs)

    # held-out wall residual (different angles than the collocation set)
    th2 = 2.0 * np.pi * (np.arange(4 * n_col) + 0.5) / (4 * n_col)
    c2, s2 = np.cos(th2), np.sin(th2)
    tx, ty = field.traction(a * c2, a * s2, c2, s2)
    rx = tx - (-(sx * c2 + txy * s2))
    ry = ty - (-(txy * c2 + sy * s2))
    scale = max(abs(sx), abs(sy), abs(txy), 1e-300)
    resid = float(np.max(np.hypot(rx, ry)) / scale)

    lead = max(np.max(np.abs(coeffs[0][:1])), np.max(np.abs(coeffs[1][:1])))
    tail = 0.0
    if n_modes > 1:
        tail = max(np.max(np.abs(coeffs[0][1:])), np.max(np.abs(coeffs[1][1:])))
    return HoleProblem(mat, a, sx, sy, txy, field, resid,
                       float(tail / lead))
