"""Lekhnitskii exterior fields for the region outside a circle.

Representation: stress function F = 2 Re sum_k F_k(z_k), z_k = x + mu_k y,
Phi_k := dF_k/dz_k.  Then

    sxx =  2 Re sum_k mu_k^2 Phi_k'(z_k)
    syy =  2 Re sum_k        Phi_k'(z_k)
    sxy = -2 Re sum_k mu_k   Phi_k'(z_k)
    ux  =  2 Re sum_k p_k Phi_k(z_k)
    uy  =  2 Re sum_k q_k Phi_k(z_k)

Decaying exterior solutions for the circle |z| = R use the conformal maps

    z_k = (R/2) [ (1 - i mu_k) zeta + (1 + i mu_k) / zeta ],

univalent on |zeta| >= 1, with the boundary r = R at zeta = e^{i theta}, and
potentials Phi_k(zeta) = sum_{m>=1} c_{k,m} zeta^{-m}.

This module never touches Lambda: it is the independent verification path for
the closed-form modal operator (and the builder of the closed-form reference
solution).  Everything numpy/float64, vectorized over evaluation points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .material import TIMaterial

__all__ = ["zeta_of_z", "ExteriorField"]


def _zmap(mu: complex, R: float, zeta: np.ndarray) -> np.ndarray:
    return (R / 2.0) * ((1.0 - 1j * mu) * zeta + (1.0 + 1j * mu) / zeta)


def _dz_dzeta(mu: complex, R: float, zeta: np.ndarray) -> np.ndarray:
    return (R / 2.0) * ((1.0 - 1j * mu) - (1.0 + 1j * mu) / zeta ** 2)


def zeta_of_z(mu: complex, R: float, z: np.ndarray) -> np.ndarray:
    """Invert the exterior map: the root of
    (R/2)(1-i mu) zeta^2 - z zeta + (R/2)(1+i mu) = 0 with |zeta| >= 1.
    The two roots multiply to (1+i mu)/(1-i mu), modulus < 1 for Im mu > 0,
    so exactly one root lies on or outside the unit circle."""
    a = (R / 2.0) * (1.0 - 1j * mu)
    c = (R / 2.0) * (1.0 + 1j * mu)
    disc = np.sqrt(z * z - 4.0 * a * c)
    r1 = (z + disc) / (2.0 * a)
    r2 = (z - disc) / (2.0 * a)
    return np.where(np.abs(r1) >= np.abs(r2), r1, r2)


@dataclass
class ExteriorField:
    """A decaying exterior elastic field outside the circle of radius R,
    given by coefficient arrays coeffs[k][m-1] = c_{k,m}, k = 0, 1."""
    mat: TIMaterial
    R: float
    coeffs: tuple[np.ndarray, np.ndarray]   # complex arrays, index m-1

    def _phi_phiprime(self, k: int, z: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        mu = (self.mat.mu1, self.mat.mu2)[k]
        zeta = zeta_of_z(mu, self.R, z)
        c = self.coeffs[k]
        phi = np.zeros_like(zeta)
        dphi_dzeta = np.zeros_like(zeta)
        zinv = 1.0 / zeta
        zpow = zinv.copy()                       # zeta^{-m}
        for m1, cm in enumerate(c):
            m = m1 + 1
            phi = phi + cm * zpow
            dphi_dzeta = dphi_dzeta - m * cm * zpow * zinv
            zpow = zpow * zinv
        dphi_dz = dphi_dzeta / _dz_dzeta(mu, self.R, zeta)
        return phi, dphi_dz

    def displacements(self, x: np.ndarray, y: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        ux = np.zeros_like(x, dtype=float)
        uy = np.zeros_like(x, dtype=float)
        for k, (p, q) in enumerate(((self.mat.p1, self.mat.q1),
                                    (self.mat.p2, self.mat.q2))):
            mu = (self.mat.mu1, self.mat.mu2)[k]
            phi, _ = self._phi_phiprime(k, x + mu * y)
            ux += 2.0 * np.real(p * phi)
            uy += 2.0 * np.real(q * phi)
        return ux, uy

    def stresses(self, x: np.ndarray, y: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sxx = np.zeros_like(x, dtype=float)
        syy = np.zeros_like(x, dtype=float)
        sxy = np.zeros_like(x, dtype=float)
        for k in (0, 1):
            mu = (self.mat.mu1, self.mat.mu2)[k]
            _, dphi = self._phi_phiprime(k, x + mu * y)
            sxx += 2.0 * np.real(mu * mu * dphi)
            syy += 2.0 * np.real(dphi)
            sxy -= 2.0 * np.real(mu * dphi)
        return sxx, syy, sxy

    def traction(self, x: np.ndarray, y: np.ndarray, nx: np.ndarray,
                 ny: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sxx, syy, sxy = self.stresses(x, y)
        return sxx * nx + sxy * ny, sxy * nx + syy * ny

    def circle_traces(self, r: float, n_theta: int
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(theta, u (2 x n), t (2 x n)) on the circle of radius r >= R,
        with t = sigma . rhat: the traction the exterior transmits inward
        across that circle (outward-of-near-field normal convention)."""
        th = 2.0 * np.pi * np.arange(n_theta) / n_theta
        x, y = r * np.cos(th), r * np.sin(th)
        ux, uy = self.displacements(x, y)
        tx, ty = self.traction(x, y, np.cos(th), np.sin(th))
        return th, np.stack([ux, uy]), np.stack([tx, ty])
