"""Work-conjugate circular endpoint gauge in the retained Fourier trace space.

Coordinates and fields use the pilot nondimensionalization. The input material
has moduli in Pa; its stiffness is divided by E_REF. Unit auxiliary force is
conjugate to shortening D=-(u(x2)-u(x1)).e12. The finite modal load is exact for
traces with modes 0..modes, rather than a smoothed point-force prescription.
"""

from operator import index

import numpy as np

from . import config as C
from sf_core.exterior_field import ExteriorField
from sf_core.material import TIMaterial


class CircularGaugeInfluence:
    """Auxiliary field outside the circle through two distinct endpoints."""

    def __init__(self, material, endpoints, modes):
        self.endpoints = np.asarray(endpoints, dtype=float).copy()
        self.modes = index(modes)
        if self.endpoints.shape != (2, 2) or not np.isfinite(self.endpoints).all():
            raise ValueError("The gauge needs two finite Cartesian endpoints")
        radii = np.linalg.norm(self.endpoints, axis=1)
        if radii[0] <= 0 or not np.isclose(radii[0], radii[1], rtol=1e-12, atol=0):
            raise ValueError("The endpoints must lie on the same nonzero circle")
        chord = self.endpoints[1] - self.endpoints[0]
        length = np.linalg.norm(chord)
        if length == 0 or self.modes < 1:
            raise ValueError("The endpoints must differ and modes must be positive")
        self.radius = float(radii[0])
        self.direction = chord / length
        self.material = TIMaterial(
            "gauge", material.Eh / C.E_REF, material.Ev / C.E_REF,
            material.nu_vh, material.nu_hh, material.Gvh / C.E_REF, material.phi)

        angles = np.arctan2(self.endpoints[:, 1], self.endpoints[:, 0])
        harmonics = np.arange(1, self.modes + 1)
        phase = np.exp(-1j * harmonics * angles[0]) - np.exp(-1j * harmonics * angles[1])
        self.traction_coefficients = np.zeros((2, self.modes + 1), dtype=complex)
        self.traction_coefficients[:, 1:] = self.direction[:, None] * phase / (2 * np.pi * self.radius)
        # Material-side normal is -e_r, so K_m=-(m/a) conjugate(Lambda).
        stiffness = -self.material.Lam.conj()
        self.displacement_coefficients = np.zeros_like(self.traction_coefficients)
        self.displacement_coefficients[:, 1:] = np.linalg.solve(
            stiffness, self.traction_coefficients[:, 1:]) * (self.radius / harmonics)
        material_map = np.array([[self.material.p1, self.material.p2],
                                 [self.material.q1, self.material.q2]])
        # ExteriorField uses potentials proportional to exp(-i*m*theta).
        potentials = np.linalg.solve(material_map, self.displacement_coefficients[:, 1:].conj())
        self.field = ExteriorField(self.material, self.radius, (potentials[0], potentials[1]))
        compliance = self.material.full_compliance()
        self._zz_map = -compliance[2, [0, 1, 5]] / compliance[2, 2]

    def _coordinates(self, xy):
        xy = np.asarray(xy, dtype=float)
        if xy.shape[-1:] != (2,) or not np.isfinite(xy).all():
            raise ValueError("Coordinates must have a finite final Cartesian axis")
        if np.any(np.linalg.norm(xy, axis=-1) < self.radius * (1 - 1e-12)):
            raise ValueError("The auxiliary field is defined outside the cavity")
        return xy[..., 0], xy[..., 1]

    def displacement(self, xy):
        """Return nondimensional Cartesian displacement, shape (..., 2)."""
        return np.stack(self.field.displacements(*self._coordinates(xy)), axis=-1)

    def stress(self, xy):
        """Return (xx, yy, zz, xy), conjugate to engineering plastic strain.

        Source-work density is sum(stress * epsilon_p, axis=-1); no extra
        factor multiplies xy when epsilon_p[..., 3] is engineering shear.
        """
        sxx, syy, sxy = self.field.stresses(*self._coordinates(xy))
        szz = np.stack((sxx, syy, sxy), axis=-1) @ self._zz_map
        return np.stack((sxx, syy, szz, sxy), axis=-1)
