"""Transversely isotropic compliance, stiffness and exterior impedance."""

from dataclasses import dataclass, field
import numpy as np
def beta0_principal(Eh: float, Ev: float, nu_vh: float, nu_hh: float,
                    Gvh: float) -> np.ndarray:
    """Plane-strain reduced compliance in bedding axes, 3x3 symmetric
    on (s_11, s_22, s_12->engineering shear) = (beta_11, beta_12, beta_16; ...)."""
    b11 = (1.0 - nu_hh ** 2) / Eh
    b22 = 1.0 / Ev - nu_vh ** 2 * Eh / Ev ** 2
    b12 = -nu_vh * (1.0 + nu_hh) / Ev
    b66 = 1.0 / Gvh
    return np.array([[b11, b12, 0.0],
                     [b12, b22, 0.0],
                     [0.0, 0.0, b66]])

def rotate_beta_congruence(beta0: np.ndarray, phi: float) -> np.ndarray:
    """Section-axes compliance via the 3x3 Voigt congruence."""
    c, s = np.cos(phi), np.sin(phi)
    Q = np.array([[c * c, s * s, 2.0 * s * c],
                  [s * s, c * c, -2.0 * s * c],
                  [-s * c, s * c, c * c - s * s]])
    return Q.T @ beta0 @ Q

def _full_compliance_principal(Eh, Ev, nu_vh, nu_hh, Gvh) -> np.ndarray:
    """6x6 Voigt compliance, principal axes: x bedding-parallel (in section),
    y bedding-normal, z the out-of-section in-plane-isotropic direction.
    Voigt order (xx, yy, zz, yz, xz, xy), engineering shear."""
    a = np.zeros((6, 6))
    a[0, 0] = 1.0 / Eh
    a[1, 1] = 1.0 / Ev
    a[2, 2] = 1.0 / Eh
    a[0, 1] = a[1, 0] = -nu_vh / Ev
    a[1, 2] = a[2, 1] = -nu_vh / Ev
    a[0, 2] = a[2, 0] = -nu_hh / Eh
    a[3, 3] = 1.0 / Gvh
    a[4, 4] = 2.0 * (1.0 + nu_hh) / Eh
    a[5, 5] = 1.0 / Gvh
    return a

_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

_TENSOR_IDX = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3,
               (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}

def _voigt_to_tensor(a: np.ndarray) -> np.ndarray:
    S = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    I, J = _TENSOR_IDX[(i, j)], _TENSOR_IDX[(k, l)]
                    fac = (0.5 if I >= 3 else 1.0) * (0.5 if J >= 3 else 1.0)
                    S[i, j, k, l] = fac * a[I, J]
    return S

def _tensor_to_voigt(S: np.ndarray) -> np.ndarray:
    a = np.zeros((6, 6))
    for I, (i, j) in enumerate(_VOIGT_PAIRS):
        for J, (k, l) in enumerate(_VOIGT_PAIRS):
            fac = (2.0 if I >= 3 else 1.0) * (2.0 if J >= 3 else 1.0)
            a[I, J] = fac * S[i, j, k, l]
    return a

def full_compliance_global(Eh: float, Ev: float, nu_vh: float, nu_hh: float,
                           Gvh: float, phi: float) -> np.ndarray:
    """Rotated 3-D compliance in engineering-Voigt order.

    This is the full counterpart of the reduced plane-strain ``beta`` chain
    and is used by the incremental elastoplastic formulation to retain ``sigma_zz``
    and ``epsilon_zz^p`` while enforcing total ``epsilon_zz = 0``.
    """

    S = _voigt_to_tensor(
        _full_compliance_principal(Eh, Ev, nu_vh, nu_hh, Gvh)
    )
    c, s = np.cos(phi), np.sin(phi)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    rotated = np.einsum("pi,qj,rk,sl,ijkl->pqrs", R, R, R, R, S)
    return _tensor_to_voigt(rotated)

def rotate_beta_tensor(Eh, Ev, nu_vh, nu_hh, Gvh, phi: float) -> np.ndarray:
    """Section-axes reduced compliance via full tensor rotation about z by
    phi (material x axis ends up at +phi from the global x axis), then the
    plane-strain reduction beta_ij = a_ij - a_i3 a_j3 / a_33, i,j in {1,2,6}."""
    a = full_compliance_global(Eh, Ev, nu_vh, nu_hh, Gvh, phi)
    idx = {"1": 0, "2": 1, "6": 5}
    out = np.zeros((3, 3))
    for r, I in enumerate(("1", "2", "6")):
        for cc, J in enumerate(("1", "2", "6")):
            i, j = idx[I], idx[J]
            out[r, cc] = a[i, j] - a[i, 2] * a[j, 2] / a[2, 2]
    return out

def apparent_inv_E(beta: np.ndarray, psi: float) -> float:
    """1/E_apparent for uniaxial stress along the in-section direction at
    angle psi: v = (c^2, s^2, sc) in (sxx, syy, txy) Voigt, 1/E = v beta v."""
    c, s = np.cos(psi), np.sin(psi)
    v = np.array([c * c, s * s, s * c])
    return float(v @ beta @ v)

def mu_roots(beta: np.ndarray) -> tuple[complex, complex]:
    """Upper-half-plane roots of the Lekhnitskii quartic, ascending Im."""
    b11, b12, b16 = beta[0, 0], beta[0, 1], beta[0, 2]
    b22, b26, b66 = beta[1, 1], beta[1, 2], beta[2, 2]
    coeffs = [b11, -2.0 * b16, 2.0 * b12 + b66, -2.0 * b26, b22]
    up = sorted([z for z in np.roots(coeffs) if z.imag > 0.0],
                key=lambda z: z.imag)
    if len(up) != 2:
        raise ValueError(f"expected 2 upper-half-plane roots, got {up}")
    return complex(up[0]), complex(up[1])

def pq_coeffs(beta: np.ndarray, mu: complex) -> tuple[complex, complex]:
    b11, b12, b16 = beta[0, 0], beta[0, 1], beta[0, 2]
    b22, b26 = beta[1, 1], beta[1, 2]
    p = b11 * mu ** 2 + b12 - b16 * mu
    q = b12 * mu + b22 / mu - b26
    return complex(p), complex(q)

def lambda_matrix(mu1: complex, mu2: complex, p1: complex, q1: complex,
                  p2: complex, q2: complex,
                  row2: tuple[float, float] = (-1.0, -1.0)) -> np.ndarray:
    """Lambda = -i B M^{-1}, with the physical second row (-1, -1)."""
    B = np.array([[mu1, mu2], [row2[0], row2[1]]], dtype=complex)
    M = np.array([[p1, p2], [q1, q2]], dtype=complex)
    return -1j * (B @ np.linalg.inv(M))

@dataclass
class TIMaterial:
    """Material bundle: constants -> beta -> roots -> pq -> Lambda."""
    name: str
    Eh: float
    Ev: float
    nu_vh: float
    nu_hh: float
    Gvh: float
    phi: float = 0.0                     # radians, x axis -> bedding-parallel
    beta: np.ndarray = field(init=False)
    mu1: complex = field(init=False)
    mu2: complex = field(init=False)
    p1: complex = field(init=False)
    q1: complex = field(init=False)
    p2: complex = field(init=False)
    q2: complex = field(init=False)
    Lam: np.ndarray = field(init=False)  # 2x2 complex, -Lam Hermitian PD

    def __post_init__(self):
        self.beta = rotate_beta_congruence(
            beta0_principal(self.Eh, self.Ev, self.nu_vh, self.nu_hh,
                            self.Gvh), self.phi)
        self.mu1, self.mu2 = mu_roots(self.beta)
        self.p1, self.q1 = pq_coeffs(self.beta, self.mu1)
        self.p2, self.q2 = pq_coeffs(self.beta, self.mu2)
        self.Lam = lambda_matrix(self.mu1, self.mu2, self.p1, self.q1,
                                 self.p2, self.q2)
        self._assert_neg_lambda_pd()

    def _assert_neg_lambda_pd(self) -> None:
        """Check that the elastic surface impedance -Lambda is Hermitian positive definite."""
        L = self.Lam
        herm = float(np.max(np.abs(L - L.conj().T)) / np.max(np.abs(L)))
        if herm > 1e-10:
            raise AssertionError(
                f"{self.name}: Lambda Hermitian defect {herm:.3e} > 1e-10")
        w = np.linalg.eigvalsh(-(L + L.conj().T) / 2.0)
        if float(w[0]) <= 0.0:
            raise AssertionError(
                f"{self.name}: -Lambda not positive definite, "
                f"eigenvalues {w}")

    @property
    def lam_minmax(self) -> tuple[float, float]:
        """Eigenvalues of -Lambda (Hermitian part), the two bound constants;
        independent of the harmonic number."""
        w = np.linalg.eigvalsh(-(self.Lam + self.Lam.conj().T) / 2.0)
        return float(w[0]), float(w[-1])

    def stiffness(self) -> np.ndarray:
        """Plane-strain stiffness D = beta^{-1} (3x3, engineering shear)."""
        return np.linalg.inv(self.beta)

    def full_compliance(self) -> np.ndarray:
        """Rotated 3-D engineering-Voigt compliance."""

        return full_compliance_global(
            self.Eh, self.Ev, self.nu_vh, self.nu_hh, self.Gvh, self.phi
        )

    def plane_strain_stiffness4(self) -> np.ndarray:
        """Stress/strain map on ``(xx, yy, zz, xy)`` with other shears zero."""

        full_stiffness = np.linalg.inv(self.full_compliance())
        indices = (0, 1, 2, 5)
        return full_stiffness[np.ix_(indices, indices)]

    def plane_strain_sigma_zz(self, sxx: float, syy: float,
                              txy: float) -> float:
        """Out-of-plane stress that makes total ``epsilon_zz`` zero."""

        compliance = self.full_compliance()
        numerator = (compliance[2, 0] * sxx + compliance[2, 1] * syy
                     + compliance[2, 5] * txy)
        return float(-numerator / compliance[2, 2])
