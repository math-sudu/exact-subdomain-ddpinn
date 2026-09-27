"""Differentiable exterior traction operator and inverse-stiffness norm."""

import numpy as np
import torch


class SfCircle:
    """S_f on the uniform circle grid of n_theta points, radius R (nondim),
    plus the dual-norm metric, as torch operations."""

    def __init__(self, Lam_nondim: np.ndarray, R: float, n_theta: int,
                 device, dtype=torch.float32, m_max: int | None = None):
        self.R = float(R)
        self.n = int(n_theta)
        self.m_max = int(m_max) if m_max is not None else self.n // 2 - 1
        if not 1 <= self.m_max < self.n // 2:
            raise ValueError("m_max must lie in [1, n_theta/2)")
        cdtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        m_all = np.arange(self.n // 2 + 1)
        keep = ((m_all >= 1) & (m_all <= self.m_max)
                & (m_all != self.n // 2))  # drop m=0, high modes, Nyquist

        # apply-side multiplier: a_t[m] = (m/R) conj(Lam) a_u[m]
        mult = np.zeros((self.n // 2 + 1, 2, 2), dtype=np.complex128)
        mult[keep] = (m_all[keep, None, None] / self.R) * np.conj(Lam_nondim)
        self.mult = torch.as_tensor(mult, dtype=cdtype, device=device)

        # metric-side quadratic form: w_m * conj((-Lam)^{-1}), m=0 -> m=1
        Qc = np.conj(np.linalg.inv(-Lam_nondim))
        w = np.zeros(self.n // 2 + 1)
        w[keep] = 4.0 * np.pi * self.R ** 2 / m_all[keep]
        w[0] = 2.0 * np.pi * self.R ** 2          # mean term, m=1 matrix weight
        self.Qc = torch.as_tensor(Qc, dtype=cdtype, device=device)
        self.wm = torch.as_tensor(w, dtype=dtype, device=device)

        # energy-pairing side: <(-S_f)u, u> = 4 pi sum m u^*(-Lambda)u
        we = np.zeros(self.n // 2 + 1)
        we[keep] = 4.0 * np.pi * m_all[keep]
        self.Qe = torch.as_tensor(np.conj(-Lam_nondim), dtype=cdtype,
                                  device=device)
        self.we = torch.as_tensor(we, dtype=dtype, device=device)

    def coeffs(self, f: torch.Tensor) -> torch.Tensor:
        """f: (2, n_theta) real -> a_m (2, n//2+1) complex."""
        return torch.fft.rfft(f, dim=1) / self.n

    def apply(self, u: torch.Tensor) -> torch.Tensor:
        """S_f u on the circle: (2, n_theta) -> (2, n_theta)."""
        a = self.coeffs(u)
        at = torch.einsum("mij,jm->im", self.mult, a)
        return torch.fft.irfft(at * self.n, n=self.n, dim=1)

    def dual_metric(self, r: torch.Tensor) -> torch.Tensor:
        """||r||_*^2, scalar, differentiable, parameter-free."""
        a = self.coeffs(r)
        qa = torch.einsum("ij,jm->im", self.Qc, a)
        per_m = torch.sum(torch.conj(a) * qa, dim=0).real
        return torch.sum(self.wm * per_m)

    def energy_pairing(self, u: torch.Tensor) -> torch.Tensor:
        """<(-S_f) u, u> = 4 pi sum_{m>=1} m u-hat_m^* (-Lambda) u-hat_m:
        twice the exterior strain energy of the trace u.  R-free (density
        pairing), positive semidefinite, differentiable, parameter-free.
        m = 0 and the Nyquist bin are dropped as in apply()."""
        a = self.coeffs(u)
        qa = torch.einsum("ij,jm->im", self.Qe, a)
        per_m = torch.sum(torch.conj(a) * qa, dim=0).real
        return torch.sum(self.we * per_m)
