"""Differentiable physical imbalance in the exterior-coupled energy norm.

The circular elastic background balances the applied loading exactly. The
remaining work residual is A*u_correction-F_plastic, using the PINN's own
displacement correction and cumulative plastic strain. The fixed inverse is
used only as a residual metric; it does not replace the network displacement.
"""

import numpy as np
import torch

from decompose_eigenstrain_response import FrozenSourceOperator


class VirtualWorkMetric:
    def __init__(self, stiffness, impedance, settings, points):
        projection = settings["projection"]
        if settings.get("elastic_prefix") != "circular" or not projection.get("centrosymmetric"):
            raise ValueError("The correction work metric requires the circular background and parity")
        nr, nt = (settings["quadrature"][name] for name in ("radial", "angular"))
        if settings["quadrature"].get("radial_power") != 1:
            raise ValueError("The work metric uses Gauss integration in the physical radius")
        self.operator = FrozenSourceOperator(
            stiffness.detach().cpu().numpy(), impedance, settings["radius"],
            projection["radial_order"], projection["angular_modes"], nr, nt)
        op = self.operator
        self.nr, self.nt = nr, nt
        self.stiffness = stiffness
        self.area = np.pi*(settings["radius"]**2-1)
        modes = op.modes[op.modes % 2 != 0]
        indices = ((modes[:, None]+op.m)*op.block + np.arange(op.block)[None]).ravel()
        matrix = op.matrix[indices][:, indices].toarray()
        self.matrix = torch.as_tensor(matrix, device=stiffness.device)
        self.factor = torch.linalg.cholesky(self.matrix)
        self.modes = torch.as_tensor(modes, device=stiffness.device)
        frequencies = modes[:, None]+np.array([-1, 1])[None]
        self.fourier_indices = torch.as_tensor(frequencies % nt, device=stiffness.device)
        basis = np.stack([[op.strains[m, shift] for shift in (-1, 1)] for m in modes])
        basis = basis.conj()*op.w[None, None, :, None, None]
        self.virtual_strains = torch.as_tensor(basis, device=stiffness.device)
        interior = points.xy_all[:points.n_pde].reshape(nr, nt, 2)
        origin = torch.atan2(interior[:, 0, 1], interior[:, 0, 0])
        frequencies = torch.as_tensor(frequencies, device=stiffness.device)
        self.angular_shift = torch.exp(-1j*origin[:, None, None]*frequencies[None])

    def source_load(self, plastic_strain):
        """Integrate the complete engineering plastic strain, including zz."""
        source = (plastic_strain@self.stiffness.T).reshape(self.nr, self.nt, 4)
        fourier = torch.fft.fft(source, dim=1)/self.nt
        shifted = fourier[:, self.fourier_indices]*self.angular_shift[..., None]
        return torch.einsum("mhrcb,rmhc->mb", self.virtual_strains, shifted).reshape(-1)

    def correction_vector(self, coefficients):
        """Convert the physical positive-frequency u coefficients to both signs."""
        values = coefficients[:, self.modes.abs(), :2].permute(1, 0, 2)
        values = torch.where(self.modes[:, None, None] < 0, values.conj(), values)
        return values.reshape(-1)

    def __call__(self, correction_coefficients, plastic_strain):
        residual = self.matrix@self.correction_vector(correction_coefficients)
        residual = residual-self.source_load(plastic_strain)
        whitened = torch.linalg.solve_triangular(self.factor, residual[:, None], upper=False)
        return whitened.abs().square().sum()/self.area
