"""Exterior-coupled virtual-work operator."""

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import splu
from pilot import plasticity as P


class FrozenSourceOperator:
    """Galerkin elasticity in the retained field space, with exact exterior blocks."""

    def __init__(self, stiffness, impedance, radius, radial_order, angular_modes,
                 radial_points=64, angular_points=512):
        self.c = stiffness
        self.rg = radius
        self.j = radial_order
        self.m = angular_modes
        self.modes = np.arange(-self.m, self.m + 1)
        self.nt = angular_points
        self.theta = np.arange(self.nt) * (2 * np.pi / self.nt)
        nodes, weights = np.polynomial.legendre.leggauss(radial_points)
        self.r = 1 + (radius - 1) * (nodes + 1) / 2
        self.w = weights * (radius - 1) / 2 * self.r * (2 * np.pi)
        self.t = np.polynomial.chebyshev.chebvander(nodes, self.j)
        derivative_coefficients = np.polynomial.chebyshev.chebder(np.eye(self.j + 1), axis=0)
        self.dt = np.polynomial.chebyshev.chebval(nodes, derivative_coefficients).T * 2 / (radius - 1)
        self.block = 2 * (self.j + 1)
        self.ndof = len(self.modes) * self.block
        self.strains = {(m, s): self.strain_basis(m, s) for m in self.modes for s in (-1, 1)}
        blocks = [[None] * len(self.modes) for _ in self.modes]
        for i, m in enumerate(self.modes):
            for n in (m - 2, m, m + 2):
                if abs(n) > self.m:
                    continue
                value = np.zeros((self.block, self.block), dtype=complex)
                for s in (-1, 1):
                    q = m + s - n
                    if q not in (-1, 1):
                        continue
                    left = self.strains[m, s]
                    right = self.strains[n, q]
                    cr = np.einsum("ij,rjb->rib", self.c, right)
                    value += np.einsum("rib,ric,r->bc", left.conj(), cr, self.w, optimize=True)
                if n == m and m:
                    a = -np.conj(impedance) if m > 0 else -impedance
                    value += np.kron(np.ones((self.j + 1, self.j + 1)), 2 * np.pi * abs(m) * a)
                blocks[i][n + self.m] = sparse.csc_matrix(value)
        stiffness_matrix = sparse.bmat(blocks, format="csc")
        # Enforce the zero Cartesian mean of the interface displacement.
        row = np.tile(np.arange(2), self.j + 1)
        column = self.m * self.block + np.arange(self.block)
        gauge = sparse.csc_matrix((np.ones(self.block), (row, column)), shape=(2, self.ndof))
        self.matrix = sparse.bmat([[stiffness_matrix, gauge.T], [gauge, None]], format="csc")
        mismatch = self.matrix - self.matrix.conj().T
        if mismatch.nnz and np.max(np.abs(mismatch.data)) > 1e-9:
            raise ValueError("The exterior-coupled bilinear form is not Hermitian")
        self.factor = splu(self.matrix, permc_spec="NATURAL")

    def strain_basis(self, m, shift):
        dr = (self.dt - shift * m * self.t / self.r[:, None]) / 2
        dy_factor = -1j * shift
        e = np.zeros((len(self.r), 4, self.block), dtype=complex)
        e[:, 0, 0::2] = dr
        e[:, 1, 1::2] = dy_factor * dr
        e[:, 3, 0::2] = dy_factor * dr
        e[:, 3, 1::2] = dr
        return e

    def loading(self, initial_stress):
        rhs = np.zeros(self.ndof + 2, dtype=complex)
        sigma = np.array([[initial_stress[0], initial_stress[3]],
                          [initial_stress[3], initial_stress[1]]])
        wall_basis = (-1.) ** np.arange(self.j + 1)
        for m in (-1, 1):
            traction = sigma @ np.array([.5, -.5j * m])
            block = 2 * np.pi * wall_basis[:, None] * traction
            start = (m + self.m) * self.block
            rhs[start:start + self.block] = block.ravel()
        return rhs

    def source_load(self, plastic):
        """plastic shape: radius, angle, load state, engineering-strain component."""
        source = np.einsum("ij,rtkj->rtki", self.c, plastic)
        return self.stress_load(source)

    def stress_load(self, source):
        """Integrate a work-conjugate stress against the virtual strain."""
        fourier = np.fft.fft(source, axis=1) / self.nt
        rhs = np.zeros((self.ndof + 2, source.shape[2]), dtype=complex)
        for i, m in enumerate(self.modes):
            for shift in (-1, 1):
                rhs[i*self.block:(i+1)*self.block] += np.einsum(
                    "rib,rki,r->bk", self.strains[m, shift].conj(),
                    fourier[:, (m+shift) % self.nt], self.w, optimize=True)
        return rhs

    def body_load(self, force):
        """Integrate a volume force against the displacement basis."""
        fourier = np.fft.fft(force, axis=1) / self.nt
        rhs = np.zeros((self.ndof + 2, force.shape[2]), dtype=complex)
        for i, m in enumerate(self.modes):
            rhs[i*self.block:(i+1)*self.block] = np.einsum(
                "rj,rkc,r->jck", self.t, fourier[:, m % self.nt], self.w,
                optimize=True).reshape(self.block, -1)
        return rhs

    def boundary_load(self, traction, radius):
        """Integrate traction on an unshifted angular grid at either boundary."""
        fourier = np.fft.fft(traction, axis=0) / len(traction)
        basis = np.polynomial.chebyshev.chebvander(
            2*(radius-1)/(self.rg-1)-1, self.j).ravel()
        rhs = np.zeros((self.ndof + 2, traction.shape[1]), dtype=complex)
        for i, m in enumerate(self.modes):
            rhs[i*self.block:(i+1)*self.block] = (
                2*np.pi*radius*basis[:, None, None]
                * fourier[m % len(traction)].T[None]).reshape(self.block, -1)
        return rhs

    def solve(self, rhs):
        coefficients = self.factor.solve(rhs)
        return coefficients[:self.ndof].reshape(len(self.modes), self.j + 1, 2, -1)

    def evaluate(self, coefficients, radius, theta):
        radius, theta = np.broadcast_arrays(radius, theta)
        t = np.polynomial.chebyshev.chebvander(2*(radius-1)/(self.rg-1)-1, self.j)
        phase = np.exp(1j * theta[:, None] * self.modes)
        u = np.einsum("pm,pj,mjck->pck", phase, t, coefficients, optimize=True)
        if np.max(np.abs(u.imag)) > 1e-9:
            raise ValueError("Conjugate Fourier modes do not reconstruct a real displacement")
        return u.real


@torch.no_grad()
def plastic_sources(bundle, model, operator, steps):
    """Return signed source strains and accumulated absolute joint shear."""
    units = np.stack((np.cos(operator.theta), np.sin(operator.theta)), axis=-1)
    xy = (operator.r[:, None, None] * units[None]).reshape(-1, 2)
    pieces = {"joint_positive": [], "joint_negative": [], "matrix": []}
    activity = []
    _, shear_projection = P._joint_geometry(bundle["params"], model.multiplier_scale)
    shear_projection = shear_projection * shear_projection.new_tensor([2., 2., 1., 1.])
    from pilot.spectral_mixed import CircularGridEvaluation, ReturnedCausalPINN
    returned_model = isinstance(model, ReturnedCausalPINN)
    chunk_size = max(operator.nt, (8192 // operator.nt) * operator.nt) if returned_model else 1024
    for start in range(0, len(xy), chunk_size):
        points = model.multiplier_scale.new_tensor(xy[start:start+chunk_size])
        if returned_model:
            grid = CircularGridEvaluation(model.increments[0], (points.reshape(-1, operator.nt, 2),))
            _, stress, multipliers, _ = model.returned_history(points, grid=grid)
        else:
            _, stress, multipliers = model(points)
        total = stress + bundle["s0"].to(stress)
        joint_increment = multipliers[..., 1:] * P.joint_flow_direction(total, bundle["params"])
        accumulated_slip = (joint_increment @ shear_projection).abs().cumsum(1)
        activity.append(accumulated_slip[:, np.array(steps)-1].cpu().numpy())
        tau = P._joint_stresses(total, bundle["params"])[1]
        joint_positive = torch.where(tau[..., None] >= 0, joint_increment, 0.).cumsum(1)
        joint_negative = torch.where(tau[..., None] < 0, joint_increment, 0.).cumsum(1)
        matrix = (multipliers[..., :1] * P.matrix_flow_direction(total, bundle["params"])).cumsum(1)
        pieces["joint_positive"].append(joint_positive[:, np.array(steps)-1].cpu().numpy())
        pieces["joint_negative"].append(joint_negative[:, np.array(steps)-1].cpu().numpy())
        pieces["matrix"].append(matrix[:, np.array(steps)-1].cpu().numpy())
    sources = {name: np.concatenate(parts).reshape(len(operator.r), operator.nt, len(steps), 4)
               for name, parts in pieces.items()}
    return sources, np.concatenate(activity).reshape(len(operator.r), operator.nt, len(steps))
