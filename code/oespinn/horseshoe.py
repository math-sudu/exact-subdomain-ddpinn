"""Geometry-mapped incremental PINN with full-load exterior-coupled work."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .horseshoe_geometry import mapped_basis
from .joint_return import joint_return
from .plasticity import matrix_yield, joint_yield
from . import config as C


class _CoupledResidual(torch.autograd.Function):
    """Cache the assembled physical tangent during a neural Krylov solve."""
    @staticmethod
    def forward(coefficients, prefix, previous_plastic, release, work):
        residual, tangent, _ = work.residual_tangent(coefficients, prefix, previous_plastic, release)
        return residual, tangent

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, tangent = output
        ctx.save_for_backward(tangent)
        ctx.mark_non_differentiable(tangent)

    @staticmethod
    def backward(ctx, gradient, tangent_gradient):
        (tangent,) = ctx.saved_tensors
        return (tangent.T@gradient).reshape(-1, 2), None, None, None, None


class MappedIncrement(nn.Module):
    """Project a periodic neural displacement onto the mapped finite space."""
    def __init__(self, geometry, radial_order, angular_modes, angular_samples,
                 hidden=(64, 64, 64), steps=12, *, device="cpu", amplitude=1.0):
        super().__init__()
        self.geometry = geometry
        self.radial_order, self.angular_modes = radial_order, angular_modes
        self.angular_samples, self.total_steps = angular_samples, steps
        self.amplitude = amplitude
        self.centrosymmetric = False
        layers, width = [], 3
        for size in hidden:
            layers.extend((nn.Linear(width, size), nn.Tanh()))
            width = size
        self.kinematic_trunk = nn.Sequential(*layers)
        # Uniform translation is removed by the projection. Omitting its
        # neural bias avoids an unidentifiable parameter and cancellation.
        self.kinematic_head = nn.Linear(width, 2, bias=False)
        nn.init.zeros_(self.kinematic_head.weight)
        j = torch.arange(radial_order+1, dtype=torch.float64, device=device)
        s = (1+torch.cos(torch.pi*j/radial_order))/2
        theta = torch.arange(angular_samples, dtype=torch.float64, device=device)*(2*torch.pi/angular_samples)
        ss, tt = torch.meshgrid(s, theta, indexing="ij")
        inputs = torch.stack((2*ss-1, tt.cos(), tt.sin()), -1).flatten(0, 1)
        endpoints = torch.ones_like(j)
        endpoints[0] = endpoints[-1] = .5
        dct = 2/radial_order*endpoints[:, None]*endpoints[None]*torch.cos(torch.pi*j[:, None]*j[None]/radial_order)
        self.register_buffer("inputs", inputs, persistent=False)
        self.register_buffer("dct", dct, persistent=False)
        self.register_buffer("multiplier_scale", torch.tensor(1., dtype=torch.float64, device=device))
        self.to(device=device, dtype=torch.float64)

    def coefficients(self):
        values = self.kinematic_head(self.kinematic_trunk(self.inputs))*self.amplitude/self.total_steps
        values = values.reshape(self.radial_order+1, self.angular_samples, 2)
        spectrum = torch.fft.rfft(values, dim=1)[:, :self.angular_modes+1]/self.angular_samples
        angular = torch.cat((spectrum[:, :1].real, 2*spectrum[:, 1:].real, -2*spectrum[:, 1:].imag), 1)
        coefficients = torch.einsum("ij,jmc->imc", self.dct, angular)
        return coefficients.flatten(0, 1)[1:]

    def forward(self, xy):
        value, _, _ = mapped_basis(self.geometry, xy, self.radial_order, self.angular_modes)
        return value @ self.coefficients()


class HorseshoeWork:
    def __init__(self, geometry, material, params, insitu_mpa, radial_order, angular_modes,
                 nr, nt, *, boundary_points=512, device="cpu", shift=0.0,
                 return_law="joint", tension_mpa=None, support_pressure_mpa=0.0):
        self.geometry, self.material, self.params = geometry, material, params
        self.j, self.m, self.nr, self.nt = radial_order, angular_modes, nr, nt
        self.device = device
        self.return_law, self.tension_mpa = return_law, tension_mpa
        self.support_pressure_mpa = support_pressure_mpa
        self.tensor = lambda value: torch.as_tensor(value, dtype=torch.float64, device=device)
        self.c = self.tensor(material.plane_strain_stiffness4()/C.E_REF)
        sx, sy, txy = insitu_mpa
        zz = material.plane_strain_sigma_zz(sx, sy, txy)
        self.s0 = self.tensor([sx, sy, zz, txy])/(C.SIGMA_REF/1e6)
        self.xy, self.w = geometry.quadrature(nr, nt, device=device, shift=shift)
        self.value, self.dx, self.dy = mapped_basis(geometry, self.xy, radial_order, angular_modes)
        self.n = self.value.shape[1]
        self.area = self.w.sum()
        self.wall_xy, self.wall_n, self.wall_w = geometry.boundary_quadrature(boundary_points, device=device, shift=shift)
        self.wall_basis = mapped_basis(geometry, self.wall_xy, radial_order, angular_modes)
        self.wall_t0 = self.traction(self.s0.expand(len(self.wall_xy), -1), self.wall_n)
        self.wall_load = self.wall_t0+support_pressure_mpa/(C.SIGMA_REF/1e6)*self.wall_n
        self.load = (self.wall_basis[0].T @ (self.wall_w[:, None]*self.wall_load)).flatten()
        theta = torch.arange(boundary_points, dtype=torch.float64, device=device)*(2*torch.pi/boundary_points)
        self.outer_n = torch.stack((theta.cos(), theta.sin()), -1)
        self.outer_xy = geometry.radius*self.outer_n
        self.outer_basis = mapped_basis(geometry, self.outer_xy, radial_order, angular_modes)
        self.lam = torch.as_tensor(material.Lam/C.E_REF, device=device)
        self.kext = self.exterior_matrix()
        self.k = self.kext.clone()
        weighted_x = self.w[:, None]*self.dx
        xx = self.dx.T@weighted_x
        xy = weighted_x.T@self.dy
        yy = self.dy.T@(self.w[:, None]*self.dy)
        self.k[0::2, 0::2] += self.c[0, 0]*xx+self.c[0, 3]*(xy+xy.T)+self.c[3, 3]*yy
        self.k[1::2, 1::2] += self.c[1, 1]*yy+self.c[1, 3]*(xy+xy.T)+self.c[3, 3]*xx
        cross = self.c[0, 1]*xy+self.c[0, 3]*xx+self.c[3, 1]*yy+self.c[3, 3]*xy.T
        self.k[0::2, 1::2] += cross
        self.k[1::2, 0::2] += cross.T
        self.k = (self.k+self.k.T)/2
        self.factor = torch.linalg.cholesky(self.k)
        self.load_norm = torch.linalg.solve_triangular(self.factor, self.load[:, None], upper=False).norm()

    @staticmethod
    def traction(stress, normal):
        return torch.stack((stress[..., 0]*normal[..., 0]+stress[..., 3]*normal[..., 1],
                            stress[..., 3]*normal[..., 0]+stress[..., 1]*normal[..., 1]), -1)

    @staticmethod
    def strain_basis(dx, dy):
        b = dx.new_zeros((len(dx), 4, 2*dx.shape[1]))
        b[:, 0, 0::2] = dx
        b[:, 1, 1::2] = dy
        b[:, 3, 0::2] = dy
        b[:, 3, 1::2] = dx
        return b

    @staticmethod
    def strain(coefficients, basis):
        _, dx, dy = basis
        return torch.stack((dx@coefficients[:, 0], dy@coefficients[:, 1],
                            dx.new_zeros(len(dx)), dy@coefficients[:, 0]+dx@coefficients[:, 1]), -1)

    def sf(self, u):
        spectrum = torch.fft.rfft(u, dim=0)/len(u)
        mode = torch.arange(len(spectrum), device=u.device, dtype=u.dtype)
        traction = torch.einsum("ij,mj...->mi...", self.lam.conj(), spectrum)
        shape = (-1,)+(1,)*(traction.ndim-1)
        traction = traction*mode.reshape(shape)/self.geometry.radius
        traction = torch.cat((traction[:-1], torch.zeros_like(traction[-1:])))
        return torch.fft.irfft(traction, n=len(u), dim=0, norm="forward")

    def exterior_matrix(self):
        v = self.outer_basis[0]
        u = v.new_zeros((len(v), 2, self.n*2))
        u[:, 0, 0::2], u[:, 1, 1::2] = v, v
        t = self.sf(u)
        return -torch.einsum("qid,qie->de", u, t)*(2*torch.pi*self.geometry.radius/len(v))

    def source_load(self, plastic):
        source = (plastic@self.c.T)*self.w[:, None]
        x = self.dx.T@source[:, 0]+self.dy.T@source[:, 3]
        y = self.dy.T@source[:, 1]+self.dx.T@source[:, 3]
        return torch.stack((x, y), -1).flatten()

    def returned(self, trial, *, strict=True):
        if self.return_law == "joint":
            return joint_return(trial, self.c, self.params, check_branch=strict)
        if self.return_law == "matrix_joint_tension":
            from .multi_surface_return import multi_surface_return
            return multi_surface_return(trial, self.c, self.params, self.tension_mpa)
        raise ValueError(self.return_law)

    def replay(self, increments, basis=None, *, strict=True, initial=None):
        if basis is None:
            basis = (self.value, self.dx, self.dy)
        plastic = self.c.new_zeros((len(basis[0]), 4))
        total = self.c.new_zeros((self.n, 2))
        if initial is not None:
            total, plastic = initial
        rows = []
        for coefficients in increments:
            total = total+coefficients
            strain = self.strain(total, basis)
            result = self.returned(self.s0+(strain-plastic)@self.c.T, strict=strict)
            plastic = plastic+result["plastic"]
            rows.append(dict(displacement=basis[0]@total, strain=strain, plastic=plastic,
                             stress=result["stress"], gamma=result["gamma"]))
        return rows

    def residual(self, coefficients, prefix, previous_plastic, release):
        if self.return_law == "matrix_joint_tension" and torch.is_grad_enabled():
            return _CoupledResidual.apply(coefficients, prefix, previous_plastic, release, self)[0]
        return self.residual_state(coefficients, prefix, previous_plastic, release)[0]

    def residual_state(self, coefficients, prefix, previous_plastic, release):
        total = prefix+coefficients
        strain = self.strain(total, (self.value, self.dx, self.dy))
        returned = self.returned(self.s0+(strain-previous_plastic)@self.c.T, strict=False)
        plastic = previous_plastic+returned["plastic"]
        residual = self.k@total.flatten()-self.source_load(plastic)-release*self.load
        return torch.linalg.solve_triangular(self.factor, residual[:, None], upper=False).flatten()/self.load_norm, returned

    def residual_tangent(self, coefficients, prefix, previous_plastic, release):
        """The common residual and consistent coefficient-space Jacobian."""
        residual, returned = self.residual_state(coefficients, prefix, previous_plastic, release)
        modulus = returned["stress_tangent"]@self.c
        tangent = self.kext.clone()
        for start in range(0, len(self.xy), 512):
            b = self.strain_basis(self.dx[start:start+512], self.dy[start:start+512])
            db = torch.einsum("qij,qjd->qid", modulus[start:start+512], b)
            tangent += torch.einsum("qid,qie,q->de", b, db, self.w[start:start+512])
        tangent = torch.linalg.solve_triangular(self.factor, tangent, upper=False)/self.load_norm
        return residual, tangent, returned

    def functional(self, points, directions):
        points, directions = self.tensor(points), self.tensor(directions)
        value = mapped_basis(self.geometry, points/self.geometry.length_scale, self.j, self.m)[0]
        return torch.einsum("qn,qc->nc", value, directions).flatten()

    def gauges(self):
        a = self.geometry.length_scale
        lower = self.geometry.boundary["bounds_min"][1]
        crown = self.geometry.boundary["bounds_max"][1]
        gauges = {
            "sidewall_shortening": self.functional([[-a, 0], [a, 0]], [[1, 0], [-1, 0]]),
            "crown_settlement": self.functional([[0, crown]], [[0, -1]]),
            "invert_heave": self.functional([[0, lower]], [[0, 1]]),
            "vertical_shortening": self.functional([[0, crown], [0, lower]], [[0, -1], [0, 1]]),
        }
        for depth, name in ((2.5, "upper_chord"), (6.0, "middle_chord"), (9.5, "lower_chord")):
            gauges[name] = self.functional(self.geometry.horizontal_chord(depth), [[1, 0], [-1, 0]])
        return gauges

    @torch.no_grad()
    def readouts(self, increments, releases, *, prefix_cache=None, evaluated=None):
        def states(name, basis):
            if evaluated is not None:
                return evaluated[name]
            if prefix_cache is None or not prefix_cache[name]:
                return self.replay(increments, basis)
            previous = prefix_cache[name]
            total = sum(increments[:-1], torch.zeros_like(increments[0]))
            return previous+self.replay(increments[-1:], basis, initial=(total, previous[-1]["plastic"]))
        fields = states("volume", None)
        wall = states("wall", self.wall_basis)
        outer = states("outer", self.outer_basis)
        gauges = self.gauges()
        adjoints = {name: torch.cholesky_solve(g[:, None], self.factor).flatten() for name, g in gauges.items()}
        scale = C.SIGMA_REF*self.geometry.length_scale/C.E_REF*1000
        total = torch.zeros_like(increments[0])
        rows = []
        for step, (coefficient, field) in enumerate(zip(increments, fields)):
            total += coefficient
            source = self.source_load(field["plastic"])
            residual = self.k@total.flatten()-source-releases[step]*self.load
            wall_res = self.traction(wall[step]["stress"]-self.s0, self.wall_n)+releases[step]*self.wall_load
            outer_res = self.traction(outer[step]["stress"]-self.s0, self.outer_n)-self.sf(outer[step]["displacement"])
            row = dict(step=step+1, release=releases[step],
                weak_relative=float(torch.linalg.solve_triangular(self.factor, residual[:, None], upper=False).norm()/self.load_norm),
                wall_rms_mpa=float((wall_res.square().sum(-1)*self.wall_w).sum().div(self.wall_w.sum()).sqrt())*C.SIGMA_REF/1e6,
                interface_rms_mpa=float(outer_res.square().sum(-1).mean().sqrt())*C.SIGMA_REF/1e6,
                maximum_matrix_yield_mpa=float(matrix_yield(field["stress"], self.params).max())*C.SIGMA_REF/1e6,
                maximum_joint_yield_mpa=float(joint_yield(field["stress"], self.params).max())*C.SIGMA_REF/1e6,
                maximum_strain=float(field["strain"].norm(dim=-1).max())*C.SIGMA_REF/C.E_REF,
                plastic_area_m2=float((field["plastic"].norm(dim=-1)>1e-9).to(self.w).dot(self.w))*self.geometry.length_scale**2)
            for name, g in gauges.items():
                a = adjoints[name]
                row[name+"_mm"] = float(g@total.flatten())*scale
                row[name+"_plastic_mm"] = float(a@source)*scale
                row[name+"_elastic_mm"] = float(a@self.load)*releases[step]*scale
                row[name+"_imbalance_mm"] = float(a@residual)*scale
            rows.append(row)
        return rows


def new_increment(previous, geometry, settings, device):
    if previous is not None:
        return deepcopy(previous).requires_grad_(True)
    return MappedIncrement(geometry, settings["radial_order"], settings["angular_modes"],
                           settings["angular_samples"], settings["hidden"], settings["load_steps"], device=device,
                           amplitude=settings.get("neural_amplitude", 1.0))
