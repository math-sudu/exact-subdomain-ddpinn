"""Geometry and spatial loading histories."""

from types import SimpleNamespace
import numpy as np
import torch
from . import config as C, plasticity as P
from .torch_sf import SfCircle


def make_geometry(cfg, device, dtype, seed, *, shift=0.0):
    """Area-uniform interior and complete, optionally shifted, FFT boundaries.

    Construct in float64 before the requested cast, including boundary coordinates.
    A shift is a fraction of a boundary cell, not an arbitrary reordered grid.
    """
    pts = SimpleNamespace(n_pde=cfg.n_pde, n_wall=cfg.n_wall, n_gam=cfg.n_gamma)
    rng = np.random.default_rng(1000 + seed)
    radius = np.sqrt(1 + (cfg.r_gamma**2 - 1) * rng.random(cfg.n_pde))
    angle = 2 * np.pi * rng.random(cfg.n_pde)
    interior = np.stack((radius*np.cos(angle), radius*np.sin(angle)), -1)
    wall_angle = 2*np.pi*(np.arange(cfg.n_wall)+0.5+shift)/cfg.n_wall
    gamma_angle = 2*np.pi*(np.arange(cfg.n_gamma)+shift)/cfg.n_gamma
    wall = np.stack((np.cos(wall_angle), np.sin(wall_angle)), -1)
    gamma = np.stack((np.cos(gamma_angle), np.sin(gamma_angle)), -1)
    tensor = lambda x: torch.as_tensor(x, device=device, dtype=dtype)
    pts.nrm_wall, pts.nrm_gam = tensor(wall), tensor(gamma)
    pts.bnd_xy = tensor(np.concatenate((wall, cfg.r_gamma*gamma)))
    pts.xy_all = torch.cat((tensor(interior), pts.bnd_xy))
    sx, sy, txy = C.nondim_insitu()
    pts.t0_wall = tensor(np.stack((sx*wall[:, 0]+txy*wall[:, 1],
                                 txy*wall[:, 0]+sy*wall[:, 1]), -1))
    mat = C.material(cfg.eh_over_ev, cfg.bedding_dip_deg)
    stiffness = tensor(mat.plane_strain_stiffness4()/C.E_REF)
    insitu = tensor((sx, sy, mat.plane_strain_sigma_zz(C.SX, C.SY, C.TXY)/C.SIGMA_REF, txy))
    sf = SfCircle(C.nondim_lambda(mat), cfg.r_gamma, cfg.n_gamma, device, dtype=dtype)
    return pts, stiffness, insitu, sf


def field_history(model, xy, stiffness, insitu, params):
    xy = xy.detach()
    directions = torch.stack((P._spatial_direction(xy, 0), P._spatial_direction(xy, 1)))
    values, derivatives = torch.vmap(lambda d: torch.func.jvp(model, (xy,), (d,)))(directions)
    u, stress, gamma = (v[0] for v in values)
    du, ds, _ = derivatives
    strain = torch.stack((du[0, ..., 0], du[1, ..., 1], torch.zeros_like(du[0, ..., 0]),
                          du[1, ..., 0]+du[0, ..., 1]), -1)
    kinematic_plastic = strain - torch.linalg.solve(stiffness, stress.unsqueeze(-1)).squeeze(-1)
    kinematic_delta = torch.diff(kinematic_plastic, dim=1, prepend=torch.zeros_like(kinematic_plastic[:, :1]))
    total = insitu+stress
    constitutive_flow = (gamma[..., :1]*P.matrix_flow_direction(total, params)
                         + gamma[..., 1:]*P.joint_flow_direction(total, params))
    scaled_gamma = gamma / model.multiplier_scale
    yield_values = torch.stack((P.matrix_yield(total, params), P.joint_yield(total, params)), -1)
    return dict(displacement=u, stress_increment=stress, stress_total=total,
                multiplier_increment=gamma, scaled_multiplier=scaled_gamma,
                strain=strain, plastic_strain=constitutive_flow.cumsum(1), delta_plastic=constitutive_flow,
                kinematic_plastic_strain=kinematic_plastic, kinematic_delta_plastic=kinematic_delta,
                yield_values=yield_values,
                equilibrium=torch.stack((ds[0, ..., 0]+ds[1, ..., 3],
                                         ds[0, ..., 3]+ds[1, ..., 1]), -1),
                flow=(kinematic_delta-constitutive_flow) @ stiffness.T,
                constitutive=(kinematic_plastic-constitutive_flow.cumsum(1)) @ stiffness.T,
                complementarity=P._fischer_burmeister(scaled_gamma, -yield_values,
                                                       params.complementarity_smoothing))
