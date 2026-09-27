"""Paired PINN interface experiments with the manuscript's exterior operator."""

from types import SimpleNamespace

import numpy as np
import torch

from pilot import config as C
from pilot import plasticity as P
from pilot.causal_mixed import IncrementFields
from pilot.mixed_plasticity import make_geometry


METRICS = ("uniform", "material", "wavelength", "compliance")


def opening_boundary(angle, axes, orientation_deg):
    """Polar ellipse coordinates, outward-from-opening normals and arc density."""
    a, b = axes
    alpha = np.deg2rad(orientation_deg)
    phi = angle - alpha
    c, s = np.cos(phi), np.sin(phi)
    q = (c / a)**2 + (s / b)**2
    radius = q**-0.5
    dr = -c * s * (1 / b**2 - 1 / a**2) * q**-1.5
    xy = radius[:, None] * np.stack((np.cos(angle), np.sin(angle)), -1)
    local_normal = np.stack((radius * c / a**2, radius * s / b**2), -1)
    rotation = np.array([[np.cos(alpha), -np.sin(alpha)],
                         [np.sin(alpha), np.cos(alpha)]])
    normal = local_normal @ rotation.T
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    return xy, normal, np.sqrt(radius**2 + dr**2)


def geometry(case, settings, device="cuda", *, evaluation=False):
    q = case.get("quadrature", settings["quadrature"])
    nr, nt, nb = (q[k] for k in ("radial", "angular", "boundary"))
    if evaluation:
        nr, nt, nb = 2 * nr, 2 * nt, 2 * nb
    radius = settings["radius"]
    cfg = SimpleNamespace(r_gamma=radius,
                  eh_over_ev=case["eh_over_ev"],
                  bedding_dip_deg=case["bedding_dip_deg"],
                  n_pde=nr * nt, n_wall=nb, n_gamma=nb)
    pts, stiffness, insitu, sf = make_geometry(cfg, device, torch.float64, 0)
    shift = 0.37 if evaluation else 0.0
    offsets = np.zeros(nr)
    if q.get("stagger_angles", False):
        offsets = (np.arange(nr) * (np.sqrt(5)-1)/2) % 1
    angle = 2 * np.pi * (np.arange(nt)[None] + shift + offsets[:, None]) / nt
    inner, _, _ = opening_boundary(angle.reshape(-1), case["opening_axes"], case["opening_angle_deg"])
    inner_radius = np.linalg.norm(inner, axis=1).reshape(nr, nt)
    z, w = np.polynomial.legendre.leggauss(nr)
    if "radial_power" in q:
        power = q["radial_power"]
        t = (z[:, None] + 1) / 2
        r = inner_radius + (radius - inner_radius) * t**power
        jacobian = r * (radius - inner_radius) * power * t**(power-1)
        area_weights = (w[:, None] * jacobian * np.pi / nt).reshape(-1)
    else:
        r = np.sqrt(inner_radius**2 + (radius**2 - inner_radius**2)
                    * (z[:, None] + 1) / 2)
        area_weights = (w[:, None] * (radius**2 - inner_radius**2)
                        * np.pi / (2 * nt)).reshape(-1)
    interior = np.stack((r * np.cos(angle), r * np.sin(angle)), -1).reshape(-1, 2)
    wall_angle = 2 * np.pi * (np.arange(nb) + 0.5 + shift) / nb
    wall, wall_normal, arc = opening_boundary(
        wall_angle, case["opening_axes"], case["opening_angle_deg"])
    gamma_angle = 2 * np.pi * (np.arange(nb) + shift) / nb
    gamma_normal = np.stack((np.cos(gamma_angle), np.sin(gamma_angle)), -1)
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    pts.xy_all = tensor(np.concatenate((interior, wall, radius * gamma_normal)))
    pts.bnd_xy = pts.xy_all[pts.n_pde:]
    pts.nrm_wall, pts.nrm_gam = tensor(wall_normal), tensor(gamma_normal)
    pts.t0_wall = P._traction4(insitu, pts.nrm_wall)
    pts.interior_weights = tensor(area_weights / area_weights.sum())
    pts.wall_weights = tensor(arc / arc.sum())
    pts.area = float(area_weights.sum())
    pts.wall_angles = tensor(wall_angle)
    return pts, stiffness, insitu, sf


def metric_value(sf, residual, metric):
    """Common retained-space metrics; full compliance / (4 pi R^2)."""
    if metric not in METRICS:
        raise ValueError(metric)
    a = sf.coeffs(residual.T)
    if metric in ("material", "compliance"):
        power = (a.conj() * (sf.Qc @ a)).sum(0).real
    else:
        power = a.abs().square().sum(0)
    weights = torch.zeros_like(power)
    weights[0] = 0.5
    weights[1:sf.m_max + 1] = 1
    if metric in ("wavelength", "compliance"):
        weights[1:sf.m_max + 1] /= torch.arange(
            1, sf.m_max + 1, device=power.device, dtype=power.dtype)
    return (weights * power).sum()


def make_elastic_model(settings, stiffness, seed):
    torch.manual_seed(seed)
    return IncrementFields(settings["hidden"], settings["radius"], 1,
                           1 / stiffness.diag().abs().max()).to(stiffness)


def elastic_fields(model, pts, stiffness, sf):
    xy = pts.xy_all
    directions = torch.eye(2, dtype=xy.dtype, device=xy.device)[:, None].expand(-1, len(xy), -1)
    values, derivatives = torch.vmap(lambda d: torch.func.jvp(model, (xy,), (d,)))(directions)
    u, stress, _ = (v[0] for v in values)
    du, ds, _ = derivatives
    strain = torch.stack((du[0, :, 0], du[1, :, 1], torch.zeros_like(du[0, :, 0]),
                          du[1, :, 0] + du[0, :, 1]), -1)
    n0, n1 = pts.n_pde, pts.n_pde + pts.n_wall
    residuals = dict(
        equilibrium=torch.stack((ds[0, :n0, 0] + ds[1, :n0, 3],
                                 ds[0, :n0, 3] + ds[1, :n0, 1]), -1),
        wall=P._traction4(stress[n0:n1], pts.nrm_wall) + pts.t0_wall,
        interface=P._traction4(stress[n1:], pts.nrm_gam) - sf.apply(u[n1:].T).T,
        constitutive=strain @ stiffness.T - stress,
        gauge=u[n1:].mean(0))
    return dict(displacement=u, stress_increment=stress, strain=strain), residuals


def regional_power(value, pts):
    n0, n1 = pts.n_pde, pts.n_pde + pts.n_wall
    power = value.square().mean(-1)
    return (0.8 * (power[:n0] * pts.interior_weights).sum()
            + 0.1 * (power[n0:n1] * pts.wall_weights).sum()
            + 0.1 * power[n1:].mean())


def objective(residuals, pts, sf, weights, metric, scale):
    terms = dict(
        equilibrium=(residuals["equilibrium"].square().sum(-1) * pts.interior_weights).sum(),
        wall=(residuals["wall"].square().sum(-1) * pts.wall_weights).sum(),
        interface=scale * metric_value(sf, residuals["interface"], metric),
        constitutive=regional_power(residuals["constitutive"], pts),
        gauge=residuals["gauge"].square().sum())
    if "complementarity" in residuals:
        terms["complementarity"] = regional_power(residuals["complementarity"], pts)
    return sum(weights[key] * value for key, value in terms.items())


@torch.no_grad()
def elastic_observables(model, pts, stiffness, sf):
    fields, r = elastic_fields(model, pts, stiffness, sf)
    n0, n1 = pts.n_pde, pts.n_pde + pts.n_wall
    wall_u = fields["displacement"][n0:n1]
    radial = pts.xy_all[n0:n1] / pts.xy_all[n0:n1].norm(dim=-1, keepdim=True)
    convergence = -(wall_u * radial).sum(-1) * C.U_REF * 1000
    mean_square = metric_value(sf, r["interface"], "uniform")
    compliance = metric_value(sf, r["interface"], "compliance")
    wall_rms = (r["wall"].square().sum(-1) * pts.wall_weights).sum().sqrt()
    equilibrium = (r["equilibrium"].square().sum(-1) * pts.interior_weights).sum().sqrt()
    return dict(convergence_max_mm=float(convergence.max()),
                convergence_mean_mm=float((convergence * pts.wall_weights).sum()),
                interface_mean_square=float(mean_square),
                interface_compliance=float(compliance),
                wall_traction_rms_mpa=float(wall_rms * C.SIGMA_REF / 1e6),
                equilibrium_rms=float(equilibrium),
                constitutive_rms=float(regional_power(r["constitutive"], pts).sqrt())), fields, r


def cpu_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}
