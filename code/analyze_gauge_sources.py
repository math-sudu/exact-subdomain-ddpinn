"""Integrate actual plastic histories against fixed endpoint influence fields.

The retained-minus-elastic-minus-source response is a diagnostic imbalance,
never a plastic source. Density arrays use axes (gauge, step, radius, angle)
and integrate with the saved nondimensional area measure to give millimetres.
"""

import argparse
import json
from operator import index
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from decompose_eigenstrain_response import plastic_sources
from pilot import config as C
from pilot.elastic_background import CircularElasticIncrement
from pilot.gauge_influence import CircularGaugeInfluence
from pilot.spectral_mixed import load_orientation_model
from publication_model import write_rows


SOURCE_NAMES = ("joint_positive", "joint_negative", "matrix")
REGION_NAMES = ("normal_tangent_positive", "normal_tangent_negative")


def integrate_oriented_regions(angular_density, normal_angle_deg):
    """Integrate the angular interpolant over fixed opposite quadrant pairs.

    The positive pair has (x.n)*(x.t)>0, with t rotated +90 degrees from n.
    Fourier integration places the boundaries at the prescribed material axes
    without assigning a sampled boundary point to one side of a moving mask.
    """
    count = angular_density.shape[-1]
    coefficients = np.fft.rfft(angular_density, axis=-1) / count
    modes = np.arange(1, coefficients.shape[-1])
    beta = np.deg2rad(normal_angle_deg)
    integrals = ((np.exp(1j * modes * (beta + np.pi / 2))
                  - np.exp(1j * modes * beta)) / (1j * modes)
                 * (1 + (-1.)**modes))
    multiplicity = np.full(len(modes), 2.)
    if count % 2 == 0:
        multiplicity[-1] = 1.
    total = 2 * np.pi * coefficients[..., 0].real
    positive = total / 2 + np.real(
        (coefficients[..., 1:] * integrals * multiplicity).sum(axis=-1))
    return np.stack((positive, total - positive), axis=-1)


def regional_rows(case_name, arrays):
    return [dict(case=case_name, load_step=int(step), gauge_deg=float(angle),
                 region=region,
                 joint_slip_area_integral=float(arrays["region_joint_slip_area_integral"][k, j]),
                 **{f"{name}_mm": float(arrays[f"region_{name}_mm"][g, k, j])
                    for name in (*SOURCE_NAMES, "source_total")})
            for k, step in enumerate(arrays["steps"])
            for g, angle in enumerate(arrays["gauge_angle_deg"])
            for j, region in enumerate(REGION_NAMES)]


def quadrature(radius, radial_points, angular_points):
    """Gauss integration in r/a and periodic integration in theta."""
    if radius <= 1 or radial_points < 1 or angular_points < 1:
        raise ValueError("Quadrature requires an annulus and positive point counts")
    nodes, weights = np.polynomial.legendre.leggauss(radial_points)
    r = 1 + (radius - 1) * (nodes + 1) / 2
    dr = weights * (radius - 1) / 2
    dtheta = 2 * np.pi / angular_points
    return SimpleNamespace(r=r, theta=np.arange(angular_points) * dtheta,
                           nt=angular_points, dr=dr, dtheta=dtheta,
                           area=np.broadcast_to((dr * r * dtheta)[:, None],
                                                (radial_points, angular_points)))


def integrate_source_work(auxiliary_stress, sources, grid):
    """Signed work using (xx, yy, zz, engineering xy); unit auxiliary force."""
    densities = {name: C.U_REF * 1000 * np.einsum(
        "grtc,rtkc->gkrt", auxiliary_stress, sources[name]) for name in SOURCE_NAMES}
    densities["source_total"] = sum(densities.values())
    arrays = {}
    for name, density in densities.items():
        arrays[f"density_{name}_mm"] = density
        arrays[f"radial_density_{name}_mm"] = density.sum(axis=-1) * grid.r * grid.dtheta
        arrays[f"angular_density_{name}_mm"] = np.einsum("gkrt,r->gkt", density, grid.dr * grid.r)
        arrays[f"{name}_mm"] = np.einsum("gkrt,rt->gk", density, grid.area)
    return arrays


@torch.no_grad()
def analyze_checkpoint(checkpoint, steps, angles=(0., 90., 104.), *,
                       radial_quadrature=64, angular_quadrature=512, device="cpu"):
    """Query only completed increments of the explicitly selected checkpoint."""
    checkpoint = Path(checkpoint)
    steps = np.array([index(step) for step in steps], dtype=int)
    angles = np.asarray(angles, dtype=float)
    bundle, saved, model = load_orientation_model(checkpoint, device)
    completed = saved["completed_steps"]
    if not len(steps) or np.any((steps < 1) | (steps > completed)):
        raise ValueError(f"Requested steps must be completed increments 1..{completed}")
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("Gauge angles must be a nonempty finite sequence")
    settings, case = saved["settings"], saved["case"]
    if settings.get("representation") != "spectral":
        raise ValueError("Gauge modes require the checkpoint's physical spectral projection")
    modes = settings["projection"]["angular_modes"]
    if angular_quadrature <= 2 * modes:
        raise ValueError("Angular quadrature must resolve the retained gauge modes")
    # Even a checkpoint with a pending next step must never query that field.
    del model.increments[int(steps.max()):]
    like = model.multiplier_scale
    elastic_field = CircularElasticIncrement(case, total_steps=1, like=like)
    material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
    if not np.allclose(bundle["c"].cpu().numpy(), material.plane_strain_stiffness4() / C.E_REF,
                       rtol=1e-12, atol=1e-12):
        raise ValueError("Checkpoint stiffness differs from the auxiliary material")
    grid = quadrature(settings["radius"], radial_quadrature, angular_quadrature)
    directions = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    endpoints = np.stack((-directions, directions), axis=1)
    xy = grid.r[:, None, None] * np.column_stack((np.cos(grid.theta), np.sin(grid.theta)))[None]
    stress = np.stack([CircularGaugeInfluence(material, pair, modes).stress(xy) for pair in endpoints])
    sources, accumulated_slip = plastic_sources(bundle, model, grid, steps)
    arrays = integrate_source_work(stress, sources, grid)
    arrays["accumulated_joint_slip_strain"] = accumulated_slip.transpose(2, 0, 1)*(C.U_REF/C.A_TUNNEL)
    activity = np.einsum("krt,r->kt", arrays["accumulated_joint_slip_strain"], grid.dr*grid.r)
    arrays["region_joint_slip_area_integral"] = integrate_oriented_regions(activity, case["joint_normal_deg"])
    for name in (*SOURCE_NAMES, "source_total"):
        arrays[f"region_{name}_mm"] = integrate_oriented_regions(
            arrays[f"angular_density_{name}_mm"], case["joint_normal_deg"])

    endpoint_xy = like.new_tensor(endpoints.reshape(-1, 2))
    retained = model(endpoint_xy)[0][:, steps - 1].cpu().numpy().reshape(len(angles), 2, len(steps), 2)
    arrays["retained_mm"] = -np.einsum("gkc,gc->gk", retained[:, 1] - retained[:, 0], directions) * C.U_REF * 1000
    elastic = elastic_field(endpoint_xy)[0].cpu().numpy().reshape(len(angles), 2, 2)
    elastic_gauge = -np.einsum("gc,gc->g", elastic[:, 1] - elastic[:, 0], directions)
    arrays["elastic_mm"] = elastic_gauge[:, None] * (steps / model.total_steps)[None] * C.U_REF * 1000
    arrays["imbalance_mm"] = arrays["retained_mm"] - arrays["elastic_mm"] - arrays["source_total_mm"]
    values = (*SOURCE_NAMES, "source_total", "retained", "elastic", "imbalance")
    rows = [dict(case=case["name"], load_step=int(step), gauge_deg=float(angle),
                 **{f"{name}_mm": float(arrays[f"{name}_mm"][g, k]) for name in values})
            for k, step in enumerate(steps) for g, angle in enumerate(angles)]
    metadata = dict(
        checkpoint=str(checkpoint.resolve()), case=case, completed_steps=int(completed),
        total_steps=model.total_steps, gauge_modes=modes,
        source="Cumulative signed flow evaluated at each increment's own total stress",
        gauge="D=-(u2-u1).e12; endpoints=(-e,+e); auxiliary force Q=1",
        imbalance="retained_mm - elastic_mm - source_total_mm; diagnostic, not a source",
        strain_components=["xx", "yy", "zz", "engineering_xy"],
        density_axes=["gauge", "step", "radius", "source_angle"],
        radial_density_axes=["gauge", "step", "radius"],
        angular_density_axes=["gauge", "step", "source_angle"],
        region_response_axes=["gauge", "step", "region"],
        accumulated_joint_slip_axes=["step", "radius", "source_angle"],
        region_joint_slip_area_integral_axes=["step", "region"],
        joint_slip_area_integral="Integral of sum_l |2 t.epsilon_p_joint_increment_l.n| dA/a^2",
        regions=dict(names=REGION_NAMES, normal_angle_deg=case["joint_normal_deg"],
                     definition="sign((x.n)*(x.t)), t is n rotated +90 degrees; fixed through unloading",
                     integration="Exact angular integration of the periodic trigonometric interpolant"),
        response_axes=["gauge", "step"],
        units=dict(response="mm", density="mm per unit dA/a^2",
                   radial_density="mm per unit d(r/a)", angular_density="mm/radian",
                   accumulated_joint_slip="strain", joint_slip_area_integral="strain integrated against dA/a^2",
                   radius="r/a", angles="degree", area_weights="dA/a^2",
                   radial_weights="d(r/a)", angular_weight="radian"),
        displacement_mm_scale=C.U_REF * 1000)
    arrays.update(steps=steps, gauge_angle_deg=angles, radius_over_a=grid.r,
                  source_angle_deg=np.rad2deg(grid.theta), area_weights=grid.area,
                  radial_weights=grid.dr, angular_weight=grid.dtheta,
                  metadata_json=json.dumps(metadata))
    return rows, arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Root containing <case>/model.pt")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--steps", nargs="+", type=int, required=True)
    parser.add_argument("--angles", nargs="+", type=float, default=[0., 90., 104.])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radial-quadrature", type=int, default=64)
    parser.add_argument("--angular-quadrature", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    regions = []
    for case in args.cases:
        result, arrays = analyze_checkpoint(args.root / case / "model.pt", args.steps, args.angles,
                                            radial_quadrature=args.radial_quadrature,
                                            angular_quadrature=args.angular_quadrature,
                                            device=args.device)
        np.savez_compressed(args.output / f"{case}.npz", **arrays)
        rows.extend(result)
        regions.extend(regional_rows(case, arrays))
        print(json.dumps(dict(case=case, steps=args.steps, gauges=args.angles)), flush=True)
    write_rows(args.output / "summary.csv", rows)
    write_rows(args.output / "regions.csv", regions)


if __name__ == "__main__":
    main()
