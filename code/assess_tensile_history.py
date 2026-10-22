"""Resolve tensile weak-plane states and their signed deformation sources.

The stress bins describe the retained coupled history. They do not remove
plastic strain, change the material law, or estimate a re-equilibrated response.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_gauge_sources import quadrature
from pilot import config as C, plasticity as P
from pilot.elastic_background import CircularElasticIncrement
from pilot.gauge_influence import CircularGaugeInfluence
from pilot.spectral_mixed import CircularGridEvaluation, load_orientation_model
from publication_model import write_rows


def conditioned_sources(normal_stress, plastic_increment, auxiliary_stress,
                        area, near_apex_stress):
    """Partition work by own-step stress and by the region at the queried state.

    Arrays use (point, step[, component]); auxiliary stresses use
    (gauge, point, engineering-stress component). Positive and negative work
    remain separate before spatial or temporal cancellation.
    """
    masks = {
        "compression": normal_stress <= 0,
        "tension": (normal_stress > 0) & (normal_stress < near_apex_stress),
        "near_apex": normal_stress >= near_apex_stress,
    }
    increment_work = np.einsum("gpc,pkc->gpk", auxiliary_stress, plastic_increment)
    increment_work *= C.U_REF * 1000 * area[None, :, None]
    cumulative_work = increment_work.cumsum(axis=-1)
    cumulative_positive = np.maximum(increment_work, 0).cumsum(axis=-1)
    cumulative_negative = np.minimum(increment_work, 0).cumsum(axis=-1)
    result = {}
    for name, mask in masks.items():
        own_step = increment_work * mask[None]
        state_region = cumulative_work * mask[None]
        result[name] = {
            "mask": mask,
            "increment_net_mm": own_step.sum(axis=1),
            "history_net_mm": own_step.sum(axis=1).cumsum(axis=-1),
            "history_positive_mm": np.maximum(own_step, 0).sum(axis=1).cumsum(axis=-1),
            "history_negative_mm": np.minimum(own_step, 0).sum(axis=1).cumsum(axis=-1),
            "state_region_net_mm": state_region.sum(axis=1),
            "state_region_positive_mm": (cumulative_positive * mask[None]).sum(axis=1),
            "state_region_negative_mm": (cumulative_negative * mask[None]).sum(axis=1),
        }
    return result


@torch.no_grad()
def assess(checkpoint, output, device="cpu", nr=96, nt=512,
           angles=(0., 90., 104.), include_pending=False):
    bundle, saved, model = load_orientation_model(checkpoint, device)
    if not include_pending and saved["completed_steps"] != model.n_steps:
        raise ValueError("Use --include-pending to assess an unaccepted increment")
    params, case = bundle["params"], saved["case"]
    q = quadrature(saved["settings"]["radius"], nr, nt)
    xy = (q.r[:, None, None] * np.column_stack((np.cos(q.theta), np.sin(q.theta)))[None]).reshape(-1, 2)
    normal, shear = P._joint_geometry(params, model.multiplier_scale)
    cq = model.return_stiffness @ shear
    friction = np.tan(np.deg2rad(params.joint_phi_deg))
    coupling = float(friction * (normal @ cq) / (shear @ cq))
    stress_scale = C.SIGMA_REF / 1e6
    apex = params.joint_cohesion * stress_scale / friction
    rounding = params.joint_smoothing * stress_scale
    near_apex = apex - rounding / friction
    normal_parts, shear_parts, plastic_parts = [], [], []
    chunk_size = max(nt, (8192 // nt) * nt)
    for start in range(0, len(xy), chunk_size):
        points = model.multiplier_scale.new_tensor(xy[start:start + chunk_size])
        grid = CircularGridEvaluation(model.increments[0], (points.reshape(-1, nt, 2),))
        _, perturbation, _, plastic = model.returned_history(points, grid=grid)
        total = perturbation + model.return_insitu
        normal_parts.append((total @ normal).cpu().numpy() * stress_scale)
        shear_parts.append((total @ shear).cpu().numpy() * stress_scale)
        plastic_parts.append(plastic.cpu().numpy())
    sigma_n, tau = np.concatenate(normal_parts), np.concatenate(shear_parts)
    plastic = np.concatenate(plastic_parts)
    if not (np.isfinite(sigma_n).all() and np.isfinite(plastic).all()):
        raise FloatingPointError("Nonfinite constitutive history")
    increments = np.diff(plastic, axis=1, prepend=np.zeros_like(plastic[:, :1]))
    directions = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    pairs = np.stack((-directions, directions), axis=1)
    material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
    modes = model.projection["angular_modes"]
    auxiliary = np.stack([CircularGaugeInfluence(material, pair, modes).stress(xy) for pair in pairs])
    partition = conditioned_sources(sigma_n, increments, auxiliary, q.area.ravel(), near_apex)
    h = params.joint_cohesion + params.joint_smoothing - friction * sigma_n / stress_scale + coupling * tau / stress_scale
    radii = np.linalg.norm(xy, axis=-1)
    polar_angles = np.rad2deg(np.arctan2(xy[:, 1], xy[:, 0])) % 360
    area_rows, source_rows, state_rows = [], [], []
    area = q.area.ravel()
    for k in range(model.n_steps):
        peak = int(sigma_n[:, k].argmax())
        minimum = int(h[:, k].argmin())
        state_rows.append(dict(load_step=k + 1, release_fraction=(k + 1) / model.total_steps,
            maximum_normal_stress_mpa=float(sigma_n[peak, k]),
            apex_normal_margin_mpa=float(apex - sigma_n[peak, k]),
            maximum_normal_radius_over_a=float(radii[peak]),
            maximum_normal_angle_deg=float(polar_angles[peak]),
            minimum_h=float(h[minimum, k]),
            minimum_h_radius_over_a=float(radii[minimum]),
            minimum_h_angle_deg=float(polar_angles[minimum])))
        for name, values in partition.items():
            mask = values["mask"][:, k]
            region_area = float(area[mask].sum())
            area_rows.append(dict(load_step=k + 1, region=name,
                area_over_a2=region_area, annulus_area_fraction=region_area / area.sum(),
                minimum_radius_over_a=float(radii[mask].min()) if mask.any() else None,
                maximum_radius_over_a=float(radii[mask].max()) if mask.any() else None))
            for g, angle in enumerate(angles):
                source_rows.append(dict(load_step=k + 1, gauge_deg=angle, region=name,
                    **{key: float(value[g, k]) for key, value in values.items() if key != "mask"}))
    endpoints = model.multiplier_scale.new_tensor(pairs.reshape(-1, 2))
    displacement = model.returned_history(endpoints)[0].cpu().numpy().reshape(len(angles), 2, model.n_steps, 2)
    retained = -np.einsum("gkc,gc->gk", displacement[:, 1] - displacement[:, 0], directions) * C.U_REF * 1000
    elastic = CircularElasticIncrement(case, 1, model.multiplier_scale)(endpoints)[0].cpu().numpy().reshape(len(angles), 2, 2)
    background = -np.einsum("gc,gc->g", elastic[:, 1] - elastic[:, 0], directions) * C.U_REF * 1000
    total_source = sum(values["history_net_mm"] for values in partition.values())
    gauge_rows = [dict(load_step=k + 1, gauge_deg=angle,
        retained_mm=float(retained[g, k]),
        source_total_mm=float(total_source[g, k]),
        imbalance_mm=float(retained[g, k] - background[g] * (k + 1) / model.total_steps - total_source[g, k]))
        for k in range(model.n_steps) for g, angle in enumerate(angles)]
    result = dict(checkpoint=str(checkpoint.resolve()), case=case,
        completed_steps=saved["completed_steps"], queried_steps=model.n_steps,
        pending_step=model.n_steps if model.n_steps > saved["completed_steps"] else None,
        integration=dict(radial=nr, angular=nt, angular_shift=0),
        normal_stress_sign="tension positive", coupling_b=coupling,
        tensile_intercept_mpa=apex, rounding_width_mpa=rounding,
        near_apex_lower_normal_stress_mpa=near_apex,
        region_definitions=dict(compression="sigma_n <= 0",
            tension="0 < sigma_n < (c-delta)/tan(phi)",
            near_apex="sigma_n >= (c-delta)/tan(phi); remaining shear strength is at most the rounding width"),
        source_definitions=dict(history="Each increment is assigned by its own returned normal stress, then summed in load order.",
            state_region="All preceding plastic increments at points belonging to the region in the queried state.",
            signs="Positive/negative denote gauge work, not slip sign; their sum is the net contribution.",
            interpretation="Source accounting in the retained coupled history, not a change of constitutive model or a re-equilibrated deletion experiment."),
        final_state=state_rows[-1], final_regions=area_rows[-3:],
        final_sources=source_rows[-3 * len(angles):], final_gauges=gauge_rows[-len(angles):])
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("states", state_rows), ("regions", area_rows),
                       ("sources", source_rows), ("gauges", gauge_rows)):
        write_rows(output / f"{name}.csv", rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(checkpoint=str(checkpoint), final_state=result["final_state"],
        final_regions=result["final_regions"], final_gauges=result["final_gauges"])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--radial-quadrature", type=int, default=96)
    parser.add_argument("--angular-quadrature", type=int, default=512)
    parser.add_argument("--include-pending", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    assess(args.checkpoint, args.output, args.device, args.radial_quadrature,
           args.angular_quadrature, include_pending=args.include_pending)


if __name__ == "__main__":
    main()
