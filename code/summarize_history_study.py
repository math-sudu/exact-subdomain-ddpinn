"""Collect deformation and exterior-admissibility evidence for a saved history."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_gauge_sources import analyze_checkpoint, regional_rows
from extract_unloading_results import sample_fields
from pilot import config as C, plasticity as P
from pilot.elastic_background import CircularElasticIncrement
from pilot.spectral_mixed import load_orientation_model
from publication_model import write_rows


@torch.no_grad()
def summarize(checkpoint, output, device="cpu", include_pending=False):
    bundle, saved, model = load_orientation_model(checkpoint, device)
    if model.n_steps != model.total_steps:
        raise ValueError("Summarize a history reaching full release")
    if saved["completed_steps"] != model.total_steps and not include_pending:
        raise ValueError("Use --include-pending to assess the unaccepted final increment")
    model.cache_projection()
    steps = list(range(1, model.total_steps+1))
    rows, arrays = analyze_checkpoint(checkpoint, steps, radial_quadrature=96,
                                      angular_quadrature=512, device=device,
                                      include_pending=include_pending)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "sources.csv", rows)
    write_rows(output / "regions.csv", regional_rows(saved["case"]["name"], arrays))
    angles = 2*np.pi*np.arange(720)/720
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    wall = sample_fields(bundle, model, directions)
    radial = -np.einsum("pkc,pc->pk", wall["displacement_mm"], directions)
    peak_slip = wall["accumulated_slip_strain"][:, -1].argmax()
    peak_closure = radial[:, -1].argmax()
    last, late_start = model.total_steps-1, 2*model.total_steps//3-1
    sources = arrays["source_total_mm"]
    side = arrays["common_region_source_total_mm"][:, -1, 0]
    near = dict(maximum_slip_percent=float(100*wall["accumulated_slip_strain"][peak_slip, -1]),
        peak_slip_angle_deg=float(np.rad2deg(angles[peak_slip])),
        maximum_diameter_mm=float(2*radial[peak_closure, -1]),
        peak_diameter_angle_deg=float(np.rad2deg(angles[peak_closure]) % 180),
        vertical_side_share=float(side[1]/sources[1, last]),
        vertical_last_third_share=float(1-sources[1, late_start]/sources[1, last]),
        source_integration=dict(radial=96, angular=512))
    direction = np.array([1., 1.])/np.sqrt(2)
    if saved["case"]["name"] == "bedding_oblique":
        ab_angles = np.deg2rad([107., 15.])
        ab_directions = np.column_stack((np.cos(ab_angles), np.sin(ab_angles)))
        radii = np.linspace(1, 3, 401)
        points = model.multiplier_scale.new_tensor(
            (radii[:, None, None]*ab_directions[None]).reshape(-1, 2))
        displacement = sum(increment(points)[0] for increment in model.increments)
        radial_ab = np.einsum("rqc,qc->rq", displacement.cpu().numpy().reshape(-1, 2, 2),
                              ab_directions)*C.U_REF*1000
        accumulation = radial_ab-radial_ab[0]
        difference = accumulation[:, 0]-accumulation[:, 1]
        changes = np.where(difference[1:-1]*difference[2:] < 0)[0]+1
        near["fixed_A_B_angles_deg"] = [107., 15.]
        near["A_minus_B_wall_convergence_mm"] = float(-radial_ab[0, 0]+radial_ab[0, 1])
        near["A_minus_B_accumulation_crossings_over_a"] = [
            float(radii[i]-difference[i]*(radii[i+1]-radii[i])/(difference[i+1]-difference[i]))
            for i in changes]

    material = C.material(saved["case"]["eh_over_ev"], saved["case"]["bedding_dip_deg"])
    radius = saved["settings"]["radius"]
    trace_angles = 2*np.pi*np.arange(1024)/1024
    ring = radius*np.column_stack((np.cos(trace_angles), np.sin(trace_angles)))
    trace = model.returned_history(model.multiplier_scale.new_tensor(ring))[0].cpu().numpy()
    exterior_angles = 2*np.pi*(np.arange(720)+.37)/720
    exterior_radii = np.unique(np.r_[radius + np.array([0, .01, .025, .05, .1, .15, .25, .5]),
                                    radius*np.geomspace(1, 4, 24)])
    x = (C.A_TUNNEL*exterior_radii[:, None]*np.cos(exterior_angles)).ravel()
    y = (C.A_TUNNEL*exterior_radii[:, None]*np.sin(exterior_angles)).ravel()
    compliance = material.full_compliance()
    exterior_rows = []
    spans = []
    inside = model.returned_history(model.multiplier_scale.new_tensor(
        np.array([1., 2.])[:, None]*direction))[0].cpu().numpy()
    elastic = CircularElasticIncrement(saved["case"], 1, model.multiplier_scale)
    elastic_radial = elastic(model.multiplier_scale.new_tensor(
        np.array([1., 2., 6.])[:, None]*direction))[0].cpu().numpy() @ direction * C.U_REF * 1000
    for k, step in enumerate(steps):
        field = P._exterior_field_from_trace(trace[:, k].T*C.U_REF,
            material, radius*C.A_TUNNEL, model.projection["angular_modes"])
        xx, yy, xy = field.stresses(x, y)
        zz = -compliance[2, [0, 1, 5]] @ np.stack((xx, yy, xy))/compliance[2, 2]
        stress = model.multiplier_scale.new_tensor(np.stack((xx, yy, zz, xy), -1))/C.SIGMA_REF + model.return_insitu
        values = (P.joint_yield(stress, bundle["params"])*C.SIGMA_REF/1e6).cpu().numpy()
        index = values.argmax()
        ir, it = divmod(index, len(exterior_angles))
        exterior_rows.append(dict(load_step=step, release_fraction=step/model.total_steps,
            maximum_joint_yield_mpa=float(values[index]), radius_over_a=float(exterior_radii[ir]),
            angle_deg=float(np.rad2deg(exterior_angles[it])),
            positive_sample_fraction=float(np.mean(values>0))))
        outer_xy = 6*C.A_TUNNEL*direction
        outer = np.asarray(field.displacements(outer_xy[0:1], outer_xy[1:2])).reshape(2)
        near_span = (inside[1, k]-inside[0, k]) @ direction * C.U_REF*1000
        far_span = (outer - inside[0, k]*C.U_REF) @ direction * 1000
        for length, total, background in ((1, near_span, elastic_radial[1]-elastic_radial[0]),
                                           (5, far_span, elastic_radial[2]-elastic_radial[0])):
            spans.append(dict(load_step=step, release_fraction=step/model.total_steps,
                span_m=length*C.A_TUNNEL, total_mm=float(total),
                elastic_mm=float(background*step/model.total_steps),
                nonelastic_mm=float(total-background*step/model.total_steps)))
    write_rows(output / "exterior.csv", exterior_rows)
    write_rows(output / "radial_spans.csv", spans)
    accepted = {row["step"]: row for row in saved["rows"] if row["completed"]}
    result = dict(checkpoint=str(checkpoint.resolve()), settings=saved["settings"],
        completed_steps=saved["completed_steps"],
        pending_step=model.n_steps if model.n_steps > saved["completed_steps"] else None,
        final_gauges=rows[-3:], final_near_field=near,
        exterior=dict(maximum_joint_yield_mpa=max(row["maximum_joint_yield_mpa"] for row in exterior_rows),
            radii_over_a=exterior_radii.tolist(), angular_samples=len(exterior_angles),
            far_field_joint_yield_mpa=float(P.joint_yield(bundle["s0"], bundle["params"])*C.SIGMA_REF/1e6)),
        computation=dict(recomputed_steps=len(accepted),
            evaluations=sum(row["evaluations"] for row in accepted.values())
                +sum(row["evaluations"] for row in saved.get("predictor_restarts", []))
                +sum(row["evaluations"] for row in saved.get("constrained_head_solves", [])),
            recorded_solve_seconds=sum(row["elapsed_s"] for row in accepted.values())
                +sum(row["elapsed_s"] for row in saved.get("predictor_restarts", []))
                +sum(row["elapsed_s"] for row in saved.get("constrained_head_solves", [])),
            predictor_restarts=saved.get("predictor_restarts", []),
            constrained_head_solves=saved.get("constrained_head_solves", []),
            response_tolerance_mm_by_step={str(k): row.get("response_tolerance_mm")
                                           for k, row in accepted.items()},
            initial_correction_scale_by_step={str(k): row.get("initial_correction_scale")
                                             for k, row in accepted.items()},
            stopping="Per-step optimizer status and any recorded response tolerance are retained; a missing tolerance is not retrospectively assigned.",
            final_trust_status_by_step={str(k): row["trust_status"] for k, row in accepted.items()}))
    control_path = checkpoint.parent / "control.json"
    if control_path.exists():
        protocol = json.loads(control_path.read_text(encoding="utf-8"))
        physical = json.loads((checkpoint.parent / "physical_history.json").read_text(encoding="utf-8"))
        accepted_physical = {row["step"]: row for row in physical if row["accepted"]}
        if set(accepted_physical) != set(accepted):
            raise ValueError("Physical acceptance and saved increment records differ")
        result["computation"]["physical_acceptance_targets"] = protocol["targets"]
        result["computation"]["response_tolerance_mm_by_step"] = {
            str(k): protocol["targets"]["response_change_mm"] for k in accepted_physical}
        result["computation"]["stopping"] = (
            "Accepted plastic increments passed the checks in physical_history.json; the final increment remains pending."
            if result["pending_step"] else
            "Every plastic increment passed the physical checks saved in physical_history.json.")
    (output / "summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(final_near_field=near, exterior=result["exterior"],
                          final_gauges=result["final_gauges"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-pending", action="store_true",
                        help="Assess an unaccepted full-release iterate without accepting it")
    args = parser.parse_args()
    torch.set_num_threads(1)
    summarize(args.checkpoint, args.output, args.device, args.include_pending)


if __name__ == "__main__":
    main()
