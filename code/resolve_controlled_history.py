"""Re-solve a fixed PINN history with common physical increment acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_gauge_sources import quadrature
from assess_history_resolution import assess as assess_weak
from interface_extension import cpu_state
from pilot import config as C, plasticity as P
from pilot.elastic_background import CircularElasticIncrement
from pilot.gauge_influence import CircularGaugeInfluence
from pilot.spectral_mixed import CircularGridEvaluation, load_orientation_model
from recompute_orientation_history import initialize
from refine_crossed_history import warm_start
from run_orientation_extension import run_return_case, save, write_rows


@torch.no_grad()
def append_predictor(destination, predictor, onset_scale):
    checkpoint = destination / "model.pt"
    _, saved, model = load_orientation_model(checkpoint)
    _, _, source = load_orientation_model(Path(saved["initialization_checkpoint"]))
    plastic_prefix = any(torch.count_nonzero(inc.coefficients()[..., :2])
                         for inc in model.increments)
    previous = (model.increments[-1] if predictor == "preceding" and plastic_prefix
                else source.increments[model.n_steps])
    warm_start(model, previous)
    current = model.increments[-1]
    if not plastic_prefix:
        current.kinematic_head.weight[:2].mul_(onset_scale)
        current.kinematic_head.bias[:2].mul_(onset_scale)
        current.background_subtraction[..., :2, :].mul_(onset_scale)
        current._coefficient_cache = None
    saved.update(model=cpu_state(model), model_steps=model.n_steps,
                 evaluations=0, step_elapsed_s=0., initial_correction_scale=1.,
                 trust_region_state=dict(solves=0, inner={}, last_result={}))
    save(checkpoint, saved)


@torch.no_grad()
def physical_readout(checkpoint, device="cpu", nr=96, nt=512):
    """Replay the complete prefix on unshifted volume and shifted wall points."""
    bundle, saved, model = load_orientation_model(checkpoint, device)
    model.cache_projection()
    params, case = bundle["params"], saved["case"]
    q = quadrature(saved["settings"]["radius"], nr, nt)
    xy = (q.r[:, None, None] * np.column_stack((np.cos(q.theta), np.sin(q.theta)))[None]).reshape(-1, 2)
    normal, shear = P._joint_geometry(params, model.multiplier_scale)
    cq = model.return_stiffness @ shear
    friction = np.tan(np.deg2rad(params.joint_phi_deg))
    coupling = friction * (normal @ cq) / (shear @ cq)
    sources, minimum = [], float("inf")
    for start in range(0, len(xy), nt * 8):
        points = model.multiplier_scale.new_tensor(xy[start:start + nt * 8])
        grid = CircularGridEvaluation(model.increments[0], (points.reshape(-1, nt, 2),))
        _, stress, _, plastic = model.returned_history(points, grid=grid)
        total = stress + model.return_insitu
        margin = params.joint_cohesion - friction * (total @ normal) + coupling * (total @ shear)
        minimum = min(minimum, float(margin.min()))
        sources.append(plastic[:, -1].cpu().numpy())
    plastic = np.concatenate(sources)
    pairs = np.array([[[-1., 0.], [1., 0.]], [[0., -1.], [0., 1.]]])
    directions = np.eye(2)
    material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
    auxiliary = np.stack([CircularGaugeInfluence(material, pair, model.projection["angular_modes"]).stress(xy)
                          for pair in pairs])
    source_mm = np.einsum("gpc,pc,p->g", auxiliary, plastic, q.area.ravel()) * C.U_REF * 1000
    endpoints = model.multiplier_scale.new_tensor(pairs.reshape(-1, 2))
    u = model.returned_history(endpoints)[0][:, -1].cpu().numpy().reshape(2, 2, 2)
    elastic = CircularElasticIncrement(case, 1, model.multiplier_scale)(endpoints)[0].cpu().numpy().reshape(2, 2, 2)
    nonelastic = -np.einsum("gc,gc->g", u[:, 1] - u[:, 0]
        - model.n_steps / model.total_steps * (elastic[:, 1] - elastic[:, 0]), directions) * C.U_REF * 1000
    theta = (torch.arange(1024, device=endpoints.device, dtype=endpoints.dtype) + .5) * (2 * torch.pi / 1024)
    wall = torch.stack((theta.cos(), theta.sin()), -1)
    _, stress, _, _ = model.returned_history(wall)
    residual = P._traction4(stress[:, -1], wall) + model.n_steps / model.total_steps * P._traction4(model.return_insitu, wall)
    result = dict(step=model.n_steps, minimum_prefix_branch_margin=minimum,
                  wall_vector_rms_mpa=float(residual.square().sum(-1).mean().sqrt()) * C.SIGMA_REF / 1e6,
                  integration=[nr, nt], wall_points=1024, wall_angular_shift=.5)
    for j, angle in enumerate((0, 90)):
        result[f"source_{angle}_mm"] = float(source_mm[j])
        result[f"nonelastic_{angle}_mm"] = float(nonelastic[j])
        result[f"gauge_imbalance_{angle}_mm"] = float(nonelastic[j] - source_mm[j])
    return result


def acceptance_reasons(current, previous, targets):
    """Separate physical acceptance from optimizer termination or stagnation."""
    physical_keys = ("wall_vector_rms_mpa",) + tuple(
        f"{key}_{angle}_mm" for key in ("source", "nonelastic", "gauge_imbalance")
        for angle in (0, 90))
    if not all(np.isfinite(current[key]) for key in physical_keys):
        return ["nonfinite_physical_readout"]
    reasons = []
    if not np.isfinite(current["minimum_prefix_branch_margin"]) or current["minimum_prefix_branch_margin"] <= 0:
        reasons.append("unsupported_return")
    if current["wall_vector_rms_mpa"] > targets["wall_rms_mpa"]:
        reasons.append("wall_traction")
    for angle in (0, 90):
        if abs(current[f"gauge_imbalance_{angle}_mm"]) > targets["gauge_imbalance_mm"]:
            reasons.append(f"gauge_{angle}")
    if previous is None:
        reasons.append("second_block_required")
    else:
        change = max(abs(current[f"{key}_{angle}_mm"] - previous[f"{key}_{angle}_mm"])
                     for key in ("source", "nonelastic") for angle in (0, 90))
        current["response_change_mm"] = change
        if change > targets["response_change_mm"]:
            reasons.append("response_change")
    if "enriched_residual_dual_norm" in current:
        if not np.isfinite(current["enriched_residual_dual_norm"]) or current["enriched_residual_dual_norm"] > targets["weak_residual_norm"]:
            reasons.append("enriched_weak_balance")
    else:
        reasons.append("enriched_test_required")
    return reasons


def needs_test_enrichment(record, targets):
    reasons = set(record["pending_reasons"])
    if reasons == {"enriched_weak_balance"}:
        return True
    return (reasons == {"response_change", "enriched_weak_balance"}
            and record["frozen_source_projection_floor"] > targets["weak_residual_norm"])


def run(source, destination, predictor, onset_scale, targets, through_step, blocks, device,
        polish_evaluations=None):
    protocol = dict(source=str(source.resolve()), predictor=predictor, onset_scale=onset_scale,
                    targets=targets, integration=[96, 512], enriched_order_increase=[8, 16],
                    training_test_policy="enrich_on_unbalanced_weak_test")
    protocol_path = destination / "control.json"
    if not (destination / "model.pt").exists():
        initialize(source, destination)
        protocol_path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    elif json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Resume with the same controlled-history protocol")
    report_path = destination / "physical_history.json"
    records = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else []
    checkpoint = destination / "model.pt"
    if polish_evaluations is not None:
        if polish_evaluations < 0:
            raise ValueError("The polishing evaluation budget must be nonnegative")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved["settings"]["trust_region"]["polish_evaluations"] = polish_evaluations
        save(checkpoint, saved)
        (destination / "config.json").write_text(json.dumps(saved["settings"], indent=2) + "\n", encoding="utf-8")

    def enrich_training_tests(saved, assessment):
        projection = saved["settings"]["projection"]
        previous_space = {key: projection[key] for key in ("radial_order", "angular_modes")}
        saved["settings"]["work_test_space"] = dict(radial_order=projection["radial_order"] + 8,
                                                    angular_modes=projection["angular_modes"] + 16)
        saved["work_test_enrichment"] = dict(step=saved["model_steps"], evaluations=saved["evaluations"],
            assessment_evaluations=assessment["evaluations"],
            previous_test_space=previous_space,
            test_space=dict(saved["settings"]["work_test_space"]),
            reason="Enriched weak balance remains above target; the response is stable or its frozen-source projection floor already exceeds the target")
        saved["trust_region_state"]["inner"] = {}
        save(checkpoint, saved)
        (destination / "config.json").write_text(json.dumps(saved["settings"], indent=2) + "\n", encoding="utf-8")
        return saved

    while True:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["completed_steps"] >= through_step:
            return
        if saved["model_steps"] == saved["completed_steps"]:
            append_predictor(destination, predictor, onset_scale)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = saved["model_steps"]
        previous = next((r for r in reversed(records) if r["step"] == step and "source_0_mm" in r), None)
        last_weak = next((r for r in reversed(records) if r["step"] == step
                         and "enriched_residual_dual_norm" in r), None)
        if (last_weak and needs_test_enrichment(last_weak, targets)
                and "work_test_space" not in saved["settings"]):
            saved = enrich_training_tests(saved, last_weak)
        for block in range(blocks):
            run_return_case(saved["case"], saved["settings"], destination, step, device,
                            trust_blocks=1, accept_step=None, initialize_checkpoint=None)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            try:
                current = physical_readout(checkpoint, device)
            except ValueError as error:
                current = dict(step=step, accepted=False, failure=str(error),
                               evaluations=saved["evaluations"])
                records.append(current)
                report_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
                raise
            current.update(evaluations=saved["evaluations"], optimizer_status=saved["trust_region_state"]["last_result"].get("status"),
                           head_solves=len(saved.get("constrained_head_solves", [])),
                           polish_evaluation_limit=saved["settings"]["trust_region"]["polish_evaluations"],
                           training_test_space=saved["settings"].get("work_test_space", saved["settings"]["projection"]))
            # Enrich once cheap response and boundary checks support freezing.
            reasons = acceptance_reasons(current, previous, targets)
            if reasons == ["enriched_test_required"] or block == blocks - 1:
                projection = saved["settings"]["projection"]
                weak = assess_weak(checkpoint, [step], projection["radial_order"] + 8,
                    projection["angular_modes"] + 16, 96, 512, device, include_pending=True,
                    decompose_projection=True)
                current["enriched_relative_residual"] = weak["rows"][0]["relative_residual_to_source"]
                current["enriched_residual_dual_norm"] = weak["rows"][0]["residual_dual_norm"]
                current["enriched_source_dual_norm"] = weak["rows"][0]["source_dual_norm"]
                current["retained_fit_dual_norm"] = weak["rows"][0]["retained_fit_dual_norm"]
                current["frozen_source_projection_floor"] = weak["rows"][0]["frozen_source_projection_floor"]
                reasons = acceptance_reasons(current, previous, targets)
            current.update(accepted=not reasons, pending_reasons=reasons)
            records.append(current)
            report_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(current), flush=True)
            if needs_test_enrichment(current, targets) and "work_test_space" not in saved["settings"]:
                saved = enrich_training_tests(saved, current)
            if current["accepted"]:
                saved["completed_steps"] = step
                saved["rows"].append(dict(saved["rows"][-1], completed=True,
                    response_change_mm=current["response_change_mm"],
                    response_tolerance_mm=targets["response_change_mm"]))
                write_rows(destination / "trajectory.csv", saved["rows"])
                save(checkpoint, saved)
                break
            previous = current
        else:
            raise RuntimeError(f"Step {step} remains pending after {blocks} additional blocks; inspect physical_history.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictor", choices=("source", "preceding"), required=True)
    parser.add_argument("--onset-scale", type=float, default=1.)
    parser.add_argument("--through-step", type=int, default=12)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--polish-evaluations", type=int,
                        help="Set the saved L-BFGS polishing budget after a budget-limited trust-region block")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weak-load-scale", type=float, required=True,
                        help="Fixed final plastic-load dual norm used to scale every increment's weak target")
    args = parser.parse_args()
    torch.set_num_threads(1)
    targets = dict(response_change_mm=1e-4, gauge_imbalance_mm=1e-3,
                   wall_rms_mpa=.01, weak_residual_norm=.02 * args.weak_load_scale)
    run(args.source, args.output, args.predictor, args.onset_scale, targets,
        args.through_step, args.blocks, args.device, args.polish_evaluations)


if __name__ == "__main__":
    main()
