"""Compare full-release PINN paths and their common physical checks."""

import csv
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/controlled_history"
NAMES = ("parallel_source", "parallel_preceding")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def measurements(summary, tensile):
    gauges = []
    for row in summary["final_gauges"]:
        if row["gauge_deg"] not in (0., 90.):
            continue
        tension = sum(item["history_net_mm"] for item in tensile["final_sources"]
                      if item["gauge_deg"] == row["gauge_deg"] and item["region"] != "compression")
        gauges.append(dict(gauge_deg=row["gauge_deg"], retained_mm=row["retained_mm"],
            source_mm=row["source_total_mm"], imbalance_mm=row["imbalance_mm"],
            end_tension_source_mm=tension,
            end_tension_source_fraction=abs(tension / row["source_total_mm"])))
    near = summary["final_near_field"]
    return dict(gauges=gauges, vertical_side_share=near["vertical_side_share"],
        vertical_last_third_share=near["vertical_last_third_share"],
        maximum_exterior_yield_mpa=summary["exterior"]["maximum_joint_yield_mpa"])


def history_assessment(directory, checks, weak):
    """Join independent weak tests to the same saved prefixes and gauge sources."""
    with (directory / "readouts/sources.csv").open(newline="", encoding="utf-8") as stream:
        sources = {(int(row["load_step"]), float(row["gauge_deg"])): row
                   for row in csv.DictReader(stream)}
    if [row["load_step"] for row in weak["rows"]] != sorted(checks):
        raise ValueError(f"Independent weak tests must cover all plastic increments in {directory}")
    rows = []
    for row in weak["rows"]:
        step = row["load_step"]
        gauges = [{"gauge_deg": angle, **{key: float(sources[step, angle][key])
                    for key in ("source_total_mm", "retained_mm", "imbalance_mm")}}
                  for angle in (0., 90., 104.)]
        rows.append(dict(row, accepted=checks[step]["accepted"], gauges=gauges))
    return rows


def main():
    baseline_path = ROOT / "results/history_resolution/parallel_r4_j48_m80/model.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    common = ("radius", "load_steps", "hidden", "projection", "quadrature", "physics",
              "weights", "elastic_prefix", "interface_metric", "constitutive_update")
    before = measurements(read(baseline_path.parent / "readouts/summary.json"),
                          read(ROOT / "results/tensile_history/parallel_space/summary.json"))
    before["relative_weak_residual"] = read(
        ROOT / "results/history_resolution/parallel_r4_j48_m80_weak_enriched.json"
    )["rows"][-1]["relative_residual_to_source"]
    before["wall_vector_rms_mpa"] = baseline["rows"][-1]["wall_traction_rms_mpa"]
    before["independent_weak_test"] = read(DIRECTORY / "retained_independent_weak.json")
    studies = {}
    targets = None
    accepted_histories = {}
    for name in NAMES:
        directory = DIRECTORY / name
        checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
        if (checkpoint["case"] != baseline["case"] or any(
                checkpoint["settings"][key] != baseline["settings"][key] for key in common)):
            raise ValueError(f"Physical problem or displacement discretization changed in {name}")
        total = checkpoint["settings"]["load_steps"]
        completed = checkpoint["completed_steps"]
        if checkpoint["model_steps"] != total or completed not in (total - 1, total):
            raise ValueError(f"The path has not reached its final increment: {name}")
        protocol = read(directory / "control.json")
        if targets is not None and protocol["targets"] != targets:
            raise ValueError("The controlled histories use different physical targets")
        targets = protocol["targets"]
        records = read(directory / "physical_history.json")
        accepted = {row["step"]: row for row in records if row["accepted"]}
        if set(accepted) != set(range(3, completed + 1)):
            raise ValueError(f"Missing plastic-increment acceptance in {name}")
        final = records[-1]
        if (final["step"] != total or final["evaluations"] != checkpoint["evaluations"]
                or "enriched_residual_dual_norm" not in final):
            raise ValueError(f"The final saved state needs its own physical assessment: {name}")
        accepted_histories[name] = {**accepted, total: final}
        independent = read(directory / "independent_weak.json")
        result = measurements(read(directory / "readouts/summary.json"),
                              read(directory / "tensile/summary.json"))
        result.update(predictor=protocol["predictor"], onset_scale=protocol["onset_scale"],
            independent_weak_test=independent,
            history_assessment=history_assessment(directory, accepted_histories[name], independent),
            accepted_plastic_increments=sorted(accepted),
            completed_steps=completed, final_accepted=final["accepted"],
            final_pending_reasons=final["pending_reasons"],
            final_weak_residual_norm=final["enriched_residual_dual_norm"],
            relative_weak_residual=final["enriched_relative_residual"],
            wall_vector_rms_mpa=final["wall_vector_rms_mpa"],
            minimum_sampled_prefix_branch_margin=min(row["minimum_prefix_branch_margin"]
                                                    for row in accepted_histories[name].values()),
            maximum_accepted_weak_residual_norm=max(row["enriched_residual_dual_norm"] for row in accepted.values()),
            maximum_accepted_wall_rms_mpa=max(row["wall_vector_rms_mpa"] for row in accepted.values()),
            training_test_enrichment=checkpoint["work_test_enrichment"])
        result["constrained_head_solves"] = checkpoint.get("constrained_head_solves", [])
        studies[name] = result
    changes = []
    for left, right in zip(*(studies[name]["gauges"] for name in NAMES)):
        if left["gauge_deg"] != right["gauge_deg"]:
            raise ValueError("The measurement directions differ")
        changes.append(dict(gauge_deg=left["gauge_deg"],
            source_difference_mm=right["source_mm"] - left["source_mm"],
            relative_source_difference=(right["source_mm"] - left["source_mm"]) / abs(left["source_mm"]),
            displacement_difference_mm=right["retained_mm"] - left["retained_mm"]))
    step_changes = [dict(step=step,
        both_accepted=all(accepted_histories[name][step]["accepted"] for name in NAMES), **{
        f"source_difference_{angle}_mm": accepted_histories[NAMES[1]][step][f"source_{angle}_mm"]
            - accepted_histories[NAMES[0]][step][f"source_{angle}_mm"]
        for angle in (0, 90)}) for step in range(3, total + 1)]
    result = dict(
        purpose="Assess full-release source repeatability with re-solved plastic prefixes and common physical checks, retaining any failed final targets.",
        fixed_problem={key: baseline["settings"][key] for key in common},
        targets=targets, baseline=before, studies=studies,
        preceding_minus_source=changes, increment_differences=step_changes)
    (DIRECTORY / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(studies=studies, preceding_minus_source=changes), indent=2))


if __name__ == "__main__":
    main()
