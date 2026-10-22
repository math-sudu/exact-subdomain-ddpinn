"""Compare final-increment continuations with their unchanged source histories."""

import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "parallel_space": ("parallel_r4_j48_m80", "feasible_head.pt"),
    "perpendicular_space": ("perpendicular_j48_m80", "model.pt"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(name, original, filename):
    directory = ROOT / "results/late_step_continuation" / name
    baseline_path = ROOT / "results/history_resolution" / original / "model.pt"
    trial_path = directory / filename
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    candidate = torch.load(trial_path, map_location="cpu", weights_only=False)
    step = candidate["model_steps"]
    if (baseline["case"] != candidate["case"]
            or baseline["settings"] != candidate["settings"]
            or baseline["completed_steps"] != step
            or candidate["completed_steps"] != step - 1):
        raise ValueError("Compare a pending continuation with the same completed problem")
    for key, value in baseline["model"].items():
        if key.startswith("increments.") and int(key.split(".")[1]) >= step - 1:
            continue
        if not torch.equal(value, candidate["model"][key]):
            raise ValueError(f"The preceding history or material changed: {key}")
    before = read_json(ROOT / "results/tensile_history" / name / "summary.json")
    after = read_json(directory / "tensile/summary.json")
    weak_before = read_json(ROOT / "results/history_resolution" / (original + "_weak_enriched.json"))
    weak_after = read_json(directory / "weak.json")
    if (weak_before["test_space"] != weak_after["test_space"]
            or weak_before["integration"] != weak_after["integration"]):
        raise ValueError("The weak assessments must use the same test and integration spaces")
    metrics = []
    for state, saved, sources, weak in (("retained", baseline, before, weak_before),
                                        ("continued", candidate, after, weak_after)):
        row = next(row for row in reversed(saved["rows"]) if row["step"] == step)
        metrics.append(dict(state=state, objective=row["objective"],
            wall_vector_rms_mpa=row["wall_traction_rms_mpa"],
            interface_component_rms_mpa=row["interface_traction_rms_mpa"],
            relative_weak_residual=weak["rows"][-1]["relative_residual_to_source"],
            gauges=sources["final_gauges"]))
    changes = []
    for old, new in zip(before["final_gauges"], after["final_gauges"]):
        if old["gauge_deg"] != new["gauge_deg"]:
            raise ValueError("Gauge definitions differ")
        changes.append(dict(gauge_deg=old["gauge_deg"],
            displacement_change_mm=new["retained_mm"] - old["retained_mm"],
            source_change_mm=new["source_total_mm"] - old["source_total_mm"],
            relative_source_change=(new["source_total_mm"] - old["source_total_mm"])
                                  / abs(old["source_total_mm"])))
    neural = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    final_neural = neural["rows"][-1]
    return dict(baseline=str(baseline_path.relative_to(ROOT)),
        candidate=str(trial_path.relative_to(ROOT)), step=step,
        frozen_prefix=list(range(1, step)), prefix_and_material_tensors_identical=True,
        candidate_status="pending optimization-sensitivity experiment",
        neural_blocks=final_neural["trust_solves"], neural_evaluations=final_neural["evaluations"],
        neural_optimizer_status=final_neural["trust_status"],
        head_solves=[r for r in candidate.get("constrained_head_solves", []) if r["step"] == step],
        test_space=weak_after["test_space"], integration=weak_after["integration"],
        metrics=metrics, changes=changes, local_branch=read_json(directory / "branch.json"))


def main():
    output = ROOT / "results/late_step_continuation/comparison.json"
    result = dict(
        purpose="Separate late-step optimization sensitivity from changes to the domain, representation and preceding history.",
        interpretation="Finite continuation experiments, not a convergence bound or replacement of the fully recomputed histories.",
        studies={name: summarize(name, *inputs) for name, inputs in STUDIES.items()})
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, study in result["studies"].items():
        print(json.dumps(dict(study=name, metrics=study["metrics"], changes=study["changes"])))


if __name__ == "__main__":
    main()
