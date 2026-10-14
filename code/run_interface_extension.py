"""Train paired interface-metric PINNs and export their common observables."""

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from interface_extension import (
    cpu_state, elastic_fields, elastic_observables, geometry,
    make_elastic_model, metric_value, objective,
)
from pilot.lbfgs import budgeted_lbfgs_step


ROOT = Path(__file__).resolve().parents[1]


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def warm_start(case, settings, seed, directory, device):
    pts, c, s0, sf = geometry(case, settings, device)
    model = make_elastic_model(settings, c, seed)
    path = directory / "initial.pt"
    if path.exists():
        stored = torch.load(path, map_location=device, weights_only=False)
        runtime_settings = lambda s: {key: value for key, value in s.items() if key not in ("cases", "question")}
        if runtime_settings(stored["settings"]) != runtime_settings(settings) or stored["case"] != case:
            raise ValueError(f"Changed experiment inputs at {path}")
        model.load_state_dict(stored["model"])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
        for _ in range(settings["warmup_updates"]):
            optimizer.zero_grad(set_to_none=True)
            _, residuals = elastic_fields(model, pts, c, sf)
            loss = objective(residuals, pts, sf, settings["weights"], "uniform", 1.)
            loss.backward()
            optimizer.step()
        save(path, dict(settings=settings, case=case, seed=seed, model=cpu_state(model)))
    with torch.no_grad():
        _, residuals = elastic_fields(model, pts, c, sf)
        baseline = float(metric_value(sf, residuals["interface"], "uniform"))
        scales = {name: baseline / float(metric_value(sf, residuals["interface"], name))
                  for name in case["metrics"]}
    return model, (pts, c, s0, sf), scales


def run_metric(case, settings, seed, metric, initial, train, scale, directory, device):
    destination = directory / metric
    destination.mkdir(exist_ok=True)
    checkpoint = destination / "model.pt"
    pts, c, s0, sf = train
    test, tc, _, tsf = geometry(case, settings, device, evaluation=True)
    model = deepcopy(initial)
    rows, adam_updates, lbfgs_evaluations, elapsed = [], 0, 0, 0.
    phase = "adam"
    stored = None
    if checkpoint.exists():
        stored = torch.load(checkpoint, map_location=device, weights_only=False)
        if stored["phase"] == "complete" and "summary" in stored:
            return stored["summary"]
        model.load_state_dict(stored["model"])
        rows = stored["rows"]
        adam_updates, lbfgs_evaluations = stored["adam_updates"], stored["lbfgs_evaluations"]
        elapsed, phase = stored["elapsed_s"], stored["phase"]
    started = time.perf_counter()
    optimizer = None

    def record():
        observables, _, _ = elastic_observables(model, test, tc, tsf)
        with torch.no_grad():
            _, residuals = elastic_fields(model, pts, c, sf)
            loss = float(objective(residuals, pts, sf, settings["weights"], metric, scale))
        row = dict(case=case["name"], seed=seed, metric=metric, phase=phase,
                   adam_updates=adam_updates, lbfgs_evaluations=lbfgs_evaluations,
                   evaluations=adam_updates + lbfgs_evaluations,
                   elapsed_s=elapsed + time.perf_counter() - started,
                   objective=loss, metric_scale=scale, **observables)
        rows.append(row)
        write_rows(destination / "trajectory.csv", rows)
        save(checkpoint, dict(case=case, settings=settings, seed=seed, metric=metric,
                             metric_scale=scale, model=cpu_state(model), phase=phase,
                             adam_updates=adam_updates, lbfgs_evaluations=lbfgs_evaluations,
                             elapsed_s=row["elapsed_s"], rows=rows,
                             optimizer=optimizer.state_dict() if optimizer else None))
        print(json.dumps(row), flush=True)

    if not rows:
        record()
    if phase == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
        if stored is not None and stored["optimizer"] is not None:
            optimizer.load_state_dict(stored["optimizer"])
        while adam_updates < settings["adam_updates"]:
            fraction = adam_updates / settings["adam_updates"]
            optimizer.param_groups[0]["lr"] = settings["learning_rate"] * (0.1 + 0.9 * (1 + np.cos(np.pi * fraction)) / 2)
            optimizer.zero_grad(set_to_none=True)
            _, residuals = elastic_fields(model, pts, c, sf)
            loss = objective(residuals, pts, sf, settings["weights"], metric, scale)
            loss.backward()
            optimizer.step()
            adam_updates += 1
            if adam_updates % settings["report_every"] == 0:
                record()
        phase = "lbfgs"
        stored = None
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=80,
                                  max_eval=100, history_size=50, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-12, tolerance_change=1e-16)
    if stored is not None and stored["phase"] == "lbfgs":
        optimizer.load_state_dict(stored["optimizer"])

    def closure():
        nonlocal lbfgs_evaluations
        optimizer.zero_grad(set_to_none=True)
        _, residuals = elastic_fields(model, pts, c, sf)
        loss = objective(residuals, pts, sf, settings["weights"], metric, scale)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Nonfinite loss for {case['name']} {metric} {seed}")
        loss.backward()
        lbfgs_evaluations += 1
        return loss

    while settings["lbfgs_evaluations"] - lbfgs_evaluations >= 27:
        before = lbfgs_evaluations
        budgeted_lbfgs_step(optimizer, closure,
                           remaining_evals=settings["lbfgs_evaluations"] - lbfgs_evaluations,
                           chunk_max_eval=100)
        record()
        if lbfgs_evaluations - before == 1:
            break
    phase = "complete"
    record()
    summary, fields, residuals = elastic_observables(model, test, tc, tsf)
    summary = dict(rows[-1], **summary)
    np.savez_compressed(destination / "fields.npz",
                        xy=test.xy_all.cpu().numpy(),
                        displacement=fields["displacement"].cpu().numpy(),
                        stress_increment=fields["stress_increment"].cpu().numpy(),
                        wall_angles=test.wall_angles.cpu().numpy(),
                        wall_normal=test.nrm_wall.cpu().numpy(),
                        interface_residual=residuals["interface"].cpu().numpy(),
                        interface_coefficients=tsf.coeffs(residuals["interface"].T).cpu().numpy(),
                        n_pde=test.n_pde, n_wall=test.n_wall,
                        insitu=s0.cpu().numpy())
    save(checkpoint, dict(case=case, settings=settings, seed=seed, metric=metric,
                         metric_scale=scale, model=cpu_state(model), phase=phase,
                         summary=summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/interface_extension.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/interface_extension")
    parser.add_argument("--case")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--metric")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for case in settings["cases"]:
        if args.case is not None and args.case != case["name"]:
            continue
        for seed in settings["seeds"]:
            if args.seed is not None and args.seed != seed:
                continue
            directory = args.output / case["name"] / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            initial, train, scales = warm_start(case, settings, seed, directory, args.device)
            for metric in case["metrics"]:
                if args.metric is not None and args.metric != metric:
                    continue
                summaries.append(run_metric(case, settings, seed, metric, initial,
                                            train, scales[metric], directory, args.device))
    all_rows = []
    for path in sorted(args.output.glob("*/seed_*/*/trajectory.csv")):
        with path.open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if rows and rows[-1]["phase"] == "complete":
            all_rows.append(rows[-1])
    if all_rows:
        write_rows(args.output / "summary.csv", all_rows)
    print(json.dumps(dict(completed=len(summaries), output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
