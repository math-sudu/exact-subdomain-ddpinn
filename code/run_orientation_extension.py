"""Causal unloading with independently varied weak-plane orientation."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from publication_model import load_model
from extract_unloading_results import sample_fields, stress_path_rows
from interface_extension import cpu_state, geometry, objective
from pilot import config as C, plasticity as P
from pilot.causal_mixed import current_residuals, prefix_cache
from pilot.spectral_mixed import make_causal_model, freeze_history
from pilot.lbfgs import budgeted_lbfgs_step
from run_interface_extension import ROOT, save, write_rows


@torch.no_grad()
def wall_response(model, params, insitu, count=720):
    angle = np.arange(count) * 2 * np.pi / count
    xy = np.stack((np.cos(angle), np.sin(angle)), -1)
    data = sample_fields(dict(params=params, s0=insitu), model, xy)
    convergence = -(data["displacement_mm"] * xy[:, None]).sum(-1)
    slip = data["accumulated_slip_strain"]
    return dict(convergence_max_mm=float(convergence[:, -1].max()),
                convergence_mean_mm=float(convergence[:, -1].mean()),
                slip_max_percent=float(slip[:, -1].max() * 100),
                convergence_angle_deg=float(np.rad2deg(angle[convergence[:, -1].argmax()])),
                slip_angle_deg=float(np.rad2deg(angle[slip[:, -1].argmax()])))


def export_case(destination, case, settings, model, c, s0, params):
    bundle = dict(params=params, c=c, s0=s0)
    radius = 1 + (settings["radius"] - 1) * np.linspace(0, 1, 129)**2
    radius = np.sqrt((radius[:-1]**2 + radius[1:]**2) / 2)
    angles = np.arange(720) * 2 * np.pi / 720
    unit = np.stack((np.cos(angles), np.sin(angles)), -1)
    xy = (radius[:, None, None] * unit[None]).reshape(-1, 2)
    data = sample_fields(bundle, model, xy)
    arrays = {key: value.reshape(len(radius), len(angles), *value.shape[1:]).astype(np.float32)
              for key, value in data.items()}
    np.savez_compressed(destination / "fields.npz", radius_over_a=radius,
                        angle_deg=np.rad2deg(angles), **arrays)
    wall = sample_fields(bundle, model, unit)
    radial = -(wall["displacement_mm"] * unit[:, None]).sum(-1)
    locations = {}
    for name, index in (("maximum_convergence", radial[:, -1].argmax()),
                        ("maximum_slip", wall["accumulated_slip_strain"][:, -1].argmax())):
        locations[name] = dict(angle_deg=float(np.rad2deg(angles[index])), xy_over_a=unit[index].tolist())
    rows = []
    for j, angle in enumerate(angles):
        for k in range(model.n_steps):
            rows.append(dict(angle_deg=float(np.rad2deg(angle)), load_step=k+1,
                             convergence_mm=float(radial[j, k]),
                             signed_slip_strain=float(wall["signed_slip_strain"][j, k]),
                             accumulated_slip_strain=float(wall["accumulated_slip_strain"][j, k])))
    write_rows(destination / "wall_profiles.csv", rows)
    write_rows(destination / "stress_paths.csv", stress_path_rows(bundle, model, locations))
    (destination / "metadata.json").write_text(json.dumps(dict(
        case=case, settings=settings, fixed_locations=locations,
        initialization="Reuse the common elastic prefix while both yield functions remain nonpositive on the independent evaluation points, with zero plastic outputs. Plastic initialization: " + settings["plastic_initialization"] + ". Retrain with the current weak-plane orientation.",
        displacement_mm_scale=C.U_REF*1000, strain_scale=C.U_REF/C.A_TUNNEL,
        insitu_stress_mpa=(s0.cpu().numpy()*C.SIGMA_REF/1e6).tolist()), indent=2)+"\n", encoding="utf-8")


def run_case(case, settings, destination, through_step, device):
    initialization = settings["plastic_initialization"]
    if initialization not in ("preceding_trained_increment", "corresponding_original_increment"):
        raise ValueError(f"Unknown plastic initialization: {initialization}")
    destination.mkdir(parents=True, exist_ok=True)
    pts, c, s0, sf = geometry(case, settings, device)
    params = P.WeakPlaneParameters(**settings["physics"], joint_normal_deg=case["joint_normal_deg"]).checked()
    run = ROOT / settings["initialization_run"]
    _, _, initial = load_model(run, run / "model.pt", device)
    if settings.get("representation") == "spectral":
        projected = make_causal_model(settings, c)
        for _ in range(initial.n_steps):
            projected.append_increment(copy_previous=False)
        projected.load_state_dict(initial.state_dict())
        initial = projected
    freeze_history(initial)
    elastic_steps = 0
    if settings["reuse_inactive_elastic_prefix"]:
        probe, _, _, _ = geometry(case, settings, device, evaluation=True)
        with torch.no_grad():
            stress = torch.cat([initial(x)[1] + s0 for x in probe.xy_all.split(2048)])
            maxima = torch.maximum(P.matrix_yield(stress, params), P.joint_yield(stress, params)).amax(0)
            for maximum in maxima:
                if maximum > 0:
                    break
                elastic_steps += 1
        del probe, stress
    model = make_causal_model(settings, c)
    checkpoint = destination / "model.pt"
    completed, rows, saved = 0, [], None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["case"] != case or saved["settings"] != settings:
            raise ValueError(f"Changed experiment inputs at {checkpoint}")
        for _ in range(saved["model_steps"]):
            model.append_increment(copy_previous=False)
        model.load_state_dict(saved["model"])
        completed, rows = saved["completed_steps"], saved["rows"]
    for step in range(completed + 1, min(through_step, settings["load_steps"]) + 1):
        resuming = saved is not None and model.n_steps == step and completed == step-1
        if resuming:
            current = model.increments[-1]
            del model.increments[-1]
        else:
            previous = (initial.increments[step-1]
                        if step <= elastic_steps or initialization == "corresponding_original_increment"
                        else model.increments[-1] if model.n_steps else initial.increments[0])
            current = deepcopy(previous).requires_grad_(True)
        elastic_reuse = step <= elastic_steps
        if elastic_reuse:
            with torch.no_grad():
                current.kinematic_head.weight[2:].zero_()
                current.kinematic_head.bias[2:].zero_()
        freeze_history(model)
        cache = prefix_cache(model, pts.xy_all, c, s0, params)
        model.increments.append(current)
        current.requires_grad_(True)
        optimizer = torch.optim.LBFGS(current.parameters(), lr=1., max_iter=80, max_eval=100,
                                      history_size=50, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-12, tolerance_change=1e-16)
        evaluations, elapsed = 0, 0.
        if resuming:
            optimizer.load_state_dict(saved["optimizer"])
            evaluations, elapsed = saved["evaluations"], saved["step_elapsed_s"]
        started = time.perf_counter()

        def record(is_complete=False):
            with torch.no_grad():
                h, residuals = current_residuals(current, cache, pts, c, s0, sf, params, step)
                loss = objective(residuals, pts, sf, settings["weights"], "compliance", 4*np.pi*sf.R**2)
            response = wall_response(model, params, s0)
            row = dict(case=case["name"], step=step,
                       phase="elastic_reuse" if elastic_reuse else "plastic_training",
                       evaluations=evaluations,
                       elapsed_s=elapsed+time.perf_counter()-started,
                       objective=float(loss), completed=is_complete, **response)
            rows.append(row)
            write_rows(destination / "trajectory.csv", rows)
            save(checkpoint, dict(case=case, settings=settings, model=cpu_state(model),
                                 model_steps=model.n_steps, completed_steps=step if is_complete else step-1,
                                 evaluations=evaluations, step_elapsed_s=row["elapsed_s"],
                                 rows=rows, optimizer=None if is_complete else optimizer.state_dict(),
                                 c=c.cpu(), s0=s0.cpu()))
            print(json.dumps(row), flush=True)

        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            _, residuals = current_residuals(current, cache, pts, c, s0, sf, params, step)
            loss = objective(residuals, pts, sf, settings["weights"], "compliance", 4*np.pi*sf.R**2)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite objective at {case['name']} load {step}")
            loss.backward()
            evaluations += 1
            return loss

        record()
        if elastic_reuse:
            record(is_complete=True)
            completed, saved = step, None
            del cache
            continue
        while settings["lbfgs_evaluations_per_step"] - evaluations >= 27:
            before = evaluations
            budgeted_lbfgs_step(optimizer, closure,
                               remaining_evals=settings["lbfgs_evaluations_per_step"] - evaluations,
                               chunk_max_eval=100)
            record()
            if evaluations - before == 1:
                break
        record(is_complete=True)
        completed, saved = step, None
        del cache
    if completed == settings["load_steps"]:
        freeze_history(model)
        export_case(destination, case, settings, model, c, s0, params)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/orientation_extension.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/orientation_extension")
    parser.add_argument("--case")
    parser.add_argument("--through-step", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1,
                        help="PyTorch CPU threads; useful for CPU training or field export.")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    for case in settings["cases"]:
        if args.case is None or case["name"] == args.case:
            completed = run_case(case, settings, args.output / case["name"], args.through_step, args.device)
            print(json.dumps(dict(case=case["name"], completed_steps=completed)), flush=True)


if __name__ == "__main__":
    main()
