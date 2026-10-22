"""Resolve a saved PINN history on an enlarged domain or refined discretization."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from interface_extension import cpu_state
from pilot.spectral_mixed import load_orientation_model, make_causal_model
from refine_crossed_history import warm_start
from run_orientation_extension import run_return_case, save, write_rows


def initialize_continuation(source, destination, step):
    """Reopen a saved increment without changing its preceding physical history."""
    _, original, model = load_orientation_model(source)
    if not 1 <= step <= original["completed_steps"]:
        raise ValueError("Restart a completed increment from its retained neural state")
    del model.increments[step:]
    saved = dict(original, model=cpu_state(model), model_steps=step,
        completed_steps=step - 1, evaluations=0, step_elapsed_s=0.,
        rows=[row for row in original["rows"] if row["step"] < step],
        initial_correction_scale=1.,
        trust_region_state=dict(solves=0, inner={}, last_result={}),
        initialization_checkpoint=str(source.resolve()))
    for key in ("predictor_restarts", "constrained_head_solves"):
        if key in saved:
            saved[key] = [row for row in saved[key] if row["step"] < step]
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / "model.pt", saved)
    (destination / "config.json").write_text(json.dumps(saved["settings"], indent=2) + "\n", encoding="utf-8")


def initialize(source, destination, *, radius=None, radial_order=None,
               angular_modes=None, radial_quadrature=None, angular_quadrature=None,
               load_steps=None):
    """Use old networks as guesses; retain only the exact elastic prefix."""
    bundle, original, old_model = load_orientation_model(source)
    settings = deepcopy(original["settings"])
    settings.pop("source_total_predictor_steps", None)
    if original["completed_steps"] != settings["load_steps"]:
        raise ValueError("The source must contain a completed loading history")
    settings["cases"] = [original["case"]]
    if radius is not None:
        settings["radius"] = radius
    if load_steps is not None:
        settings["load_steps"] = load_steps
    if settings["load_steps"] % original["settings"]["load_steps"]:
        raise ValueError("Uniform step refinement requires an integer subdivision")
    if radial_order is not None:
        settings["projection"]["radial_order"] = radial_order
    if angular_modes is not None:
        settings["projection"]["angular_modes"] = angular_modes
        settings["projection"]["angular_samples"] = max(
            settings["projection"]["angular_samples"], 4 * angular_modes)
    if radial_quadrature is not None:
        settings["quadrature"]["radial"] = radial_quadrature
    if angular_quadrature is not None:
        settings["quadrature"]["angular"] = angular_quadrature
    settings["weights"]["virtual_work"] = np.pi * (settings["radius"]**2 - 1)
    settings["question"] = "Resolve the saved neural history with the specified domain, projection, quadrature and uniform load increments."
    model = make_causal_model(settings, bundle["c"], original["case"])
    subdivisions = settings["load_steps"] // old_model.total_steps
    for previous in old_model.increments:
        if torch.count_nonzero(previous.coefficients()[..., :2]):
            break
        for _ in range(subdivisions):
            warm_start(model, previous)
    saved = dict(case=original["case"], settings=settings, model=cpu_state(model),
        model_steps=model.n_steps, completed_steps=model.n_steps,
        evaluations=0, step_elapsed_s=0., rows=[], c=bundle["c"], s0=bundle["s0"],
        trust_region_state=dict(solves=0, inner={}, last_result={}),
        initialization_checkpoint=str(source.resolve()))
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / "model.pt", saved)
    (destination / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def source_total_predictor(model, source):
    """Start at the source's cumulative displacement on the same physical domain."""
    current, old = model.increments[-1], source.increments[0]
    if (current.input_scale != old.input_scale or current.radial_order < old.radial_order
            or current.angular_modes < old.angular_modes):
        raise ValueError("Source-total prediction requires the same domain and a containing projection space")
    whole, remainder = divmod(model.n_steps * source.total_steps, model.total_steps)
    old_total = sum((inc.coefficients()[..., :2] for inc in source.increments[:whole]),
                    torch.zeros_like(old.coefficients()[..., :2]))
    if remainder:
        old_total = old_total + (remainder / model.total_steps) * source.increments[whole].coefficients()[..., :2]
    target = torch.zeros_like(current.coefficients()[..., :2])
    target[:old_total.shape[0], :old_total.shape[1]] = old_total
    target -= sum((inc.coefficients()[..., :2] for inc in model.increments[:-1]),
                  torch.zeros_like(target))
    subtraction = torch.view_as_complex(current.background_subtraction)
    subtraction[..., :2] += current.coefficients()[..., :2] - target
    current._coefficient_cache = None


def configure_source_total_predictor(destination, step):
    """Configure a future predictor or retain and restart the current increment."""
    checkpoint = destination / "model.pt"
    _, saved, model = load_orientation_model(checkpoint)
    steps = saved["settings"].get("source_total_predictor_steps", [])
    if step in steps:
        return
    if not model.n_steps <= step <= model.total_steps:
        raise ValueError("Select the current or a future increment, without rewinding a frozen prefix")
    saved["settings"]["source_total_predictor_steps"] = sorted(steps + [step])
    if step == model.n_steps:
        _, _, source = load_orientation_model(Path(saved["initialization_checkpoint"]))
        source_total_predictor(model, source)
        if saved["evaluations"]:
            backup = destination / f"before_source_total_step_{step}.pt"
            if backup.exists():
                raise FileExistsError(backup)
            shutil.copyfile(checkpoint, backup)
            shutil.copyfile(destination / "trajectory.csv", backup.with_suffix(".csv"))
            last = saved["rows"][-1]
            saved.setdefault("predictor_restarts", []).append({key: last[key] for key in (
                "step", "evaluations", "elapsed_s", "objective", "wall_traction_rms_mpa",
                "residual_projection_0_mm", "residual_projection_90_mm")})
        saved.update(model=cpu_state(model), completed_steps=step-1, evaluations=0,
            step_elapsed_s=0., initial_correction_scale=1.,
            rows=[row for row in saved["rows"] if row["step"] < step],
            trust_region_state=dict(solves=0, inner={}, last_result={}))
        write_rows(destination / "trajectory.csv", saved["rows"])
    save(checkpoint, saved)
    (destination / "config.json").write_text(json.dumps(saved["settings"], indent=2)+"\n", encoding="utf-8")


def append_guess(destination):
    _, saved, model = load_orientation_model(destination / "model.pt")
    _, _, original = load_orientation_model(Path(saved["initialization_checkpoint"]))
    subdivisions = model.total_steps // original.total_steps
    source_index = model.n_steps // subdivisions
    warm_start(model, original.increments[source_index])
    if model.n_steps in saved["settings"].get("source_total_predictor_steps", []):
        source_total_predictor(model, original)
    saved.update(model=cpu_state(model), model_steps=model.n_steps,
        evaluations=0, step_elapsed_s=0., initial_correction_scale=1.,
        trust_region_state=dict(solves=0, inner={}, last_result={}))
    save(destination / "model.pt", saved)


def resolve_through(destination, through_step, device, max_blocks, response_atol):
    """Freeze increments after their selected responses stabilize between solves.

    This is a response stopping rule for a resolution experiment, not an error
    bound or a constitutive/pointwise-equilibrium convergence claim.
    """
    checkpoint = destination / "model.pt"
    keys = ("nonelastic_0_mm", "nonelastic_90_mm", "source_training_0_mm", "source_training_90_mm")
    while True:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["completed_steps"] >= through_step:
            return
        if saved["model_steps"] == saved["completed_steps"]:
            append_guess(destination)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        step = saved["model_steps"]
        for _ in range(max_blocks):
            previous = saved["rows"][-1] if saved["rows"] and saved["rows"][-1]["step"] == step else None
            run_return_case(saved["case"], saved["settings"], destination, step,
                            device, trust_blocks=1, accept_step=None, initialize_checkpoint=None)
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            current = saved["rows"][-1]
            if previous is None or saved["trust_region_state"]["solves"] < 2:
                continue
            change = max(abs(current[key] - previous[key]) for key in keys)
            if change <= response_atol:
                row = dict(current, completed=True, response_change_mm=change,
                           response_tolerance_mm=response_atol)
                saved["completed_steps"] = step
                saved["rows"].append(row)
                write_rows(destination / "trajectory.csv", saved["rows"])
                save(checkpoint, saved)
                print(json.dumps(dict(step=step, response_change_mm=change,
                                      response_tolerance_mm=response_atol, completed=True)), flush=True)
                break
        else:
            raise RuntimeError(f"Step {step} remains pending after {max_blocks} additional solves")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float)
    parser.add_argument("--radial-order", type=int)
    parser.add_argument("--angular-modes", type=int)
    parser.add_argument("--radial-quadrature", type=int)
    parser.add_argument("--angular-quadrature", type=int)
    parser.add_argument("--load-steps", type=int)
    parser.add_argument("--restart-step", type=int,
                        help="Continue a retained increment on its unchanged prefix and discretization in a new output directory")
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--accept-step", type=int)
    parser.add_argument("--source-total-predictor-step", type=int,
                        help="Match cumulative source displacement at this step; retain any current trial before restarting")
    parser.add_argument("--through-step", type=int,
                        help="Recompute through this step using response stability between solves")
    parser.add_argument("--response-atol-mm", type=float, default=1e-4)
    parser.add_argument("--max-blocks-per-step", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = args.output / "model.pt"
    if not checkpoint.exists():
        changes = {key: getattr(args, key) for key in (
            "radius", "radial_order", "angular_modes", "radial_quadrature",
            "angular_quadrature", "load_steps")}
        if args.restart_step is not None:
            if any(value is not None for value in changes.values()):
                raise ValueError("Restart keeps the source domain and discretization")
            initialize_continuation(args.source, args.output, args.restart_step)
        else:
            initialize(args.source, args.output, **changes)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if Path(saved["initialization_checkpoint"]).resolve() != args.source.resolve():
        raise ValueError("Resume with the original source checkpoint")
    actual = dict(radius=saved["settings"]["radius"],
        load_steps=saved["settings"]["load_steps"],
        radial_order=saved["settings"]["projection"]["radial_order"],
        angular_modes=saved["settings"]["projection"]["angular_modes"],
        radial_quadrature=saved["settings"]["quadrature"]["radial"],
        angular_quadrature=saved["settings"]["quadrature"]["angular"])
    for key, value in actual.items():
        requested = getattr(args, key)
        if requested is not None and requested != value:
            raise ValueError(f"Cannot change {key} when resuming this history")
    if args.source_total_predictor_step is not None:
        configure_source_total_predictor(args.output, args.source_total_predictor_step)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if args.accept_step is not None:
        step = args.accept_step
        head_solved = any(row["step"] == step for row in saved.get("constrained_head_solves", []))
        if (saved["model_steps"] != step or saved["completed_steps"] != step - 1
                or not (saved["trust_region_state"]["solves"] or head_solved)
                or not saved["rows"] or saved["rows"][-1]["step"] != step):
            raise ValueError("Accept only an assessed pending increment")
        saved["completed_steps"] = step
        saved["rows"].append(dict(saved["rows"][-1], completed=True))
        write_rows(args.output / "trajectory.csv", saved["rows"])
        save(checkpoint, saved)
    if args.through_step is not None:
        if (not 1 <= args.through_step <= saved["settings"]["load_steps"]
                or args.response_atol_mm <= 0 or args.max_blocks_per_step < 1):
            raise ValueError("Specify an available load step and positive response tolerance")
        resolve_through(args.output, args.through_step, args.device,
                        args.max_blocks_per_step, args.response_atol_mm)
        return
    if saved["completed_steps"] < saved["settings"]["load_steps"]:
        if saved["model_steps"] == saved["completed_steps"]:
            append_guess(args.output)
        through_step = saved["completed_steps"] + 1
    else:
        through_step = saved["completed_steps"]
    run_return_case(saved["case"], saved["settings"], args.output, through_step,
                    args.device, trust_blocks=args.blocks, accept_step=None,
                    initialize_checkpoint=None)


if __name__ == "__main__":
    main()
