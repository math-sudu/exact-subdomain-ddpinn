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
from interface_extension import cpu_state, geometry, objective, regional_power
from pilot import config as C, plasticity as P
from pilot.augmented_constraints import RegionalAugmentation
from pilot.causal_mixed import current_residuals, prefix_cache
from pilot.spectral_mixed import CircularGridEvaluation, make_causal_model, freeze_history, load_orientation_model
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
    initialization = ("Accept the common circular elastic displacement and stress increments while both yield functions remain nonpositive. Train neural corrections to that elastic background with the current weak-plane orientation."
                      if settings.get("elastic_prefix") == "circular" else
                      "Train every increment with the current weak-plane orientation. Plastic initialization: " + settings["plastic_initialization"] + ".")
    (destination / "metadata.json").write_text(json.dumps(dict(
        case=case, settings=settings, fixed_locations=locations,
        initialization=initialization,
        displacement_mm_scale=C.U_REF*1000, strain_scale=C.U_REF/C.A_TUNNEL,
        insitu_stress_mpa=(s0.cpu().numpy()*C.SIGMA_REF/1e6).tolist()), indent=2)+"\n", encoding="utf-8")


def run_return_case(case, settings, destination, through_step, device, *,
                    trust_blocks, accept_step, initialize_checkpoint):
    """Continue neural displacement increments with exact local weak-plane return."""
    from pilot.orientation_tangent import neural_primal_step
    from pilot.returned_orientation import ReturnedOrientationResidual

    checkpoint = destination/"model.pt"
    destination.mkdir(parents=True, exist_ok=True)
    controls = settings["trust_region"]
    if checkpoint.exists():
        bundle, saved, model = load_orientation_model(checkpoint, device)
        resume_settings = deepcopy(saved["settings"])
        # The dispatch list may grow without changing this case's physics.
        resume_settings.pop("cases", None)
        changed_equilibrium = (resume_settings.get("equilibrium_form", "strong")
                               != settings.get("equilibrium_form", "strong"))
        if "equilibrium_form" in settings:
            resume_settings["equilibrium_form"] = settings["equilibrium_form"]
        else:
            resume_settings.pop("equilibrium_form", None)
        for key in ("linear_iterations", "evaluations_per_solve", "polish_evaluations"):
            resume_settings["trust_region"][key] = controls[key]
        if (saved["case"] != case
                or resume_settings != {key: value for key, value in settings.items() if key != "cases"}):
            raise ValueError(f"Changed returned-history inputs at {checkpoint}")
        if changed_equilibrium:
            saved["trust_region_state"] = dict(solves=0, inner={}, last_result={})
    else:
        if initialize_checkpoint is None:
            raise ValueError("A returned-history run requires --initialize-checkpoint at plastic onset")
        bundle, saved, initial = load_orientation_model(initialize_checkpoint, device)
        for key in ("hidden", "radius", "load_steps", "projection", "physics", "elastic_prefix"):
            if saved["settings"][key] != settings[key]:
                raise ValueError(f"Changed physical initializer setting: {key}")
        onset = saved["model_steps"]
        if (saved["case"] != case or onset < 1
                or saved["completed_steps"] not in (onset-1, onset)):
            raise ValueError("Initialize from the same case with a pending or completed onset increment")
        for increment in initial.increments[:onset-1]:
            if torch.count_nonzero(increment.coefficients()):
                raise ValueError("Mixed-to-return conversion requires the exact elastic prefix")
        model = make_causal_model(settings, bundle["c"].to(device), case)
        for _ in range(saved["model_steps"]):
            model.append_increment(copy_previous=False)
        state = model.state_dict()
        state.update(initial.state_dict())
        model.load_state_dict(state)
        # A completed mixed onset supplies weights, not return-history acceptance.
        saved = dict(saved, completed_steps=onset-1, rows=[], evaluations=0, step_elapsed_s=0.,
                     trust_region_state=dict(solves=0, inner={}, last_result={}),
                     initialization_checkpoint=str(initialize_checkpoint.resolve()))
        del initial
    saved["settings"] = settings
    completed, rows = saved["completed_steps"], saved["rows"]
    pending = model.n_steps == completed+1
    if accept_step is not None and (not pending or accept_step != model.n_steps
            or not saved["trust_region_state"]["solves"] or accept_step > through_step):
        raise ValueError("Accept only the assessed pending returned-history increment")
    for step in range(completed+1, min(through_step, settings["load_steps"])+1):
        if model.n_steps < step:
            model.append_increment(copy_previous=True)
            saved.update(evaluations=0, step_elapsed_s=0.,
                         trust_region_state=dict(solves=0, inner={}, last_result={}))
        freeze_history(model)
        problem = ReturnedOrientationResidual(model, saved, bundle["params"])
        current = model.increments[-1]
        current.requires_grad_(True)
        state = saved["trust_region_state"]
        evaluations, elapsed = saved["evaluations"], saved["step_elapsed_s"]
        started = time.perf_counter()

        def record(is_complete=False):
            with torch.no_grad():
                coefficients = current.coefficients()
                assessment = problem.assessment(coefficients)
                objective_value = float(problem(coefficients).square().sum())
            row = dict(case=case["name"], step=step, phase="returned_neural_training",
                equilibrium_form=problem.equilibrium_form,
                evaluations=evaluations, elapsed_s=elapsed+time.perf_counter()-started,
                objective=objective_value, completed=is_complete, trust_solves=state["solves"],
                trust_status=state["last_result"].get("status"),
                primal_optimality=state["last_result"].get("optimality"), **assessment)
            rows.append(row)
            write_rows(destination/"trajectory.csv", rows)
            payload = dict(case=case, settings=settings, model=cpu_state(model),
                model_steps=model.n_steps, completed_steps=step if is_complete else step-1,
                evaluations=evaluations, step_elapsed_s=row["elapsed_s"], rows=rows,
                c=problem.c.cpu(), s0=problem.s0.cpu(), trust_region_state=state,
                initialization_checkpoint=saved["initialization_checkpoint"])
            save(checkpoint, payload)
            print(json.dumps(row), flush=True)

        record()
        if accept_step == step:
            record(is_complete=True)
            completed = step
            continue
        for _ in range(trust_blocks):
            before = evaluations
            state["last_result"] = {}
            def accepted(intermediate_result):
                nonlocal evaluations
                evaluations = before+int(intermediate_result.nfev)
                record()
            result = neural_primal_step(problem, controls, state["inner"], accepted)
            evaluations = before+result["evaluations"]
            state["solves"] += 1
            state["last_result"] = result
            if result["status"] == 1:
                state["inner"] = {}
            record()
        return completed
    if completed == settings["load_steps"]:
        freeze_history(model)
        export_case(destination, case, settings, model, bundle["c"], bundle["s0"], bundle["params"])
    return completed


def run_case(case, settings, destination, through_step, device, *, trust_blocks=1, accept_step=None,
             initialize_checkpoint=None):
    if settings.get("constitutive_update") == "joint_return":
        return run_return_case(case, settings, destination, through_step, device,
            trust_blocks=trust_blocks, accept_step=accept_step, initialize_checkpoint=initialize_checkpoint)
    if initialize_checkpoint is not None:
        raise ValueError("--initialize-checkpoint is for mixed-to-return onset conversion")
    trust_controls = settings.get("trust_region")
    if trust_controls is not None:
        if not 0 < trust_controls["gradient_reduction"] < 1:
            raise ValueError("gradient_reduction must lie between zero and one")
        if trust_blocks < 0:
            raise ValueError("trust_blocks cannot be negative")
    initialization = settings["plastic_initialization"]
    if initialization not in ("preceding_trained_increment", "corresponding_original_increment"):
        raise ValueError(f"Unknown plastic initialization: {initialization}")
    destination.mkdir(parents=True, exist_ok=True)
    pts, c, s0, sf = geometry(case, settings, device)
    params = P.WeakPlaneParameters(**settings["physics"], joint_normal_deg=case["joint_normal_deg"]).checked()
    model = make_causal_model(settings, c, case)
    work_metric = None
    if "virtual_work" in settings["weights"]:
        from pilot.virtual_work import VirtualWorkMetric
        material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
        work_metric = VirtualWorkMetric(c, C.nondim_lambda(material), settings, pts)
    initial = None
    if settings.get("elastic_prefix") != "circular" or initialization == "corresponding_original_increment":
        run = ROOT / settings["initialization_run"]
        _, _, initial = load_model(run, run / "model.pt", device)
        if settings.get("representation") == "spectral":
            initial_settings = {key: value for key, value in settings.items() if key != "elastic_prefix"}
            initial_settings["projection"] = {key: value for key, value in settings["projection"].items()
                                              if key != "stress_coefficient_correction"}
            projected = make_causal_model(initial_settings, c, case)
            for _ in range(initial.n_steps):
                projected.append_increment(copy_previous=False)
            projected.load_state_dict(initial.state_dict())
            initial = projected
        freeze_history(initial)
    elastic_steps = 0
    if settings.get("elastic_prefix") == "circular":
        probe, _, _, _ = geometry(case, settings, device, evaluation=True)
        elastic_increment = model.make_increment()
        with torch.no_grad():
            ds = torch.cat([elastic_increment(x)[1] for x in probe.xy_all.split(2048)])
            for step in range(1, settings["load_steps"]+1):
                stress = step*ds+s0
                maximum = torch.maximum(P.matrix_yield(stress, params), P.joint_yield(stress, params)).max()
                if maximum > 0:
                    break
                elastic_steps += 1
        del probe, stress, ds, elastic_increment
    checkpoint = destination / "model.pt"
    completed, rows, saved = 0, [], None
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        resume_settings = deepcopy(saved["settings"])
        if trust_controls is not None and "trust_region" in resume_settings:
            # Resource limits may change while the same pending inner problem,
            # column scales and stopping target continue from the checkpoint.
            for key in ("linear_iterations", "evaluations_per_solve"):
                resume_settings["trust_region"][key] = trust_controls[key]
            if "polish_evaluations" in trust_controls:
                resume_settings["trust_region"]["polish_evaluations"] = trust_controls["polish_evaluations"]
        if saved["case"] != case or resume_settings != settings:
            raise ValueError(f"Changed experiment inputs at {checkpoint}")
        for _ in range(saved["model_steps"]):
            model.append_increment(copy_previous=False)
        model.load_state_dict(saved["model"])
        completed, rows = saved["completed_steps"], saved["rows"]
    if accept_step is not None and (trust_controls is None or saved is None or
            accept_step != saved["model_steps"] or accept_step != completed+1 or
            not saved["trust_region_state"]["solves"] or accept_step > through_step):
        raise ValueError("Accept only the assessed pending trust-region increment")
    for step in range(completed + 1, min(through_step, settings["load_steps"]) + 1):
        resuming = saved is not None and model.n_steps == step and completed == step-1
        if resuming:
            current = model.increments[-1]
            del model.increments[-1]
        else:
            if step <= elastic_steps:
                current = model.make_increment()
            elif initial is None:
                current = (deepcopy(model.increments[-1]).requires_grad_(True)
                           if model.n_steps else model.make_increment())
            else:
                previous = (initial.increments[step-1]
                            if initialization == "corresponding_original_increment"
                            else model.increments[-1] if model.n_steps else initial.increments[0])
                if settings.get("elastic_prefix") == "circular":
                    current = model.make_increment()
                    current.initialize_from_total(previous)
                else:
                    current = deepcopy(previous).requires_grad_(True)
        elastic_reuse = step <= elastic_steps
        if elastic_reuse:
            with torch.no_grad():
                for head in (current.kinematic_head, current.stress_head):
                    head.weight.zero_()
                    head.bias.zero_()
        freeze_history(model)
        cache = prefix_cache(model, pts.xy_all, c, s0, params)
        if getattr(current, "elastic_background", None) is not None:
            with torch.no_grad():
                cache["increment_background"] = current.elastic_background.spatial_fields(pts.xy_all)
                n0, n1 = pts.n_pde, pts.n_pde+pts.n_wall
                nt = case.get("quadrature", settings["quadrature"])["angular"]
                cache["increment_grid"] = CircularGridEvaluation(current, (
                    pts.xy_all[:n0].reshape(-1, nt, 2),
                    pts.xy_all[n0:n1][None], pts.xy_all[n1:][None]))
        if work_metric is not None:
            with torch.no_grad():
                cache["displacement_correction"] = sum(
                    (inc.coefficients()[..., :2] for inc in model.increments),
                    torch.zeros_like(current.coefficients()[..., :2]))
        model.increments.append(current)
        current.requires_grad_(True)
        optimizer = torch.optim.LBFGS(current.parameters(), lr=1., max_iter=80, max_eval=100,
                                      history_size=50, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-12, tolerance_change=1e-16)
        augmentations = {}
        for name, components in (("complementarity", 2), ("constitutive", 4)):
            key = name + "_augmentation"
            if key in settings:
                augmentations[name] = RegionalAugmentation(
                    pts, settings["weights"][name],
                    settings[key]["evaluations_per_update"], components)
        evaluations, elapsed = 0, 0.
        trust_state = dict(warmup_complete=False, solves=0, inner={}, last_result={})
        if resuming:
            if saved["optimizer"] is not None:
                optimizer.load_state_dict(saved["optimizer"])
            evaluations, elapsed = saved["evaluations"], saved["step_elapsed_s"]
            for name, augmentation in augmentations.items():
                augmentation.load_state_dict(saved[name + "_augmentation"])
            if trust_controls is not None:
                trust_state = saved["trust_region_state"]
        started = time.perf_counter()

        def evaluate_loss():
            coefficients = current.coefficients() if work_metric is not None else None
            h, residuals = current_residuals(current, cache, pts, c, s0, sf, params, step,
                                            projected_coefficients=coefficients)
            loss = objective(residuals, pts, sf, settings["weights"], "compliance", 4*np.pi*sf.R**2)
            work = None
            if work_metric is not None:
                correction = cache["displacement_correction"]+coefficients[..., :2]
                work = work_metric(correction, h["plastic_strain"][:pts.n_pde])
                loss = loss+settings["weights"]["virtual_work"]*work
            base_objective = loss
            for name, augmentation in augmentations.items():
                loss = loss + augmentation.linear_term(residuals[name])
            return loss, work, base_objective, residuals

        def record(is_complete=False, update_duals=()):
            with torch.no_grad():
                loss, work, base_objective, residuals = evaluate_loss()
                if update_duals:
                    for name in update_duals:
                        augmentations[name].update(residuals[name], evaluations)
                    # Function values and secant pairs belong to the old dual.
                    optimizer.state.clear()
                    loss = base_objective + sum(
                        augmentation.linear_term(residuals[name])
                        for name, augmentation in augmentations.items())
            response = wall_response(model, params, s0)
            row = dict(case=case["name"], step=step,
                       phase="exact_elastic" if elastic_reuse else "plastic_training",
                       evaluations=evaluations,
                       elapsed_s=elapsed+time.perf_counter()-started,
                       objective=float(loss), completed=is_complete, **response)
            if work is not None:
                row["virtual_work"] = float(work)
            if augmentations:
                row["base_objective"] = float(base_objective)
            for name, augmentation in augmentations.items():
                row[name + "_rms"] = float(regional_power(residuals[name], pts).sqrt())
                count_key = "dual_updates" if name == "complementarity" else name + "_dual_updates"
                row[count_key] = augmentation.dual_updates
            if trust_controls is not None:
                row.update(trust_solves=trust_state["solves"],
                           trust_linear_iterations=trust_controls["linear_iterations"],
                           trust_evaluation_limit=trust_controls["evaluations_per_solve"],
                           primal_polish_limit=trust_controls.get("polish_evaluations", 0),
                           trust_status=trust_state["last_result"].get("status"),
                           primal_optimality=trust_state["last_result"].get("optimality"),
                           primal_target=trust_state["inner"].get("gradient_tolerance"))
            rows.append(row)
            write_rows(destination / "trajectory.csv", rows)
            payload = dict(case=case, settings=settings, model=cpu_state(model),
                           model_steps=model.n_steps, completed_steps=step if is_complete else step-1,
                           evaluations=evaluations, step_elapsed_s=row["elapsed_s"],
                           rows=rows, optimizer=None if is_complete else optimizer.state_dict(),
                           c=c.cpu(), s0=s0.cpu())
            for name, augmentation in augmentations.items():
                payload[name + "_augmentation"] = augmentation.state_dict()
            if trust_controls is not None:
                payload["trust_region_state"] = trust_state
            save(checkpoint, payload)
            print(json.dumps(row), flush=True)

        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            loss, _, _, _ = evaluate_loss()
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
        if trust_controls is not None:
            if accept_step == step:
                record(is_complete=True)
                completed, saved = step, None
                del cache
                continue
            if not trust_state["warmup_complete"]:
                while trust_controls["warmup_evaluations"] - evaluations >= 27:
                    before = evaluations
                    budgeted_lbfgs_step(optimizer, closure,
                        remaining_evals=trust_controls["warmup_evaluations"]-evaluations,
                        chunk_max_eval=100)
                    record()
                    if evaluations-before == 1:
                        break
                optimizer.state.clear()
                trust_state["warmup_complete"] = True
                record()
            from pilot.orientation_tangent import OrientationResidual, neural_primal_step
            problem = OrientationResidual(model, dict(settings=settings, case=case), params, duals=False)
            problem.duals = {name: augmentation.dual for name, augmentation in augmentations.items()}
            for _ in range(trust_blocks):
                before = evaluations
                trust_state["last_result"] = {}

                def accepted(intermediate_result):
                    nonlocal evaluations
                    evaluations = before+int(intermediate_result.nfev)
                    record()

                result = neural_primal_step(problem, trust_controls, trust_state["inner"], accepted)
                evaluations = before+result["evaluations"]
                trust_state["solves"] += 1
                trust_state["last_result"] = result
                update_duals = tuple(augmentations) if result["status"] == 1 else ()
                # Save the accepted neural fields and the corresponding dual
                # update together. A resource stop preserves the same inner target.
                if result["status"] == 1:
                    trust_state["inner"] = {}
                record(update_duals=update_duals)
            # A finite compute block is a resumable candidate, not a frozen
            # plastic history. Assess it before explicitly accepting this step.
            return completed
        while settings["lbfgs_evaluations_per_step"] - evaluations >= 27:
            before = evaluations
            budgeted_lbfgs_step(optimizer, closure,
                               remaining_evals=settings["lbfgs_evaluations_per_step"] - evaluations,
                               chunk_max_eval=100)
            stationary = evaluations - before == 1
            update_duals = [name for name, augmentation in augmentations.items()
                            if stationary or evaluations >= augmentation.next_update_evaluation]
            record(update_duals=update_duals)
            if stationary and not augmentations:
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
    parser.add_argument("--trust-blocks", type=int, default=1,
                        help="Fixed-dual solve blocks before saving a pending increment for assessment.")
    parser.add_argument("--accept-step", type=int,
                        help="Freeze this already assessed pending trust-region increment, then continue.")
    parser.add_argument("--initialize-checkpoint", type=Path,
                        help="Initialize local-return training from a pending or completed mixed onset checkpoint.")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    for case in settings["cases"]:
        if args.case is None or case["name"] == args.case:
            completed = run_case(case, settings, args.output / case["name"], args.through_step, args.device,
                                 trust_blocks=args.trust_blocks, accept_step=args.accept_step,
                                 initialize_checkpoint=args.initialize_checkpoint)
            print(json.dumps(dict(case=case["name"], completed_steps=completed)), flush=True)


if __name__ == "__main__":
    main()
