"""Train and resume mapped neural increments; each saved history replays in space."""
import argparse
import csv
import json
import time
from itertools import count
from pathlib import Path

import numpy as np
import torch

from oespinn import config as C
from oespinn.horseshoe import HorseshoeWork, MappedIncrement, new_increment
from oespinn.horseshoe_geometry import HorseshoeGeometry
from oespinn.plasticity import WeakPlaneParameters

ROOT = Path(__file__).resolve().parents[1]


def make_work(settings, device, *, quadrature=None):
    geometry = HorseshoeGeometry(ROOT/settings["geometry"], settings["length_scale_m"], settings["interface_radius"])
    if "elastic_constants" in settings:
        from sf_core.material import TIMaterial
        constants = settings["elastic_constants"]
        material = TIMaterial(settings["case"], constants["Eh_gpa"]*1e9, constants["Ev_gpa"]*1e9,
                              constants["nu_vh"], constants["nu_hh"], constants["Gvh_gpa"]*1e9,
                              np.deg2rad(constants["bedding_dip_deg"]))
    else:
        material = C.material(**settings["material"])
    params = WeakPlaneParameters(**settings["physics"])
    q = quadrature or settings["quadrature"]
    work = HorseshoeWork(geometry, material, params, settings["insitu_mpa"],
        settings["radial_order"], settings["angular_modes"], q["radial"], q["angular"],
        boundary_points=q["boundary"], device=device, shift=q.get("shift", 0.0),
        return_law=settings.get("return_law", "joint"), tension_mpa=settings.get("tension_mpa"),
        support_pressure_mpa=settings.get("support_pressure_mpa", 0.0))
    return geometry, work


def load_models(saved, geometry, device):
    settings = saved["settings"]
    models = []
    for state in saved["models"]:
        model = new_increment(None, geometry, settings, device)
        model.load_state_dict({key: value for key, value in state.items() if key != "kinematic_head.bias"})
        models.append(model.requires_grad_(False))
    return models


def load_coefficients(saved, geometry, device, *, count=None):
    """Read the physical history from either supported solution parameterization."""
    count = saved["completed_steps"] if count is None else count
    if saved.get("solver") in ("direct_spectral", "reduced_spectral"):
        return [c.to(device) for c in saved["coefficients"][:count]], None
    models = load_models(saved, geometry, device)[:count]
    with torch.no_grad():
        retained = [model.coefficients() for model in models]
    difference = max(float((c-stored.to(c)).abs().max())
                     for c, stored in zip(retained, saved["coefficients"]))
    if any(not torch.allclose(c, stored.to(c), atol=1e-11, rtol=1e-12)
           for c, stored in zip(retained, saved["coefficients"])):
        raise ValueError("The retained neural networks and saved projections differ")
    return retained, difference


def write_results(output, work, coefficients, releases, *, prefix_cache=None, evaluated=None):
    rows = work.readouts(coefficients, releases, prefix_cache=prefix_cache, evaluated=evaluated)
    with (output/"responses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output/"summary.json").write_text(json.dumps(rows[-1], indent=2)+"\n", "utf-8")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--through-step", type=int)
    parser.add_argument("--blocks", type=int, help="Optional blocks for one invocation; default continues to response stability")
    parser.add_argument("--optimizer", choices=("lbfgs", "tangent"), default="lbfgs")
    parser.add_argument("--tangent-evaluations", type=int, default=15)
    parser.add_argument("--linear-iterations", type=int, default=500)
    parser.add_argument("--equilibrium-tolerance-mm", type=float,
                        help="Stop inner optimization on gauge residual work, retaining the full residual for assessment")
    parser.add_argument("--response-relative-tolerance", type=float, default=0.,
                        help="Relative stability tolerance added to the configured absolute response tolerance")
    parser.add_argument("--initialize-from", help="Use corresponding neural states as predictors; replay and retrain every increment")
    parser.add_argument("--assess", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    settings = json.loads((ROOT/args.config).read_text("utf-8"))
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    geometry, work = make_work(settings, args.device)
    print(json.dumps({"event": "assembled", "seconds": time.perf_counter()-started,
                      "degrees_of_freedom": work.n*2, "area_m2": float(work.area)*geometry.length_scale**2}), flush=True)
    saved = torch.load(output/"model.pt", map_location=args.device, weights_only=False) if (output/"model.pt").exists() else None
    if saved and saved["settings"] != settings:
        raise ValueError("Resume configuration differs from the saved calculation")
    models = load_models(saved, geometry, args.device) if saved else []
    completed = saved["completed_steps"] if saved else 0
    trajectory = saved["trajectory"] if saved else []
    predictor = torch.load(ROOT/args.initialize_from, map_location=args.device, weights_only=False) if args.initialize_from else None
    if args.assess:
        coefficients = [model.coefficients().detach() for model in models]
        write_results(output, work, coefficients, [(i+1)/settings["load_steps"] for i in range(len(models))])
        return
    target = args.through_step or settings["load_steps"]
    for step in range(completed+1, target+1):
        if len(models) < step:
            if predictor is None:
                model = new_increment(models[-1] if models else None, geometry, settings, args.device)
            else:
                source_step = int(np.ceil(step/settings["load_steps"]*predictor["settings"]["load_steps"]))-1
                if source_step >= len(predictor["models"]):
                    raise ValueError("The predictor has no neural increment for the required release")
                model = new_increment(None, geometry, settings, args.device)
                model.load_state_dict({key: value for key, value in predictor["models"][source_step].items()
                                       if key != "kinematic_head.bias"})
            models.append(model)
        current = models[-1].requires_grad_(True)
        coefficients = [model.coefficients().detach() for model in models[:-1]]
        prefix = sum(coefficients, torch.zeros((work.n, 2), device=args.device))
        with torch.no_grad():
            prefix_cache = {name: work.replay(coefficients, basis) for name, basis in
                            (("volume", None), ("wall", work.wall_basis), ("outer", work.outer_basis))}
        previous_plastic = prefix_cache["volume"][-1]["plastic"] if coefficients else work.c.new_zeros((len(work.xy), 4))
        # A predictor is only an initial guess. Back it towards the retained
        # preceding state if its large strain increment has no admissible return.
        damping = 1.
        with torch.no_grad():
            while True:
                try:
                    work.residual(current.coefficients(), prefix, previous_plastic, step/settings["load_steps"])
                    break
                except FloatingPointError:
                    if damping == 0.:
                        raise
                    factor = .5 if damping > 2**-20 else 0.
                    current.kinematic_head.weight.mul_(factor)
                    damping *= factor
            if damping != 1.:
                print(json.dumps({"event": "predictor_damping", "step": step, "factor": damping}), flush=True)
        optimizer = torch.optim.LBFGS(current.parameters(), lr=1., max_iter=settings["training"]["iterations_per_block"],
            max_eval=settings["training"]["iterations_per_block"]*5//4,
            tolerance_grad=1e-12, tolerance_change=1e-15, history_size=100, line_search_fn="strong_wolfe")
        releases = [(i+1)/settings["load_steps"] for i in range(step)]
        def checkpoint(progress=None):
            torch.save(dict(settings=settings, models=[m.state_dict() for m in models],
                            coefficients=[c.cpu() for c in coefficients+[current.coefficients().detach()]],
                            completed_steps=completed, initialization=args.initialize_from,
                            predictor_completed_steps=predictor["completed_steps"] if predictor else None,
                            trajectory=trajectory, optimizer_progress=progress), output/"model.next.pt")
            (output/"model.next.pt").replace(output/"model.pt")
        response_names = [name+suffix for suffix in ("_mm", "_plastic_mm") for name in work.gauges()]
        previous_response = (np.array([trajectory[-1][name] for name in response_names])
                             if trajectory and trajectory[-1]["step"] == step else None)
        for block in range(args.blocks) if args.blocks is not None else count():
            evaluations = 0
            rejected_returns = 0
            before = time.perf_counter()
            def closure():
                nonlocal evaluations
                optimizer.zero_grad(set_to_none=True)
                r = work.residual(current.coefficients(), prefix, previous_plastic, releases[-1])
                loss = .5*r.square().sum()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Unsupported constitutive trial during neural optimization")
                loss.backward()
                evaluations += 1
                return loss
            if args.optimizer == "lbfgs":
                optimizer.step(closure)
            else:
                from oespinn.orientation_tangent import NeuralCoordinates, linear_operator, column_scale, trust_region
                coordinates = NeuralCoordinates(current)
                point = coordinates.zero()
                def function(delta):
                    nonlocal rejected_returns
                    try:
                        return work.residual(coordinates(delta), prefix, previous_plastic, releases[-1])
                    except FloatingPointError:
                        if torch.is_grad_enabled():
                            raise
                        # An unsupported trial cannot enter the retained history.
                        # A large objective rejects it in the trust-region ratio.
                        rejected_returns += 1
                        return delta.new_full((2*work.n,), 1e10)
                _, value, reverse = linear_operator(function, point, torch.ones_like(point))
                scale = column_scale(reverse, value, point, samples=16)
                del reverse, value
                def accepted(intermediate_result):
                    coordinates.apply(point.new_tensor(intermediate_result.x))
                    progress = dict(step=step, evaluations=int(intermediate_result.nfev),
                                    weak_relative=float(np.sqrt(2*intermediate_result.cost)))
                    checkpoint(progress)
                    print(json.dumps({"event": "iterate", **progress}), flush=True)
                fit = trust_region(function, point, scale, linear_iterations=args.linear_iterations,
                    evaluations=args.tangent_evaluations, gradient_tolerance=1e-13,
                    linear_tolerance=1e-5, callback=accepted)
                coordinates.apply(point.new_tensor(fit.x))
                evaluations = fit.nfev
            all_coefficients = coefficients+[current.coefficients().detach()]
            try:
                rows = write_results(output, work, all_coefficients, releases, prefix_cache=prefix_cache)
            except FloatingPointError as error:
                progress = dict(step=step, evaluations=evaluations,
                    weak_relative=float(work.residual(current.coefficients(), prefix, previous_plastic, releases[-1]).detach().norm()),
                    replay_failure=str(error))
                checkpoint(progress)
                print(json.dumps({"event": "pending_replay", **progress}), flush=True)
                # An intermediate volume iterate may still be inadmissible at
                # wall replay points. Retain it as pending and continue the
                # volume optimization; it cannot complete a history increment.
                continue
            response = np.array([rows[-1][name] for name in response_names])
            change = None if previous_response is None else float(np.max(np.abs(response-previous_response)))
            tolerance = settings["training"]["response_change_mm"]+args.response_relative_tolerance*np.abs(response)
            scaled_change = None if previous_response is None else float(np.max(np.abs(response-previous_response)/tolerance))
            record = dict(step=step, block=len([r for r in trajectory if r["step"] == step])+1,
                          evaluations=evaluations, seconds=time.perf_counter()-before,
                          rejected_return_trials=rejected_returns,
                          response_change_mm=change, **{k:v for k,v in rows[-1].items() if k != "step"})
            record["equilibrium_tolerance_mm"] = args.equilibrium_tolerance_mm
            record["response_relative_tolerance"] = args.response_relative_tolerance
            record["response_scaled_change"] = scaled_change
            trajectory.append(record)
            previous_response = response
            accurate = rows[-1]["weak_relative"] < settings["training"]["weak_relative"]
            if args.equilibrium_tolerance_mm is not None:
                accurate = max(abs(rows[-1][name+"_imbalance_mm"]) for name in work.gauges()) < args.equilibrium_tolerance_mm
            stable = scaled_change is not None and scaled_change < 1.
            if accurate and stable:
                completed = step
            checkpoint()
            (output/"config.json").write_text(json.dumps(settings, indent=2)+"\n", "utf-8")
            (output/"trajectory.json").write_text(json.dumps(trajectory, indent=2)+"\n", "utf-8")
            print(json.dumps({"event": "block", **record, "completed_steps": completed}), flush=True)
            if completed == step:
                current.requires_grad_(False)
                break
        if completed != step:
            print(json.dumps({"event": "pending", "step": step}), flush=True)
            break


if __name__ == "__main__":
    main()
