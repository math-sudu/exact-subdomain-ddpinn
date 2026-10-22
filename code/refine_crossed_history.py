"""Refine the crossed history and, when required, enlarge its nonlinear domain."""
import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from interface_extension import cpu_state
from pilot import config as C, plasticity as P
from pilot.returned_orientation import ReturnedOrientationResidual
from pilot.joint_return import joint_return
from pilot.causal_mixed import IncrementFields
from pilot.spectral_mixed import load_orientation_model, make_causal_model
from run_orientation_extension import run_return_case, save, wall_response, write_rows

SOURCE = ROOT / "results/orientation_extension/bedding_crossed/model.pt"
REFINED_R3 = ROOT / "results/history_studies/crossed_refined/bedding_crossed"
DESTINATION = REFINED_R3


def initialize():
    _, saved, model = load_orientation_model(SOURCE)
    del model.increments[11:]
    prefix = {key: value for key, value in saved["model"].items()
              if not key.startswith("increments.11.")}
    state = cpu_state(model)
    assert state.keys() == prefix.keys()
    assert all(torch.equal(value, prefix[key]) for key, value in state.items())
    saved.update(model=state, model_steps=11, completed_steps=10,
                 rows=[row for row in saved["rows"] if row["step"] <= 10],
                 evaluations=0, step_elapsed_s=0.,
                 trust_region_state=dict(solves=0, inner={}, last_result={}))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    save(DESTINATION / "model.pt", saved)


@torch.no_grad()
def warm_start(model, original_increment):
    """Transfer a network to a new projection interval and uniform load step."""
    increment = model.append_increment(copy_previous=False)
    state = {key: value.clone() for key, value in original_increment.state_dict().items()}
    # The saved elastic potentials contain the original load-step fraction.
    # Keep the newly constructed background for the target number of steps.
    for key, value in increment.state_dict().items():
        if key.startswith("elastic_background."):
            state[key] = value.clone()
    factor = increment.input_scale / original_increment.input_scale
    for name in ("kinematic_trunk.0.weight", "stress_trunk.0.weight"):
        state[name] *= factor
    if torch.count_nonzero(state["background_subtraction"]):
        background = original_increment.project(torch.cat(original_increment.elastic_background(
            original_increment.projection_nodes), -1))
        subtraction = torch.view_as_complex(state["background_subtraction"])
        if torch.allclose(subtraction, background, rtol=1e-12, atol=1e-12):
            background = increment.project(torch.cat(increment.elastic_background(increment.projection_nodes), -1))
        else:
            # Retain an actual predictor offset, including a scaled correction
            # or the difference between cumulative source and new histories.
            values = original_increment.evaluate(subtraction, increment.projection_nodes)
            background = increment.project(values) * original_increment.total_steps / increment.total_steps
        state["background_subtraction"] = torch.view_as_real(background)
    else:
        state["background_subtraction"] = torch.zeros_like(increment.background_subtraction)
    increment.load_state_dict(state)
    xy = increment.projection_nodes
    fraction = original_increment.total_steps / increment.total_steps
    torch.testing.assert_close(IncrementFields.forward(increment, xy)[0],
                               fraction * IncrementFields.forward(original_increment, xy)[0],
                               rtol=1e-12, atol=1e-12)


def initialize_expanded():
    bundle, original, old_model = load_orientation_model(SOURCE)
    settings = deepcopy(original["settings"])
    settings.update(radius=4, question="Recompute the crossed history with an elastic exterior beyond 4a after slip reached the original 3a interface.")
    settings["cases"] = [original["case"]]
    settings["projection"]["radial_order"] = 40
    settings["quadrature"]["radial"] = 64
    settings["weights"]["virtual_work"] = np.pi * (settings["radius"] ** 2 - 1)
    model = make_causal_model(settings, bundle["c"], original["case"])
    for increment in old_model.increments[:4]:
        warm_start(model, increment)
    assert all(torch.count_nonzero(inc.coefficients()) == 0 for inc in model.increments[:3])
    saved = dict(case=original["case"], settings=settings, model=cpu_state(model),
        model_steps=4, completed_steps=3, evaluations=0, step_elapsed_s=0., rows=[],
        c=bundle["c"], s0=bundle["s0"],
        trust_region_state=dict(solves=0, inner={}, last_result={}),
        initialization_checkpoint=str(SOURCE))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    save(DESTINATION / "model.pt", saved)
    (DESTINATION.parent / "config.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")


def initialize_next_increment(radius):
    """Use a corresponding retained network on the newly solved prefix."""
    _, saved, model = load_orientation_model(DESTINATION / "model.pt")
    step = model.n_steps + 1
    source = REFINED_R3 / "model.pt" if radius == 4 and step >= 11 else SOURCE
    _, _, original = load_orientation_model(source)
    warm_start(model, original.increments[step - 1])
    saved.update(model=cpu_state(model), model_steps=step, completed_steps=step - 1,
                 evaluations=0, step_elapsed_s=0.,
                 trust_region_state=dict(solves=0, inner={}, last_result={}))
    save(DESTINATION / "model.pt", saved)


def resolve_final_quadrature():
    """Resolve the near-wall return branch missed by the coarser final solve."""
    _, saved, model = load_orientation_model(DESTINATION / "model.pt")
    if model.n_steps != 12:
        raise ValueError("This refinement requires the twelve-increment crossed history")
    previous = DESTINATION.parent / "before_final_quadrature/bedding_crossed"
    previous.mkdir(parents=True, exist_ok=True)
    save(previous / "model.pt", saved)
    _, _, original = load_orientation_model(SOURCE)
    del model.increments[11:]
    warm_start(model, original.increments[11])
    settings = deepcopy(saved["settings"])
    settings["quadrature"].update(radial=96, angular=512)
    settings["question"] = "Resolve the final crossed increment's near-wall local return with 96 radial and 512 angular integration points; steps 1–11 retain their solved fields."
    saved.update(model=cpu_state(model), settings=settings, completed_steps=11,
        rows=[row for row in saved["rows"] if row["step"] <= 11],
        evaluations=0, step_elapsed_s=0.,
        trust_region_state=dict(solves=0, inner={}, last_result={}))
    save(DESTINATION / "model.pt", saved)
    (DESTINATION.parent / "final_quadrature_config.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def exterior_assessment(bundle, saved, model):
    """Check the elastic exterior after the late plastic zone has expanded."""
    radius = saved["settings"]["radius"]
    theta = 2 * np.pi * np.arange(1024) / 1024
    xy = radius * np.column_stack((np.cos(theta), np.sin(theta)))
    displacement, _, gamma = model(model.multiplier_scale.new_tensor(xy))
    case = saved["case"]
    material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
    field = P._exterior_field_from_trace(displacement[:, -1].cpu().numpy().T * C.U_REF,
        material, radius * C.A_TUNNEL, saved["settings"]["projection"]["angular_modes"])
    angles = 2 * np.pi * (np.arange(720) + .37) / 720
    radii = radius * np.geomspace(1, 4, 24)
    x = (C.A_TUNNEL * radii[:, None] * np.cos(angles)).ravel()
    y = (C.A_TUNNEL * radii[:, None] * np.sin(angles)).ravel()
    xx, yy, xy = field.stresses(x, y)
    compliance = material.full_compliance()
    zz = -compliance[2, [0, 1, 5]] @ np.stack((xx, yy, xy)) / compliance[2, 2]
    stress = torch.from_numpy(np.stack((xx, yy, zz, xy), -1)) / C.SIGMA_REF + bundle["s0"]
    stress = torch.cat((stress, bundle["s0"][None]))
    return dict(interface_cumulative_gamma_max=float(gamma[..., 1].sum(1).max()),
        interface_active_points=int((gamma[:, -1, 1] > 0).sum()),
        exterior_joint_yield_max_mpa=float(P.joint_yield(stress, bundle["params"]).max() * C.SIGMA_REF / 1e6),
        exterior_matrix_yield_max_mpa=float(P.matrix_yield(stress, bundle["params"]).max() * C.SIGMA_REF / 1e6),
        radial_range_over_a=[float(radii[0]), float(radii[-1])],
        radial_samples=len(radii), angular_samples=len(angles), includes_far_field=True)


@torch.no_grad()
def near_wall_branch(model, radial_range=(1., 1.05), angular_range=(40., 46.)):
    """Locate the return margin in the region where the actual export failed."""
    from scipy.optimize import minimize
    model.cache_projection()
    c, params = model.return_stiffness, model.return_params
    normal, shear = P._joint_geometry(params, c)
    cq = c @ shear
    friction = np.tan(np.deg2rad(params.joint_phi_deg))
    a = friction * (normal @ cq) / (shear @ cq)

    def margin(polar):
        polar = np.atleast_2d(polar)
        theta = np.deg2rad(polar[:, 1])
        xy = c.new_tensor(polar[:, :1] * np.column_stack((np.cos(theta), np.sin(theta))))
        strain = plastic = xy.new_zeros((len(xy), 4))
        for k, increment in enumerate(model.increments):
            du = increment.spatial_fields(xy)[1][0]
            strain = strain + torch.stack((du[0, :, 0], du[1, :, 1],
                torch.zeros_like(du[0, :, 0]), du[1, :, 0] + du[0, :, 1]), -1)
            trial = model.return_insitu + (strain - plastic) @ c.T
            if k < model.n_steps - 1:
                plastic = plastic + joint_return(trial, c, params)["plastic"]
        h = params.joint_cohesion + params.joint_smoothing - friction * (trial @ normal) + a * (trial @ shear)
        return h.cpu().numpy()

    r, theta = np.meshgrid(np.linspace(*radial_range, 31), np.linspace(*angular_range, 25), indexing="ij")
    points = np.column_stack((r.ravel(), theta.ravel()))
    values = margin(points)
    point = points[values.argmin()]
    fit = minimize(lambda p: float(margin(p)[0]), point, method="L-BFGS-B",
                   bounds=(radial_range, angular_range))
    if fit.fun < values.min():
        point = fit.x
    return dict(minimum_h=float(margin(point)[0]), epsilon=params.joint_smoothing,
                radius_over_a=float(point[0]), angle_deg=float(point[1]),
                search_radius_over_a=list(radial_range), search_angle_deg=list(angular_range))


def backtrack_final_update(device):
    """Halve the actual neural update that crossed the local return branch."""
    checkpoint = DESTINATION / "model.pt"
    bundle, saved, model = load_orientation_model(checkpoint, device)
    accepted = torch.load(DESTINATION.parent / "branch_valid/bedding_crossed/model.pt",
                          map_location="cpu", weights_only=False)
    parameters = {"increments.11." + name for name, _ in model.increments[-1].named_parameters()}
    changed = {name for name in saved["model"]
               if not torch.equal(saved["model"][name], accepted["model"][name])}
    if not changed or not changed <= parameters:
        raise ValueError("Backtrack only the final neural parameters on an unchanged prefix")
    proposal = DESTINATION.parent / "before_feasible_backtrack/bedding_crossed"
    proposal.mkdir(parents=True, exist_ok=True)
    save(proposal / "model.pt", saved)
    state = {name: (accepted["model"][name] + .5 * (value - accepted["model"][name])
                    if name in changed else value) for name, value in saved["model"].items()}
    model.load_state_dict(state)
    branch = near_wall_branch(model)
    if branch["minimum_h"] <= branch["epsilon"]:
        raise ValueError(f"The half update still crosses the return branch: {branch}")
    problem = ReturnedOrientationResidual(model, saved, bundle["params"])
    with torch.no_grad():
        coefficients = problem.current.coefficients()
        assessment = problem.assessment(coefficients)
        objective = float(problem(coefficients).square().sum())
    if objective >= accepted["rows"][-1]["objective"]:
        raise ValueError("The feasible backtrack did not improve the coupled objective")
    row = dict(saved["rows"][-1], **assessment)
    row.update(evaluations=saved["evaluations"] + 1,
               phase="returned_neural_feasible_backtrack", objective=objective,
               completed=False, trust_status=None, primal_optimality=None,
               neural_step_fraction=.5, start_evaluations=accepted["evaluations"],
               proposed_evaluations=saved["evaluations"])
    saved["rows"].append(row)
    saved.update(model=cpu_state(model), completed_steps=11,
                 evaluations=row["evaluations"])
    saved["trust_region_state"].update(inner={}, last_result={})
    saved["feasible_backtrack"] = dict(fraction=.5, start_evaluations=accepted["evaluations"],
        proposed_evaluations=row["proposed_evaluations"], near_wall_return=branch)
    save(checkpoint, saved)
    write_rows(DESTINATION / "trajectory.csv", saved["rows"])
    print(json.dumps(dict(objective=objective, **saved["feasible_backtrack"])), flush=True)


@torch.no_grad()
def assess(device):
    bundle, saved, model = load_orientation_model(DESTINATION / "model.pt", device)
    problem = ReturnedOrientationResidual(model, saved, bundle["params"])
    coefficients = problem.current.coefficients()
    history, _ = problem.fields(coefficients, equilibrium=False)
    blocks = problem.blocks(coefficients)
    rows = []
    for response in problem.gauge_response(coefficients, history["plastic_strain"]):
        rows.append(dict(step=model.n_steps, evaluations=saved["evaluations"],
            trust_solves=saved["trust_region_state"]["solves"], **response,
            radial_quadrature=saved["settings"]["quadrature"]["radial"],
            angular_quadrature=saved["settings"]["quadrature"]["angular"],
            **{name + "_objective": float(value.square().sum()) for name, value in blocks.items()}))
    path = DESTINATION.parent / "refinement.csv"
    previous = list(csv.DictReader(path.open())) if path.exists() else []
    write_rows(path, previous + rows)
    print(json.dumps(rows), flush=True)
    response = wall_response(model, bundle["params"], bundle["s0"])
    exterior = exterior_assessment(bundle, saved, model)
    assessment = dict(checkpoint=str((DESTINATION / "model.pt").relative_to(ROOT)),
        step=model.n_steps, completed_steps=saved["completed_steps"],
        gauges=rows, wall=response, exterior=exterior)
    if model.n_steps == 12:
        assessment["near_wall_return"] = near_wall_branch(model)
    (DESTINATION / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in assessment.items()
                      if key not in ("checkpoint", "step", "completed_steps", "gauges")}), flush=True)


def main():
    global DESTINATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius", type=int, choices=(3, 4), default=3)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--accept-step", type=int)
    parser.add_argument("--assess-only", action="store_true")
    parser.add_argument("--resolve-final-quadrature", action="store_true")
    parser.add_argument("--backtrack-final-update", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.radius == 4:
        DESTINATION = ROOT / "results/orientation_extension/bedding_crossed"
    if not (DESTINATION / "model.pt").exists():
        initialize_expanded() if args.radius == 4 else initialize()
    if args.resolve_final_quadrature:
        resolve_final_quadrature()
    if args.backtrack_final_update:
        backtrack_final_update(args.device)
    if not args.assess_only:
        saved = torch.load(DESTINATION / "model.pt", map_location="cpu", weights_only=False)
        if args.accept_step is not None:
            step = args.accept_step
            if (saved["model_steps"] != step or saved["completed_steps"] != step - 1
                    or not saved["trust_region_state"]["solves"]):
                raise ValueError("Accept only the assessed pending increment")
            # Its saved fields and response assessment have not changed since
            # the preceding solve; acceptance does not require a new assembly.
            saved["completed_steps"] = step
            saved["rows"].append(dict(saved["rows"][-1], completed=True))
            write_rows(DESTINATION / "trajectory.csv", saved["rows"])
            save(DESTINATION / "model.pt", saved)
        if saved["completed_steps"] < 12:
            if saved["model_steps"] == saved["completed_steps"]:
                initialize_next_increment(args.radius)
            run_return_case(saved["case"], saved["settings"], DESTINATION, 12, args.device,
                            trust_blocks=args.blocks, accept_step=None, initialize_checkpoint=None)
        elif args.accept_step == 12:
            run_return_case(saved["case"], saved["settings"], DESTINATION, 12, args.device,
                            trust_blocks=0, accept_step=None, initialize_checkpoint=None)
    assess(args.device)


if __name__ == "__main__":
    main()
