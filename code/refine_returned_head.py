"""Refine a pending neural displacement head with explicit return feasibility."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import LinearConstraint, minimize
import torch

from interface_extension import cpu_state
from analyze_gauge_sources import quadrature
from pilot import plasticity as P
from pilot.orientation_tangent import NeuralCoordinates, column_scale, linear_operator
from pilot.returned_orientation import ReturnedOrientationResidual
from pilot.spectral_mixed import CircularGridEvaluation, load_orientation_model
from recompute_orientation_history import source_total_predictor
from run_orientation_extension import save


def branch_margin(problem, coefficients, extra_points=None, extra_grid=None):
    """h-delta is affine in the displacement-head weights at a frozen trunk."""
    if extra_points is None:
        gradient = problem.cache["increment_grid"](coefficients[..., :2], second=False)[1]
        gradient = gradient + problem.cache["increment_background"][1][0]
        previous = problem.cache["stress_increment"]
    else:
        total = problem.prefix_coefficients + coefficients[..., :2]
        gradient = (problem.current._spatial_expansion(total, extra_points)[1]
                    if extra_grid is None else extra_grid(total, second=False)[1])
        gradient = gradient + problem.step * problem.current.elastic_background.spatial_fields(extra_points)[1][0]
        # The h combination annihilates accumulated plastic strain parallel to q.
        previous = 0.
    strain = torch.stack((gradient[0, :, 0], gradient[1, :, 1],
        torch.zeros_like(gradient[0, :, 0]), gradient[1, :, 0]+gradient[0, :, 1]), -1)
    trial = problem.s0 + previous + strain @ problem.c.T
    normal, shear = P._joint_geometry(problem.params, problem.c)
    cq = problem.c @ shear
    friction = np.tan(np.deg2rad(problem.params.joint_phi_deg))
    coupling = friction * (normal @ cq) / (shear @ cq)
    return problem.params.joint_cohesion - friction*(trial @ normal) + coupling*(trial @ shear)


def refine(checkpoint, output, device, iterations, source_start=False, patch=None, floor=1e-9,
           independent=False, patch_samples=(101, 121), extra_patches=()):
    if output.exists():
        raise FileExistsError(output)
    bundle, saved, model = load_orientation_model(checkpoint, device)
    if source_start:
        _, _, source = load_orientation_model(Path(saved["initialization_checkpoint"]), device)
        source_total_predictor(model, source)
    problem = ReturnedOrientationResidual(model, saved, bundle["params"])
    coordinates = NeuralCoordinates(problem.current, "displacement_head")
    zero = coordinates.zero()
    residual = lambda delta: problem(coordinates(delta))
    extra_points = None
    patches = ([] if patch is None else [patch]) + list(extra_patches)
    if patches:
        points = []
        for bounds in patches:
            radii = torch.linspace(bounds[0], bounds[1], patch_samples[0], device=zero.device, dtype=zero.dtype)
            angles = torch.linspace(bounds[2], bounds[3], patch_samples[1], device=zero.device, dtype=zero.dtype)*np.pi/180
            points.append((radii[:, None, None]*torch.stack((angles.cos(), angles.sin()), -1)[None]).reshape(-1, 2))
        extra_points = torch.cat(points)
    base_margin = lambda delta: branch_margin(problem, coordinates(delta))
    extra_margin = lambda delta: branch_margin(problem, coordinates(delta), extra_points)
    functions = [base_margin]
    independent_margin = None
    if independent:
        q = quadrature(saved["settings"]["radius"], 96, 512)
        rings = zero.new_tensor(q.r[:, None, None] * np.column_stack((np.cos(q.theta), np.sin(q.theta)))[None])
        independent_grid = CircularGridEvaluation(problem.current, (rings,))
        independent_points = rings.reshape(-1, 2)
        independent_margin = lambda delta: branch_margin(problem, coordinates(delta),
                                                          independent_points, independent_grid)
        functions.append(independent_margin)
    if extra_points is not None:
        functions.append(extra_margin)
    margin = lambda delta: torch.cat([function(delta) for function in functions])
    _, value, reverse = linear_operator(residual, zero, torch.ones_like(zero))
    scale = column_scale(reverse, value, zero).detach()
    initial_objective = float(value.square().sum().detach())
    del value, reverse
    initial_margin = margin(zero).detach()
    if base_margin(zero).min() <= 0 or not torch.isfinite(initial_margin).all():
        raise ValueError("The constrained head solve needs a strictly feasible starting field")
    matrix = torch.func.jacfwd(base_margin)(zero).detach()
    if independent_margin is not None:
        matrix = torch.cat((matrix, torch.func.jacfwd(independent_margin)(zero).detach()))
    if extra_points is not None:
        pieces = []
        for tangents in torch.eye(len(zero), device=zero.device, dtype=zero.dtype).split(4):
            pieces.append(torch.vmap(lambda tangent: torch.func.jvp(extra_margin,
                (zero,), (tangent,))[1])(tangents).detach())
        matrix = torch.cat((matrix, torch.cat(pieces).T))
    matrix = matrix * scale[None]
    constraint = LinearConstraint(matrix.cpu().numpy(),
                                  (floor-initial_margin).cpu().numpy(), np.inf)
    del matrix
    evaluations, completed = 0, 0
    started = time.perf_counter()

    def objective(values):
        nonlocal evaluations
        point = zero.new_tensor(values).requires_grad_(True)
        result = residual(point*scale)
        loss = result.square().sum()
        derivative, = torch.autograd.grad(loss, point)
        evaluations += 1
        if not torch.isfinite(loss) or not torch.isfinite(derivative).all():
            raise FloatingPointError("Constrained head trial left the supported return branch")
        return float(loss.detach()), derivative.detach().cpu().numpy()

    def report(values):
        nonlocal completed
        completed += 1
        if completed == 1 or completed % 5 == 0:
            with torch.no_grad():
                delta = zero.new_tensor(values)*scale
                print(json.dumps(dict(iteration=completed, evaluations=evaluations,
                    objective=float(residual(delta).square().sum()),
                    minimum_branch_margin=float(margin(delta).min()))), flush=True)

    print(json.dumps(dict(initial_objective=initial_objective,
        minimum_branch_margin=float(initial_margin.min()), variables=len(zero),
        constraints=len(initial_margin))), flush=True)
    fit = minimize(objective, np.zeros(len(zero)), jac=True, method="SLSQP",
        constraints=[constraint], callback=report,
        options=dict(maxiter=iterations, ftol=1e-10))
    delta = zero.new_tensor(fit.x)*scale
    with torch.no_grad():
        final_margin = float(margin(delta).min())
        final_objective = float(residual(delta).square().sum())
    restores_feasibility = float(initial_margin.min()) <= 0 < final_margin
    if (final_margin <= 0 or not np.isfinite(final_objective)
            or (final_objective >= initial_objective and not restores_feasibility)):
        raise ValueError("The constrained head solve did not produce a supported improvement")
    coordinates.apply(delta)
    step = model.n_steps
    if (saved["rows"] and saved["rows"][-1]["step"] == step
            and saved["rows"][-1]["evaluations"]):
        saved.setdefault("predictor_restarts", []).append({key: saved["rows"][-1][key] for key in (
            "step", "evaluations", "elapsed_s", "objective", "wall_traction_rms_mpa",
            "residual_projection_0_mm", "residual_projection_90_mm")})
    result = dict(step=step, evaluations=evaluations, elapsed_s=time.perf_counter()-started,
        initial_objective=initial_objective, objective=final_objective,
        minimum_branch_margin=final_margin, margin_floor=floor,
        optimizer_status=int(fit.status), optimizer_message=str(fit.message),
        variables=len(zero), constraints=len(initial_margin),
        source_total_start=source_start, guard_patch=patch,
        additional_guard_patches=list(extra_patches),
        guard_patch_samples=patch_samples if patch is not None else None,
        independent_quadrature=[96, 512] if independent else None,
        initial_minimum_branch_margin=float(initial_margin.min()))
    saved.setdefault("constrained_head_solves", []).append(result)
    with torch.no_grad():
        assessment = problem.assessment(problem.current.coefficients())
    row = dict(case=saved["case"]["name"], step=step, phase="constrained_head_assessment",
        equilibrium_form=problem.equilibrium_form, evaluations=0, elapsed_s=0.,
        objective=final_objective, completed=False, trust_solves=0,
        initial_correction_scale=1., trust_status=None, primal_optimality=None, **assessment)
    saved.update(model=cpu_state(model), completed_steps=step-1, evaluations=0,
        step_elapsed_s=0., rows=[old for old in saved["rows"] if old["step"] < step]+[row],
        initial_correction_scale=1.,
        trust_region_state=dict(solves=0, inner={}, last_result={}))
    save(output, saved)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--source-total-start", action="store_true")
    parser.add_argument("--guard-patch", type=float, nargs=4, metavar=("R_MIN", "R_MAX", "THETA_MIN", "THETA_MAX"))
    parser.add_argument("--additional-guard-patch", type=float, nargs=4, action="append", default=[])
    parser.add_argument("--margin-floor", type=float, default=1e-9)
    parser.add_argument("--guard-independent", action="store_true")
    parser.add_argument("--patch-samples", type=int, nargs=2, default=[101, 121])
    args = parser.parse_args()
    torch.set_num_threads(1)
    refine(args.checkpoint, args.output, args.device, args.iterations,
           args.source_total_start, args.guard_patch, args.margin_floor,
           args.guard_independent, args.patch_samples, args.additional_guard_patch)


if __name__ == "__main__":
    main()
