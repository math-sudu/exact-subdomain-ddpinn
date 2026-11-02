"""Solve the same incremental equations directly in mapped spectral coefficients."""
import argparse
import json
import time

import torch

from run_horseshoe import ROOT, make_work, write_results
from oespinn import config as C


@torch.no_grad()
def solve_increment(work, prefix, previous_plastic, release, predictor, *, tolerance=1e-4, progress=None):
    """Consistent Newton steps in elastic energy coordinates, with Armijo search."""
    coefficient = predictor.clone()
    records = []
    evaluations = 0
    for iteration in range(80):
        residual, tangent, returned = work.residual_tangent(
            coefficient, prefix, previous_plastic, release)
        evaluations += 1
        norm = float(residual.norm())
        if norm <= tolerance:
            return coefficient, returned, records, evaluations
        # y = L^T d / ||L^-1 F||, so the elastic Jacobian is the identity.
        jacobian = torch.linalg.solve_triangular(
            work.factor, tangent.T, upper=False).T*work.load_norm
        merit = .5*norm**2
        normal = None
        # At a difficult active-set transition a Newton line search can stall.
        # Regularized least-squares directions retain descent of the same merit.
        for regularization in range(9):
            damping = 0.
            if regularization == 0:
                direction_y, info = torch.linalg.solve_ex(jacobian, -residual)
                if int(info) != 0:
                    continue
            else:
                if normal is None:
                    normal = jacobian.T@jacobian
                    gradient = jacobian.T@residual
                damping = float(normal.diag().max())*10.**(regularization-7)
                direction_y = torch.linalg.solve(
                    normal+damping*torch.eye(len(residual), device=residual.device), -gradient)
            direction = torch.linalg.solve_triangular(
                work.factor.T, direction_y[:, None], upper=True).reshape(-1, 2)*work.load_norm
            slope = float(residual@(jacobian@direction_y))
            if not bool(torch.isfinite(direction).all()) or slope >= 0:
                continue
            alpha = 1.
            accepted = False
            for backtrack in range(16):
                trial = coefficient+alpha*direction
                try:
                    value, _ = work.residual_state(trial, prefix, previous_plastic, release)
                    trial_merit = float(.5*value.square().sum())
                except FloatingPointError:
                    trial_merit = float("inf")
                evaluations += 1
                if trial_merit <= merit+1e-4*alpha*slope:
                    accepted = True
                    break
                alpha *= .5
            if accepted:
                break
        else:
            raise RuntimeError(f"Armijo search failed at release {release}, residual {norm}")
        coefficient = trial
        records.append(dict(iteration=iteration+1, residual=norm, alpha=alpha,
                            trial_residual=(2*trial_merit)**.5, backtracks=backtrack, damping=damping))
        if progress is not None:
            progress(records[-1], coefficient)
    raise RuntimeError(f"Newton iteration did not converge at release {release}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--residual-work-bound-mm", type=float, default=.15)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    settings = json.loads((ROOT/args.config).read_text("utf-8"))
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    geometry, work = make_work(settings, args.device)
    displacement_scale = C.SIGMA_REF*geometry.length_scale/C.E_REF*1000
    bound_factors = {name: float(torch.linalg.solve_triangular(
        work.factor, gauge[:, None], upper=False).norm()*work.load_norm)*displacement_scale
        for name, gauge in work.gauges().items()}
    tolerance = args.residual_work_bound_mm/max(bound_factors.values())
    increments, trajectory = [], []
    prefix = work.c.new_zeros((work.n, 2))
    plastic = work.c.new_zeros((len(work.xy), 4))
    elastic = torch.cholesky_solve(work.load[:, None], work.factor).reshape(-1, 2)/settings["load_steps"]
    saved_path = output/"model.pt"
    from_scratch = not saved_path.exists() and not (output/"pending.pt").exists()
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    setup_seconds = time.perf_counter()-started
    if saved_path.exists():
        saved = torch.load(saved_path, map_location=args.device, weights_only=False)
        if saved["settings"] != settings or saved["tolerance"] > tolerance:
            raise ValueError("Resume settings differ from the saved direct calculation")
        increments = saved["coefficients"]
        trajectory = saved["trajectory"]
        if increments:
            with torch.no_grad():
                prefix = sum(increments, prefix)
                plastic = work.replay(increments)[-1]["plastic"]
    for step in range(len(increments)+1, settings["load_steps"]+1):
        before = time.perf_counter()
        pending_path = output/"pending.pt"
        predictor = increments[-1] if increments else elastic
        if pending_path.exists():
            pending = torch.load(pending_path, map_location=args.device, weights_only=False)
            if pending["step"] == step and pending["settings"] == settings:
                predictor = pending["coefficient"]
        def progress(record, coefficient):
            torch.save(dict(step=step, settings=settings, coefficient=coefficient.cpu(),
                            iteration=record), pending_path)
            print(json.dumps(dict(step=step, **record)), flush=True)
        coefficient, returned, records, evaluations = solve_increment(
            work, prefix, plastic, step/settings["load_steps"],
            predictor, tolerance=tolerance, progress=progress)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = time.perf_counter()-before
        increments.append(coefficient)
        prefix = prefix+coefficient
        plastic = plastic+returned["plastic"]
        record = dict(step=step, seconds=seconds, residual_evaluations=evaluations,
                      tolerance=tolerance, newton_iterations=len(records), iterations=records)
        trajectory.append(record)
        saved = dict(solver="direct_spectral", settings=settings, tolerance=tolerance,
                     completed_steps=step, coefficients=[c.cpu() for c in increments],
                     trajectory=trajectory, setup_seconds=setup_seconds,
                     device=torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu")
        torch.save(saved, output/"model.next.pt")
        (output/"model.next.pt").replace(saved_path)
        if pending_path.exists():
            pending_path.unlink()
        (output/"trajectory.json").write_text(json.dumps(trajectory, indent=2)+"\n", "utf-8")
        print(json.dumps({k:v for k,v in record.items() if k != "iterations"}), flush=True)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    computation_seconds = time.perf_counter()-started if from_scratch else None
    with torch.no_grad():
        rows = write_results(output, work, increments,
                             [(i+1)/settings["load_steps"] for i in range(len(increments))])
    metadata = dict(solver="direct_spectral", config=args.config, device=saved["device"],
                    coefficient_count=work.n*2, setup_seconds=setup_seconds,
                    retained_step_solve_seconds=sum(r["seconds"] for r in trajectory),
                    computation_seconds=computation_seconds, from_scratch=from_scratch,
                    timing_scope=("Complete uninterrupted assembly and all incremental solves, including checkpoint writes; "
                                  "excludes response export and regional source integration." if from_scratch else
                                  "Sum of retained step executions; separate interrupted or failed invocations are excluded."),
                    newton_iterations=sum(r["newton_iterations"] for r in trajectory),
                    residual_evaluations=sum(r["residual_evaluations"] for r in trajectory),
                    maximum_cuda_memory_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                    tolerance=tolerance, residual_work_bound_mm=args.residual_work_bound_mm,
                    residual_work_bound_factors_mm=bound_factors, final=rows[-1])
    (output/"comparison_run.json").write_text(json.dumps(metadata, indent=2)+"\n", "utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
