"""Evaluate a saved history against independently integrated, enriched weak tests."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import splu

from decompose_eigenstrain_response import FrozenSourceOperator, plastic_sources
from pilot import config as C
from pilot.spectral_mixed import load_orientation_model


@torch.no_grad()
def local_sector(checkpoint, center_deg, order=128, *, include_pending=False):
    """Integrate momentum over a fixed small sector around a documented defect."""
    from causal_sector_balance import sector_quadrature
    from pilot import plasticity as P
    bundle, saved, model = load_orientation_model(checkpoint)
    if model.n_steps != model.total_steps or (not include_pending and saved["completed_steps"] != model.total_steps):
        raise ValueError("Assess the final completed state")
    model.cache_projection()
    interval = [center_deg-5, center_deg+5]
    q = sector_quadrature(1., 1.1, *np.deg2rad(interval), order, model.multiplier_scale)
    stress = model.returned_history(q["contour_xy"])[1][:, -1]
    traction = P._traction4(stress, q["contour_normal"])
    actual = (traction*q["contour_weights"][:, None]).sum(0)
    prescribed = -P._traction4(bundle["s0"], q["contour_normal"][:order])
    applied = actual+((prescribed-traction[:order])*q["contour_weights"][:order, None]).sum(0)
    scale = (prescribed.norm(dim=-1)*q["contour_weights"][:order]).sum()
    return dict(checkpoint=str(checkpoint.resolve()), completed_steps=saved["completed_steps"],
        pending_step=model.n_steps if model.n_steps > saved["completed_steps"] else None,
        radius_range_over_a=[1., 1.1],
        angle_range_deg=interval, quadrature_per_edge=order,
        actual_resultant_mpa_m=(actual*C.SIGMA_REF*C.A_TUNNEL/1e6).tolist(),
        prescribed_wall_resultant_mpa_m=(applied*C.SIGMA_REF*C.A_TUNNEL/1e6).tolist(),
        wall_loading_scale_mpa_m=float(scale*C.SIGMA_REF*C.A_TUNNEL/1e6),
        relative_actual_resultant=float(actual.norm()/scale),
        relative_prescribed_wall_resultant=float(applied.norm()/scale))


@torch.no_grad()
def assess(checkpoint, steps, radial_order, angular_modes, nr, nt, device="cpu", *, include_pending=False,
           decompose_projection=False):
    bundle, saved, model = load_orientation_model(checkpoint, device)
    available = model.n_steps if include_pending else saved["completed_steps"]
    if not steps or min(steps) < 1 or max(steps) > available:
        raise ValueError("Assess completed increments, or explicitly include the pending iterate")
    del model.increments[max(steps):]
    projection = saved["settings"]["projection"]
    if radial_order < projection["radial_order"] or angular_modes < projection["angular_modes"]:
        raise ValueError("Weak test space must contain the displacement space")
    material = C.material(saved["case"]["eh_over_ev"], saved["case"]["bedding_dip_deg"])
    operator = FrozenSourceOperator(bundle["c"].cpu().numpy(), C.nondim_lambda(material),
        saved["settings"]["radius"], radial_order, angular_modes, nr, nt)
    model.cache_projection()
    sources, _ = plastic_sources(bundle, model, operator, steps)
    loads = operator.source_load(sum(sources.values()))
    accumulated = np.cumsum(np.stack([increment.coefficients()[..., :2].cpu().numpy()
                                     for increment in model.increments]), axis=0)
    if decompose_projection:
        if not projection.get("centrosymmetric"):
            raise ValueError("Projection decomposition requires the odd displacement space")
        retained = np.concatenate([
            index * operator.block + np.arange(2 * (projection["radial_order"] + 1))
            for index, mode in enumerate(operator.modes)
            if mode % 2 and abs(mode) <= projection["angular_modes"]])
        retained_matrix = operator.matrix[retained][:, retained].tocsc()
        retained_factor = splu(retained_matrix)
    rows = []
    for column, step in enumerate(steps):
        coefficients = accumulated[step - 1]
        vector = np.zeros(operator.ndof + 2, dtype=complex)
        padded = vector[:operator.ndof].reshape(2*angular_modes+1, radial_order+1, 2)
        for index, mode in enumerate(operator.modes):
            if abs(mode) <= projection["angular_modes"]:
                values = coefficients[:, abs(mode)]
                padded[index, :len(values)] = values.conj() if mode < 0 else values
        source = loads[:, column]
        residual = operator.matrix @ vector - source
        correction = operator.factor.solve(residual)
        source_response = operator.factor.solve(source)
        residual_norm = np.sqrt(max(0., np.vdot(residual, correction).real))
        source_norm = np.sqrt(max(0., np.vdot(source, source_response).real))
        rows.append(dict(load_step=step, residual_dual_norm=float(residual_norm),
                         source_dual_norm=float(source_norm),
                         relative_residual_to_source=float(residual_norm/source_norm)
                         if source_norm else None))
        if decompose_projection:
            load = source[retained]
            best = retained_factor.solve(load)
            difference = vector[retained] - best
            fit_energy = float(np.vdot(difference, retained_matrix @ difference).real)
            floor_energy = float(np.vdot(source, source_response).real - np.vdot(load, best).real)
            rows[-1].update(retained_fit_dual_norm=float(np.sqrt(max(0., fit_energy))),
                frozen_source_projection_floor=float(np.sqrt(max(0., floor_energy))),
                decomposition_energy_error=float(residual_norm**2 - fit_energy - floor_energy))
    return dict(checkpoint=str(checkpoint.resolve()), completed_steps=saved["completed_steps"],
        pending_step=max(steps) if max(steps) > saved["completed_steps"] else None,
        displacement_space=projection, test_space=dict(radial_order=radial_order, angular_modes=angular_modes),
        integration=dict(radial=nr, angular=nt),
        meaning="Dual energy norm of the weak residual at the saved displacement and its own plastic source; not a nonlinear solution error bound.",
        rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[6, 9, 12])
    parser.add_argument("--radial-order", type=int)
    parser.add_argument("--angular-modes", type=int)
    parser.add_argument("--local-sector-center", type=float,
                        help="Assess a 10-degree sector from a to 1.1a instead of the weak-test norm")
    parser.add_argument("--radial-quadrature", type=int, default=96)
    parser.add_argument("--angular-quadrature", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-pending", action="store_true",
                        help="Evaluate the unaccepted current iterate without changing its acceptance state")
    parser.add_argument("--decompose-projection", action="store_true",
                        help="Separate retained-space imbalance from the frozen-source projection floor")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.local_sector_center is not None:
        result = local_sector(args.checkpoint, args.local_sector_center, include_pending=args.include_pending)
    else:
        if args.radial_order is None or args.angular_modes is None:
            parser.error("Specify the enriched test orders or a local sector center")
        result = assess(args.checkpoint, args.steps, args.radial_order, args.angular_modes,
                        args.radial_quadrature, args.angular_quadrature, args.device,
                        include_pending=args.include_pending, decompose_projection=args.decompose_projection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
