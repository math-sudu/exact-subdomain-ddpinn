"""Separate endpoint-source differences using the actual training work matrix.

The saved neural history is only evaluated. Its endpoint response is separated
into plastic-source work and the adjoint projection of its weak residual.
Uniform-grid integration then isolates quadrature and auxiliary-space effects.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_gauge_sources import integrate_oriented_regions, integrate_source_work, quadrature
from decompose_eigenstrain_response import plastic_sources
from interface_extension import geometry
from pilot import config as C
from pilot.discrete_gauge import DiscreteGaugeInfluence
from pilot.elastic_background import CircularElasticIncrement
from pilot.gauge_influence import CircularGaugeInfluence
from pilot.spectral_mixed import load_orientation_model
from pilot.virtual_work import VirtualWorkMetric
from publication_model import write_rows

ROOT = Path(__file__).resolve().parents[1]
SCALE = C.U_REF * 1000


@torch.no_grad()
def training_plastic(model, points, steps):
    parts = [model.returned_history(xy)[3][:, steps - 1].cpu().numpy()
             for xy in points.split(1024)]
    return np.concatenate(parts)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/orientation_extension")
    parser.add_argument("--cases", nargs="+", default=["bedding_crossed", "bedding_oblique", "bedding_aligned"])
    parser.add_argument("--steps", nargs="+", type=int, default=list(range(1, 13)))
    parser.add_argument("--angles", nargs="+", type=float, default=[0., 90., 104.])
    parser.add_argument("--radial-quadrature", type=int, default=64)
    parser.add_argument("--angular-quadrature", type=int, default=512)
    parser.add_argument("--output", type=Path, default=ROOT / "results/history_analysis/gauge_diagnostics")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    angles, steps = np.asarray(args.angles), np.asarray(args.steps)
    directions = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    endpoints = np.stack((-directions, directions), axis=1)
    summaries, regions = [], []
    metric = None
    functional_error = identity_error = work_error = 0.
    for case_name in args.cases:
        print(f"Loading {case_name}", flush=True)
        checkpoint = args.root / case_name / "model.pt"
        bundle, saved, model = load_orientation_model(checkpoint)
        if np.any((steps < 1) | (steps > saved["completed_steps"])):
            raise ValueError("Only completed increments may be evaluated")
        del model.increments[int(steps.max()):]
        settings, case = saved["settings"], saved["case"]
        points, stiffness, _, _ = geometry(case, settings, "cpu")
        torch.testing.assert_close(bundle["c"], stiffness, rtol=1e-12, atol=1e-12)
        material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
        if metric is None:
            print("Assembling training matrix and endpoint adjoints", flush=True)
            metric = VirtualWorkMetric(stiffness, C.nondim_lambda(material), settings, points)
            influences = [DiscreteGaugeInfluence(metric, pair) for pair in endpoints]
            adjoints = torch.stack([field.adjoint for field in influences], dim=1)
            functionals = torch.stack([field.functional for field in influences], dim=1)
            grid = quadrature(settings["radius"], args.radial_quadrature, args.angular_quadrature)
            modes = settings["projection"]["angular_modes"]
            if args.angular_quadrature <= 2 * modes:
                raise ValueError("Source quadrature must resolve the retained gauge modes")
            xy = grid.r[:, None, None] * np.column_stack((np.cos(grid.theta), np.sin(grid.theta)))[None]
            discrete_stress = np.stack([field.stress_on_rings(grid.r, grid.theta) for field in influences])
            continuous_stress = np.stack([CircularGaugeInfluence(material, pair, modes).stress(xy) for pair in endpoints])
            train_xy = points.xy_all[:points.n_pde].numpy().reshape(metric.nr, metric.nt, 2)
            train_angles = np.arctan2(train_xy[..., 1], train_xy[..., 0])
            training_stress = np.stack([field.stress_on_rings(metric.operator.r, train_angles) for field in influences])
            first_settings = settings
            first_points = points.xy_all[:points.n_pde].clone()
        else:
            np.testing.assert_array_equal(stiffness.numpy(), metric.operator.c)
            assert settings["projection"] == first_settings["projection"]
            assert settings["quadrature"] == first_settings["quadrature"]
            assert settings["radius"] == first_settings["radius"]
            torch.testing.assert_close(points.xy_all[:points.n_pde], first_points, rtol=0, atol=0)

        print(f"Evaluating training-grid plastic history: {case_name}", flush=True)
        plastic = training_plastic(model, points.xy_all[:points.n_pde], steps)
        loads = torch.stack([metric.source_load(torch.as_tensor(plastic[:, k])) for k in range(len(steps))], dim=1)
        cumulative = torch.stack([inc.coefficients()[..., :2] for inc in model.increments]).cumsum(0)
        vectors = torch.stack([metric.correction_vector(cumulative[k - 1]) for k in steps], dim=1)
        residual = metric.matrix @ vectors - loads
        source_training = (adjoints.mH @ loads).real.numpy() * SCALE
        residual_projection = (adjoints.mH @ residual).real.numpy() * SCALE
        correction = (functionals.mH @ vectors).real.numpy() * SCALE
        source_integral = np.einsum("grtc,rtkc,r->gk", training_stress,
            plastic.reshape(metric.nr, metric.nt, len(steps), 4), metric.operator.w / metric.nt) * SCALE
        work_error = max(work_error, float(np.max(np.abs(source_integral - source_training))))

        endpoint_xy = model.multiplier_scale.new_tensor(endpoints.reshape(-1, 2))
        displacement = model(endpoint_xy)[0][:, steps - 1].numpy().reshape(len(angles), 2, len(steps), 2)
        direct = -np.einsum("gkc,gc->gk", displacement[:, 1] - displacement[:, 0], directions) * SCALE
        elastic_field = CircularElasticIncrement(case, total_steps=1, like=model.multiplier_scale)
        elastic_u = elastic_field(endpoint_xy)[0].numpy().reshape(len(angles), 2, 2)
        elastic = (-np.einsum("gc,gc->g", elastic_u[:, 1] - elastic_u[:, 0], directions)[:, None]
                   * (steps / model.total_steps)[None]) * SCALE
        functional_error = max(functional_error, float(np.max(np.abs(direct - elastic - correction))))
        identity_error = max(identity_error, float(np.max(np.abs(correction - source_training - residual_projection))))

        print(f"Evaluating {args.radial_quadrature}x{args.angular_quadrature} source history: {case_name}", flush=True)
        sources, activity = plastic_sources(bundle, model, grid, steps)
        discrete = integrate_source_work(discrete_stress, sources, grid)
        continuous = integrate_source_work(continuous_stress, sources, grid)
        quadrature_part = source_training - discrete["source_total_mm"]
        auxiliary_part = discrete["source_total_mm"] - continuous["source_total_mm"]
        activity_density = np.einsum("rtk,r->kt", activity, grid.dr * grid.r) * C.U_REF / C.A_TUNNEL
        regional_activity = integrate_oriented_regions(activity_density, 135.)
        for operator_name, arrays in (("continuous", continuous), ("discrete", discrete)):
            regional = integrate_oriented_regions(arrays["angular_density_source_total_mm"], 135.)
            for g, angle in enumerate(angles):
                for k, step in enumerate(steps):
                    for j, region in enumerate(("side", "top_bottom")):
                        regions.append(dict(case=case_name, load_step=int(step), gauge_deg=float(angle),
                            operator=operator_name, region=region, source_mm=float(regional[g, k, j]),
                            joint_slip_area_integral=float(regional_activity[k, j])))
        for g, angle in enumerate(angles):
            for k, step in enumerate(steps):
                summaries.append(dict(case=case_name, load_step=int(step), gauge_deg=float(angle),
                    direct_mm=float(direct[g, k]), elastic_mm=float(elastic[g, k]),
                    correction_functional_mm=float(correction[g, k]),
                    source_training_mm=float(source_training[g, k]),
                    source_discrete_mm=float(discrete["source_total_mm"][g, k]),
                    source_continuous_mm=float(continuous["source_total_mm"][g, k]),
                    residual_projection_mm=float(residual_projection[g, k]),
                    quadrature_component_mm=float(quadrature_part[g, k]),
                    auxiliary_component_mm=float(auxiliary_part[g, k]),
                    direct_minus_elastic_minus_continuous_mm=float(direct[g, k] - elastic[g, k] - continuous["source_total_mm"][g, k])))
        print(json.dumps([row for row in summaries if row["case"] == case_name and row["load_step"] == int(steps.max())]), flush=True)
        write_rows(args.output / "summary.csv", summaries)
        write_rows(args.output / "regions.csv", regions)

    metadata = dict(producer="code/diagnose_gauge_sources.py", checkpoints=str(args.root.resolve()),
        cases=args.cases, steps=args.steps, gauge_angles_deg=args.angles,
        training_quadrature=first_settings["quadrature"],
        source_quadrature=dict(radial=args.radial_quadrature, angular=args.angular_quadrature),
        projection=first_settings["projection"],
        identity="D-D_el-S_cont = Re(z^H r_V) + (S_train-S_discrete) + (S_discrete-S_cont)",
        discrete="H^H z=g with the exact VirtualWorkMetric matrix and D_c=Re(g^H d)",
        region_definition="side: |x|>|y|; top_bottom: |y|>|x|; fixed globally for all orientations",
        region_integration="Exact angular integration of the uniform-grid trigonometric interpolant",
        units=dict(response="mm", activity="strain integrated against dA/a^2"),
        maximum_endpoint_functional_error_mm=functional_error,
        maximum_discrete_identity_error_mm=identity_error,
        maximum_training_source_integral_error_mm=work_error)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
