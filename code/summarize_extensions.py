"""Export comparison tables consumed by the extended manuscript."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from interface_extension import elastic_fields, geometry, make_elastic_model, metric_value, regional_power
from pilot import config as C, plasticity as P
from pilot.mixed_plasticity import field_history
from pilot.spectral_mixed import load_orientation_model
from run_interface_extension import write_rows


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def elastic():
    root = ROOT / "results/interface_extension"
    settings = json.loads((ROOT / "configs/interface_extension.json").read_text(encoding="utf-8"))
    summaries, comparisons, shapes = [], [], []
    for case in settings["cases"]:
        baseline = None
        for metric in case["metrics"]:
            rows = [read(root / case["name"] / f"seed_{seed}" / metric / "trajectory.csv")[-1]
                    for seed in settings["seeds"]]
            if any(row["phase"] != "complete" for row in rows):
                raise RuntimeError(f"Incomplete trials for {case['name']} / {metric}")
            points, stiffness, _, exterior = geometry(case, settings, "cpu")
            for seed, row in zip(settings["seeds"], rows):
                saved = torch.load(root / case["name"] / f"seed_{seed}" / metric / "model.pt",
                                   map_location="cpu", weights_only=False)
                model = make_elastic_model(settings, stiffness, seed)
                model.load_state_dict(saved["model"])
                with torch.no_grad():
                    _, residuals = elastic_fields(model, points, stiffness, exterior)
                    row["training_equilibrium_rms"] = float((residuals["equilibrium"].square().sum(-1)
                                                             * points.interior_weights).sum().sqrt())
            summaries.extend(rows)
            if metric == "uniform":
                baseline = np.array([float(row["interface_compliance"]) for row in rows])
            entry = dict(case=case["name"], metric=metric, initializations=len(rows))
            for key in ("convergence_max_mm", "convergence_mean_mm", "interface_compliance",
                        "interface_mean_square", "wall_traction_rms_mpa", "equilibrium_rms",
                        "constitutive_rms", "training_equilibrium_rms"):
                values = np.array([float(row[key]) for row in rows])
                entry.update({f"{key}_{stat}": float(fn(values)) for stat, fn in
                              (("median", np.median), ("min", np.min), ("max", np.max))})
            ratios = (np.array([float(row["interface_compliance"]) for row in rows])/baseline
                      if baseline is not None else np.full(len(rows), np.nan))
            entry.update({f"paired_energy_ratio_to_uniform_{stat}": float(fn(ratios))
                          for stat, fn in (("median", np.median), ("min", np.min), ("max", np.max))})
            comparisons.append(entry)
            if metric == "compliance":
                for seed, row in zip(settings["seeds"], rows):
                    data = np.load(root / case["name"] / f"seed_{seed}" / metric / "fields.npz")
                    n0, n1 = int(data["n_pde"]), int(data["n_pde"])+int(data["n_wall"])
                    xy = data["xy"][n0:n1]
                    radial = xy/np.linalg.norm(xy, axis=1, keepdims=True)
                    closure = -(data["displacement"][n0:n1]*radial).sum(-1)*C.U_REF*1000
                    index = closure.argmax()
                    shapes.append(dict(case=case["name"], seed=seed,
                                       eh_over_ev=case["eh_over_ev"],
                                       opening_axis_ratio=case["opening_axes"][0]/case["opening_axes"][1],
                                       opening_angle_deg=case["opening_angle_deg"],
                                       convergence_max_mm=float(row["convergence_max_mm"]),
                                       convergence_mean_mm=float(row["convergence_mean_mm"]),
                                       convergence_angle_deg=float(np.rad2deg(np.arctan2(xy[index, 1], xy[index, 0])) % 360)))
    write_rows(root / "summary.csv", summaries)
    write_rows(root / "comparison.csv", comparisons)
    write_rows(root / "geometry_summary.csv", shapes)
    print(json.dumps(dict(elastic_trials=len(summaries), comparison_rows=len(comparisons))))


@torch.no_grad()
def orientation_history_residuals(checkpoint, device, *, require_complete=True):
    """Evaluate completed increments at their original fractions of total loading."""
    bundle, saved, model = load_orientation_model(checkpoint, device)
    settings, case = saved["settings"], saved["case"]
    if require_complete and saved["completed_steps"] != settings["load_steps"]:
        raise RuntimeError(f"Incomplete orientation history: {checkpoint}")
    model.increments = model.increments[:saved["completed_steps"]]
    rows = []
    for evaluation in (False, True):
        pts, c, s0, sf = geometry(case, settings, device, evaluation=evaluation)
        keys = ("displacement", "stress_increment", "equilibrium", "constitutive",
                "complementarity", "yield_values")
        parts = [field_history(model, xy, c, s0, bundle["params"]) for xy in pts.xy_all.split(2048)]
        fields = {key: torch.cat([part[key] for part in parts]) for key in keys}
        del parts
        n0, n1 = pts.n_pde, pts.n_pde + pts.n_wall
        for k in range(model.n_steps):
            wall = (P._traction4(fields["stress_increment"][n0:n1, k], pts.nrm_wall)
                    + (k+1)/model.total_steps*pts.t0_wall)
            interface = (P._traction4(fields["stress_increment"][n1:, k], pts.nrm_gam)
                         - sf.apply(fields["displacement"][n1:, k].T).T)
            rows.append(dict(case=case["name"], step=k+1,
                sampling="independent" if evaluation else "training",
                equilibrium_rms=float((fields["equilibrium"][:n0, k].square().sum(-1)
                                       * pts.interior_weights).sum().sqrt()),
                constitutive_rms=float(regional_power(fields["constitutive"][:, k], pts).sqrt()),
                complementarity_rms=float(regional_power(fields["complementarity"][:, k], pts).sqrt()),
                wall_traction_rms_mpa=float((wall.square().sum(-1)*pts.wall_weights).sum().sqrt()*C.SIGMA_REF/1e6),
                interface_compliance=float(metric_value(sf, interface, "compliance")),
                interface_yield_max_mpa=float(fields["yield_values"][n1:, k].max()*C.SIGMA_REF/1e6)))
    return rows


@torch.no_grad()
def exterior_yield_rows(checkpoint, device):
    """Check the elastic domain assumed by the same coupled exterior model."""
    bundle, saved, model = load_orientation_model(checkpoint, device)
    settings, case = saved["settings"], saved["case"]
    model.increments = model.increments[:saved["completed_steps"]]
    mat = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
    radius = settings["radius"]
    n_gamma = settings["quadrature"]["boundary"]
    trace_angle = 2*np.pi*np.arange(n_gamma)/n_gamma
    trace_xy = radius*np.stack((np.cos(trace_angle), np.sin(trace_angle)), -1)
    displacement = model(model.multiplier_scale.new_tensor(trace_xy))[0].cpu().numpy()*C.U_REF
    last_mode = settings.get("projection", {}).get("angular_modes", n_gamma//2-1)
    angle = 2*np.pi*(np.arange(720)+.37)/720
    radii = radius*np.geomspace(1, 4, 24)
    x = (C.A_TUNNEL*radii[:, None]*np.cos(angle)).reshape(-1)
    y = (C.A_TUNNEL*radii[:, None]*np.sin(angle)).reshape(-1)
    compliance = mat.full_compliance()
    zz_coefficients = -compliance[2, [0, 1, 5]]/compliance[2, 2]
    rows = []
    for k in range(model.n_steps):
        field = P._exterior_field_from_trace(displacement[:, k].T, mat, radius*C.A_TUNNEL, last_mode)
        xx, yy, xy = field.stresses(x, y)
        zz = zz_coefficients @ np.stack((xx, yy, xy))
        stress = torch.from_numpy(np.stack((xx, yy, zz, xy), -1))/C.SIGMA_REF+bundle["s0"]
        stress = torch.cat((stress, bundle["s0"][None]))
        rows.append(dict(case=case["name"], step=k+1,
            max_matrix_yield_mpa=float(P.matrix_yield(stress, bundle["params"]).max()*C.SIGMA_REF/1e6),
            max_joint_yield_mpa=float(P.joint_yield(stress, bundle["params"]).max()*C.SIGMA_REF/1e6),
            radial_samples=len(radii), angular_samples=len(angle),
            minimum_radius_over_a=float(radii[0]), maximum_radius_over_a=float(radii[-1]),
            includes_far_field=True))
    return rows


def orientation_response(source, case, load_steps):
    """Summarize the exported wall response without rerunning field diagnostics."""
    paths = read(source / "stress_paths.csv")
    a, b = (next(row for row in paths if row["location"] == name and
                 int(row["load_step"]) == load_steps)
            for name in ("maximum_convergence", "maximum_slip"))
    wall = [row for row in read(source / "wall_profiles.csv")
            if int(row["load_step"]) == load_steps]
    return dict(case=case["name"], joint_normal_deg=case["joint_normal_deg"],
        convergence_max_mm=float(a["closure_mm"]),
        convergence_mean_mm=float(np.mean([float(row["convergence_mm"]) for row in wall])),
        slip_max_percent=100*float(b["accumulated_slip_strain"]),
        convergence_at_slip_max_mm=float(b["closure_mm"]),
        convergence_ratio_B_A=float(b["closure_mm"])/float(a["closure_mm"]),
        convergence_angle_deg=float(a["angle_deg"]),
        slip_angle_deg=float(b["angle_deg"]),
        angular_separation_mod180_deg=abs((float(a["angle_deg"])-float(b["angle_deg"])+90) % 180-90),
        joint_yield_A_mpa=float(a["joint_yield_MPa"]),
        joint_yield_B_mpa=float(b["joint_yield_MPa"]))


def orientation(root, config_path, device, histories=None):
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    summaries, residual_rows, exterior_rows = [], [], []
    for case in settings["cases"]:
        source = histories[case["name"]] if histories is not None else root / case["name"]
        residuals = orientation_history_residuals(source / "model.pt", device)
        residual_rows.extend(residuals)
        exterior = exterior_yield_rows(source / "model.pt", device)
        exterior_rows.extend(exterior)
        final_residuals = {row["sampling"]: row for row in residuals if row["step"] == settings["load_steps"]}
        values = {f"{sampling}_{quantity}": row[quantity]
                  for sampling, row in final_residuals.items()
                  for quantity in ("equilibrium_rms", "constitutive_rms", "complementarity_rms")}
        values.update({key: final_residuals["independent"][key] for key in
                       ("wall_traction_rms_mpa", "interface_compliance", "interface_yield_max_mpa")})
        values["exterior_max_yield_mpa"] = max(max(row["max_matrix_yield_mpa"], row["max_joint_yield_mpa"])
                                               for row in exterior)
        summaries.append(dict(orientation_response(source, case, settings["load_steps"]), **values))
    write_rows(root / "summary.csv", summaries)
    write_rows(root / "residuals.csv", residual_rows)
    write_rows(root / "exterior_yield.csv", exterior_rows)
    print(json.dumps(dict(orientation_cases=len(summaries))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("elastic", "orientation", "all"), default="all")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--orientation-data", type=Path,
                        help="Alternative history root; defaults to the manuscript's selected histories.")
    parser.add_argument("--orientation-output", type=Path, default=ROOT / "results/history_analysis")
    parser.add_argument("--orientation-config", type=Path, default=ROOT / "configs/orientation_extension.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.study in ("elastic", "all"):
        elastic()
    if args.study in ("orientation", "all"):
        from current_histories import HISTORIES
        destination = (args.orientation_data or args.orientation_output).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        orientation(destination, args.orientation_config.resolve(), args.device,
                    histories=HISTORIES if args.orientation_data is None else None)


if __name__ == "__main__":
    main()
