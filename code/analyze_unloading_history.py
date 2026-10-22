"""Extract slip sectors, section gauges and whole-space radial histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from current_histories import HISTORIES
from extract_unloading_results import plane_shear, sample_fields, stress_path_rows
from pilot import plasticity as P
from pilot.spectral_mixed import ReturnedCausalPINN, load_orientation_model
from publication_model import write_rows
from sf_core.material import TIMaterial

ROOT = Path(__file__).resolve().parents[1]
CASES = ("bedding_oblique", "bedding_aligned", "bedding_crossed")
OUTPUT = ROOT / "results/history_analysis"


def material_from_metadata(metadata):
    m = metadata["material"]
    return TIMaterial(name="retained_anisotropic_material", Eh=m["Eh_GPa"] * 1e9, Ev=m["Ev_GPa"] * 1e9,
                      Gvh=m["Gvh_GPa"] * 1e9, nu_vh=m["nu_vh"],
                      nu_hh=m["nu_hh"], phi=np.deg2rad(m["bedding_dip_deg"]))


@torch.no_grad()
def wall_history(bundle, model, angles, scale):
    units = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    points = model.multiplier_scale.new_tensor(units)
    u, stress, multiplier = (model.returned_history(points)[:3]
        if isinstance(model, ReturnedCausalPINN) else model(points))
    total = stress + bundle["s0"].to(stress)
    joint = multiplier[..., 1:] * P.joint_flow_direction(total, bundle["params"])
    delta = plane_shear(joint, bundle["params"].joint_normal_deg).cpu().numpy()
    delta *= scale["displacement_mm"] / (1000 * scale["tunnel_radius_m"])
    sn, tau, _, _ = P._joint_stresses(total, bundle["params"])
    u = u.cpu().numpy() * scale["displacement_mm"]
    return dict(displacement_mm=u, convergence_mm=-np.einsum("aki,ai->ak", u, units),
                slip_increment=delta, signed_slip=np.cumsum(delta, axis=1),
                accumulated_slip=np.cumsum(np.abs(delta), axis=1),
                sigma_n_MPa=sn.cpu().numpy() * scale["stress_MPa"],
                tau_j_MPa=tau.cpu().numpy() * scale["stress_MPa"])


def slip_sectors(case, angles, history):
    """Track the strongest positive/negative shear lobes in each half circle.

    Each output retains its actual 360-degree angle. A lobe is identified by
    shear sign and its spatial half, never by its instantaneous global rank.
    """
    rows = []
    for step in range(history["accumulated_slip"].shape[1]):
        signed = history["signed_slip"][:, step]
        for half in (0, 1):
            for sign, name in ((1, "positive"), (-1, "negative")):
                indices = np.flatnonzero((angles // 180 == half) & (sign * signed > 0))
                if not len(indices):
                    continue
                j = indices[np.argmax(history["accumulated_slip"][indices, step])]
                d = indices[np.argmax(np.abs(history["slip_increment"][indices, step]))]
                rows.append(dict(case=case, load_step=step + 1,
                                 branch=f"{name}_{half}", angle_deg=angles[j],
                                 accumulated_slip=history["accumulated_slip"][j, step],
                                 signed_slip=history["signed_slip"][j, step],
                                 increment_at_cumulative_peak=history["slip_increment"][j, step],
                                 increment_peak_angle_deg=angles[d],
                                 peak_increment=history["slip_increment"][d, step]))
    return rows


def gauge_histories(case, angles, history, radius):
    units = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    half = len(angles) // 2
    x1, x2 = radius * units[:half], radius * units[half:]
    direction = (x2 - x1) / np.linalg.norm(x2 - x1, axis=1)[:, None]
    delta = history["displacement_mm"][half:] - history["displacement_mm"][:half]
    shortening = -np.einsum("aki,ai->ak", delta, direction)
    np.testing.assert_allclose(shortening, history["convergence_mm"][:half]
                               + history["convergence_mm"][half:], atol=1e-14)
    rows = []
    for j in [int(np.argmin(np.abs(angles - a))) for a in (0, 45, 90, 135)]:
        for k in range(shortening.shape[1]):
            rows.append(dict(case=case, angle_deg=angles[j], load_step=k + 1,
                             release_fraction=(k + 1) / shortening.shape[1],
                             x1_m=x1[j, 0], y1_m=x1[j, 1],
                             x2_m=x2[j, 0], y2_m=x2[j, 1],
                             shortening_mm=shortening[j, k]))
    return shortening, rows


@torch.no_grad()
def whole_space(model, metadata, angles):
    """Use the decaying exterior and the same zero-mean translation gauge."""
    rg = metadata["radius_gamma_over_a"]
    scale = metadata["scales"]["displacement_mm"]
    units = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    trace = model(model.multiplier_scale.new_tensor(rg * units))[0].cpu().numpy()
    mean = trace.mean(axis=0)
    mat = material_from_metadata(metadata)
    inside = 1 + (rg - 1) * np.linspace(0, 1, 257) ** 2
    outside = np.geomspace(rg, 100, 161)
    radii = np.r_[inside, outside[1:]]
    histories = []
    labels = ("final_maximum_closure", "final_maximum_slip")
    directions = [metadata["fixed_locations"][name]["xy_over_a"] for name in labels]
    fields = [P._exterior_field_from_trace(trace[:, k].T, mat, rg,
                                          model.projection["angular_modes"])
              for k in range(model.n_steps)]
    for direction in np.array(directions):
        xy = inside[:, None] * direction
        inner = model(model.multiplier_scale.new_tensor(xy))[0].cpu().numpy() - mean[None]
        outer = []
        xy = outside[:, None] * direction
        for k, field in enumerate(fields):
            uv = np.stack(field.displacements(xy[:, 0], xy[:, 1]), axis=-1)
            np.testing.assert_allclose(uv[0], inner[-1, k], atol=2e-12)
            outer.append(uv)
        uv = np.concatenate((inner, np.stack(outer, axis=1)[1:]), axis=0)
        histories.append(np.einsum("rki,i->rk", uv, direction) * scale)
    ur = np.stack(histories)
    return dict(radius_over_a=radii, radial_displacement_mm=ur,
                cumulative_mm=ur - ur[:, :1], tail_mm=-ur,
                convergence_mm=-ur[:, 0], interface_mean_mm=mean * scale)


@torch.no_grad()
def slip_inset(bundle, model, checkpoint, grid):
    """Keep the common display window when a computational interface moves."""
    radius, angles = grid["radius_over_a"], grid["angle_rad"]
    if bundle["spec"]["r_gamma"] == grid["radius_edges_over_a"][-1]:
        with np.load(checkpoint.with_name("fields.npz")) as fields:
            np.testing.assert_allclose(fields["radius_over_a"], radius, atol=1e-12)
            np.testing.assert_allclose(np.deg2rad(fields["angle_deg"]), angles, atol=1e-12)
            values = fields["accumulated_slip_strain"][:, :, -1]
    else:
        unit = np.column_stack((np.cos(angles), np.sin(angles)))
        xy = (radius[:, None, None] * unit[None]).reshape(-1, 2)
        fields = sample_fields(bundle, model, xy)
        values = fields["accumulated_slip_strain"][:, -1].reshape(len(radius), len(angles)).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite final slip in the display window: {checkpoint}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--root", type=Path,
                        help="Override the current histories with one root containing <case>/model.pt")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    angles = np.arange(720) * .5
    metadata = json.loads((ROOT / "results/unloading/metadata.json").read_text())
    grid = np.load(ROOT / "results/unloading/fields.npz")
    histories, sectors, gauges, summaries, models = [], [], [], [], []
    paths, insets = [], []
    for case in CASES:
        directory = args.root.resolve() / case if args.root else HISTORIES[case]
        checkpoint = directory / "model.pt"
        bundle, saved, model = load_orientation_model(checkpoint)
        if saved["completed_steps"] != model.total_steps:
            raise ValueError(f"Incomplete unloading history: {checkpoint}")
        h = wall_history(bundle, model, angles, metadata["scales"])
        histories.append(h)
        sectors.extend(slip_sectors(case, angles, h))
        diametric, rows = gauge_histories(case, angles, h, metadata["scales"]["tunnel_radius_m"])
        gauges.extend(rows)
        h["diametric_convergence_mm"] = diametric
        j = int(np.argmax(diametric[:, -1]))
        summaries.append(dict(case=case, maximum_diametric_mm=diametric[j, -1],
                              maximum_diametric_angle_deg=angles[j],
                              horizontal_mm=diametric[0, -1], vertical_mm=diametric[180, -1],
                              diagonal_45_mm=diametric[90, -1], diagonal_135_mm=diametric[270, -1]))
        paths.append(checkpoint.relative_to(ROOT).as_posix())
        models.append((bundle, model))
        insets.append(slip_inset(bundle, model, checkpoint, grid))
        print(case, summaries[-1], flush=True)
    arrays = {key: np.stack([h[key] for h in histories]) for key in histories[0]}
    np.savez_compressed(args.output / "wall_history.npz", angle_deg=angles, cases=CASES, **arrays)
    np.savez_compressed(args.output / "slip_insets.npz", cases=CASES,
                        radius_over_a=grid["radius_over_a"], angle_rad=grid["angle_rad"],
                        accumulated_slip_strain=np.stack(insets))
    write_rows(args.output / "slip_sectors.csv", sectors)
    write_rows(args.output / "gauge_histories.csv", gauges)
    write_rows(args.output / "gauge_summary.csv", summaries)
    bundle, model = models[0]
    full = whole_space(model, metadata, angles)
    np.savez_compressed(args.output / "whole_space_radial.npz", **full)
    points = dict(metadata["fixed_locations"])
    onset = next(row for row in sectors
                 if row["case"] == CASES[0] and row["branch"] == "negative_0")
    for name, angle in (("early_negative_lobe", onset["angle_deg"]),):
        points[name] = dict(angle_deg=angle, xy_over_a=[np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    write_rows(args.output / "fixed_sector_stress_paths.csv", stress_path_rows(bundle, model, points))
    provenance = dict(checkpoints=paths, producer="code/analyze_unloading_history.py",
                      slip_insets="Final slip on the baseline a<r<3a grid; expanded histories are queried at these coordinates",
                      angles="Counterclockwise from +x; original 0–360 degree fields retained",
                      branch_definition="Shear sign and fixed 180-degree half; no global-rank tracking",
                      scales=metadata["scales"],
                      whole_space_gauge="Subtract interface Cartesian mean at each increment from inner displacement; exterior decays at infinity",
                      fixed_locations=points)
    (args.output / "metadata.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("Interface translation removed (maximum norm, mm):", np.linalg.norm(full["interface_mean_mm"], axis=1).max())
    for k in (5, 8, 9, 12):
        print("Baseline sectors", k, [(r["branch"], r["angle_deg"], r["accumulated_slip"])
                                      for r in sectors if r["case"] == CASES[0] and r["load_step"] == k])


if __name__ == "__main__":
    main()
