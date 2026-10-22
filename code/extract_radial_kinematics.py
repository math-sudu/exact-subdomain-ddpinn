"""Query the retained spatial history for radial kinematic figure panels."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from publication_model import load_model, write_rows
from pilot.spectral_mixed import load_orientation_model

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/unloading"
LOCATIONS = ("final_maximum_closure", "final_maximum_slip")


def radial_history(model, radii, direction, displacement_mm, radius_m):
    """Return u and its radial derivative at fixed spatial coordinates."""
    rho = model.multiplier_scale.new_tensor(radii).requires_grad_(True)
    unit = model.multiplier_scale.new_tensor(direction)
    u = model(rho[:, None] * unit)[0]
    ur = (u * unit).sum(-1)
    derivatives = torch.stack([
        torch.autograd.grad(ur[:, k].sum(), rho, retain_graph=k < model.n_steps - 1)[0]
        for k in range(model.n_steps)
    ], -1)
    # rho = r/a; epsilon_rr = (U_ref/a) d(u_hat_r)/d(rho).
    return (u.detach().cpu().numpy() * displacement_mm,
            ur.detach().cpu().numpy() * displacement_mm,
            derivatives.detach().cpu().numpy() * displacement_mm / (1000 * radius_m))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    formulation = parser.add_mutually_exclusive_group()
    formulation.add_argument('--orientation', action='store_true', default=False)
    formulation.add_argument('--pointwise', action='store_false', dest='orientation')
    args = parser.parse_args()
    data = args.data.resolve()
    torch.set_num_threads(1)
    metadata = json.loads((data / "metadata.json").read_text(encoding="utf-8"))
    if args.orientation:
        _, saved, model = load_orientation_model(ROOT / metadata["checkpoint"], "cpu")
    else:
        _, saved, model = load_model(ROOT / metadata["source_run"],
                                     ROOT / metadata["checkpoint"], "cpu")
    if ("algorithm_state" in saved and saved["algorithm_state"]["event"] != metadata["event"]) or model.n_steps != 12:
        raise ValueError("Use the twelve-increment checkpoint bound to the field export.")
    rg = metadata["radius_gamma_over_a"]
    scale = metadata["scales"]["displacement_mm"]
    radius_m = metadata["scales"]["tunnel_radius_m"]
    radii = 1 + (rg - 1) * np.linspace(0, 1, 257)**2
    angles = np.arange(720) * .5
    units = np.column_stack((np.cos(np.deg2rad(angles)), np.sin(np.deg2rad(angles))))
    with torch.no_grad():
        xy = model.multiplier_scale.new_tensor(np.concatenate((units, rg * units)))
        boundary_u = model(xy)[0].cpu().numpy() * scale
    wall = -(boundary_u[:720] * units[:, None]).sum(-1)
    interface = -(boundary_u[720:] * units[:, None]).sum(-1)
    annulus = wall - interface

    # Integrate spatial derivatives independently along A/B using the same
    # radial partition as the original calculation, resolving the narrow
    # near-wall derivative with 32 Gauss points per panel.
    nodes, weights = np.polynomial.legendre.leggauss(32)
    edges = 1 + (rg - 1) * np.linspace(0, 1, 33)**2
    qr = (edges[:-1, None] + np.diff(edges)[:, None] * (nodes + 1) / 2).ravel()
    qw = (np.diff(edges)[:, None] * weights / 2).ravel()
    u, ur, strain, contributions, rows = [], [], [], [], []
    for name, letter in zip(LOCATIONS, ("A", "B")):
        location = metadata["fixed_locations"][name]
        direction = location["xy_over_a"]
        uv, rv, ev = radial_history(model, radii, direction, scale, radius_m)
        _, _, qe = radial_history(model, qr, direction, scale, radius_m)
        integral = np.einsum("r,rk->k", qw, qe) * radius_m * 1000
        c, motion = -rv[0], -rv[-1]
        # This checks the kinematic extraction and unit conversion, not the
        # field equations or a comparison with another solution method.
        np.testing.assert_allclose(c, motion + integral, rtol=1e-7, atol=1e-9)
        index = int(round(location["angle_deg"] / .5))
        np.testing.assert_allclose(wall[index], c, rtol=1e-10, atol=1e-12)
        u.append(uv)
        ur.append(rv)
        strain.append(ev)
        contributions.append(np.stack((c, motion, integral), -1))
        for k in range(12):
            rows.append(dict(location=letter, angle_deg=location["angle_deg"],
                             load_step=k+1, release_fraction=(k+1)/12,
                             wall_convergence_mm=c[k], interface_contribution_mm=motion[k],
                             annulus_contribution_mm=integral[k]))
    np.savez_compressed(data / "radial_kinematics.npz", radius_over_a=radii,
                        angle_deg=angles, displacement_mm=np.array(u),
                        radial_displacement_mm=np.array(ur), radial_strain=np.array(strain),
                        wall_convergence_mm=wall, interface_motion_mm=interface,
                        annulus_integral_mm=annulus, fixed_contributions_mm=np.array(contributions))
    write_rows(data / "radial_contributions.csv", rows)
    metadata["radial_kinematics"] = dict(
        source="code/extract_radial_kinematics.py", arrays=(data / "radial_kinematics.npz").relative_to(ROOT).as_posix(),
        fixed_contributions=(data / "radial_contributions.csv").relative_to(ROOT).as_posix(),
        location_order=["A", "B"], radial_samples=257, angle_spacing_deg=.5,
        strain_definition="Full strain of the excavation displacement: d(u_r)/d(r), from spatial AD.",
        annulus_integral="Full angular profiles use u_r(R_Gamma)-u_r(a); A/B use 32 quadratic radial panels with 32 Gauss points each.",
        cumulative_integral="From the wall to radius r: u_r(r)-u_r(a), in mm.",
        contribution_order=["wall convergence", "interface motion", "annulus radial-strain integral"],
        units="Displacement and integrated contributions: mm. Strain: dimensionless. Coordinates: r/a.",
    )
    (data / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    values = np.array(contributions)
    gap = values[0, -1] - values[1, -1]
    print("Final A/B [convergence, interface, annulus] (mm):", values[:, -1].tolist())
    print("A minus B (mm):", gap.tolist())
    print("Interface / annulus fractions of the final gap:", (gap[1:] / gap[0]).tolist())
    print("Radial strain ranges (microstrain):", [(float(e.min()*1e6), float(e.max()*1e6)) for e in strain])


if __name__ == "__main__":
    main()
