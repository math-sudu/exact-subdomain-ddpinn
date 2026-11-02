"""Evaluate displacement, stress and plastic strain from the shared histories."""
import argparse
import csv
from pathlib import Path

import torch

from oespinn import config as C
from oespinn.horseshoe_geometry import mapped_basis
from reconstruct_shared_history import load_history
from run_horseshoe import make_work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--radial-points", type=int, default=32)
    parser.add_argument("--angular-points", type=int, default=128)
    parser.add_argument("--radius-m", type=float, default=19.)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    settings, increments = load_history(args.case, args.device)
    geometry, work = make_work(settings, args.device)
    steps = args.steps or [len(increments)]
    if any(step < 1 or step > len(increments) for step in steps):
        raise ValueError(f"Steps must lie between 1 and {len(increments)}")
    theta = torch.arange(args.angular_points, device=args.device) * (2 * torch.pi / args.angular_points)
    wall = geometry.wall(theta)[0] * geometry.length_scale
    if not float(wall.max()) < args.radius_m <= geometry.radius * geometry.length_scale:
        raise ValueError("Sampling radius must enclose the wall and stay within the coupling interface")
    fraction = torch.linspace(0, 1, args.radial_points, device=args.device)
    radius = wall[None] + fraction[:, None] * (args.radius_m - wall[None])
    xy_m = torch.stack((radius * theta.cos(), radius * theta.sin()), -1).reshape(-1, 2)
    strain_scale = C.SIGMA_REF / C.E_REF
    columns = ["step", "x_m", "y_m", "ux_mm", "uy_mm", "sigma_xx_mpa", "sigma_yy_mpa",
               "sigma_zz_mpa", "sigma_xy_mpa", "plastic_xx", "plastic_yy", "plastic_zz", "plastic_gamma_xy"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream, torch.no_grad():
        writer = csv.writer(stream)
        writer.writerow(columns)
        for points in xy_m.split(128):
            basis = mapped_basis(geometry, points / geometry.length_scale, work.j, work.m)
            states = work.replay(increments, basis)
            for step in steps:
                state = states[step - 1]
                values = torch.cat((points, state["displacement"] * strain_scale * geometry.length_scale * 1000,
                                    state["stress"] * C.SIGMA_REF / 1e6, state["plastic"] * strain_scale), dim=1)
                writer.writerows([step, *row] for row in values.cpu().tolist())
    print(f"Wrote {len(xy_m) * len(steps)} field samples to {args.output}")


if __name__ == "__main__":
    main()
