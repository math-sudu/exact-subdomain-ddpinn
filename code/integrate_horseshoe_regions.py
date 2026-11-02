"""Replay plastic sources on radial quadrature fitted to the depth partitions."""
import argparse
import csv
import json
import time

import numpy as np
import torch

from assess_horseshoe import extend, source_rows
from oespinn.horseshoe_geometry import mapped_basis
from run_horseshoe import ROOT, load_coefficients, make_work


def depth_grid(geometry, nr, nt, device):
    """Gauss integration in each physical depth interval; periodic angles."""
    tensor = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    nodes, weights = np.polynomial.legendre.leggauss(nr)
    nodes, weights = tensor((nodes+1)/2), tensor(weights/2)
    theta = (torch.arange(nt, device=device, dtype=torch.float64)+.5)*(2*torch.pi/nt)
    wall = geometry.wall(theta)[0]
    edges = (torch.zeros_like(wall), torch.full_like(wall, 2/geometry.length_scale),
             torch.full_like(wall, 5/geometry.length_scale), geometry.radius-wall)
    grids = []
    for low, high in zip(edges[:-1], edges[1:]):
        if bool((high <= low).any()):
            raise ValueError("The interface must enclose all requested depth bands")
        radius = wall[None]+low[None]+nodes[:, None]*(high-low)[None]
        xy = torch.stack((radius*theta.cos(), radius*theta.sin()), -1).flatten(0, 1)
        w = (radius*(high-low)[None]*weights[:, None]*(2*torch.pi/nt)).flatten()
        grids.append((xy, w))
    return torch.cat([g[0] for g in grids]), torch.cat([g[1] for g in grids])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nr", type=int, default=24, help="Gauss points per depth interval")
    parser.add_argument("--nt", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    saved = torch.load(ROOT/args.checkpoint, map_location=args.device, weights_only=False)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    settings = dict(saved["settings"])
    if saved["completed_steps"] != settings["load_steps"]:
        raise ValueError("Regional integration requires the completed history")
    j0, m0 = settings["radial_order"], settings["angular_modes"]
    settings.update(radial_order=24, angular_modes=48)
    _, work = make_work(settings, args.device,
                        quadrature=dict(radial=64, angular=512, boundary=1024, shift=.5))
    retained, _ = load_coefficients(saved, work.geometry, args.device)
    coefficients = [extend(c, j0, m0, work.j, work.m) for c in retained]
    xy, weights = depth_grid(work.geometry, args.nr, args.nt, args.device)
    area = float(weights.sum())*work.geometry.length_scale**2
    original_area = float(work.area)*work.geometry.length_scale**2
    influences = {name: torch.cholesky_solve(g[:, None], work.factor).reshape(-1, 2)
                  for name, g in work.gauges().items()}
    # Keep the common independently integrated elastic influence operator.
    # Only plastic history queries and source-integration weights change.
    rows = {}
    for first in range(0, len(xy), args.chunk_size):
        work.xy, work.w = xy[first:first+args.chunk_size], weights[first:first+args.chunk_size]
        work.value, work.dx, work.dy = mapped_basis(work.geometry, work.xy, work.j, work.m)
        fields = work.replay(coefficients)
        for row in source_rows(work, coefficients, fields, influence_vectors=influences):
            key = tuple(row[k] for k in ("step", "gauge", "mechanism", "region"))
            if key not in rows:
                rows[key] = row
            else:
                for column in ("net_mm", "positive_mm", "negative_mm"):
                    rows[key][column] += row[column]
        print(json.dumps(dict(points=min(first+args.chunk_size, len(xy)), total=len(xy))), flush=True)
    output = ROOT/args.output
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    computation_seconds = time.perf_counter()-start
    output.mkdir(parents=True, exist_ok=True)
    with (output/"sources.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())
    metadata = dict(checkpoint=args.checkpoint, steps=saved["completed_steps"],
        radial_points_per_band=args.nr, angles=args.nt, shift=.5,
        chunk_size=args.chunk_size,
        depth_breaks_m=[0, 2, 5], outer_boundary="circular coupling interface",
        influence_space=[24, 48], influence_quadrature=[64, 512],
        computation_seconds=computation_seconds,
        timing_scope="Influence assembly, saved-history replay and source integration; checkpoint loading and output writing excluded",
        integrated_area_m2=area, common_area_m2=original_area,
        area_relative_difference=abs(area/original_area-1))
    (output/"assessment.json").write_text(json.dumps(metadata, indent=2)+"\n", "utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
