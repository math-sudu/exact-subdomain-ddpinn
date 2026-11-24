"""Compare external traction on a common physical circle after an interface change."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from assess_horseshoe import exterior_field, exterior_rows
from run_horseshoe import make_work


@torch.no_grad()
def tractions(saved, common_radius):
    _, work = make_work(saved["settings"], "cpu")
    increments = saved["coefficients"]
    theta = np.arange(768)*(2*np.pi/768)
    nx, ny = np.cos(theta), np.sin(theta)
    u = torch.zeros_like(increments[0])
    values = []
    for delta in increments:
        u += delta
        sx, sy, txy = exterior_field(work, u).stresses(common_radius*nx, common_radius*ny)
        values.append(np.column_stack([sx*nx+txy*ny, txy*nx+sy*ny])/1.e6)
    margins = pd.DataFrame(exterior_rows(work, increments)).groupby("step").max()
    return np.array(values), margins.filter(like="yield_mpa").max(axis=1).to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("closer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    saved = [torch.load(p, map_location="cpu", weights_only=False) for p in (args.original,args.closer)]
    radius = max(s["settings"]["interface_radius"]*s["settings"]["length_scale_m"] for s in saved)
    (t0, f0), (t1, f1) = [tractions(s, radius) for s in saved]
    def rms(v):
        return np.sqrt((v**2).sum(axis=-1).mean(axis=-1))
    pd.DataFrame(dict(step=np.arange(len(t0))+1, radius_m=radius,
                      baseline_rms_mpa=rms(t0), closer_rms_mpa=rms(t1),
                      difference_rms_mpa=rms(t1-t0), relative_difference_pct=100*rms(t1-t0)/rms(t0),
                      baseline_max_exterior_yield_mpa=f0, closer_max_exterior_yield_mpa=f1)
                 ).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
