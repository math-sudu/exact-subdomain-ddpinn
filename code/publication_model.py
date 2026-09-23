"""Load the final spatial networks without training checkpoints or point sets."""

import csv

import torch

from pilot.causal_mixed import CausalMixedPINN
from pilot.plasticity import WeakPlaneParameters


def load_model(run, checkpoint=None, device="cpu"):
    saved = torch.load(checkpoint or run / "model.pt", map_location="cpu", weights_only=True)
    spec = saved["spec"]
    with torch.random.fork_rng(devices=[]):
        model = CausalMixedPINN(spec["hidden"], spec["input_scale"],
                               spec["load_steps"], saved["c"]).to(saved["c"])
        for _ in range(saved["load_step"]):
            model.append_increment(copy_previous=False)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    bundle = dict(spec=spec, c=saved["c"], s0=saved["s0"],
                  params=WeakPlaneParameters(**spec["physics"]))
    return bundle, saved, model


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
