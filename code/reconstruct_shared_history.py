"""Load shared spectral coefficients or write a history for material replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_history(case, device="cpu"):
    histories = json.loads((ROOT / "data/histories.json").read_text("utf-8"))
    record = histories[case]
    settings = json.loads((ROOT / record["config"]).read_text("utf-8"))
    values = np.loadtxt(ROOT / "data" / case / "coefficients.csv", delimiter=",", skiprows=1)
    dimension = 2 * (settings["radial_order"] + 1) * (2 * settings["angular_modes"] + 1) - 2
    coefficients = values[:, 2:4].reshape(record["steps"], dimension // 2, 2)
    increments = [torch.as_tensor(row, dtype=torch.float64, device=device) for row in coefficients]
    return settings, increments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings, coefficients = load_history(args.case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(settings=settings, coefficients=coefficients,
                    completed_steps=len(coefficients), solver="shared_spectral_history"), args.output)
    print(f"Wrote {len(coefficients)} increments to {args.output}")


if __name__ == "__main__":
    main()
