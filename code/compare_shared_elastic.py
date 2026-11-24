"""Compare the archived first-release displacements with two independent FE meshes."""
import argparse
import json
from pathlib import Path

import pandas as pd

from compare_horseshoe_elastic_fem import compare_first_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    settings = json.loads((root / "configs/refined.json").read_text(encoding="utf-8"))
    geometry = json.loads((root / settings["geometry"]).read_text(encoding="utf-8"))
    contour = next(b for b in geometry["boundaries"] if b["role"] == "initial_support_outer")
    direct = pd.read_csv(root / "data/accuracy_reference/responses.csv", nrows=1).iloc[0]
    network = pd.read_csv(root / "data/full_network/responses.csv", nrows=1).iloc[0]
    comparison = compare_first_release(settings, contour, direct, network)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
