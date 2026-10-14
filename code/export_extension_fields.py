"""Reconstruct study field grids from the supplied final networks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from interface_extension import elastic_observables, geometry, make_elastic_model
from pilot.spectral_mixed import load_orientation_model
from run_orientation_extension import export_case


ROOT = Path(__file__).resolve().parents[1]


@torch.no_grad()
def export_elastic(directory, settings, case, seed, metric, device):
    saved = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    points, stiffness, insitu, exterior = geometry(case, settings, device, evaluation=True)
    model = make_elastic_model(settings, stiffness, seed)
    model.load_state_dict(saved["model"])
    model.eval()
    _, fields, residuals = elastic_observables(model, points, stiffness, exterior)
    np.savez_compressed(
        directory / "fields.npz",
        xy=points.xy_all.cpu().numpy(),
        displacement=fields["displacement"].cpu().numpy(),
        stress_increment=fields["stress_increment"].cpu().numpy(),
        wall_angles=points.wall_angles.cpu().numpy(),
        wall_normal=points.nrm_wall.cpu().numpy(),
        interface_residual=residuals["interface"].cpu().numpy(),
        interface_coefficients=exterior.coeffs(residuals["interface"].T).cpu().numpy(),
        n_pde=points.n_pde, n_wall=points.n_wall,
        insitu=insitu.cpu().numpy(),
    )
    print(f"Exported {case['name']} / seed {seed} / {metric}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("elastic", "orientation", "all"), default="all")
    parser.add_argument("--case", help="Export only this case name.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    exported = 0
    for study, name in (("elastic", "interface_extension"), ("orientation", "orientation_extension")):
        if args.study not in (study, "all"):
            continue
        settings = json.loads((ROOT / "configs" / f"{name}.json").read_text(encoding="utf-8"))
        for case in settings["cases"]:
            if args.case is not None and args.case != case["name"]:
                continue
            directory = ROOT / "results" / name / case["name"]
            if study == "elastic":
                for seed in settings["seeds"]:
                    for metric in case["metrics"]:
                        export_elastic(directory / f"seed_{seed}" / metric,
                                       settings, case, seed, metric, args.device)
                        exported += 1
            else:
                bundle, saved, model = load_orientation_model(directory / "model.pt", args.device)
                export_case(directory, case, saved["settings"], model,
                            bundle["c"], bundle["s0"], bundle["params"])
                print(f"Exported {case['name']}", flush=True)
                exported += 1
    if not exported:
        parser.error(f"No matching case: {args.case}")


if __name__ == "__main__":
    main()
