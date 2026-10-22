"""Sample the completed baseline for the spatial locator in Figure 7."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_unloading_history import material_from_metadata
from current_histories import HISTORIES
from pilot import plasticity as P
from pilot.elastic_background import CircularElasticIncrement
from pilot.spectral_mixed import load_orientation_model

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/unloading"
OUTPUT = ROOT / "results/history_analysis/kinematics_figure.npz"


@torch.no_grad()
def inner_kinematics(model, xy, scale):
    """Sum the exact derivatives of the saved displacement increments."""
    displacements, strains = [], []
    for block in np.array_split(xy, max(1, int(np.ceil(len(xy) / 2048)))):
        points = model.multiplier_scale.new_tensor(block)
        direction = points / points.norm(dim=-1, keepdim=True)
        u = torch.zeros_like(points)
        derivative = points.new_zeros((2, len(points), 2))
        for increment in model.increments:
            values, derivatives = increment.spatial_fields(points)
            u += values[0]
            derivative += derivatives[0]
        err = torch.einsum("ni,inj,nj->n", direction, derivative, direction)
        displacements.append(u.cpu().numpy() * scale["displacement_mm"])
        strains.append(err.cpu().numpy() * scale["displacement_mm"]
                       / (1000 * scale["tunnel_radius_m"]))
    return np.concatenate(displacements), np.concatenate(strains)


def exterior_radial_strain(field, xy, scale):
    """Differentiate the same decaying complex potentials used by the solver."""
    direction = xy / np.linalg.norm(xy, axis=-1, keepdims=True)
    c, s = direction.T
    derivative = np.zeros(len(xy))
    for k, (p, q, mu) in enumerate(((field.mat.p1, field.mat.q1, field.mat.mu1),
                                    (field.mat.p2, field.mat.q2, field.mat.mu2))):
        _, dphi = field._phi_phiprime(k, xy[:, 0] + mu * xy[:, 1])
        derivative += 2 * np.real((p * c + q * s) * dphi * (c + mu * s))
    return derivative * scale["displacement_mm"] / (1000 * scale["tunnel_radius_m"])


@torch.no_grad()
def extract(output):
    torch.set_num_threads(4)
    metadata = json.loads((BASELINE / "metadata.json").read_text())
    locations = json.loads((OUTPUT.parent / "metadata.json").read_text())["fixed_locations"]
    checkpoint = HISTORIES["bedding_oblique"] / "model.pt"
    _, saved, model = load_orientation_model(checkpoint, "cpu")
    if saved["completed_steps"] != 12 or model.n_steps != 12:
        raise ValueError("Figure 7 requires the completed twelve-step baseline.")
    model.cache_projection()
    scale = metadata["scales"]
    with np.load(BASELINE / "fields.npz") as grid:
        radii = grid["radius_over_a"]
        angles = grid["angle_rad"]
        unit = np.column_stack((np.cos(angles), np.sin(angles)))
        xy = (radii[:, None, None] * unit).reshape(-1, 2)
        u, strain = inner_kinematics(model, xy, scale)
        np.testing.assert_allclose(u.reshape(*grid["displacement_mm"].shape[:2], 2),
                                   grid["displacement_mm"][:, :, -1], atol=1.5e-8, rtol=1e-7)
        edges = grid["radius_edges_over_a"]
    rg = metadata["radius_gamma_over_a"]
    trace = model(model.multiplier_scale.new_tensor(rg * unit))[0][:, -1].numpy()
    field = P._exterior_field_from_trace(trace.T, material_from_metadata(metadata),
                                         rg, model.projection["angular_modes"])
    recovered = np.stack(field.displacements(rg * unit[:, 0], rg * unit[:, 1]), -1)
    np.testing.assert_allclose(recovered, trace - trace.mean(axis=0), atol=2e-12)
    outer_edges = np.linspace(rg, 4, 49)
    outer_radii = (outer_edges[1:] + outer_edges[:-1]) / 2
    outer_xy = (outer_radii[:, None, None] * unit).reshape(-1, 2)
    outer_strain = exterior_radial_strain(field, outer_xy, scale)
    directions = np.array([locations[name]["xy_over_a"] for name in
                           ("final_maximum_closure", "final_maximum_slip")])
    with np.load(BASELINE / "radial_kinematics.npz") as radial:
        ray_xy = (directions[:, None, :] * radial["radius_over_a"][None, :, None]).reshape(-1, 2)
        _, ray_strain = inner_kinematics(model, ray_xy, scale)
        np.testing.assert_allclose(ray_strain.reshape(2, -1), radial["radial_strain"][:, :, -1],
                                   rtol=2e-10, atol=2e-13)
    # Check the exterior derivative against the displacement on representative rays.
    probe = (directions[:, None, :] * np.array([3.2, 4., 8.])[None, :, None]).reshape(-1, 2)
    probe_unit = probe / np.linalg.norm(probe, axis=-1, keepdims=True)
    h = 1e-4
    def ur(points):
        uv = np.stack(field.displacements(points[:, 0], points[:, 1]), -1)
        return (uv * probe_unit).sum(-1) * scale["displacement_mm"]
    difference = (ur(probe + h * probe_unit) - ur(probe - h * probe_unit)) / (2 * h * 1000 * scale["tunnel_radius_m"])
    np.testing.assert_allclose(exterior_radial_strain(field, probe, scale), difference,
                               rtol=2e-7, atol=1e-12)
    elastic = CircularElasticIncrement(saved["case"], 1, model.multiplier_scale)
    wall = model.multiplier_scale.new_tensor(directions)
    elastic_wall = -(elastic(wall)[0].numpy() * directions).sum(-1) * scale["displacement_mm"]
    total_wall = -(model(wall)[0][:, -1].numpy() * directions).sum(-1) * scale["displacement_mm"]
    with np.load(OUTPUT.parent / "whole_space_radial.npz") as radial:
        np.testing.assert_allclose(total_wall, radial["convergence_mm"][:, -1], atol=1e-12)
    values = np.concatenate((strain.reshape(len(radii), -1),
                             outer_strain.reshape(len(outer_radii), -1)))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite radial strain in the Figure 7 field.")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, radius_over_a=np.r_[radii, outer_radii],
                        radius_edges_over_a=np.r_[edges, outer_edges[1:]], angle_rad=angles,
                        radial_strain=values, total_wall_mm=total_wall, elastic_wall_mm=elastic_wall,
                        load_step=12, interface_radius_over_a=rg,
                        checkpoint=checkpoint.relative_to(ROOT).as_posix())
    print(f"Wrote {output}: {values.shape}; strain range {values.min()*1e6:.3f} to {values.max()*1e6:.3f} microstrain")
    print("A-B total / elastic / nonelastic (mm):", float(np.diff(total_wall)[0] * -1),
          float(np.diff(elastic_wall)[0] * -1), float(np.diff(total_wall-elastic_wall)[0] * -1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    extract(parser.parse_args().output)
