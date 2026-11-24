"""Compare two constitutive integrations of the same near-wall strain path."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import torch

from oespinn import config as C
from oespinn.horseshoe import HorseshoeWork
from oespinn.horseshoe_geometry import HorseshoeGeometry, mapped_basis
from oespinn.multi_surface_return import multi_surface_return, surfaces
from oespinn.plasticity import WeakPlaneParameters
from reconstruct_shared_history import load_history
from run_horseshoe import ROOT
from sf_core.material import TIMaterial


def in_bedding_axes(vector, axes, *, strain=False):
    xx, yy, zz, xy = vector
    shear = xy / 2 if strain else xy
    tensor = axes.T @ np.array([[xx, shear], [shear, yy]]) @ axes
    return np.array([tensor[0, 0], tensor[1, 1], zz,
                     tensor[0, 1] * (2 if strain else 1)])


class BeddingReturn:
    """Zero-dilation update in principal material axes, with stresses in MPa."""

    def __init__(self, elastic, physics, tension):
        if physics["matrix_psi_deg"] != 0 or physics["joint_psi_deg"] != 0:
            raise ValueError("The principal-axis reduction requires zero dilation.")
        eh, ev = 1000 * elastic["Eh_gpa"], 1000 * elastic["Ev_gpa"]
        vh, hh = elastic["nu_vh"] / ev, elastic["nu_hh"] / eh
        self.compliance = np.array([[1 / eh, -vh, -hh], [-vh, 1 / ev, -vh],
                                    [-hh, -vh, 1 / eh]])
        self.normal = np.linalg.inv(self.compliance)
        self.shear = 1000 * elastic["Gvh_gpa"]
        self.elastic = np.zeros((4, 4))
        self.elastic[:3, :3], self.elastic[3, 3] = self.normal, self.shear
        self.deviator = np.eye(3) - np.ones((3, 3)) / 3
        tangent = np.tan(np.deg2rad(physics["matrix_phi_deg"]))
        denominator = np.sqrt(9 + 12 * tangent**2)
        self.alpha = tangent / denominator
        self.strength = 3 * physics["matrix_cohesion_mpa"] / denominator
        self.matrix_rounding = physics["matrix_smoothing_mpa"]
        self.slip_rounding = physics["joint_smoothing_mpa"]
        self.friction = np.tan(np.deg2rad(physics["joint_phi_deg"]))
        self.cohesion = physics["joint_cohesion_mpa"]
        self.tension = tension

    def state(self, trial, a):
        # a = matrix multiplier / (2 sqrt(J2 + rounding^2)).
        system = np.eye(3) + a * self.normal @ self.deviator
        normal = np.linalg.solve(system, trial[:3])
        opening = 0.
        if normal[1] > self.tension:
            direction = np.linalg.solve(system, self.normal[:, 1])
            opening = (normal[1] - self.tension) / direction[1]
            normal -= opening * direction

        # Bedding shear decouples elastically from all three normal stresses.
        capacity = self.cohesion - self.friction * normal[1]
        shear_limit = np.sqrt(capacity * (capacity + 2 * self.slip_rounding))
        shear = np.clip(trial[3] / (1 + 2 * self.shear * a), -shear_limit, shear_limit)
        stress = np.r_[normal, shear]
        dev = self.deviator @ normal
        radius = np.sqrt(.5 * (dev @ dev) + shear**2 + self.matrix_rounding**2)
        matrix_yield = radius - self.matrix_rounding + self.alpha * normal.sum() - self.strength
        return stress, radius, matrix_yield, opening

    def update(self, trial):
        a = 0.
        if self.state(trial, a)[2] > 1e-12:
            upper = 1 / self.shear
            while self.state(trial, upper)[2] > 0:
                upper *= 2
                if upper > 1e6:
                    raise ValueError("No admissible matrix return on the prescribed path.")
            a = brentq(lambda factor: self.state(trial, factor)[2], 0, upper, xtol=1e-15)
        stress, radius, _, opening = self.state(trial, a)
        shear = stress[3]
        plastic = np.zeros((3, 4))
        plastic[0, :3] = a * (self.deviator @ stress[:3])
        plastic[0, 3] = 2 * a * shear
        plastic[1, 3] = (trial[3] - shear) / self.shear - plastic[0, 3]
        plastic[2, 1] = opening
        slip = plastic[1, 3] * np.hypot(shear, self.slip_rounding) / shear if shear else 0.
        return stress, np.array([2 * a * radius, slip, opening]), plastic


def point_history(settings, coefficients, angle_deg, depth_m):
    geometry = HorseshoeGeometry(ROOT / settings["geometry"], settings["length_scale_m"],
                                 settings["interface_radius"])
    theta = torch.tensor([np.deg2rad(angle_deg)], dtype=torch.float64)
    radius = geometry.wall(theta)[0] + depth_m / geometry.length_scale
    points = torch.stack((radius * theta.cos(), radius * theta.sin()), dim=1)
    basis = mapped_basis(geometry, points, settings["radial_order"], settings["angular_modes"])
    return np.array([HorseshoeWork.strain(c, basis)[0].numpy() for c in coefficients])


@torch.no_grad()
def compare(settings, increments):
    elastic, physics = settings["elastic_constants"], settings["physics"]
    dip = np.deg2rad(elastic["bedding_dip_deg"])
    if not np.isclose(np.sin(np.deg2rad(physics["joint_normal_deg"]) - dip), 1):
        raise ValueError("The weak-plane normal must coincide with the bedding normal.")
    axes = np.array([[np.cos(dip), -np.sin(dip)], [np.sin(dip), np.cos(dip)]])
    reference = BeddingReturn(elastic, physics, settings["tension_mpa"])
    parameters = WeakPlaneParameters(**physics)
    material = TIMaterial("shaft", elastic["Eh_gpa"] * 1e9, elastic["Ev_gpa"] * 1e9,
                          elastic["nu_vh"], elastic["nu_hh"], elastic["Gvh_gpa"] * 1e9, dip)
    stiffness = torch.tensor(material.plane_strain_stiffness4() / C.E_REF, dtype=torch.float64)
    sx, sy, txy = settings["insitu_mpa"]
    stress_unit, strain_unit = C.SIGMA_REF / 1e6, C.SIGMA_REF / C.E_REF
    initial = np.array([sx, sy, material.plane_strain_sigma_zz(sx, sy, txy), txy])
    initial_local = in_bedding_axes([sx, sy, 0, txy], axes)
    initial_local[2] = -(reference.compliance[2, :2] @ initial_local[:2]) / reference.compliance[2, 2]
    rows = []
    for subdivisions in (1, 2):
        global_stress = torch.tensor(initial / stress_unit)[None]
        local_stress = initial_local.copy()
        for i, increment in enumerate(np.repeat(increments / subdivisions, subdivisions, axis=0), 1):
            trial = global_stress + torch.tensor(increment) @ stiffness.T
            result = multi_surface_return(trial, stiffness, parameters, settings["tension_mpa"])
            global_stress = result["stress"]
            flow = surfaces(global_stress, parameters, settings["tension_mpa"] / stress_unit)[1][0]
            gamma = result["gamma"][0].numpy() * strain_unit
            parts = [in_bedding_axes(f * g, axes, strain=True) for f, g in zip(flow.numpy(), gamma)]
            local_trial = local_stress + reference.elastic @ in_bedding_axes(increment * strain_unit, axes, strain=True)
            local_stress, local_gamma, local_parts = reference.update(local_trial)
            original = in_bedding_axes(global_stress[0].numpy() * stress_unit, axes)
            for method, stress, multipliers, plastic in (
                ("coupled", original, gamma, parts),
                ("bedding_axes", local_stress, local_gamma, local_parts),
            ):
                row = dict(subdivisions=subdivisions, step=i, release=i / (subdivisions * len(increments)), method=method)
                row.update(zip(("sigma_hh_mpa", "sigma_vv_mpa", "sigma_zz_mpa", "sigma_hv_mpa"), stress))
                for name, multiplier, part in zip(("matrix", "slip", "opening"), multipliers, plastic):
                    row[name + "_multiplier"] = multiplier
                    for component, value in zip(("hh", "vv", "zz", "gamma_hv"), part):
                        row[name + "_" + component] = value
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Use a native history in place of the shared full_network case.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        settings, coefficients = saved["settings"], saved["coefficients"][:saved["completed_steps"]]
    else:
        settings, coefficients = load_history("full_network")
    increments = point_history(settings, coefficients, angle_deg=135, depth_m=1.)
    rows = compare(settings, increments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
