"""Independent integration, enriched weak tests and exterior-yield readouts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from run_horseshoe import ROOT, make_work, write_results, load_coefficients
from oespinn import config as C
from oespinn.plasticity import matrix_yield, joint_yield
from oespinn.horseshoe_geometry import mapped_basis
from sf_core.exterior_field import ExteriorField


def extend(coefficients, j0, m0, j1, m1):
    raw = torch.cat((coefficients.new_zeros((1, 2)), coefficients)).reshape(j0+1, 2*m0+1, 2)
    target = coefficients.new_zeros((j1+1, 2*m1+1, 2))
    target[:j0+1, :m0+1] = raw[:, :m0+1]
    target[:j0+1, m1+1:m1+m0+1] = raw[:, m0+1:]
    return target.flatten(0, 1)[1:]


def exterior_rows(work, increments):
    mat, geo = work.material, work.geometry
    mapping = np.array([[mat.p1, mat.p2], [mat.q1, mat.q2]])
    angles = np.arange(768)*2*np.pi/768
    radii = np.array([1., 1.01, 1.05, 1.2, 1.5, 2., 4., 8.])*geo.radius*geo.length_scale
    x, y = (radii[:, None]*np.cos(angles)).ravel(), (radii[:, None]*np.sin(angles)).ravel()
    total = torch.zeros_like(increments[0])
    rows = []
    for step, coefficient in enumerate(increments, 1):
        total += coefficient
        u = (work.outer_basis[0]@total).cpu().numpy()*C.SIGMA_REF*geo.length_scale/C.E_REF
        spectrum = np.fft.rfft(u, axis=0)/len(u)
        potentials = np.linalg.solve(mapping, spectrum[1:work.m+1].conj().T)
        field = ExteriorField(mat, geo.radius*geo.length_scale, tuple(potentials))
        sx, sy, txy = field.stresses(x, y)
        zz = mat.plane_strain_sigma_zz(1., 0., 0.)*sx+mat.plane_strain_sigma_zz(0., 1., 0.)*sy+mat.plane_strain_sigma_zz(0., 0., 1.)*txy
        stress = work.tensor(np.stack((sx, sy, zz, txy), -1)/C.SIGMA_REF)+work.s0
        joint = joint_yield(stress, work.params).reshape(len(radii), -1)*C.SIGMA_REF/1e6
        matrix = matrix_yield(stress, work.params).reshape(len(radii), -1)*C.SIGMA_REF/1e6
        if work.return_law == "matrix_joint_tension":
            from oespinn.multi_surface_return import surfaces
            tension = surfaces(stress, work.params, work.tension_mpa/work.params.stress_scale_mpa)[0][:, 2]
            tension = tension.reshape(len(radii), -1)*C.SIGMA_REF/1e6
        for i, radius in enumerate(radii):
            row = dict(step=step, radius_m=float(radius),
                       maximum_joint_yield_mpa=float(joint[i].max()),
                       maximum_matrix_yield_mpa=float(matrix[i].max()))
            if work.return_law == "matrix_joint_tension":
                row["maximum_tensile_yield_mpa"] = float(tension[i].max())
            rows.append(row)
    return rows


@torch.no_grad()
def source_rows(work, increments, fields, *, influence_vectors=None):
    angle = torch.atan2(work.xy[:, 1], work.xy[:, 0])
    rad = work.xy.norm(dim=-1)
    rwall = work.geometry.wall(angle)[0]
    depth = (rad-rwall)*work.geometry.length_scale
    masks = {"all": torch.ones_like(depth, dtype=torch.bool),
             "crown_sector": work.xy[:, 1] > work.xy[:, 0].abs(),
             "invert_sector": -work.xy[:, 1] > work.xy[:, 0].abs(),
             "left_sector": -work.xy[:, 0] >= work.xy[:, 1].abs(),
             "right_sector": work.xy[:, 0] >= work.xy[:, 1].abs(),
             "depth_0_2m": depth < 2, "depth_2_5m": (depth >= 2) & (depth < 5),
             "depth_over_5m": depth >= 5}
    scale = C.SIGMA_REF*work.geometry.length_scale/C.E_REF*1000
    previous = torch.zeros_like(fields[0]["plastic"])
    rows = []
    if influence_vectors is None:
        influence_vectors = {name: torch.cholesky_solve(gauge[:, None], work.factor).reshape(-1, 2)
                             for name, gauge in work.gauges().items()}
    virtual_stresses = {}
    for name, adjoint in influence_vectors.items():
        virtual_stresses[name] = work.strain(adjoint, (work.value, work.dx, work.dy))@work.c.T
    stresses = torch.stack(list(virtual_stresses.values()))
    region_weights = torch.stack(list(masks.values())).to(work.w.dtype)
    for step, field in enumerate(fields, 1):
        increment = field["plastic"]-previous
        mechanisms = {"total": increment}
        if work.return_law == "matrix_joint_tension":
            from oespinn.multi_surface_return import surfaces
            _, flow, _, _ = surfaces(field["stress"], work.params, work.tension_mpa/work.params.stress_scale_mpa)
            for i, name in enumerate(("matrix", "joint", "tension")):
                mechanisms[name] = field["gamma"][:, i, None]*flow[:, i]
        density = torch.einsum('gnc,mnc->mgn', stresses, torch.stack(list(mechanisms.values())))
        density = density*work.w*scale
        values = torch.stack([part@region_weights.T for part in
                              (density, density.clamp_min(0), density.clamp_max(0))], -1).cpu().tolist()
        for mi, mechanism in enumerate(mechanisms):
            for gi, name in enumerate(virtual_stresses):
                for ri, region in enumerate(masks):
                    net, positive, negative = values[mi][gi][ri]
                    rows.append(dict(step=step, gauge=name, mechanism=mechanism, region=region,
                                     net_mm=net, positive_mm=positive, negative_mm=negative))
        previous = field["plastic"]
    return rows


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def angular_readouts(work, increments):
    angular_mode = torch.cat((torch.arange(work.m+1, device=work.device),
                              torch.arange(1, work.m+1, device=work.device)))
    mode = angular_mode.repeat(work.j+1)[1:]
    odd = (mode % 2 == 1)[:, None]
    total = torch.zeros_like(increments[0])
    scale = C.SIGMA_REF*work.geometry.length_scale/C.E_REF*1000
    rows = []
    for step, increment in enumerate(increments, 1):
        total += increment
        for name, gauge in work.gauges().items():
            rows.append(dict(step=step, gauge=name,
                odd_modes_mm=float(gauge@(total*odd).flatten())*scale,
                even_and_mean_modes_mm=float(gauge@(total*~odd).flatten())*scale))
    return rows


@torch.no_grad()
def kinematics_and_return(work, increments, fields):
    rows = []
    total = torch.zeros_like(increments[0])
    geo = work.geometry
    pairs = {"sidewall_shortening": [[-geo.length_scale, 0], [geo.length_scale, 0]],
             "vertical_shortening": [[0, geo.boundary["bounds_min"][1]], [0, geo.boundary["bounds_max"][1]]]}
    pairs.update({name: geo.horizontal_chord(depth) for name, depth in
                  (("upper_chord", 2.5), ("middle_chord", 6.), ("lower_chord", 9.5))})
    pair_points = {name: work.tensor(points) for name, points in pairs.items()}
    pair_basis = {name: mapped_basis(geo, points/geo.length_scale, work.j, work.m)[0]
                  for name, points in pair_points.items()}
    for step, (coefficient, field) in enumerate(zip(increments, fields), 1):
        total += coefficient
        gradient = torch.stack((work.dx@total, work.dy@total), -1)*C.SIGMA_REF/C.E_REF
        linear = (gradient+gradient.transpose(-1, -2))/2
        quadratic = gradient.transpose(-1, -2)@gradient/2
        linear_energy = (linear.square().sum((-1, -2))*work.w).sum()
        quadratic_energy = (quadratic.square().sum((-1, -2))*work.w).sum()
        row = dict(step=step, maximum_displacement_gradient=float(gradient.norm(dim=(-1, -2)).max()),
                   maximum_quadratic_strain=float(quadratic.norm(dim=(-1, -2)).max()),
                   quadratic_to_linear_l2=float((quadratic_energy/linear_energy).sqrt()))
        active = field["plastic"].norm(dim=-1)*C.SIGMA_REF/C.E_REF > 1e-7
        row["plastic_extent_threshold_strain"] = 1e-7
        row["maximum_plastic_radius_m"] = float(work.xy[active].norm(dim=-1).max())*geo.length_scale if bool(active.any()) else 0.
        for name, points in pair_points.items():
            displacement = pair_basis[name]@total*C.SIGMA_REF*geo.length_scale/C.E_REF
            initial = points[1]-points[0]
            change = displacement[1]-displacement[0]
            linear_closure = -(initial/initial.norm())@change
            exact_closure = initial.norm()-(initial+change).norm()
            row[name+"_span_linearization_mm"] = float(linear_closure-exact_closure)*1000
        if work.return_law == "matrix_joint_tension":
            from oespinn.multi_surface_return import surfaces
            f, flow, _, _ = surfaces(field["stress"], work.params, work.tension_mpa/work.params.stress_scale_mpa)
            row["maximum_yield_violation_mpa"] = float(f.clamp_min(0).max())*C.SIGMA_REF/1e6
            row["minimum_multiplier"] = float(field["gamma"].min())
            row["maximum_complementarity"] = float((f*field["gamma"]).abs().max())
            for i, mechanism in enumerate(("matrix", "joint", "tension")):
                row[mechanism+"_active_area_m2"] = float(((field["gamma"][:, i]>1e-7)*work.w).sum())*work.geometry.length_scale**2
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--nt", type=int, default=512)
    parser.add_argument("--test-j", type=int)
    parser.add_argument("--test-m", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--include-pending", action="store_true")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    saved = torch.load(ROOT/args.checkpoint, weights_only=False, map_location=args.device)
    settings = dict(saved["settings"])
    j0, m0 = settings["radial_order"], settings["angular_modes"]
    settings["radial_order"] = args.test_j or j0
    settings["angular_modes"] = args.test_m or m0
    _, work = make_work(settings, args.device, quadrature=dict(radial=args.nr, angular=args.nt, boundary=1024, shift=.5))
    count = len(saved["coefficients"]) if args.include_pending else saved["completed_steps"]
    retained, projection_difference = load_coefficients(saved, work.geometry, args.device, count=count)
    coefficients = [extend(c.to(args.device), j0, m0, settings["radial_order"], settings["angular_modes"]) for c in retained]
    coefficients = [c/args.substeps for c in coefficients for _ in range(args.substeps)]
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        fields = {name: work.replay(coefficients, basis) for name, basis in
                  (("volume", None), ("wall", work.wall_basis), ("outer", work.outer_basis))}
        rows = write_results(output, work, coefficients, [(i+1)/(settings["load_steps"]*args.substeps) for i in range(len(coefficients))], evaluated=fields)
        write_csv(output/"exterior.csv", exterior_rows(work, coefficients))
        write_csv(output/"sources.csv", source_rows(work, coefficients, fields["volume"]))
        write_csv(output/"kinematics_return.csv", kinematics_and_return(work, coefficients, fields["volume"]))
        write_csv(output/"angular_modes.csv", angular_readouts(work, coefficients))
    (output/"assessment.json").write_text(json.dumps(dict(checkpoint=args.checkpoint,
        saved_completed_steps=saved["completed_steps"], assessed_steps=len(coefficients),
        maximum_neural_projection_difference=projection_difference,
        history_substeps=args.substeps, subdivision_role="linear subdivision of saved increments, followed by local replay and equilibrium assessment",
        radial_quadrature=args.nr, angular_quadrature=args.nt, shift=.5,
        test_radial_order=settings["radial_order"], test_angular_modes=settings["angular_modes"],
        final=rows[-1]), indent=2)+"\n", "utf-8")
    print(json.dumps(rows[-1]))


if __name__ == "__main__":
    main()
