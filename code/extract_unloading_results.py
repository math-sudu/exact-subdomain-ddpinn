"""Export the saved twelve-step fields for unloading figures and tables.

No training or checkpoint writes occur here. Plastic strain follows the saved
multiplier/flow representation. Slip is a continuum shear strain resolved on
the weak plane, not a displacement jump across a discrete interface.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from publication_model import load_model, write_rows
from pilot import config as C, plasticity as P
from pilot.spectral_mixed import load_orientation_model

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/model'
OUTPUT = ROOT / 'results/unloading'


def equivalent_increment(strain):
    """sqrt(2/3 dev(e):dev(e)); input uses engineering xy shear."""
    dev = strain[..., :3] - strain[..., :3].mean(-1, keepdim=True)
    return (2 / 3 * (dev.square().sum(-1) + 0.5 * strain[..., 3].square())).sqrt()


def plane_shear(strain, normal_deg):
    """2 t.e.n with n=(cos(beta),sin(beta)), t=(-sin(beta),cos(beta))."""
    beta = np.deg2rad(normal_deg)
    nx, ny = np.cos(beta), np.sin(beta)
    return 2 * nx * ny * (strain[..., 1] - strain[..., 0]) + (nx*nx - ny*ny)*strain[..., 3]


@torch.no_grad()
def sample_fields(bundle, model, xy):
    parts = []
    for chunk in np.array_split(xy, max(1, int(np.ceil(len(xy) / 2048)))):
        points = torch.as_tensor(chunk, device=model.multiplier_scale.device,
                                 dtype=model.multiplier_scale.dtype)
        u, stress, multiplier = model(points)
        total = stress + bundle['s0'].to(stress)
        scale = C.U_REF / C.A_TUNNEL
        matrix = multiplier[..., :1] * P.matrix_flow_direction(total, bundle['params']) * scale
        joint = multiplier[..., 1:] * P.joint_flow_direction(total, bundle['params']) * scale
        slip_increment = plane_shear(joint, bundle['params'].joint_normal_deg)
        part = dict(
            displacement_mm=u * C.U_REF * 1000,
            plastic_strain=(matrix + joint).cumsum(1),
            accumulated_equivalent=equivalent_increment(matrix + joint).cumsum(1),
            matrix_accumulated_equivalent=equivalent_increment(matrix).cumsum(1),
            joint_accumulated_equivalent=equivalent_increment(joint).cumsum(1),
            signed_slip_strain=slip_increment.cumsum(1),
            accumulated_slip_strain=slip_increment.abs().cumsum(1),
        )
        parts.append({key: value.cpu().numpy() for key, value in part.items()})
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


@torch.no_grad()
def stress_path_rows(bundle, model, locations):
    """Resolve total stress and flow at the same fixed wall coordinates."""
    xy = model.multiplier_scale.new_tensor([p['xy_over_a'] for p in locations.values()])
    u, stress, multiplier = model(xy)
    total = stress + bundle['s0'].to(stress)
    sigma_n, tau, _, _ = P._joint_stresses(total, bundle['params'])
    yield_value = P.joint_yield(total, bundle['params'])
    strain_scale = C.U_REF / C.A_TUNNEL
    joint = multiplier[..., 1:] * P.joint_flow_direction(total, bundle['params']) * strain_scale
    slip = plane_shear(joint, bundle['params'].joint_normal_deg)
    accumulated = slip.abs().cumsum(1)
    closure = -(u * xy[:, None]).sum(-1) * C.U_REF * 1000
    stress_scale = C.SIGMA_REF / 1e6
    s0 = bundle['s0'].to(stress)
    sn0, tau0, _, _ = P._joint_stresses(s0, bundle['params'])
    f0 = P.joint_yield(s0, bundle['params'])
    rows = []
    for i, (name, location) in enumerate(locations.items()):
        for step in range(model.n_steps + 1):
            k = step - 1
            sigma = total[i, k] if step else s0
            rows.append(dict(
                location=name, angle_deg=location['angle_deg'], load_step=step,
                release_fraction=step/model.total_steps,
                **{f'sigma_{component}_MPa': float(sigma[j] * stress_scale)
                   for j, component in enumerate(('xx', 'yy', 'zz', 'xy'))},
                sigma_n_MPa=float((sigma_n[i, k] if step else sn0) * stress_scale),
                tau_j_MPa=float((tau[i, k] if step else tau0) * stress_scale),
                joint_yield_MPa=float((yield_value[i, k] if step else f0) * stress_scale),
                joint_multiplier_increment=float(multiplier[i, k, 1] * strain_scale) if step else 0.,
                signed_slip_increment=float(slip[i, k]) if step else 0.,
                accumulated_slip_strain=float(accumulated[i, k]) if step else 0.,
                closure_mm=float(closure[i, k]) if step else 0.,
            ))
    return rows


def export(run, checkpoint, output, device, stress_paths_only=False, orientation=False):
    torch.set_num_threads(1)
    if orientation:
        bundle, saved, model = load_orientation_model(checkpoint, device)
        if saved["completed_steps"] != saved["settings"]["load_steps"]:
            raise ValueError("Finish the orientation history before extracting its figures.")
    else:
        bundle, saved, model = load_model(run, checkpoint, device)
    if model.n_steps != 12:
        raise ValueError('The unloading extraction requires all twelve saved increments.')
    output.mkdir(parents=True, exist_ok=True)
    if stress_paths_only:
        metadata = json.loads((output/'metadata.json').read_text(encoding='utf-8'))
        if ((ROOT/metadata['source_run']).resolve() != run
                or (ROOT/metadata['checkpoint']).resolve() != checkpoint
                or ('algorithm_state' in saved and saved['algorithm_state']['event'] != metadata['event'])):
            raise ValueError('The stress paths must use the checkpoint of the existing field export.')
        write_rows(output/'fixed_location_stress_paths.csv',
                   stress_path_rows(bundle, model, metadata['fixed_locations']))
        return
    radius_gamma = bundle['spec']['r_gamma']
    # More radial samples near the wall; angles cover the full circumference.
    radius_edges = 1 + (radius_gamma - 1) * np.linspace(0, 1, 129)**2
    radius = np.sqrt((radius_edges[:-1]**2 + radius_edges[1:]**2) / 2)
    angle = np.arange(720) * 2 * np.pi / 720
    unit = np.stack((np.cos(angle), np.sin(angle)), -1)
    xy = (radius[:, None, None] * unit[None]).reshape(-1, 2)
    fields = sample_fields(bundle, model, xy)
    fields = {key: value.reshape(len(radius), len(angle), *value.shape[1:])
              for key, value in fields.items()}
    wall = sample_fields(bundle, model, unit)
    tangent = np.stack((-unit[:, 1], unit[:, 0]), -1)
    closure = -(wall['displacement_mm'] * unit[:, None]).sum(-1)
    tangential = (wall['displacement_mm'] * tangent[:, None]).sum(-1)
    area = np.pi * np.diff(radius_edges**2)[:, None] / len(angle) * C.A_TUNNEL**2
    # This is a named plotting contour, not a yield/acceptance criterion.
    footprint_level = 10e-6
    rows = []
    for k in range(12):
        ic = int(closure[:, k].argmax())
        islip = int(wall['accumulated_slip_strain'][:, k].argmax())
        footprint = fields['accumulated_equivalent'][:, :, k] >= footprint_level
        rows.append(dict(
            load_step=k+1, release_fraction=(k+1)/12,
            maximum_wall_closure_mm=float(closure[ic, k]),
            maximum_closure_angle_deg=ic*0.5,
            mean_wall_closure_mm=float(closure[:, k].mean()),
            maximum_wall_slip_strain=float(wall['accumulated_slip_strain'][islip, k]),
            maximum_slip_angle_deg=islip*0.5,
            maximum_accumulated_equivalent=float(max(
                fields['accumulated_equivalent'][:, :, k].max(),
                wall['accumulated_equivalent'][:, k].max())),
            matrix_maximum_accumulated_equivalent=float(max(
                fields['matrix_accumulated_equivalent'][:, :, k].max(),
                wall['matrix_accumulated_equivalent'][:, k].max())),
            footprint_10microstrain_area_m2=float((footprint*area).sum()),
            footprint_10microstrain_outer_radius_over_a=float(
                np.broadcast_to(radius[:, None], footprint.shape)[footprint].max()
                if footprint.any() else 0),
        ))
    write_rows(output/'load_summary.csv', rows)
    wall_rows = []
    for k in range(12):
        for j in range(len(angle)):
            wall_rows.append(dict(load_step=k+1, release_fraction=(k+1)/12,
                angle_deg=j*0.5, closure_mm=float(closure[j, k]),
                tangential_mm=float(tangential[j, k]),
                signed_slip_strain=float(wall['signed_slip_strain'][j, k]),
                accumulated_slip_strain=float(wall['accumulated_slip_strain'][j, k]),
                accumulated_equivalent=float(wall['accumulated_equivalent'][j, k])))
    write_rows(output/'wall_profiles.csv', wall_rows)
    location_indices = {'final_maximum_closure': int(closure[:, -1].argmax()),
                        'final_maximum_slip': int(wall['accumulated_slip_strain'][:, -1].argmax())}
    histories = []
    for name, j in location_indices.items():
        for k in range(12):
            histories.append(dict(location=name, **wall_rows[k*len(angle)+j]))
    write_rows(output/'fixed_location_histories.csv', histories)
    np.savez_compressed(output/'fields.npz', radius_over_a=radius,
                        radius_edges_over_a=radius_edges, angle_rad=angle,
                        **{key: value.astype(np.float32) for key, value in fields.items()})
    material = C.material()
    # The saved stiffness is the source; this equality identifies its engineering constants.
    np.testing.assert_allclose(bundle['c'].numpy(), material.plane_strain_stiffness4()/C.E_REF)
    provenance = dict(representation=bundle['spec'].get('representation', 'pointwise'))
    if orientation:
        provenance.update(completed_steps=saved['completed_steps'],
                          projection=bundle['spec'].get('projection'))
    elif 'algorithm_state' in saved:
        provenance.update(accepted=saved['algorithm_state']['accepted'],
                          event=saved['algorithm_state']['event'])
    metadata = dict(
        source_run=run.relative_to(ROOT).as_posix(),
        checkpoint=checkpoint.relative_to(ROOT).as_posix(),
        **provenance,
        load_steps=model.n_steps, radius_gamma_over_a=radius_gamma,
        material=dict(Eh_GPa=material.Eh/1e9, Ev_GPa=material.Ev/1e9,
                      Gvh_GPa=material.Gvh/1e9, nu_vh=material.nu_vh,
                      nu_hh=material.nu_hh, bedding_dip_deg=np.rad2deg(material.phi)),
        physics={key: getattr(bundle['params'], key) for key in bundle['spec']['physics']},
        scales=dict(tunnel_radius_m=C.A_TUNNEL, stress_MPa=C.SIGMA_REF/1e6,
                    modulus_GPa=C.E_REF/1e9, displacement_mm=C.U_REF*1000),
        insitu_stress_MPa=(bundle['s0']*C.SIGMA_REF/1e6).tolist(),
        grid=dict(radial_cells=len(radius), angular_cells=len(angle),
                  radial_rule='quadratic edges; midpoints in squared radius',
                  wall_angle_spacing_deg=0.5, footprint_level=footprint_level),
        storage='fields.npz uses float32 for plotting; all evaluations and CSV summaries use float64',
        definitions=dict(
            accumulated_equivalent='sum_k sqrt(2/3 dev(delta epsilon_p_k):dev(delta epsilon_p_k))',
            plastic_strain='sum_k [gamma_m grad(g_m) + gamma_j grad(g_j)]; engineering xy shear',
            signed_slip_strain='sum_k 2 t.delta epsilon_p_joint_k.n',
            accumulated_slip_strain='sum_k abs(2 t.delta epsilon_p_joint_k.n)',
            footprint='area where accumulated equivalent plastic strain >= 10 microstrain; display contour only',
            units='displacements in mm; strains dimensionless; coordinates divided by tunnel radius',
        ),
        fixed_locations={name: dict(angle_deg=j*0.5, xy_over_a=unit[j].tolist())
                         for name, j in location_indices.items()},
    )
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    write_rows(output/'fixed_location_stress_paths.csv',
               stress_path_rows(bundle, model, metadata['fixed_locations']))
    print(json.dumps(dict(final=rows[-1], locations=metadata['fixed_locations']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=RUN)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--orientation', action='store_true',
                        help='Read a completed run_orientation_extension checkpoint.')
    parser.add_argument('--stress-paths-only', action='store_true',
                        help='Query the existing fixed locations without regenerating the field grid.')
    args = parser.parse_args()
    export(args.run.resolve(), (args.checkpoint or args.run/'model.pt').resolve(),
           args.output.resolve(), args.device, args.stress_paths_only, args.orientation)
