"""Solve a complete history with causal feature learning and reduced continuation."""
import argparse
import json
import time

import torch

from run_horseshoe import ROOT, make_work, write_results
from solve_horseshoe_coefficients import solve_increment
from learn_horseshoe_features import FeatureFitError, fit_features
from oespinn.neural_reduction import (hidden_coefficients, vector_features, energy_basis,
                                    ReducedIncrement, solve_reduced, current_residual_enrichment)
from oespinn import config as C


@torch.no_grad()
def enriched_basis(model, work, gauges):
    candidate = vector_features(hidden_coefficients(model, all_layers=True))
    forces = torch.stack([work.load, *gauges.values()], 1)
    lift = torch.cholesky_solve(forces, work.factor)
    return energy_basis(work, torch.cat((candidate, lift), 1))[0]


def select_prefix_width(geometry, work, settings, increments, gauges, tolerance, args, output):
    """Predict the last *directly observed* increment from the earlier prefix."""
    train = increments[:-1]
    prefix = sum(train, torch.zeros_like(train[0]))
    with torch.no_grad():
        plastic = work.replay(train)[-1]['plastic']
    scale = C.SIGMA_REF*geometry.length_scale/C.E_REF*1000
    candidates = []
    for width in (32, 64):
        torch.manual_seed(args.seed)
        candidate_settings = dict(settings, hidden=[width]*3)
        try:
            model, _ = fit_features(geometry, work, candidate_settings, train,
                output=output/f'validation_width{width}', source='direct prefix excluding its last increment')
        except FeatureFitError as error:
            candidates.append(dict(width=width, fit_error=str(error)))
            continue
        with torch.no_grad():
            basis = enriched_basis(model, work, gauges)
            problem = ReducedIncrement(work, basis, prefix, plastic, len(increments)/settings['load_steps'])
            initial = basis.T@work.k@train[-1].flatten()
            coordinate, _, _ = solve_reduced(problem, initial, tolerance=tolerance)
            error = basis@coordinate - increments[-1].flatten()
            errors = {name: float(gauge@error)*scale for name, gauge in gauges.items()}
            candidates.append(dict(width=width, errors_mm=errors,
                                   maximum_mm=max(abs(e) for e in errors.values())))
    fitted = [row for row in candidates if 'maximum_mm' in row]
    if not fitted:
        raise FeatureFitError('Neither candidate width fitted the early training increments.')
    accepted = [row for row in fitted if row['maximum_mm'] <= .15]
    selected = min(accepted, key=lambda row: row['width']) if accepted else min(
        fitted, key=lambda row: row['maximum_mm'])
    return dict(training_steps=len(train), validation_step=len(increments),
                candidates=candidates, selected_width=selected['width'],
                validated_at_015mm=bool(accepted))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--prefix-step', type=int, default=8)
    parser.add_argument('--early-selection', action='store_true',
                        help='Determine the switch and width using only the current direct prefix.')
    parser.add_argument('--current-enrichment', action='store_true',
                        help='Enrich the early-selected space with the current out-of-space equilibrium defect.')
    parser.add_argument('--width', type=int, help='Fixed width for a controlled comparison.')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    settings = json.loads((ROOT/args.config).read_text('utf-8'))
    if args.early_selection and args.width:
        parser.error('--width is for fixed-width runs')
    if args.current_enrichment and not args.early_selection:
        parser.error('--current-enrichment requires --early-selection')
    if args.width:
        settings['hidden'] = [args.width]*3
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    geometry, work = make_work(settings, 'cuda')
    scale = C.SIGMA_REF*geometry.length_scale/C.E_REF*1000
    gauges = work.gauges()
    bound = max(float(torch.linalg.solve_triangular(work.factor, g[:, None], upper=False).norm()*work.load_norm)*scale
                for g in gauges.values())
    tolerance = .15/bound
    stages = dict(setup=time.perf_counter()-started)
    increments, records = [], []
    prefix = work.c.new_zeros((work.n, 2))
    plastic = work.c.new_zeros((len(work.xy), 4))
    predictor = torch.cholesky_solve(work.load[:, None], work.factor).reshape(-1, 2)/settings['load_steps']
    elastic_increment = predictor
    departures = []
    with torch.no_grad():
        last_direct = settings['load_steps'] if args.early_selection else args.prefix_step
        for step in range(1, last_direct+1):
            before = time.perf_counter()
            coefficient, returned, iterations, _ = solve_increment(work, prefix, plastic,
                step/settings['load_steps'], predictor, tolerance=1e-6)
            seconds = time.perf_counter()-before
            increments.append(coefficient)
            prefix += coefficient
            plastic += returned['plastic']
            predictor = coefficient
            records.append(dict(step=step, seconds=seconds, iterations=iterations))
            print(json.dumps(dict(stage='prefix', step=step, seconds=seconds)), flush=True)
            if args.early_selection:
                change = (coefficient-elastic_increment).flatten()
                departures.append(max(abs(float(gauge@change))*scale for gauge in gauges.values()))
                if len(departures) >= 2 and min(departures[-2:]) > .15:
                    break
    prefix_steps = len(increments)
    if prefix_steps >= settings['load_steps']:
        raise RuntimeError('No withheld release remains after the direct prefix.')
    stages['prefix'] = sum(r['seconds'] for r in records)
    torch.save(dict(solver='direct_spectral', settings=settings,
        completed_steps=prefix_steps, coefficients=[c.cpu() for c in increments]), output/'prefix.pt')
    if args.early_selection:
        before = time.perf_counter()
        selection = select_prefix_width(geometry, work, settings, increments, gauges, tolerance, args, output)
        selection['increment_departures_from_elastic_mm'] = departures
        settings['hidden'] = [selection['selected_width']]*3
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        stages['selection'] = time.perf_counter()-before
        (output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n', 'utf-8')
        print(json.dumps(dict(stage='early_selection', **selection)), flush=True)
    before = time.perf_counter()
    model, training = fit_features(geometry, work, settings, increments,
        output=output/'features', source=str((output/'prefix.pt').relative_to(ROOT)).replace('\\', '/'))
    stages['learning'] = time.perf_counter()-before
    method = 'current_tangent_neural_features' if args.current_enrichment else 'causal_neural_features'
    with torch.no_grad():
        before = time.perf_counter()
        basis = enriched_basis(model, work, gauges)
        stages['basis'] = time.perf_counter()-before
        for step in range(prefix_steps+1, settings['load_steps']+1):
            before = time.perf_counter()
            problem = ReducedIncrement(work, basis, prefix, plastic, step/settings['load_steps'])
            initial = basis.T@work.k@increments[-1].flatten()
            coordinate, returned, iterations = solve_reduced(problem, initial, tolerance=tolerance)
            coefficient = (basis@coordinate).reshape(-1, 2)
            update = {}
            if args.current_enrichment:
                del problem
                enriched, defect = current_residual_enrichment(
                    work, basis, coefficient, prefix, plastic, step/settings['load_steps'], tolerance)
                update = dict(initial_full_residual=defect, added_modes=enriched.shape[1]-basis.shape[1])
                if update['added_modes']:
                    basis = enriched
                    problem = ReducedIncrement(work, basis, prefix, plastic, step/settings['load_steps'])
                    initial = torch.cat((coordinate, coordinate.new_zeros(1)))
                    coordinate, returned, corrector = solve_reduced(problem, initial, tolerance=tolerance)
                    update['corrector_iterations'] = corrector
                    coefficient = (basis@coordinate).reshape(-1, 2)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-before
            increments.append(coefficient)
            prefix += coefficient
            plastic += returned['plastic']
            records.append(dict(step=step, seconds=seconds, iterations=iterations, **update))
            torch.save(dict(solver='reduced_spectral', method=method, settings=settings,
                completed_steps=step, prefix_steps=prefix_steps, seed=args.seed,
                coefficients=[c.cpu() for c in increments], trial_basis=basis.cpu(), trajectory=records), output/'model.pt')
            print(json.dumps(dict(stage='reduced', step=step, seconds=seconds,
                                  dimension=basis.shape[1])), flush=True)
        stages['continuation'] = sum(r['seconds'] for r in records[prefix_steps:])
        elapsed = time.perf_counter()-started
        metadata = dict(method=method, seed=args.seed, config=args.config,
            prefix_steps=prefix_steps, width=settings['hidden'][0],
            completed_steps=settings['load_steps'], dimension=basis.shape[1],
            stage_seconds=stages, computation_seconds=sum(stages.values()),
            wall_seconds_to_final_solution=elapsed, device=torch.cuda.get_device_name(),
            timing_scope='From zero history through feature learning and final equilibrium; final output replay excluded.',
            trajectory=records)
        (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n', 'utf-8')
        print(json.dumps({k:v for k,v in metadata.items() if k!='trajectory'}), flush=True)
        write_results(output, work, increments, [(i+1)/settings['load_steps'] for i in range(len(increments))])


if __name__ == '__main__':
    main()
