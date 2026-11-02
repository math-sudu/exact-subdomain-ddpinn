"""Solve a complete history with causal feature learning and reduced continuation."""
import argparse
import json
import time

import torch

from run_horseshoe import ROOT, make_work, write_results
from solve_horseshoe_coefficients import solve_increment
from learn_horseshoe_features import fit_features
from oespinn.neural_reduction import hidden_coefficients, vector_features, energy_basis, ReducedIncrement, solve_reduced
from oespinn import config as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--prefix-step', type=int, default=8)
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    settings = json.loads((ROOT/args.config).read_text('utf-8'))
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
    with torch.no_grad():
        for step in range(1, args.prefix_step+1):
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
    stages['prefix'] = sum(r['seconds'] for r in records)
    torch.save(dict(solver='direct_spectral', settings=settings,
        completed_steps=args.prefix_step, coefficients=[c.cpu() for c in increments]), output/'prefix.pt')
    before = time.perf_counter()
    model, training = fit_features(geometry, work, settings, increments,
        output=output/'features', source=str((output/'prefix.pt').relative_to(ROOT)).replace('\\', '/'))
    stages['learning'] = time.perf_counter()-before
    with torch.no_grad():
        before = time.perf_counter()
        candidate = vector_features(hidden_coefficients(model, all_layers=True))
        forces = torch.stack([work.load, *gauges.values()], 1)
        lift = torch.cholesky_solve(forces, work.factor)
        basis, _ = energy_basis(work, torch.cat((candidate, lift), 1))
        stages['basis'] = time.perf_counter()-before
        for step in range(args.prefix_step+1, settings['load_steps']+1):
            before = time.perf_counter()
            problem = ReducedIncrement(work, basis, prefix, plastic, step/settings['load_steps'])
            initial = basis.T@work.k@increments[-1].flatten()
            coordinate, returned, iterations = solve_reduced(problem, initial, tolerance=tolerance)
            coefficient = (basis@coordinate).reshape(-1, 2)
            torch.cuda.synchronize()
            seconds = time.perf_counter()-before
            increments.append(coefficient)
            prefix += coefficient
            plastic += returned['plastic']
            records.append(dict(step=step, seconds=seconds, iterations=iterations))
            torch.save(dict(solver='reduced_spectral', method='causal_neural_features', settings=settings,
                completed_steps=step, prefix_steps=args.prefix_step, seed=args.seed,
                coefficients=[c.cpu() for c in increments], trial_basis=basis.cpu(), trajectory=records), output/'model.pt')
            print(json.dumps(dict(stage='reduced', step=step, seconds=seconds)), flush=True)
        stages['continuation'] = sum(r['seconds'] for r in records[args.prefix_step:])
        elapsed = time.perf_counter()-started
        metadata = dict(method='causal_neural_features', seed=args.seed, config=args.config,
            prefix_steps=args.prefix_step, completed_steps=settings['load_steps'], dimension=basis.shape[1],
            stage_seconds=stages, computation_seconds=sum(stages.values()),
            wall_seconds_to_final_solution=elapsed, device=torch.cuda.get_device_name(),
            timing_scope='From zero history through feature learning and final equilibrium; final output replay excluded.',
            trajectory=records)
        (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n', 'utf-8')
        print(json.dumps({k:v for k,v in metadata.items() if k!='trajectory'}), flush=True)
        write_results(output, work, increments, [(i+1)/settings['load_steps'] for i in range(len(increments))])


if __name__ == '__main__':
    main()
