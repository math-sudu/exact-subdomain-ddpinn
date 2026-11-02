"""Compare learned, untrained, spectral and prefix-enriched trial spaces."""
import argparse
import json
import time
import torch

from run_horseshoe import ROOT, make_work, write_results
from oespinn.neural_reduction import hidden_coefficients, vector_features, energy_basis, ReducedIncrement, solve_reduced
from oespinn.horseshoe import new_increment
from oespinn import config as C


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['neural', 'snapshots', 'random', 'spectral', 'mixed'], required=True)
    parser.add_argument('--prefix-step', type=int, default=8)
    parser.add_argument('--through-step', type=int, default=12)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source', default='results/horseshoe/direct_refined/model.pt')
    parser.add_argument('--features')
    args = parser.parse_args()
    if args.method=='neural' and not args.features:
        parser.error('--method neural requires --features')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    torch.manual_seed(12)
    saved = torch.load(ROOT/args.source, map_location='cuda', weights_only=False)
    settings = saved['settings']
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    before = time.perf_counter()
    geometry, work = make_work(settings, 'cuda')
    increments = saved['coefficients'][:args.prefix_step]
    prefix = sum(increments, torch.zeros_like(increments[0]))
    plastic = work.replay(increments)[-1]['plastic']
    common_setup = time.perf_counter()-before
    before = time.perf_counter()
    if args.method in ('neural', 'random'):
        model = new_increment(None, geometry, settings, 'cuda')
        if args.method=='neural':
            learned = torch.load(ROOT/args.features, map_location='cuda', weights_only=False)
            if learned['source'] != args.source or learned['prefix_steps'] > args.prefix_step:
                raise ValueError('Learned features do not belong to the available prefix')
            model.kinematic_trunk.load_state_dict({k.removeprefix('kinematic_trunk.'):v
                for k,v in learned['state'].items() if k.startswith('kinematic_trunk.')})
        candidate = vector_features(hidden_coefficients(model, all_layers=True))
    elif args.method=='snapshots':
        candidate = torch.stack(increments).flatten(1).T
    else:
        indices = torch.arange((work.j+1)*(2*work.m+1), device='cuda').reshape(work.j+1, -1)
        columns = torch.cat((torch.arange(13, device='cuda'), torch.arange(work.m+1, work.m+13, device='cuda')))
        scalar_indices = indices[:8, columns].flatten()[1:]-1
        candidate = work.c.new_zeros((2*work.n, 2*len(scalar_indices)))
        candidate[2*scalar_indices, torch.arange(0, candidate.shape[1], 2, device='cuda')] = 1
        candidate[2*scalar_indices+1, torch.arange(1, candidate.shape[1], 2, device='cuda')] = 1
        if args.method == 'mixed':
            candidate = torch.cat((candidate, torch.stack(increments).flatten(1).T), 1)
    forces = torch.stack([work.load, *work.gauges().values()], 1)
    lift = torch.cholesky_solve(forces, work.factor)
    basis, _ = energy_basis(work, torch.cat((candidate, lift), 1))
    basis_seconds = time.perf_counter()-before
    records = []
    scale = C.SIGMA_REF*geometry.length_scale/C.E_REF*1000
    gauges = work.gauges()
    bound = max(float(torch.linalg.solve_triangular(work.factor, g[:, None], upper=False).norm()*work.load_norm)*scale
                for g in gauges.values())
    for step in range(args.prefix_step+1, args.through_step+1):
        before = time.perf_counter()
        problem = ReducedIncrement(work, basis, prefix, plastic, step/settings['load_steps'])
        initial = basis.T@work.k@increments[-1].flatten()
        coordinate, returned, iterations = solve_reduced(problem, initial, tolerance=.15/bound)
        coefficient = (basis@coordinate).reshape(-1, 2)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-before
        prefix += coefficient
        plastic += returned['plastic']
        increments.append(coefficient)
        record = dict(step=step, seconds=seconds, iterations=iterations,
            final_readouts_mm={name:float(g@prefix.flatten())*scale for name,g in gauges.items()})
        records.append(record)
        torch.save(dict(solver='reduced_spectral', method=args.method, settings=settings,
            coefficients=[c.cpu() for c in increments], completed_steps=step,
            prefix_source=args.source, feature_source=args.features, prefix_steps=args.prefix_step,
            trial_basis=basis.cpu(), trajectory=records), output/'model.pt')
        print(json.dumps({k:v for k,v in record.items() if k!='iterations'}), flush=True)
    metadata = dict(method=args.method, source=args.source, feature_source=args.features,
        solve_space='reduced', linear_solver='dense_reduced',
        prefix_steps=args.prefix_step, completed_steps=args.through_step, dimension=basis.shape[1],
        common_setup_seconds=common_setup, basis_seconds=basis_seconds,
        solve_seconds=sum(r['seconds'] for r in records), trajectory=records)
    (output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n', 'utf-8')
    write_results(output, work, increments, [(i+1)/settings['load_steps'] for i in range(len(increments))])


if __name__ == '__main__':
    main()
