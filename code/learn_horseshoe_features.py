"""Fit spatial neural features to directly solved early increments."""

import json
import time
import torch
from oespinn.horseshoe import new_increment
from oespinn.neural_reduction import hidden_coefficients
def fit_features(geometry, work, settings, increments, *, output, source):
    model = new_increment(None, geometry, settings, work.device)
    target = torch.stack(increments).flatten(1).T
    count = target.shape[1]
    model.kinematic_head = torch.nn.Linear(sum(settings['hidden']), 2*count,
                                          bias=False, device=work.device)
    torch.nn.init.zeros_(model.kinematic_head.weight)
    norms = torch.linalg.vector_norm(work.factor.T@target, dim=0)
    target = target/norms
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, max_eval=125,
        tolerance_grad=1e-10, tolerance_change=1e-13, line_search_fn='strong_wolfe', history_size=50)
    before = time.perf_counter()
    evaluations, history = 0, []
    def errors():
        features = hidden_coefficients(model, all_layers=True)
        predicted = features@model.kinematic_head.weight.T
        predicted = predicted.reshape(work.n, count, 2).permute(0, 2, 1).reshape(2*work.n, count)
        return work.factor.T@(predicted-target)
    def closure():
        nonlocal evaluations
        optimizer.zero_grad(set_to_none=True)
        loss = errors().square().sum()/count/2
        loss.backward()
        evaluations += 1
        return loss
    output.mkdir(parents=True, exist_ok=True)
    for block in range(80):
        optimizer.step(closure)
        with torch.no_grad():
            error = torch.linalg.vector_norm(errors(), dim=0)
        torch.cuda.synchronize()
        record = dict(block=block+1, seconds=time.perf_counter()-before,
                      evaluations=evaluations, relative_energy_errors=error.tolist())
        history.append(record)
        torch.save(dict(source=source, prefix_steps=len(increments), settings=settings,
            state=model.state_dict(), history=history), output/'features.pt')
        (output/'training.json').write_text(json.dumps(history, indent=2)+'\n', 'utf-8')
        print(json.dumps(record), flush=True)
        if float(error.max()) < .03:
            break
    else:
        raise RuntimeError('Feature fit did not attain the specified prefix approximation')
    return model, history
