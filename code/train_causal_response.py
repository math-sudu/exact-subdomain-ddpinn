"""Damped Gauss--Newton update for a causal load increment."""

import torch
from causal_linearization_cache import cached_residual_jvp
from numerical import StreamedProjection, projected_step

SETTINGS = dict(initial_krylov=240)


def propose(problem, diagonal, state, check):
    factor = lambda p: diagonal*p
    with cached_residual_jvp(problem):
        projection = StreamedProjection(problem, factor, check)
        projection.seed(SETTINGS['initial_krylov'])
        gram, gradient = projection.quadratic()
        rows, slack = gradient.new_empty((0, len(gradient))), gradient.new_empty(0)
        mu = state['mu']
        while True:
            y = projected_step(gram, gradient, rows, slack, mu, check)
            if float(y.norm()) <= state['radius']:
                break
            mu *= 2
        direction = factor(projection.V@y)
        tangent = problem.jv(direction)
    base = float(problem.base.square().sum())
    alpha, backtracks = 1., 0
    while True:
        check()
        trial = problem.theta.detach()+alpha*direction
        if torch.equal(trial, problem.theta.detach()):
            raise ArithmeticError('The candidate no longer changes the installed parameters')
        with torch.no_grad():
            actual = float(problem.residual(trial).square().sum())
        prediction = float((problem.base+alpha*tangent).square().sum())
        if actual < base and prediction < base:
            return trial, dict(accepted=True, reason='merit_descent',
                mu=mu, radius=state['radius'], coordinate_step_norm=float(alpha*y.norm()),
                prediction_ratio=(base-actual)/(base-prediction),
                actual_decrease=base-actual, predicted_decrease=base-prediction,
                objective=actual, backtracks=backtracks, trial_fraction=alpha,
                krylov_directions=projection.V.shape[1],
                load_step=problem.step, load_steps=12)
        alpha *= .5
        backtracks += 1
