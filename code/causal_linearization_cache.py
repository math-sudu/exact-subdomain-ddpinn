"""Reusable exact residual JVPs for a single installed numerical linearization."""

from contextlib import contextmanager

import torch


class FixedResidualJVP:
    """Transpose the retained VJP graph while keeping the physical residual fixed."""
    def __init__(self, problem):
        self.problem = problem
        self.theta, self.vec, self.base = problem.theta, problem.vec, problem.base
        self.theta_version = self.theta._version
        self.selection = problem.cache, problem.pts, problem.sf
        self.cotangent = torch.zeros_like(self.base, requires_grad=True)
        self.pullback = torch.autograd.grad(self.vec, self.theta, self.cotangent,
                                           create_graph=True, retain_graph=True)[0]

    def __call__(self, direction):
        p = self.problem
        if (p.theta is not self.theta or p.theta._version != self.theta_version
                or p.vec is not self.vec or p.base is not self.base):
            raise RuntimeError('The installed linearization changed; rebuild the residual JVP cache')
        if any(a is not b for a, b in zip(self.selection, (p.cache, p.pts, p.sf))):
            raise RuntimeError('The selected residual fields changed; use their own linearization')
        return torch.autograd.grad(self.pullback, self.cotangent, direction,
                                   retain_graph=True)[0].detach()


@contextmanager
def cached_residual_jvp(problem):
    """Use a fixed JVP only inside this scope, restoring the original method."""
    absent = object()
    previous = vars(problem).get('jv', absent)
    cache = FixedResidualJVP(problem)
    problem.jv = cache
    try:
        yield cache
    finally:
        if previous is absent:
            del problem.jv
        else:
            problem.jv = previous
