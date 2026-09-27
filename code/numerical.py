"""Streamed Krylov projection."""

import torch


def reorthogonalize(x, basis):
    if not basis.shape[1]:
        return x, x.new_zeros(0)
    a = basis.T@x
    x = x-basis@a
    b = basis.T@x
    return x-basis@b, a+b


def require_finite(*values):
    if any(not bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError('Nonfinite residual, parameter or constraint quantity')


class StreamedProjection:
    """Store V and B J^T J B V, never a residual-by-direction matrix.

    A projection belongs to one fixed parameter linearization. Appended
    constraint directions reuse all previous normal images.
    """
    def __init__(self, problem, factor, check=lambda: None):
        self.problem, self.factor, self.check = problem, factor, check
        self.gradient = factor(problem.jt(problem.base))
        self.V = problem.theta.new_zeros((problem.theta.numel(), 0))
        self.H = self.V.clone()
        self.rows_seen = 0

    def append(self, raw):
        self.check()
        vector, _ = reorthogonalize(raw, self.V)
        norm = vector.norm()
        if norm <= torch.finfo(vector.dtype).eps*max(self.V.shape)*raw.norm():
            return False
        vector = vector/norm
        image = self.problem.jv(self.factor(vector))
        self.check()
        normal = self.factor(self.problem.jt(image))
        require_finite(normal)
        self.V = torch.cat((self.V, vector[:, None]), dim=1)
        self.H = torch.cat((self.H, normal[:, None]), dim=1)
        return True

    def seed(self, directions):
        for _ in range(min(directions, self.problem.theta.numel())):
            raw = -self.gradient if not self.V.shape[1] else self.H[:, -1]
            if not self.append(raw):
                break

    def add_rows(self, rows):
        for row in rows[self.rows_seen:]:
            self.append(row)
        self.rows_seen = len(rows)

    def quadratic(self):
        gram = self.V.T@self.H
        return (gram+gram.T)/2, self.V.T@self.gradient


def projected_step(gram, gradient, rows, slack, mu, check):
    """Unconstrained damped Gauss--Newton step in projected coordinates."""
    check()
    if len(rows):
        raise ValueError("This solver uses an unconstrained projected step")
    matrix = gram + mu * torch.eye(len(gradient), device=gram.device, dtype=gram.dtype)
    chol = torch.linalg.cholesky(matrix)
    return torch.cholesky_solve(-gradient[:, None], chol)[:, 0]
