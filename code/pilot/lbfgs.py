"""Shared L-BFGS helpers for enforcing closure-evaluation budgets."""

from __future__ import annotations

from collections.abc import Callable

import torch


# PyTorch 2.2.2's strong-Wolfe search performs at most 26 closure calls in
# one line search. ``LBFGS.step`` checks max_eval only after that search, so a
# step can overshoot max_eval by 25 calls. Reserve that overshoot before the
# final chunk to stay within the configured evaluation budget.
_STRONG_WOLFE_MAX_OVERSHOOT = 25
_MIN_SAFE_STEP_EVALS = 1 + 26


def budgeted_lbfgs_step(
    optimizer: torch.optim.LBFGS,
    closure: Callable[[], torch.Tensor],
    *,
    remaining_evals: int,
    chunk_max_eval: int,
) -> torch.Tensor | None:
    """Take one complete strong-Wolfe step without exceeding a hard budget.

    ``None`` means that the remaining allowance is too small to guarantee a
    complete line search. The caller then keeps the last fully accepted model
    state, which is valid because the contract specifies an upper bound.
    """

    if remaining_evals < _MIN_SAFE_STEP_EVALS:
        return None
    optimizer.param_groups[0]["max_eval"] = min(
        chunk_max_eval,
        remaining_evals - _STRONG_WOLFE_MAX_OVERSHOOT,
    )
    return optimizer.step(closure)
