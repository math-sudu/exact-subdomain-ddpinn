"""Pointwise dual variables for fixed regional physical constraints."""

import torch


def regional_inner(left, right, pts):
    """Use the same regional masses and component mean as regional_power."""
    n0, n1 = pts.n_pde, pts.n_pde + pts.n_wall
    product = (left * right).mean(-1)
    return (0.8 * (product[:n0] * pts.interior_weights).sum()
            + 0.1 * (product[n0:n1] * pts.wall_weights).sum()
            + 0.1 * product[n1:].mean())


class RegionalAugmentation:
    """Dual state for the current increment, ordered like pts.xy_all.

    The caller selects accepted primal endpoints for updates. For L-BFGS,
    the evaluation interval starts again at each update; an inner stationary
    point may update earlier. The trust-region path uses its inner gradient
    reduction instead. Any optimizer history for the old dual is discarded.
    """

    def __init__(self, pts, weight, evaluations_per_update, components):
        if evaluations_per_update <= 0 or int(evaluations_per_update) != evaluations_per_update:
            raise ValueError("evaluations_per_update must be a positive integer")
        self.pts, self.weight = pts, float(weight)
        self.evaluations_per_update = int(evaluations_per_update)
        self.dual = pts.xy_all.new_zeros((len(pts.xy_all), components))
        self.dual_updates = 0
        self.next_update_evaluation = self.evaluations_per_update

    def linear_term(self, residual):
        return regional_inner(self.dual, residual, self.pts)

    @torch.no_grad()
    def update(self, residual, evaluations):
        self.dual.add_(residual, alpha=2 * self.weight)
        self.dual_updates += 1
        self.next_update_evaluation = evaluations + self.evaluations_per_update

    def state_dict(self):
        return dict(dual=self.dual.detach().cpu().clone(),
                    dual_updates=self.dual_updates,
                    next_update_evaluation=self.next_update_evaluation)

    @torch.no_grad()
    def load_state_dict(self, state):
        if state["dual"].shape != self.dual.shape:
            raise ValueError("Dual state has a different fixed sampling shape")
        self.dual.copy_(state["dual"])
        self.dual_updates = state["dual_updates"]
        self.next_update_evaluation = state["next_update_evaluation"]
