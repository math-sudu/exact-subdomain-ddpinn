"""Regional integration measures."""

import math


def region_slices(points):
    n0, n1 = points.n_pde, points.n_pde + points.n_wall
    return {"interior": slice(0, n0), "wall": slice(n0, n1),
            "interface": slice(n1, n1 + points.n_gam)}


def regional_mean_square(value, points, weights):
    """Average regional means with declared masses, independent of point counts."""
    regions = region_slices(points)
    if (set(weights) != set(regions)
            or not all(math.isfinite(w) and w >= 0 for w in weights.values())
            or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)):
        raise ValueError("regional masses must be finite, nonnegative and sum to one")
    if value.shape[0] != points.n_pde + points.n_wall + points.n_gam:
        raise ValueError("regional residual does not match the point layout")
    return sum(weights[name] * value[sl].square().mean()
               for name, sl in regions.items())
