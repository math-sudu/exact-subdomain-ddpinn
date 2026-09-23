"""Construct the causal model and its integration points from numerical settings."""

from types import SimpleNamespace

import torch

from pilot.causal_mixed import CausalMixedPINN
from pilot.causal_sampling import apply_quadrature
from pilot.causal_composite_quadrature import apply_composite_quadrature
from pilot.mixed_plasticity import make_geometry
from pilot.plasticity import WeakPlaneParameters


def make_case(spec, device="cpu"):
    cfg = SimpleNamespace(
        r_gamma=spec["r_gamma"], hidden=tuple(spec["hidden"]),
        load_steps=spec["load_steps"], n_pde=spec["n_pde"],
        n_wall=spec["boundary"], n_gamma=spec["boundary"],
        eh_over_ev=2.625, bedding_dip_deg=45.0,
    )
    params = WeakPlaneParameters(**spec["physics"]).checked()
    points, c, s0, sf = make_geometry(
        cfg, device, torch.float64, spec["training_seed"], shift=spec["training_shift"])
    if "composite_area_quadrature" in spec:
        apply_composite_quadrature(points, cfg.r_gamma, **spec["composite_area_quadrature"])
    elif "interior_quadrature" in spec:
        apply_quadrature(points, cfg.r_gamma, spec["interior_quadrature"])
    else:
        points.interior_weights = c.new_full((cfg.n_pde,), 1 / cfg.n_pde)
    model = CausalMixedPINN(cfg.hidden, spec["input_scale"], cfg.load_steps, c).to(c)
    return model, params, points, c, s0, sf
