"""Exact local return for the rounded, zero-dilation weak-plane branch.

This eliminates the local stress/multiplier equations at a supplied strain.
The caller must check the matrix yield and solve the coupled boundary problem.
"""

import numpy as np
import torch

from . import plasticity as P


def joint_return(trial_stress, stiffness, params, *, check_branch=True):
    """Return stress, plastic increment, multiplier and consistent tangent.

    With zero dilation the plastic strain is kappa*q, where q is the fixed
    engineering-shear direction. Put k=q.C.q, b=p.C.q and a=tan(phi)*b/k.
    Eliminating kappa gives sqrt(tau**2+eps**2)+a*tau=h. The root on the
    trial's shear branch gives kappa=(tau_trial-tau)/k and gamma=kappa/alpha.
    All spatial/parameter derivatives pass through this piecewise exact map.
    """
    if params.joint_psi_deg != 0:
        raise ValueError("This return map requires zero weak-plane dilation")
    normal, shear = P._joint_geometry(params, trial_stress)
    cq = stiffness@shear
    k = shear@cq
    friction = float(np.tan(np.deg2rad(params.joint_phi_deg)))
    a = friction*(normal@cq)/k
    if abs(float(a)) >= 1:
        raise ValueError("The current shear-return branch requires abs(a) < 1")
    epsilon = params.joint_smoothing
    tau_trial, normal_trial = trial_stress@shear, trial_stress@normal
    active = P.joint_yield(trial_stress, params) > 0
    h = params.joint_cohesion+epsilon-friction*normal_trial+a*tau_trial
    # The inactive branch does not evaluate an unrelated yield-surface root.
    h = torch.where(active, h, torch.ones_like(h)*(1+epsilon))
    discriminant = h.square()-(1-a.square())*epsilon**2
    # Spatial vmap/JVP callers validate the untransformed field separately.
    if check_branch and torch.any(active & (h <= epsilon)):
        raise ValueError("Shear return reached the rounded apex; another branch is required")
    sign = torch.where(tau_trial < 0, -1., 1.)
    tau = (-a*h+sign*discriminant.sqrt())/(1-a.square())
    kappa = torch.where(active, (tau_trial-tau)/k, torch.zeros_like(tau))
    stress = trial_stress-kappa[..., None]*cq
    returned_tau = stress@shear
    alpha = returned_tau/(returned_tau.square()+epsilon**2).sqrt()
    gamma = kappa/torch.where(active, alpha, torch.ones_like(alpha))
    plastic = kappa[..., None]*shear
    normal_f = P.joint_yield_gradient(stress, params)
    denominator = normal_f@cq
    denominator = torch.where(active, denominator, torch.ones_like(denominator))
    correction = cq[..., None]*(normal_f@stiffness)[..., None, :]/denominator[..., None, None]
    tangent = stiffness-torch.where(active[..., None, None], correction, torch.zeros_like(correction))
    return dict(stress=stress, plastic=plastic, gamma=gamma, kappa=kappa,
                active=active, tangent=tangent)
