"""Implicit simultaneous matrix, weak-plane shear and tensile-cutoff return."""
import numpy as np
import torch
from itertools import combinations

from .plasticity import _dp_constants, _joint_geometry


def surfaces(stress, params, tension):
    """Yield values, flow directions, yield gradients and flow Hessians."""
    normal, shear = _joint_geometry(params, stress)
    metric = torch.diag(stress.new_tensor([.5, .5, .5, 1.]))
    metric[:3, :3] -= 1/6
    pressure = stress.new_tensor([1., 1., 1., 0.])
    qstress = stress@metric.T
    radius = ((stress*qstress).sum(-1)+params.matrix_smoothing**2).sqrt()
    devgrad = qstress/radius[..., None]
    hessian = metric/radius[..., None, None]-qstress[..., :, None]*qstress[..., None, :]/radius[..., None, None]**3
    alpha, cohesion = _dp_constants(params.matrix_phi_deg, params.matrix_cohesion)
    dilation, _ = _dp_constants(params.matrix_psi_deg)
    fm = radius-params.matrix_smoothing+alpha*(stress@pressure)-cohesion
    gm = devgrad+dilation*pressure
    nm = devgrad+alpha*pressure
    tau = stress@shear
    rj = (tau.square()+params.joint_smoothing**2).sqrt()
    sheargrad = tau[..., None]/rj[..., None]*shear
    hj = params.joint_smoothing**2/rj[..., None, None]**3*shear[:, None]*shear[None]
    friction = np.tan(np.deg2rad(params.joint_phi_deg))
    dilationj = np.tan(np.deg2rad(params.joint_psi_deg))
    fj = rj-params.joint_smoothing+friction*(stress@normal)-params.joint_cohesion
    gj = sheargrad+dilationj*normal
    nj = sheargrad+friction*normal
    nt = normal.expand_as(stress)
    ft = stress@normal-tension
    return (torch.stack((fm, fj, ft), -1), torch.stack((gm, gj, nt), -2),
            torch.stack((nm, nj, nt), -2),
            torch.stack((hessian, hj, torch.zeros_like(hessian)), -3))


def equations(state, trial, stiffness, params, tension, scale):
    stress, multiplier = state[..., :4], state[..., 4:]
    f, flow, normal, hessian = surfaces(stress, params, tension)
    cq = torch.einsum("ij,qaj->qia", stiffness, flow)/scale
    balance = stress-trial+torch.einsum("qia,qa->qi", cq, multiplier)
    length = (multiplier.square()+f.square()).sqrt().clamp_min(1e-15)
    complementarity = length-multiplier+f
    residual = torch.cat((balance, complementarity), -1)
    jacobian = state.new_zeros((len(state), 7, 7))
    jacobian[:, :4, :4] = torch.eye(4, device=state.device, dtype=state.dtype)+torch.einsum(
        "ij,qa,qajk->qik", stiffness, multiplier/scale, hessian)
    jacobian[:, :4, 4:] = cq
    jacobian[:, 4:, :4] = (1+f/length)[..., None]*normal
    jacobian[:, 4:, 4:] = torch.diag_embed(multiplier/length-1)
    return residual, jacobian


def active_set_return(trial, stiffness, params, tension, scale):
    """Resolve stalled intersections with active sets and local root predictors."""
    solution = trial.new_zeros((len(trial), 7))
    solved = torch.zeros(len(trial), dtype=torch.bool, device=trial.device)
    for count in (1, 2, 3):
        for active in combinations(range(3), count):
            indices = (~solved).nonzero().flatten()
            if not len(indices):
                return solution
            local_trial = trial[indices]
            state = torch.cat((local_trial.clone(), local_trial.new_zeros((len(indices), 3))), -1)
            mask = torch.zeros(3, dtype=torch.bool, device=trial.device)
            mask[list(active)] = True

            def active_equations(value):
                residual, jacobian = equations(value, local_trial, stiffness, params, tension, scale)
                f, _, normal, _ = surfaces(value[:, :4], params, tension)
                residual[:, 4:] = torch.where(mask, f, value[:, 4:])
                jacobian[:, 4:, :4] = torch.where(mask[None, :, None], normal, 0.)
                jacobian[:, 4:, 4:] = torch.diag((~mask).to(trial))
                return residual, jacobian

            for _ in range(60):
                residual, jacobian = active_equations(state)
                error = residual.square().sum(-1)
                if float(error.max()) < 1e-18:
                    break
                delta, info = torch.linalg.solve_ex(jacobian, -residual[..., None])
                delta = delta.squeeze(-1)
                usable = (info == 0) & torch.isfinite(delta).all(-1)
                delta = torch.where(usable[:, None], delta, 0.)
                alpha = torch.ones_like(error)
                for _ in range(24):
                    candidate_residual, _ = active_equations(state+alpha[:, None]*delta)
                    good = (candidate_residual.square().sum(-1) <= error*(1-1e-4*alpha)) | (error < 1e-18) | ~usable
                    if bool(good.all()):
                        break
                    alpha = torch.where(good, alpha, alpha/2)
                state += alpha[:, None]*delta
            residual, _ = equations(state, local_trial, stiffness, params, tension, scale)
            valid = residual.abs().amax(-1) < 1e-8
            solution[indices[valid]] = state[valid]
            solved[indices[valid]] = True
    if not bool(solved.all()):
        from scipy.optimize import least_squares
        for index in (~solved).nonzero().flatten():
            local_trial = trial[index:index+1]
            def residual(value):
                state = torch.as_tensor(value, dtype=trial.dtype, device=trial.device)[None]
                return equations(state, local_trial, stiffness, params, tension, scale)[0][0].cpu().numpy()
            def jacobian(value):
                state = torch.as_tensor(value, dtype=trial.dtype, device=trial.device)[None]
                return equations(state, local_trial, stiffness, params, tension, scale)[1][0].cpu().numpy()
            if params.joint_psi_deg == 0:
                from .joint_return import joint_return
                try:
                    shear_seed = joint_return(local_trial, stiffness, params)
                except (ValueError, FloatingPointError):
                    shear_seed = None
                if shear_seed is not None:
                    initial = np.concatenate((shear_seed["stress"][0].cpu().numpy(),
                        [0., float(shear_seed["gamma"][0]*scale), 0.]))
                    polished = least_squares(residual, initial, jac=jacobian,
                        ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=1000)
                    if np.max(np.abs(polished.fun)) < 1e-8:
                        solution[index] = torch.as_tensor(polished.x, dtype=trial.dtype, device=trial.device)
                        continue
            initial = np.concatenate((local_trial[0].cpu().numpy(), np.zeros(3)))
            lower = np.concatenate((np.full(4, -np.inf), np.zeros(3)))
            # First approach an admissible multiplier region, then remove the
            # bound barrier to resolve inactive zero multipliers accurately.
            bounded = least_squares(residual, initial, jac=jacobian,
                bounds=(lower, np.full(7, np.inf)), ftol=1e-13, xtol=1e-13, gtol=1e-13, max_nfev=1000)
            polished = least_squares(residual, bounded.x, jac=jacobian,
                ftol=1e-14, xtol=1e-14, gtol=1e-14, max_nfev=1000)
            if np.max(np.abs(polished.fun)) >= 1e-8:
                raise FloatingPointError(f"No admissible local return for trial {local_trial[0].tolist()}")
            solution[index] = torch.as_tensor(polished.x, dtype=trial.dtype, device=trial.device)
    return solution


class _ImplicitReturn(torch.autograd.Function):
    @staticmethod
    def forward(trial, stiffness, params, tension):
        scale = stiffness.diag().max()
        state = torch.cat((trial.clone(), trial.new_zeros((len(trial), 3))), -1)
        pending = torch.arange(len(trial), device=trial.device)
        previous_error = trial.new_full((len(trial),), float("inf"))
        stagnation = torch.zeros(len(trial), dtype=torch.long, device=trial.device)

        def resolve(indices):
            # Stalled intersections involve small batches; solve the same
            # active-set equations on the CPU and return their admissible root.
            return active_set_return(trial[indices].cpu(), stiffness.cpu(), params,
                                     tension, scale.cpu()).to(trial.device)

        for _ in range(80):
            residual, jacobian = equations(state[pending], trial[pending], stiffness, params, tension, scale)
            error = residual.square().sum(-1)
            unresolved = error >= 1e-18
            pending = pending[unresolved]
            if not len(pending):
                break
            residual, jacobian, error = residual[unresolved], jacobian[unresolved], error[unresolved]
            stagnation[pending] = torch.where(error >= .99*previous_error[pending], stagnation[pending]+1, 0)
            previous_error[pending] = error
            stalled = stagnation[pending] >= 6
            if bool(stalled.any()):
                state[pending[stalled]] = resolve(pending[stalled])
                pending = pending[~stalled]
                residual, jacobian, error = residual[~stalled], jacobian[~stalled], error[~stalled]
                if not len(pending):
                    break
            delta, info = torch.linalg.solve_ex(jacobian, -residual[..., None])
            delta = delta.squeeze(-1)
            usable = (info == 0) & torch.isfinite(delta).all(-1)
            if not bool(usable.all()):
                failed = pending[~usable]
                state[failed] = resolve(failed)
                pending, delta, error = pending[usable], delta[usable], error[usable]
                if not len(pending):
                    break
            alpha = torch.ones_like(error)
            for _ in range(30):
                candidate = state[pending]+alpha[:, None]*delta
                candidate_residual, _ = equations(candidate, trial[pending], stiffness, params, tension, scale)
                good = candidate_residual.square().sum(-1) <= error*(1-1e-4*alpha)
                if bool(good.all()):
                    break
                alpha = torch.where(good, alpha, alpha/2)
            state[pending] += alpha[:, None]*delta
        residual, jacobian = equations(state, trial, stiffness, params, tension, scale)
        unresolved = residual.abs().amax(-1) > 1e-8
        if bool(unresolved.any()):
            state[unresolved] = resolve(unresolved.nonzero().flatten())
            residual, jacobian = equations(state, trial, stiffness, params, tension, scale)
        right = torch.eye(7, device=trial.device, dtype=trial.dtype)[:, :4]
        tangent = torch.linalg.solve(jacobian, right.expand(len(trial), -1, -1))
        return state, tangent

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, tangent = output
        ctx.save_for_backward(tangent)
        ctx.mark_non_differentiable(tangent)

    @staticmethod
    def backward(ctx, grad_output, grad_tangent):
        (tangent,) = ctx.saved_tensors
        return torch.einsum("qij,qi->qj", tangent, grad_output), None, None, None


def multi_surface_return(trial, stiffness, params, tension_mpa):
    state, tangent = _ImplicitReturn.apply(trial, stiffness, params, tension_mpa/params.stress_scale_mpa)
    stress = state[..., :4]
    plastic = torch.linalg.solve(stiffness, (trial-stress).T).T
    return dict(stress=stress, plastic=plastic, gamma=state[..., 4:]/stiffness.diag().max(),
                stress_tangent=tangent[:, :4])
