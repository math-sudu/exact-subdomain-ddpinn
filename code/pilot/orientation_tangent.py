"""Coupled tangent diagnostics for the projected causal PINN objective."""

import numpy as np
import torch

from interface_extension import geometry, regional_power
from . import config as C
from .causal_mixed import current_residuals, prefix_cache
from .spectral_mixed import CircularGridEvaluation
from .virtual_work import VirtualWorkMetric
from . import plasticity as P


def joint_active_tangent(stress, gamma, stiffness, params):
    """Consistent local tangent conditional on joint-only plastic loading.

    Differentiate s=C(e-ep_old-gamma*grad(g(s))) together with f(s)=0.
    This is a branch diagnostic, not an activation rule or a stress update.
    Engineering shear and the flow-potential Hessian use the existing model.
    """
    flow = P.joint_flow_direction(stress, params)
    normal = P.joint_yield_gradient(stress, params)
    hessian = torch.func.jacrev(lambda value: P.joint_flow_direction(value, params))(stress)
    modulus = torch.linalg.solve(torch.eye(4, device=stress.device, dtype=stress.dtype)
                                  +gamma*stiffness@hessian, stiffness)
    return modulus-torch.outer(modulus@flow, normal@modulus)/(normal@modulus@flow)


class OrientationResidual:
    """Real residual vector whose squared norm is the training objective.

    Saved duals enter by completing the square, adding only a constant to the
    objective. The completed prefix and all own-step flow directions are kept.
    """
    local_constraints = ("constitutive", "complementarity")

    def __init__(self, model, saved, params, *, evaluation=False, duals=True):
        self.settings, self.case = saved["settings"], saved["case"]
        self.step = model.n_steps
        self.current = model.increments[-1]
        device = self.current.multiplier_scale.device
        self.pts, self.c, self.s0, self.sf = geometry(
            self.case, self.settings, device, evaluation=evaluation)
        self.params = params
        self.work = VirtualWorkMetric(self.c, C.nondim_lambda(C.material(
            self.case["eh_over_ev"], self.case["bedding_dip_deg"])),
            self.settings, self.pts) if not evaluation else None
        # prefix_cache only reads the supplied model. Restore its ModuleList
        # before exposing this object to callers.
        del model.increments[-1]
        try:
            self.cache = prefix_cache(model, self.pts.xy_all, self.c, self.s0, params)
            self.prefix_coefficients = sum(
                (inc.coefficients()[..., :2] for inc in model.increments),
                torch.zeros_like(self.current.coefficients()[..., :2]))
        finally:
            model.increments.append(self.current)
        with torch.no_grad():
            self.cache["increment_background"] = self.current.elastic_background.spatial_fields(
                self.pts.xy_all)
        n0, n1 = self.pts.n_pde, self.pts.n_pde+self.pts.n_wall
        nt = self.settings["quadrature"]["angular"]*(2 if evaluation else 1)
        self.angular_points = nt
        self.cache["increment_grid"] = CircularGridEvaluation(self.current, (
            self.pts.xy_all[:n0].reshape(-1, nt, 2),
            self.pts.xy_all[n0:n1][None], self.pts.xy_all[n1:][None]))
        self.regional_weights = torch.cat((.8*self.pts.interior_weights,
            .1*self.pts.wall_weights,
            self.pts.xy_all.new_full((len(self.pts.xy_all)-n1,), .1/(len(self.pts.xy_all)-n1))))
        self.interface_factor = torch.linalg.cholesky(self.sf.Qc).mH
        self.duals = {}
        if duals and not evaluation:
            for name in self.local_constraints:
                if name+"_augmentation" in saved:
                    self.duals[name] = saved[name+"_augmentation"]["dual"].to(self.c)

    def fields(self, coefficients):
        return current_residuals(self.current, self.cache, self.pts, self.c,
            self.s0, self.sf, self.params, self.step, projected_coefficients=coefficients)

    def blocks(self, coefficients):
        h, r = self.fields(coefficients)
        return self.weighted_blocks(coefficients, h, r)

    def weighted_blocks(self, coefficients, h, r):
        """Weight the selected field equations and the exterior-coupled work."""
        w = self.settings["weights"]
        interface = self.interface_factor@self.sf.coeffs(r["interface"].T)
        interface = interface*(w["interface"]*self.sf.wm).sqrt()[None]
        blocks = {}
        if "equilibrium" in r:
            blocks["equilibrium"] = r["equilibrium"]*(w["equilibrium"]*self.pts.interior_weights).sqrt()[:, None]
        blocks.update({
            "wall": r["wall"]*(w["wall"]*self.pts.wall_weights).sqrt()[:, None],
            "interface": torch.view_as_real(interface),
            "gauge": w["gauge"]**.5*r["gauge"],
        })
        for name in self.local_constraints:
            value = r[name]
            if name in self.duals:
                value = value+self.duals[name]/(2*w[name])
            blocks[name] = value*(w[name]*self.regional_weights/value.shape[-1]).sqrt()[:, None]
        if self.work is not None:
            correction = self.prefix_coefficients+coefficients[..., :2]
            imbalance = (self.work.matrix@self.work.correction_vector(correction)
                         -self.work.source_load(h["plastic_strain"][:self.pts.n_pde]))
            whitened = torch.linalg.solve_triangular(self.work.factor, imbalance[:, None], upper=False)
            blocks["virtual_work"] = torch.view_as_real(whitened)*(w["virtual_work"]/self.work.area)**.5
        return blocks

    def __call__(self, coefficients):
        return torch.cat([value.reshape(-1) for value in self.blocks(coefficients).values()])

    @torch.no_grad()
    def assessment(self, coefficients):
        h, r = self.fields(coefficients)
        n0 = self.pts.n_pde
        point = torch.argmax((h["multiplier_increment"][:n0, 1].clamp_min(0)
                              *h["yield_values"][:n0, 1].abs()))
        point_xy = self.pts.xy_all[point]
        constitutive = r["constitutive"][:n0].reshape(-1, self.angular_points, 4)
        spectrum = torch.fft.rfft(constitutive, dim=1)
        spectrum[:, self.current.angular_modes+1:] = 0
        low = torch.fft.irfft(spectrum, n=self.angular_points, dim=1).reshape(-1, 4)
        high = r["constitutive"][:n0]-low
        return dict(
            equilibrium_rms=float((r["equilibrium"].square().sum(-1)*self.pts.interior_weights).sum().sqrt()),
            constitutive_rms=float(regional_power(r["constitutive"], self.pts).sqrt()),
            complementarity_rms=float(regional_power(r["complementarity"], self.pts).sqrt()),
            wall_traction_rms_mpa=float((r["wall"].square().sum(-1)*self.pts.wall_weights).sum().sqrt()*C.SIGMA_REF/1e6),
            constitutive_retained_angular_rms_mpa=float((low.square().mean(-1)*self.pts.interior_weights).sum().sqrt()*C.SIGMA_REF/1e6),
            constitutive_unretained_angular_rms_mpa=float((high.square().mean(-1)*self.pts.interior_weights).sum().sqrt()*C.SIGMA_REF/1e6),
            principal_point=dict(radius=float(point_xy.norm()),
                angle_deg=float(torch.atan2(point_xy[1], point_xy[0])*180/torch.pi)%360,
                constitutive_mpa=(r["constitutive"][point]*C.SIGMA_REF/1e6).tolist(),
                joint_yield_mpa=float(h["yield_values"][point, 1]*C.SIGMA_REF/1e6)),
            minimum_yy_constitutive_mpa=float(r["constitutive"][:n0, 1].min()*C.SIGMA_REF/1e6))


class PhysicalCoordinates:
    """Independent real coordinates of the allowed projected field modes."""

    def __init__(self, increment, channels="all"):
        self.base = increment.coefficients().detach().clone()
        mask = torch.ones_like(self.base.real, dtype=torch.bool)
        modes = increment.angular_indices.long()
        if increment.centrosymmetric:
            mask &= ((modes % 2)[:, None] == (torch.arange(8, device=modes.device)<2)[None])[None]
        if channels == "stress":
            mask[..., :2] = False
            mask[..., 6:] = False
        elif channels != "all":
            raise ValueError(channels)
        mask = torch.stack((mask, mask & (modes>0)[None, :, None]), -1)
        self.indices = mask.reshape(-1).nonzero().flatten()
        self.shape = mask.shape

    def __call__(self, delta):
        correction = delta.new_zeros(np.prod(self.shape)).scatter(0, self.indices, delta)
        return self.base+torch.view_as_complex(correction.reshape(self.shape))

    def zero(self):
        return self.base.real.new_zeros(len(self.indices))


class NeuralCoordinates:
    """Joint updates of the existing increment's actual neural parameters."""

    class CoefficientMap(torch.nn.Module):
        def __init__(self, increment):
            super().__init__()
            self.increment = increment

        def forward(self):
            return self.increment.coefficients()

    def __init__(self, increment, channels="all"):
        increment.requires_grad_(True)
        self.base = increment.coefficients().detach().clone()
        self.mapping = self.CoefficientMap(increment)
        if channels not in ("all", "stress", "displacement", "displacement_head"):
            raise ValueError(channels)
        self.parameters = {name: p.detach().clone() for name, p in self.mapping.named_parameters()
                           if channels=="all" or (channels=="stress" and name.startswith("increment.stress"))
                           or (channels=="displacement" and name.startswith("increment.kinematic"))
                           or (channels=="displacement_head" and name.startswith("increment.kinematic_head."))}
        self.indices = {}
        for name, value in self.parameters.items():
            mask = torch.ones_like(value, dtype=torch.bool)
            if channels.startswith("displacement") and name.startswith("increment.kinematic_head."):
                mask[2:] = False
            if increment.centrosymmetric and name == "increment.kinematic_head.bias":
                mask[:2] = False  # Constant displacements vanish under odd parity.
            if name == "increment.stress_spectral_correction":
                if increment.centrosymmetric:
                    mask[:, 1::2] = False
                mask[:, 0, :, 1] = False
            self.indices[name] = mask.reshape(-1).nonzero().flatten()
        self.counts = [len(index) for index in self.indices.values()]

    def __call__(self, delta):
        return torch.func.functional_call(self.mapping, self.parameter_values(delta), ())

    def parameter_values(self, delta):
        return {name: value+piece.new_zeros(value.numel()).scatter(
                    0, self.indices[name], piece).reshape(value.shape)
                  for (name, value), piece in zip(self.parameters.items(), delta.split(self.counts))}

    @torch.no_grad()
    def apply(self, delta):
        """Write the accepted coordinates into the actual causal increment."""
        parameters = dict(self.mapping.named_parameters())
        for name, value in self.parameter_values(delta).items():
            parameters[name].copy_(value)

    def zero(self):
        return self.base.real.new_zeros(sum(self.counts))


def linear_operator(function, point, scale):
    """Cache the fixed tangent and its transpose for a linear subproblem."""
    from scipy.sparse.linalg import LinearOperator

    value, reverse = torch.func.vjp(function, point)
    # The pullback is linear in its cotangent. Transposing it once gives J*v
    # while reusing the fixed primal intermediates throughout the Krylov solve.
    # This avoids repeating the neural/physical forward pass for every product.
    _, forward = torch.func.vjp(lambda cotangent: reverse(cotangent)[0],
                                torch.zeros_like(value))

    def matvec(vector):
        tangent = torch.as_tensor(vector, device=point.device, dtype=point.dtype).reshape(-1)*scale
        return forward(tangent)[0].detach().cpu().numpy()

    def rmatvec(vector):
        cotangent = torch.as_tensor(vector, device=point.device, dtype=point.dtype).reshape(-1)
        return (reverse(cotangent)[0]*scale).detach().cpu().numpy()

    return LinearOperator((value.numel(), point.numel()), matvec=matvec,
                          rmatvec=rmatvec, dtype=np.float64), value.detach(), reverse


def column_scale(reverse, value, coordinates, samples=32):
    """Estimate diag(J.T*J) from fixed Rademacher residual projections."""
    generator = torch.Generator(device=value.device).manual_seed(1729)
    diagonal = torch.zeros_like(coordinates)
    for _ in range(samples):
        sign = torch.randint(2, value.shape, generator=generator, device=value.device,
                             dtype=torch.int64).to(value.dtype)*2-1
        diagonal += reverse(sign)[0].square().detach()/samples
    # A zero head or saturated neural feature can give an exactly zero column.
    # Unit scaling leaves that null direction unchanged in the linear solve,
    # as in SciPy's Jacobian column scaling, without discarding other columns.
    return torch.where(diagonal == 0, torch.ones_like(diagonal), diagonal).rsqrt()


def trust_region(function, point, scale, *, linear_iterations, evaluations, callback=None,
                 gradient_tolerance=None, linear_tolerance=1e-6):
    """Use SciPy's regularized LSMR trust-region step on the actual residual."""
    from scipy.optimize import least_squares

    def residual(values):
        with torch.no_grad():
            return function(torch.as_tensor(values, device=point.device)).cpu().numpy()

    def jacobian(values):
        current = torch.as_tensor(values, device=point.device)
        return linear_operator(function, current, torch.ones_like(current))[0]

    termination = ({} if gradient_tolerance is None else
                   dict(ftol=None, xtol=None, gtol=gradient_tolerance))
    return least_squares(residual, point.cpu().numpy(), jac=jacobian,
        method="trf", tr_solver="lsmr", x_scale=scale.cpu().numpy(),
        tr_options=dict(maxiter=linear_iterations, atol=linear_tolerance, btol=linear_tolerance),
        max_nfev=evaluations, callback=callback, **termination)


def neural_primal_step(problem, controls, inner_state, callback=None):
    """Advance one fixed-dual inner solve and retain its stopping target.

    The relative gradient reduction controls the inexact inner solve, not
    physical acceptance. An evaluation limit never requests a dual update.
    """
    coordinates = NeuralCoordinates(problem.current, getattr(problem, "neural_channels", "all"))
    point = coordinates.zero()
    function = lambda delta: problem(coordinates(delta))
    _, value, reverse = linear_operator(function, point, torch.ones_like(point))
    if not inner_state:
        gradient = reverse(value)[0]
        initial = float(gradient.detach().abs().max())
        inner_state.update(initial_optimality=initial,
                           gradient_tolerance=max(controls["gradient_reduction"]*initial,
                                                  10*np.finfo(np.float64).eps))
        del gradient
    # Newly activated neural features need their current Jacobian scale.
    # Preserve the inner solve's original gradient stopping target.
    inner_state["scale"] = column_scale(reverse, value, point).cpu()
    del value, reverse

    def accepted(intermediate_result):
        coordinates.apply(point.new_tensor(intermediate_result.x))
        if callback is not None:
            callback(intermediate_result)

    fit = trust_region(function, point, inner_state["scale"].to(point),
        linear_iterations=controls["linear_iterations"],
        evaluations=controls["evaluations_per_solve"], callback=accepted,
        gradient_tolerance=inner_state["gradient_tolerance"],
        linear_tolerance=controls.get("linear_tolerance", 1e-6))
    polish_limit = controls.get("polish_evaluations", 0)
    if fit.status == 0 and polish_limit:
        from .lbfgs import budgeted_lbfgs_step

        scale = inner_state["scale"].to(point)
        scaled = torch.nn.Parameter(point.new_tensor(fit.x)/scale)
        optimizer = torch.optim.LBFGS([scaled], lr=1., max_iter=80, max_eval=100,
            history_size=50, line_search_fn="strong_wolfe",
            tolerance_grad=0., tolerance_change=0.)
        evaluations, before = 0, fit.nfev

        def closure():
            nonlocal evaluations
            optimizer.zero_grad(set_to_none=True)
            loss = .5*function(scaled*scale).square().sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite primal polishing objective")
            loss.backward()
            evaluations += 1
            return loss

        while polish_limit-evaluations >= 28:
            previous = scaled.detach().clone()
            budgeted_lbfgs_step(optimizer, closure,
                remaining_evals=polish_limit-evaluations-1, chunk_max_eval=100)
            # Strong-Wolfe's last trial need not be its accepted endpoint.
            # Re-evaluate the actual endpoint before checking the raw gradient.
            loss = closure()
            fit.x = (scaled*scale).detach().cpu().numpy()
            fit.cost = float(loss.detach())
            fit.nfev = before+evaluations
            fit.optimality = float((scaled.grad/scale).abs().max())
            if fit.optimality < inner_state["gradient_tolerance"]:
                fit.status = 1
                fit.message = "Gradient tolerance satisfied after scaled L-BFGS polishing."
            accepted(fit)
            if fit.status == 1:
                break
            if torch.equal(scaled.detach(), previous):
                fit.message = "Scaled L-BFGS returned unchanged parameters; inspect the local objective."
                break
    coordinates.apply(point.new_tensor(fit.x))
    return dict(status=int(fit.status), message=fit.message, evaluations=int(fit.nfev),
                optimality=float(fit.optimality),
                initial_optimality=inner_state["initial_optimality"],
                gradient_tolerance=inner_state["gradient_tolerance"])
