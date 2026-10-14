"""Strong and weak PINN equations after local weak-plane condensation."""

import torch

from . import config as C, plasticity as P
from .joint_return import joint_return
from .orientation_tangent import NeuralCoordinates, OrientationResidual


def active_set_line_probe(problem, scale=None):
    """Measure the selected residual change across the nearest loading switch.

    The descent direction belongs to the actual neural parameters. No model
    weights change; lengths bracket the first linearized active-set crossing.
    """
    coordinates = NeuralCoordinates(problem.current, "displacement")
    point = coordinates.zero()
    function = lambda value: problem(coordinates(value))
    value, reverse = torch.func.vjp(function, point)
    gradient = reverse(value)[0].detach()
    direction = -gradient if scale is None else -scale.to(point).square()*gradient
    if direction.norm() == 0:
        return dict(gradient_max=0., nearest_crossings=[], line=[])
    direction = direction/direction.norm()

    def trial_yield(delta):
        return P.joint_yield(problem.fields(coordinates(delta))[0]["trial_stress"], problem.params)

    yields, derivative = torch.func.jvp(trial_yield, (point,), (direction,))
    crossing = -yields/derivative
    candidates = torch.where(torch.isfinite(crossing) & (crossing > 0))[0]
    order = candidates[torch.argsort(crossing[candidates])]
    lengths = [0.]
    bracketed_crossing = None
    if len(order):
        first = float(crossing[order[0]].detach())
        index = int(order[0])
        with torch.no_grad():
            def selected_yield(length):
                return float(trial_yield(point+length*direction)[index])
            if selected_yield(0.)*selected_yield(2*first) < 0:
                from scipy.optimize import brentq
                first = brentq(selected_yield, 0., 2*first,
                    xtol=max(first*1e-8, torch.finfo(point.dtype).tiny))
                bracketed_crossing = dict(index=index, length=first)
        lengths += [first*.1, first*.999, first*1.001, first*10]
    rows = []
    with torch.no_grad():
        for length in lengths:
            delta = point+length*direction
            blocks = problem.blocks(coordinates(delta))
            changed = torch.where((trial_yield(delta)>0) != (yields>0))[0]
            rows.append(dict(length=length,
                objective=float(sum(block.square().sum() for block in blocks.values())),
                block_objectives={name: float(block.square().sum()) for name, block in blocks.items()},
                changed_active=changed.tolist()))
    return dict(direction="negative_gradient" if scale is None else "jacobi_scaled_negative_gradient",
        gradient_max=float(gradient.abs().max()), directional_slope=float(2*(gradient*direction).sum()),
        bracketed_crossing=bracketed_crossing,
        nearest_crossings=[dict(index=int(i), xy_over_a=problem.pts.xy_all[i].tolist(),
            trial_yield=float(yields[i].detach()), slope=float(derivative[i].detach()),
            crossing=float(crossing[i].detach())) for i in order[:2]], line=rows)


class ReturnedOrientationResidual(OrientationResidual):
    neural_channels = "displacement"
    local_constraints = ()

    def __init__(self, model, saved, params, *, evaluation=False):
        self.equilibrium_form = saved["settings"].get("equilibrium_form", "strong")
        if self.equilibrium_form not in ("strong", "virtual_work"):
            raise ValueError(f"Unknown equilibrium form: {self.equilibrium_form}")
        if self.equilibrium_form == "virtual_work" and saved["settings"]["weights"]["virtual_work"] <= 0:
            raise ValueError("Weak equilibrium requires a positive virtual-work weight")
        super().__init__(model, saved, params, evaluation=evaluation, duals=False)
        current = model.increments[-1]
        del model.increments[-1]
        try:
            with torch.no_grad():
                if model.n_steps:
                    gradients = []
                    for xy in self.pts.xy_all.split(1024):
                        gradients.append(torch.stack([torch.func.jvp(
                            lambda points: model(points)[1][:, -1], (xy,),
                            (torch.eye(2, device=xy.device, dtype=xy.dtype)[axis].expand_as(xy),))[1]
                            for axis in range(2)]))
                    stress_gradient = torch.cat(gradients, dim=1)
                else:
                    stress_gradient = self.c.new_zeros((2, len(self.pts.xy_all), 4))
                self.previous_elastic_gradient = torch.linalg.solve(
                    self.c, stress_gradient.unsqueeze(-1)).squeeze(-1)
                self.background_strain_gradient = torch.linalg.solve(self.c,
                    self.cache["increment_background"][1][1].unsqueeze(-1)).squeeze(-1)
        finally:
            model.increments.append(current)

    def blocks(self, coefficients):
        # Returned stress is continuous at activation. Its spatial derivative
        # jumps, so weak equilibrium avoids a discontinuous collocation loss.
        h, r = self.fields(coefficients, equilibrium=self.equilibrium_form == "strong")
        return self.weighted_blocks(coefficients, h, r)

    def fields(self, coefficients, *, equilibrium=True):
        grid = self.cache["increment_grid"](coefficients[..., :2], second=equilibrium)
        values, gradient = grid[:2]
        background, dbackground = self.cache["increment_background"]
        u = self.cache["displacement"]+values+background[0]
        gradient = gradient+dbackground[0]
        increment_strain = torch.stack((gradient[0, :, 0], gradient[1, :, 1],
            torch.zeros_like(gradient[0, :, 0]), gradient[1, :, 0]+gradient[0, :, 1]), -1)
        trial = self.s0+self.cache["stress_increment"]+increment_strain@self.c.T
        returned = joint_return(trial, self.c, self.params, check_branch=False)
        total = returned["stress"]
        stress = total-self.s0
        plastic = self.cache["plastic_strain"]+returned["plastic"]
        strain = self.cache["strain"]+increment_strain
        gamma = torch.stack((torch.zeros_like(returned["gamma"]), returned["gamma"]), -1)
        yields = torch.stack((P.matrix_yield(total, self.params), P.joint_yield(total, self.params)), -1)
        constitutive = (strain-plastic)@self.c.T-stress
        n0, n1 = self.pts.n_pde, self.pts.n_pde+self.pts.n_wall
        r = dict(wall=P._traction4(stress[n0:n1], self.pts.nrm_wall)+(self.step/self.current.total_steps)*self.pts.t0_wall,
            interface=P._traction4(stress[n1:], self.pts.nrm_gam)-self.sf.apply(u[n1:].T).T,
            constitutive=constitutive, flow=constitutive,
            complementarity=P._fischer_burmeister(gamma/self.current.multiplier_scale,
                -yields, self.params.complementarity_smoothing), gauge=u[n1:].mean(0))
        if equilibrium:
            hessian = grid[2]
            strain_gradient = torch.stack((hessian[:, 0, :, 0], hessian[:, 1, :, 1],
                torch.zeros_like(hessian[:, 0, :, 0]), hessian[:, 1, :, 0]+hessian[:, 0, :, 1]), -1)
            strain_gradient = strain_gradient+self.background_strain_gradient+self.previous_elastic_gradient
            stress_gradient = torch.einsum("nij,anj->ani", returned["tangent"], strain_gradient)
            r["equilibrium"] = torch.stack((stress_gradient[0, :, 0]+stress_gradient[1, :, 3],
                                           stress_gradient[0, :, 3]+stress_gradient[1, :, 1]), -1)[:n0]
        h = dict(displacement=u, stress_increment=stress, stress_total=total, strain=strain,
            plastic_strain=plastic, multiplier_increment=gamma,
            scaled_multiplier=gamma/self.current.multiplier_scale, yield_values=yields,
            kinematic_plastic_strain=strain-torch.linalg.solve(self.c, stress.unsqueeze(-1)).squeeze(-1),
            trial_stress=trial)
        return h, r

    @torch.no_grad()
    def assessment(self, coefficients):
        h, r = self.fields(coefficients)
        joint_return(h["trial_stress"], self.c, self.params)
        matrix_max = h["yield_values"][:, 0].max()
        result = super().assessment(coefficients)
        peak = r["equilibrium"].norm(dim=-1).argmax()
        xy = self.pts.xy_all[peak]
        result["principal_point"] = dict(radius=float(xy.norm()),
            angle_deg=float(torch.atan2(xy[1], xy[0])*180/torch.pi)%360,
            equilibrium_mpa_per_m=(r["equilibrium"][peak]*C.SIGMA_REF/C.A_TUNNEL/1e6).tolist(),
            constitutive_mpa=(r["constitutive"][peak]*C.SIGMA_REF/1e6).tolist(),
            joint_yield_mpa=float(h["yield_values"][peak, 1]*C.SIGMA_REF/1e6))
        result.update(maximum_matrix_yield_mpa=float(matrix_max*C.SIGMA_REF/1e6),
            maximum_joint_yield_mpa=float(h["yield_values"][:, 1].max()*C.SIGMA_REF/1e6),
            interface_traction_rms_mpa=float(r["interface"].square().mean().sqrt()*C.SIGMA_REF/1e6),
            active_area_fraction=float((self.pts.interior_weights*
                (h["multiplier_increment"][:self.pts.n_pde, 1] > 0)).sum()))
        active = self.pts.xy_all[:self.pts.n_pde][h["multiplier_increment"][:self.pts.n_pde, 1] > 0]
        result["active_radius_range"] = ([float(active.norm(dim=-1).min()), float(active.norm(dim=-1).max())]
                                         if len(active) else None)
        return result
