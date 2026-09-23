"""Functional current-increment residual with a material-derived interface norm."""

from copy import deepcopy

import torch

from pilot.causal_mixed import current_residuals, prefix_cache
from pilot.plastic_assessment import region_slices


def material_dual_vector(sf, traction_residual):
    """Factor a^H Q a as ||L^H a||^2 using the exterior operator's modal weights."""
    coefficients = sf.coeffs(traction_residual.T)
    factor = torch.linalg.cholesky(sf.Qc)
    transformed = (factor.mH @ coefficients) * sf.wm.sqrt()[None]
    return torch.cat((transformed.real.flatten(), transformed.imag.flatten()))


def weighted_vector(r, pts, spec, sf):
    w = pts.interior_weights
    weights, masses = spec["weights"], spec["regional_masses"]
    vectors = [(r["equilibrium"] * (w * weights[0]).sqrt()[:, None]).flatten()]
    vectors.extend(((r["wall"] * (weights[1] / pts.n_wall)**.5).flatten(),
                    material_dual_vector(sf, r["interface"]) * weights[2]**.5))
    for key, components, coefficient in [("constitutive", 4, weights[3]),
                                          ("complementarity", 2, weights[4])]:
        for name, sl in region_slices(pts).items():
            value = r[key][sl]
            measure = w if name == "interior" else w.new_full((len(value),), 1 / len(value))
            vectors.append((value * (measure * masses[name] * coefficient / components)
                            .sqrt()[:, None]).flatten())
    vectors.append(r["gauge"].flatten())
    return torch.cat(vectors)


class CausalResponseProblem:
    def __init__(self, model, spec, points_and_operators, stiffness, insitu, params):
        if not 1 <= model.n_steps <= model.total_steps:
            raise ValueError("The model must contain a current load increment")
        self.step = model.n_steps
        self.spec = deepcopy(spec)
        self.c, self.s0, self.params = stiffness, insitu, params
        self.model = deepcopy(model)
        self.current = self.model.increments[-1]
        del self.model.increments[-1]
        self.model.requires_grad_(False)
        self.current.requires_grad_(True)
        self.shapes = [(name, p.shape, p.numel()) for name, p in self.current.named_parameters()]
        self.start_theta = torch.cat([p.detach().flatten() for p in self.current.parameters()])
        self.original_prefix = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.pts, self.sf = points_and_operators["train"]
        self.cache = prefix_cache(self.model, self.pts.xy_all, self.c, self.s0, self.params)
        self.install(self.start_theta)

    def parameter_state(self, theta):
        if theta.shape != self.start_theta.shape:
            raise ValueError("Current-increment parameter shape mismatch")
        state, offset = {}, 0
        for name, shape, size in self.shapes:
            state[name] = theta[offset:offset + size].reshape(shape)
            offset += size
        return state

    def probe(self, theta):
        state = self.parameter_state(theta)
        current = self.current

        class Probe:
            total_steps = current.total_steps
            multiplier_scale = current.multiplier_scale

            def __call__(self, xy):
                return torch.func.functional_call(current, state, (xy,))

        return Probe()

    def fields(self, theta):
        return current_residuals(self.probe(theta), self.cache, self.pts, self.c,
                                 self.s0, self.sf, self.params, self.step)

    def residual(self, theta):
        return weighted_vector(self.fields(theta)[1], self.pts, self.spec, self.sf)

    def install(self, theta):
        self.parameter_state(theta)
        if not bool(torch.isfinite(theta).all()):
            raise FloatingPointError("Nonfinite current-increment parameters")
        self.vec = self.base = None
        self.theta = theta.detach().clone().requires_grad_(True)
        self.vec = self.residual(self.theta)
        self.base = self.vec.detach()

    def jv(self, direction):
        return torch.func.jvp(self.residual, (self.theta.detach(),), (direction,))[1].detach()

    def jt(self, value):
        return torch.autograd.grad(self.vec, self.theta, value, retain_graph=True)[0].detach()

    @torch.no_grad()
    def assemble(self, theta):
        """Return a full spatial model with the supplied current-increment parameters."""
        state = self.parameter_state(theta)
        model, current = deepcopy(self.model), deepcopy(self.current)
        for name, parameter in current.named_parameters():
            parameter.copy_(state[name])
        model.increments.append(current)
        return model

    def protected_unchanged(self):
        return (all(torch.equal(value, self.model.state_dict()[name])
                    for name, value in self.original_prefix.items())
                and torch.equal(self.start_theta,
                                torch.cat([p.detach().flatten() for p in self.current.parameters()]))
                and not any(p.requires_grad for p in self.model.parameters()))
