"""Neural parameter coordinates and matrix-free trust-region least squares."""

import numpy as np
import torch
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
