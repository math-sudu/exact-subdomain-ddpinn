"""A resolved polar projection layer for the causal mixed PINN fields.

The trainable pointwise networks supply nodal values. A fixed Chebyshev--Fourier
projection defines the spatial fields and their derivatives between nodes.
This prevents unresolved network features from producing hidden strain spikes.
"""

import torch

from .causal_mixed import CausalMixedPINN, IncrementFields
from .plasticity import WeakPlaneParameters


class SpectralIncrementFields(IncrementFields):
    def __init__(self, hidden, input_scale, total_steps, multiplier_scale,
                 radial_order, angular_modes, angular_samples):
        super().__init__(hidden, input_scale, total_steps, multiplier_scale)
        self.radial_order = int(radial_order)
        self.angular_modes = int(angular_modes)
        self.angular_samples = int(angular_samples)
        if not 0 < self.angular_modes < self.angular_samples // 2:
            raise ValueError("Angular modes must exclude the projection-grid Nyquist mode")
        tensor = dict(dtype=multiplier_scale.dtype, device=multiplier_scale.device)
        index = torch.arange(self.radial_order + 1, **tensor)
        angle = torch.arange(self.angular_samples, **tensor) * (2*torch.pi/self.angular_samples)
        radius = 1 + (self.input_scale-1)*(1+torch.cos(torch.pi*index/self.radial_order))/2
        nodes = radius[:, None, None] * torch.stack((angle.cos(), angle.sin()), -1)[None]
        endpoints = torch.ones_like(index)
        endpoints[0] = endpoints[-1] = .5
        dct = ((2/self.radial_order) * endpoints[:, None] * endpoints[None]
               * torch.cos(torch.pi*index[:, None]*index[None]/self.radial_order))
        self.register_buffer("projection_nodes", nodes.reshape(-1, 2), persistent=False)
        self.register_buffer("radial_transform", dct, persistent=False)
        self.register_buffer("angular_indices", torch.arange(self.angular_modes+1, **tensor), persistent=False)
        self._coefficient_cache = None

    def project(self, nodal_values):
        values = nodal_values.reshape(self.radial_order+1, self.angular_samples, -1)
        fourier = torch.fft.rfft(values, dim=1)[:, :self.angular_modes+1] / self.angular_samples
        return torch.einsum("ij,jmc->imc", self.radial_transform.to(fourier.dtype), fourier)

    def coefficients(self):
        if self._coefficient_cache is not None:
            return self._coefficient_cache
        u, stress, gamma = super().forward(self.projection_nodes)
        return self.project(torch.cat((u, stress, gamma), -1))

    @torch.no_grad()
    def cache_projection(self):
        if any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError("Only a frozen increment can cache its projection")
        self._coefficient_cache = self.coefficients().detach()

    def requires_grad_(self, requires_grad=True):
        if requires_grad:
            self._coefficient_cache = None
        return super().requires_grad_(requires_grad)

    def _apply(self, fn, recurse=True):
        self._coefficient_cache = None
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._coefficient_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def evaluate(self, coefficients, xy):
        radius = xy.norm(dim=-1)
        coordinate = 2*(radius-1)/(self.input_scale-1)-1
        radial = [torch.ones_like(coordinate), coordinate]
        for _ in range(2, self.radial_order+1):
            radial.append(2*coordinate*radial[-1]-radial[-2])
        basis = torch.stack(radial, -1)
        angle = torch.atan2(xy[:, 1], xy[:, 0])
        phase = torch.exp(1j*angle[:, None]*self.angular_indices[None])
        value = torch.einsum("nj,jmc,nm->nc", basis.to(coefficients.dtype), coefficients, phase)
        mean = basis @ coefficients[:, 0].real
        return 2*value.real-mean

    def forward(self, xy):
        fields = self.evaluate(self.coefficients(), xy)
        return fields[:, :2], fields[:, 2:6], fields[:, 6:]

    def spatial_fields(self, xy):
        """Differentiate the finite basis while retaining parameter gradients."""
        coefficients = self.coefficients()
        radius = xy.norm(dim=-1)
        coordinate = 2*(radius-1)/(self.input_scale-1)-1
        radial = [torch.ones_like(coordinate), coordinate]
        radial_derivative = [torch.zeros_like(coordinate), torch.ones_like(coordinate)]
        for _ in range(2, self.radial_order+1):
            radial_derivative.append(2*radial[-1]+2*coordinate*radial_derivative[-1]-radial_derivative[-2])
            radial.append(2*coordinate*radial[-1]-radial[-2])
        basis = torch.stack(radial, -1).to(coefficients.dtype)
        dbasis = (2/(self.input_scale-1))*torch.stack(radial_derivative, -1).to(coefficients.dtype)
        angle = torch.atan2(xy[:, 1], xy[:, 0])
        phase = torch.exp(1j*angle[:, None]*self.angular_indices[None])
        amplitudes = torch.einsum("nj,jmc->nmc", basis, coefficients)
        radial_amplitudes = torch.einsum("nj,jmc->nmc", dbasis, coefficients)
        fields = 2*torch.einsum("nmc,nm->nc", amplitudes, phase).real-amplitudes[:, 0].real
        dr = 2*torch.einsum("nmc,nm->nc", radial_amplitudes, phase).real-radial_amplitudes[:, 0].real
        dt = 2*torch.einsum("nmc,nm->nc", amplitudes, phase*(1j*self.angular_indices)).real
        dx = dr*(xy[:, 0]/radius)[:, None]-dt*(xy[:, 1]/radius.square())[:, None]
        dy = dr*(xy[:, 1]/radius)[:, None]+dt*(xy[:, 0]/radius.square())[:, None]
        derivatives = torch.stack((dx, dy))
        return ((fields[:, :2], fields[:, 2:6], fields[:, 6:]),
                (derivatives[..., :2], derivatives[..., 2:6], derivatives[..., 6:]))


class SpectralCausalPINN(CausalMixedPINN):
    def __init__(self, hidden, input_scale, total_steps, stiffness, projection):
        super().__init__(hidden, input_scale, total_steps, stiffness)
        self.projection = dict(projection)

    def make_increment(self):
        return SpectralIncrementFields(self.hidden, self.input_scale, self.total_steps,
                                       self.multiplier_scale, **self.projection).to(self.multiplier_scale)

    def cache_projection(self):
        for increment in self.increments:
            increment.cache_projection()


def make_causal_model(settings, stiffness):
    args = (settings["hidden"], settings["radius"], settings["load_steps"], stiffness)
    representation = settings.get("representation", "pointwise")
    if representation == "spectral":
        return SpectralCausalPINN(*args, settings["projection"]).to(stiffness)
    if representation == "pointwise":
        return CausalMixedPINN(*args).to(stiffness)
    raise ValueError(f"Unknown field representation: {representation}")


def freeze_history(model):
    model.requires_grad_(False)
    if isinstance(model, SpectralCausalPINN):
        model.cache_projection()
    return model


def load_orientation_model(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    settings, case = saved["settings"], saved["case"]
    model = make_causal_model(settings, saved["c"].to(device))
    for _ in range(saved["model_steps"]):
        model.append_increment(copy_previous=False)
    model.load_state_dict(saved["model"])
    freeze_history(model)
    spec = dict(hidden=settings["hidden"], input_scale=settings["radius"],
                r_gamma=settings["radius"], load_steps=settings["load_steps"],
                physics=dict(settings["physics"], joint_normal_deg=case["joint_normal_deg"]),
                representation=settings.get("representation", "pointwise"))
    if "projection" in settings:
        spec["projection"] = settings["projection"]
    bundle = dict(spec=spec, c=saved["c"], s0=saved["s0"],
                  params=WeakPlaneParameters(**spec["physics"]).checked())
    return bundle, saved, model
