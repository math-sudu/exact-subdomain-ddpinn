"""A resolved polar projection layer for the causal mixed PINN fields.

The trainable pointwise networks supply nodal values. A fixed Chebyshev--Fourier
projection defines the spatial fields and their derivatives between nodes.
An optional linear spectral-feature branch supplies stress corrections in
that same space without passing them through the pointwise MLP parameter map.
"""

import torch

from .causal_mixed import CausalMixedPINN, IncrementFields
from .plasticity import WeakPlaneParameters


class CircularGridEvaluation:
    """Evaluate fixed, uniformly sampled rings with inverse Fourier transforms.

    Each group has shape (rings, angles, 2). Rings may have distinct angular
    origins, as in the staggered training quadrature. All coordinate factors
    are fixed; gradients with respect to the projected coefficients remain live.
    """

    def __init__(self, increment, groups):
        self.groups = []
        modes = increment.angular_indices
        for xy in groups:
            radius = xy[:, 0].norm(dim=-1)
            origin = torch.atan2(xy[:, 0, 1], xy[:, 0, 0])
            count = xy.shape[1]
            angle = origin[:, None]+2*torch.pi*torch.arange(
                count, device=xy.device, dtype=xy.dtype)/count
            expected = radius[:, None, None]*torch.stack((angle.cos(), angle.sin()), -1)
            if count <= 2*increment.angular_modes or not torch.allclose(xy, expected, rtol=1e-12, atol=1e-12):
                raise ValueError("Grid evaluation requires resolved, uniformly sampled circular rings")
            coordinate = 2*(radius-1)/(increment.input_scale-1)-1
            radial = [torch.ones_like(coordinate), coordinate]
            derivative = [torch.zeros_like(coordinate), torch.ones_like(coordinate)]
            second = [torch.zeros_like(coordinate), torch.zeros_like(coordinate)]
            for _ in range(2, increment.radial_order+1):
                second.append(4*derivative[-1]+2*coordinate*second[-1]-second[-2])
                derivative.append(2*radial[-1]+2*coordinate*derivative[-1]-derivative[-2])
                radial.append(2*coordinate*radial[-1]-radial[-2])
            phase = torch.exp(1j*origin[:, None]*modes[None])
            basis = torch.stack(radial, -1).to(phase.dtype)
            dbasis = (2/(increment.input_scale-1))*torch.stack(derivative, -1).to(phase.dtype)
            ddbasis = (2/(increment.input_scale-1))**2*torch.stack(second, -1).to(phase.dtype)
            direction = xy/radius[:, None, None]
            self.groups.append((basis, dbasis, ddbasis, phase, count, direction, radius))
        self.modes = modes

    def __call__(self, coefficients, *, second=False):
        fields, dx, dy, hessians = [], [], [], []
        for basis, dbasis, ddbasis, phase, count, direction, radius in self.groups:
            spectrum = torch.einsum("rj,jmc->rmc", basis, coefficients)*phase[..., None]
            radial_spectrum = torch.einsum("rj,jmc->rmc", dbasis, coefficients)*phase[..., None]
            value = torch.fft.irfft(spectrum, n=count, dim=1, norm="forward")
            dr = torch.fft.irfft(radial_spectrum, n=count, dim=1, norm="forward")
            dt = torch.fft.irfft(spectrum*(1j*self.modes)[None, :, None],
                                 n=count, dim=1, norm="forward")/radius[:, None, None]
            fields.append(value.flatten(0, 1))
            dx.append((dr*direction[..., :1]-dt*direction[..., 1:]).flatten(0, 1))
            dy.append((dr*direction[..., 1:]+dt*direction[..., :1]).flatten(0, 1))
            if second:
                drr = torch.fft.irfft(torch.einsum("rj,jmc->rmc", ddbasis, coefficients)
                                     *phase[..., None], n=count, dim=1, norm="forward")
                dtt = torch.fft.irfft(spectrum*(-self.modes.square())[None, :, None],
                                     n=count, dim=1, norm="forward")/radius[:, None, None]**2
                drt = torch.fft.irfft(radial_spectrum*(1j*self.modes)[None, :, None],
                                     n=count, dim=1, norm="forward")/radius[:, None, None]
                c, s = direction[..., :1], direction[..., 1:]
                tangential = dtt+dr/radius[:, None, None]
                mixed = drt-dt/radius[:, None, None]
                xx = c*c*drr+s*s*tangential-2*c*s*mixed
                yy = s*s*drr+c*c*tangential+2*c*s*mixed
                xy = c*s*(drr-tangential)+(c*c-s*s)*mixed
                hessians.append(torch.stack((torch.stack((xx, xy)),
                                              torch.stack((xy, yy)))).flatten(2, 3))
        result = torch.cat(fields), torch.stack((torch.cat(dx), torch.cat(dy)))
        return (*result, torch.cat(hessians, dim=2)) if second else result


class SpectralIncrementFields(IncrementFields):
    def __init__(self, hidden, input_scale, total_steps, multiplier_scale,
                 radial_order, angular_modes, angular_samples, centrosymmetric=False,
                 elastic_background=None, stress_coefficient_correction=False):
        super().__init__(hidden, input_scale, total_steps, multiplier_scale)
        self.radial_order = int(radial_order)
        self.angular_modes = int(angular_modes)
        self.angular_samples = int(angular_samples)
        self.centrosymmetric = bool(centrosymmetric)
        self.elastic_background = elastic_background
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
        self.register_parameter("stress_spectral_correction", None)
        if stress_coefficient_correction:
            self.stress_spectral_correction = torch.nn.Parameter(multiplier_scale.new_zeros(
                (self.radial_order+1, self.angular_modes+1, 4, 2)))
        if self.elastic_background is not None:
            # Initializers predict total increments. Store their fixed background
            # subtraction separately from the trainable neural correction.
            self.register_buffer("background_subtraction", multiplier_scale.new_zeros(
                (self.radial_order+1, self.angular_modes+1, 8, 2)))
        self._coefficient_cache = None

    def project(self, nodal_values):
        values = nodal_values.reshape(self.radial_order+1, self.angular_samples, -1)
        fourier = torch.fft.rfft(values, dim=1)[:, :self.angular_modes+1] / self.angular_samples
        coefficients = torch.einsum("ij,jmc->imc", self.radial_transform.to(fourier.dtype), fourier)
        return self.physical_modes(coefficients)

    def physical_modes(self, coefficients, displacement_channels=2):
        if self.centrosymmetric:
            odd_modes = (self.angular_indices.to(torch.int64) % 2).bool()
            odd_channels = torch.arange(coefficients.shape[-1], device=coefficients.device) < displacement_channels
            coefficients = coefficients*(odd_modes[:, None] == odd_channels[None])[None]
        return coefficients

    def coefficients(self):
        if self._coefficient_cache is not None:
            return self._coefficient_cache
        u, stress, gamma = super().forward(self.projection_nodes)
        coefficients = self.project(torch.cat((u, stress, gamma), -1))
        if self.stress_spectral_correction is not None:
            # The mean mode is real; remaining modes have conjugate partners.
            real, imaginary = self.stress_spectral_correction.unbind(-1)
            imaginary = imaginary*(self.angular_indices > 0)[None, :, None]
            correction = self.physical_modes(torch.complex(real, imaginary), 0)
            coefficients = torch.cat((coefficients[..., :2],
                                      coefficients[..., 2:6]+correction,
                                      coefficients[..., 6:]), -1)
        if self.elastic_background is not None:
            coefficients = coefficients-torch.view_as_complex(self.background_subtraction)
        return coefficients

    @torch.no_grad()
    def initialize_from_total(self, initial):
        for name in ("kinematic_trunk", "kinematic_head", "stress_trunk", "stress_head"):
            getattr(self, name).load_state_dict(getattr(initial, name).state_dict())
        initial_correction = getattr(initial, "stress_spectral_correction", None)
        if initial_correction is not None:
            if self.stress_spectral_correction is None:
                raise ValueError("The initializer contains a spectral correction branch")
            self.stress_spectral_correction.copy_(initial_correction)
        elif self.stress_spectral_correction is not None:
            self.stress_spectral_correction.zero_()
        background = self.project(torch.cat(self.elastic_background(self.projection_nodes), -1))
        self.background_subtraction.copy_(torch.view_as_real(background))
        self._coefficient_cache = None

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
        values = fields[:, :2], fields[:, 2:6], fields[:, 6:]
        if self.elastic_background is not None:
            values = tuple(value+base for value, base in zip(values, self.elastic_background(xy)))
        return values

    def spatial_fields(self, xy, background_fields=None, *, grid=None, coefficients=None):
        """Differentiate the finite basis while retaining parameter gradients."""
        if coefficients is None:
            coefficients = self.coefficients()
        if grid is None:
            fields, derivatives = self._spatial_expansion(coefficients, xy)
        else:
            fields, derivatives = grid(coefficients)
        values = fields[:, :2], fields[:, 2:6], fields[:, 6:]
        derivatives = derivatives[..., :2], derivatives[..., 2:6], derivatives[..., 6:]
        if self.elastic_background is not None:
            base, dbase = (self.elastic_background.spatial_fields(xy)
                           if background_fields is None else background_fields)
            values = tuple(value+background for value, background in zip(values, base))
            derivatives = tuple(value+background for value, background in zip(derivatives, dbase))
        return values, derivatives

    def _spatial_expansion(self, coefficients, xy):
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
        return fields, torch.stack((dx, dy))


class SpectralCausalPINN(CausalMixedPINN):
    def __init__(self, hidden, input_scale, total_steps, stiffness, projection, elastic_case=None,
                 increment_hidden=None):
        super().__init__(hidden, input_scale, total_steps, stiffness)
        self.projection = dict(projection)
        self.elastic_case = elastic_case
        self.increment_hidden = increment_hidden or {}

    def make_increment(self):
        from .elastic_background import CircularElasticIncrement
        background = (None if self.elastic_case is None else
                      CircularElasticIncrement(self.elastic_case, self.total_steps, self.multiplier_scale))
        hidden = self.increment_hidden.get(str(self.n_steps + 1), self.hidden)
        return SpectralIncrementFields(hidden, self.input_scale, self.total_steps,
                                       self.multiplier_scale, **self.projection,
                                       elastic_background=background).to(self.multiplier_scale)

    def cache_projection(self):
        for increment in self.increments:
            increment.cache_projection()


class ReturnedCausalPINN(SpectralCausalPINN):
    """Neural displacements with an own-step, differentiable local stress return."""

    def __init__(self, *args, insitu, params, **kwargs):
        super().__init__(*args, **kwargs)
        if self.elastic_case is None:
            raise ValueError("The returned history requires the circular elastic background")
        if params.joint_psi_deg != 0:
            raise ValueError("This return map requires zero weak-plane dilation")
        self.register_buffer("return_stiffness", args[3].detach().clone())
        self.register_buffer("return_insitu", insitu.detach().clone())
        self.return_params = params

    def returned_history(self, xy, *, check_branch=True, grid=None):
        """Replay every local return; reject an unsupported path by default.

        Displacement-only evaluation of individual increments is kinematic.
        Use this entry point when claiming a valid constitutive history at xy.
        check_branch=False supports transformed training calls; unsupported
        active returns remain nonfinite and cannot serve as valid field data.
        """
        from .joint_return import joint_return
        displacement = xy.new_zeros((len(xy), 2))
        strain = plastic = xy.new_zeros((len(xy), 4))
        rows = []
        background = (self.increments[0].elastic_background.spatial_fields(xy)
                      if grid is not None else None)
        for increment in self.increments:
            values, derivatives = increment.spatial_fields(xy, background, grid=grid)
            du = derivatives[0]
            displacement = displacement+values[0]
            strain = strain+torch.stack((du[0, :, 0], du[1, :, 1],
                torch.zeros_like(du[0, :, 0]), du[1, :, 0]+du[0, :, 1]), -1)
            returned = joint_return(self.return_insitu+(strain-plastic)@self.return_stiffness.T,
                                    self.return_stiffness, self.return_params, check_branch=check_branch)
            plastic = plastic+returned["plastic"]
            gamma = torch.stack((torch.zeros_like(returned["gamma"]), returned["gamma"]), -1)
            rows.append((displacement, returned["stress"]-self.return_insitu, gamma, plastic))
        return tuple(torch.stack([row[i] for row in rows], dim=1) for i in range(4))

    def forward(self, xy):
        # vmap/JVP use a functional residual; explicit history queries validate.
        return self.returned_history(xy, check_branch=False)[:3]


def make_causal_model(settings, stiffness, case=None):
    args = (settings["hidden"], settings["radius"], settings["load_steps"], stiffness)
    representation = settings.get("representation", "pointwise")
    if settings.get("constitutive_update") not in (None, "joint_return"):
        raise ValueError(f"Unknown constitutive update: {settings['constitutive_update']}")
    if settings.get("constitutive_update") == "joint_return" and representation != "spectral":
        raise ValueError("The current local-return solver requires the spectral displacement representation")
    if representation == "spectral":
        elastic_case = None
        if settings.get("elastic_prefix") == "circular":
            if case is None:
                raise ValueError("Circular unloading requires the actual material and opening case")
            elastic_case = case
        if settings.get("constitutive_update") == "joint_return":
            from . import config as C
            material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
            sx, sy, txy = C.nondim_insitu()
            insitu = stiffness.new_tensor((sx, sy,
                material.plane_strain_sigma_zz(C.SX, C.SY, C.TXY)/C.SIGMA_REF, txy))
            params = WeakPlaneParameters(**dict(settings["physics"],
                joint_normal_deg=case["joint_normal_deg"])).checked()
            return ReturnedCausalPINN(*args, settings["projection"], elastic_case,
                                      insitu=insitu, params=params,
                                      increment_hidden=settings.get("increment_hidden")).to(stiffness)
        return SpectralCausalPINN(*args, settings["projection"], elastic_case,
                                 increment_hidden=settings.get("increment_hidden")).to(stiffness)
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
    model = make_causal_model(settings, saved["c"].to(device), case)
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
    if settings.get("elastic_prefix") == "circular":
        spec.update(elastic_prefix="circular", case=case)
    if "constitutive_update" in settings:
        spec["constitutive_update"] = settings["constitutive_update"]
    if "increment_hidden" in settings:
        spec["increment_hidden"] = settings["increment_hidden"]
    bundle = dict(spec=spec, c=saved["c"], s0=saved["s0"],
                  params=WeakPlaneParameters(**spec["physics"]).checked())
    return bundle, saved, model
