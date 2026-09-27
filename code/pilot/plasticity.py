"""Two yield surfaces, non-associated flow and complementarity."""

from dataclasses import dataclass
import numpy as np
import torch
from . import config as C


@dataclass(frozen=True)
class WeakPlaneParameters:
    """Matrix and weak-plane plasticity parameters in MPa and degrees."""

    matrix_cohesion_mpa: float = 2.0
    matrix_phi_deg: float = 32.5
    matrix_psi_deg: float = 2.5
    joint_cohesion_mpa: float = 0.2
    joint_phi_deg: float = 25.0
    joint_psi_deg: float = 0.0
    joint_normal_deg: float = 135.0
    matrix_smoothing_mpa: float = 0.01
    joint_smoothing_mpa: float = 0.01
    gamma_scale: float = 0.05
    complementarity_smoothing: float = 1.0e-8
    exterior_yield_tolerance: float = 5.0e-3

    def checked(self) -> "WeakPlaneParameters":
        if self.matrix_cohesion_mpa <= 0.0 or self.joint_cohesion_mpa < 0.0:
            raise ValueError("cohesions must be non-negative and matrix c positive")
        for name, phi, psi in (
            ("matrix", self.matrix_phi_deg, self.matrix_psi_deg),
            ("joint", self.joint_phi_deg, self.joint_psi_deg),
        ):
            if not 0.0 <= psi <= phi < 90.0:
                raise ValueError(f"{name} angles must satisfy 0 <= psi <= phi < 90")
        if self.matrix_smoothing_mpa <= 0.0 or self.joint_smoothing_mpa <= 0.0:
            raise ValueError("yield smoothing widths must be positive")
        if self.gamma_scale <= 0.0:
            raise ValueError("gamma_scale must be positive")
        return self

    @property
    def stress_scale_mpa(self) -> float:
        return C.SIGMA_REF / 1.0e6

    @property
    def matrix_cohesion(self) -> float:
        return self.matrix_cohesion_mpa / self.stress_scale_mpa

    @property
    def joint_cohesion(self) -> float:
        return self.joint_cohesion_mpa / self.stress_scale_mpa

    @property
    def matrix_smoothing(self) -> float:
        return self.matrix_smoothing_mpa / self.stress_scale_mpa

    @property
    def joint_smoothing(self) -> float:
        return self.joint_smoothing_mpa / self.stress_scale_mpa


def _dp_constants(phi_deg: float, cohesion: float = 0.0) -> tuple[float, float]:
    tangent = np.tan(np.deg2rad(phi_deg))
    denominator = np.sqrt(9.0 + 12.0 * tangent * tangent)
    return float(tangent / denominator), float(3.0 * cohesion / denominator)


def _deviatoric(stress: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = stress[..., :3].mean(dim=-1, keepdim=True)
    normal_dev = stress[..., :3] - mean
    j2 = 0.5 * torch.sum(normal_dev.square(), dim=-1) + stress[..., 3].square()
    return normal_dev, j2


def matrix_yield(stress: torch.Tensor, params: WeakPlaneParameters) -> torch.Tensor:
    """Rounded plane-strain-matched DP approximation to matrix MC."""

    params = params.checked()
    alpha, k = _dp_constants(params.matrix_phi_deg, params.matrix_cohesion)
    _, j2 = _deviatoric(stress)
    q = torch.sqrt(j2 + params.matrix_smoothing ** 2) - params.matrix_smoothing
    i1 = torch.sum(stress[..., :3], dim=-1)
    return q + alpha * i1 - k


def matrix_yield_gradient(
    stress: torch.Tensor, params: WeakPlaneParameters
) -> torch.Tensor:
    params = params.checked()
    alpha, _ = _dp_constants(params.matrix_phi_deg)
    normal_dev, j2 = _deviatoric(stress)
    denominator = torch.sqrt(j2 + params.matrix_smoothing ** 2)
    gradient = torch.empty_like(stress)
    gradient[..., :3] = normal_dev / (2.0 * denominator[..., None]) + alpha
    gradient[..., 3] = stress[..., 3] / denominator
    return gradient


def matrix_flow_direction(
    stress: torch.Tensor, params: WeakPlaneParameters
) -> torch.Tensor:
    params = params.checked()
    alpha, _ = _dp_constants(params.matrix_psi_deg)
    normal_dev, j2 = _deviatoric(stress)
    denominator = torch.sqrt(j2 + params.matrix_smoothing ** 2)
    direction = torch.empty_like(stress)
    direction[..., :3] = normal_dev / (2.0 * denominator[..., None]) + alpha
    direction[..., 3] = stress[..., 3] / denominator
    return direction


def _joint_geometry(params: WeakPlaneParameters, like: torch.Tensor):
    beta = np.deg2rad(params.joint_normal_deg)
    nx, ny = float(np.cos(beta)), float(np.sin(beta))
    normal_gradient = torch.tensor(
        [nx * nx, ny * ny, 0.0, 2.0 * nx * ny], dtype=like.dtype, device=like.device
    )
    shear_gradient = torch.tensor([
        -nx * ny,
        nx * ny,
        0.0,
        nx * nx - ny * ny,
    ], dtype=like.dtype, device=like.device)
    return normal_gradient, shear_gradient


def _joint_stresses(stress: torch.Tensor, params: WeakPlaneParameters):
    normal_gradient, shear_gradient = _joint_geometry(params, stress)
    sigma_normal = torch.sum(stress * normal_gradient, dim=-1)
    shear = torch.sum(stress * shear_gradient, dim=-1)
    return sigma_normal, shear, normal_gradient, shear_gradient


def joint_yield(stress: torch.Tensor, params: WeakPlaneParameters) -> torch.Tensor:
    params = params.checked()
    sigma_normal, shear, _, _ = _joint_stresses(stress, params)
    rounded_shear = torch.sqrt(shear.square() + params.joint_smoothing ** 2) \
        - params.joint_smoothing
    return (rounded_shear + sigma_normal * np.tan(np.deg2rad(params.joint_phi_deg))
            - params.joint_cohesion)


def joint_yield_gradient(
    stress: torch.Tensor, params: WeakPlaneParameters
) -> torch.Tensor:
    params = params.checked()
    _, shear, normal_gradient, shear_gradient = _joint_stresses(stress, params)
    shear_factor = shear / torch.sqrt(shear.square() + params.joint_smoothing ** 2)
    return (shear_factor[..., None] * shear_gradient
            + np.tan(np.deg2rad(params.joint_phi_deg)) * normal_gradient)


def joint_flow_direction(
    stress: torch.Tensor, params: WeakPlaneParameters
) -> torch.Tensor:
    params = params.checked()
    _, shear, normal_gradient, shear_gradient = _joint_stresses(stress, params)
    shear_factor = shear / torch.sqrt(shear.square() + params.joint_smoothing ** 2)
    return (shear_factor[..., None] * shear_gradient
            + np.tan(np.deg2rad(params.joint_psi_deg)) * normal_gradient)


def _fischer_burmeister(a: torch.Tensor, b: torch.Tensor,
                        smoothing: float) -> torch.Tensor:
    return torch.sqrt(a.square() + b.square() + smoothing ** 2) - a - b


def _spatial_direction(xy: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.tensor(
        (1.0, 0.0) if axis == 0 else (0.0, 1.0), dtype=xy.dtype, device=xy.device
    ).expand_as(xy)


def _traction4(stress: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    tx = stress[..., 0] * normal[..., 0] + stress[..., 3] * normal[..., 1]
    ty = stress[..., 3] * normal[..., 0] + stress[..., 1] * normal[..., 1]
    return torch.stack((tx, ty), dim=-1)


def _exterior_field_from_trace(
    displacement: np.ndarray,
    mat,
    radius: float,
    m_max: int | None,
):
    from sf_core.exterior_field import ExteriorField

    n = displacement.shape[1]
    coefficients = np.fft.rfft(displacement, axis=1) / n
    last = min(coefficients.shape[1] - 1,
               m_max if m_max is not None else coefficients.shape[1] - 1)
    potential = np.zeros((2, last), dtype=complex)
    material_map = np.array(
        [[mat.p1, mat.p2], [mat.q1, mat.q2]], dtype=complex
    )
    inverse_map = np.linalg.inv(material_map)
    for harmonic in range(1, last + 1):
        potential[:, harmonic - 1] = inverse_map @ np.conj(
            coefficients[:, harmonic]
        )
    return ExteriorField(mat, radius, (potential[0], potential[1]))
