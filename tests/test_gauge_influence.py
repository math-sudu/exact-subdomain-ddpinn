"""Endpoint work, signed traction and reciprocal source work of the gauge."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from pilot import config as C
from pilot.gauge_influence import CircularGaugeInfluence


def gauge(radius=1.3, angles=(.27, 2.14), modes=11, dip=37.):
    endpoints = radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return CircularGaugeInfluence(C.material(2.625, dip), endpoints, modes)


def wall_traction(auxiliary, angles):
    normal = np.column_stack((np.cos(angles), np.sin(angles)))
    stress = auxiliary.stress(auxiliary.radius * normal)
    return -np.column_stack((stress[:, 0] * normal[:, 0] + stress[:, 3] * normal[:, 1],
                             stress[:, 3] * normal[:, 0] + stress[:, 1] * normal[:, 1]))


@pytest.mark.parametrize("angles", [(.27, 2.14), (np.pi / 2, 3 * np.pi / 2)])
def test_actual_wall_load_is_work_conjugate_to_general_retained_trace(angles):
    auxiliary = gauge(angles=angles)
    rng = np.random.default_rng(41)
    coefficients = rng.normal(size=(2, auxiliary.modes + 1)) + 1j * rng.normal(
        size=(2, auxiliary.modes + 1))
    coefficients[:, 0] = [3.1, -.6]  # The mean translation must do no work.
    def trace(theta):
        phase = np.exp(1j * np.asarray(theta)[:, None] * np.arange(1, auxiliary.modes + 1))
        return coefficients[:, 0].real + 2 * np.real(phase @ coefficients[:, 1:].T)
    theta = (np.arange(256) + .31) * 2 * np.pi / 256
    actual_work = 2 * np.pi * auxiliary.radius * np.mean(np.sum(wall_traction(auxiliary, theta) * trace(theta), axis=-1))
    endpoints = trace(angles)
    shortening = -(endpoints[1] - endpoints[0]) @ auxiliary.direction
    np.testing.assert_allclose(actual_work, shortening, atol=2e-12, rtol=2e-12)


def test_wall_traction_has_the_endpoint_force_sign_and_actual_radius():
    auxiliary = gauge(radius=1.7, modes=9)
    theta = (np.arange(128) + .19) * 2 * np.pi / 128
    ends = np.arctan2(auxiliary.endpoints[:, 1], auxiliary.endpoints[:, 0])
    harmonics = np.arange(1, auxiliary.modes + 1)
    density = (np.cos((theta[:, None] - ends[0]) * harmonics)
               - np.cos((theta[:, None] - ends[1]) * harmonics)).sum(axis=1)
    expected = density[:, None] * auxiliary.direction / (np.pi * auxiliary.radius)
    np.testing.assert_allclose(wall_traction(auxiliary, theta), expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_array_equal(auxiliary.traction_coefficients[:, 0], 0.)
    np.testing.assert_allclose(expected.mean(axis=0), 0., atol=1e-14)


@pytest.mark.parametrize("dip", [17., 45.])
def test_displacement_and_four_component_stress_obey_the_actual_material(dip):
    auxiliary = gauge(dip=dip)
    material = C.material(2.625, dip)
    stiffness = material.plane_strain_stiffness4() / C.E_REF
    np.testing.assert_allclose(auxiliary.material.plane_strain_stiffness4(), stiffness, rtol=2e-14, atol=2e-14)
    xy = auxiliary.radius * np.array([[1.2, .3], [-.6, 1.4], [-1.5, -.4], [.4, -1.3]])
    h = 2e-5 * auxiliary.radius
    derivatives = []
    for direction in np.eye(2):
        derivatives.append((-auxiliary.displacement(xy + 2 * h * direction)
                            + 8 * auxiliary.displacement(xy + h * direction)
                            - 8 * auxiliary.displacement(xy - h * direction)
                            + auxiliary.displacement(xy - 2 * h * direction)) / (12 * h))
    dx, dy = derivatives
    strain = np.column_stack((dx[:, 0], dy[:, 1], np.zeros(len(xy)), dy[:, 0] + dx[:, 1]))
    stress = auxiliary.stress(xy)
    np.testing.assert_allclose(stress, strain @ stiffness.T, rtol=2e-8, atol=2e-10)
    full_stress = np.zeros((len(xy), 6))
    full_stress[:, [0, 1, 2, 5]] = stress
    full_strain = full_stress @ auxiliary.material.full_compliance().T
    np.testing.assert_allclose(full_strain[:, 2:5], 0., atol=2e-14)


def test_reciprocal_source_work_with_zero_exterior_trace_and_nonzero_zz_source():
    auxiliary = gauge(modes=9)
    a, b = auxiliary.radius, 3.2
    nodes, weights = np.polynomial.legendre.leggauss(96)
    radius = a + (b - a) * (nodes + 1) / 2
    theta = np.arange(512) * 2 * np.pi / 512
    r, angle = np.meshgrid(radius, theta, indexing="ij")
    xy = np.stack((r * np.cos(angle), r * np.sin(angle)), axis=-1)
    # A compatible strain whose displacement vanishes at r=b.
    f, df = (b - r)**2, -2 * (b - r)
    shape = np.stack((np.cos(angle) + .27 * np.sin(3 * angle),
                      .4 * np.sin(2 * angle) + .2 * np.cos(5 * angle)), axis=-1)
    dshape = np.stack((-np.sin(angle) + .81 * np.cos(3 * angle),
                       .8 * np.cos(2 * angle) - np.sin(5 * angle)), axis=-1)
    dx = df[..., None] * np.cos(angle)[..., None] * shape - (f / r)[..., None] * np.sin(angle)[..., None] * dshape
    dy = df[..., None] * np.sin(angle)[..., None] * shape + (f / r)[..., None] * np.cos(angle)[..., None] * dshape
    plastic = np.stack((dx[..., 0], dy[..., 1], np.zeros_like(r), dy[..., 0] + dx[..., 1]), axis=-1)
    # Adding C^-1 (0,0,q,0) changes only physical sigma_zz, so the same
    # displacement still solves equilibrium and both traction conditions.
    stress = auxiliary.stress(xy)
    q = (b - r) * stress[..., 2]
    zz_compliance = np.linalg.solve(auxiliary.material.plane_strain_stiffness4(), [0., 0., 1., 0.])
    plastic += q[..., None] * zz_compliance
    area = weights[:, None] * (b - a) / 2 * r * (2 * np.pi / len(theta))
    actual = np.sum(np.sum(stress * plastic, axis=-1) * area)
    ends = np.arctan2(auxiliary.endpoints[:, 1], auxiliary.endpoints[:, 0])
    endpoint_u = (b - a)**2 * np.column_stack((np.cos(ends) + .27 * np.sin(3 * ends),
                                             .4 * np.sin(2 * ends) + .2 * np.cos(5 * ends)))
    expected = -(endpoint_u[1] - endpoint_u[0]) @ auxiliary.direction
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
    zz_work = np.sum(stress[..., 2] * plastic[..., 2] * area)
    assert zz_work > 0  # Dropping zz breaks this otherwise unchanged response.
    assert not np.isclose(actual - zz_work, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("endpoints", [[[1., 0.], [1., 0.]], [[1., 0.], [0., 2.]]])
def test_invalid_endpoint_geometry_is_rejected(endpoints):
    with pytest.raises(ValueError):
        CircularGaugeInfluence(C.material(2.625, 45.), endpoints, 8)
