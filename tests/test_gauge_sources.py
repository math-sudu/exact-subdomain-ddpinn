"""Signed source integration and checkpoint-to-artifact behavior."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import analyze_gauge_sources as producer
from pilot import config as C, plasticity as P


def test_signed_work_keeps_zz_and_engineering_shear_with_integrable_densities():
    grid = producer.quadrature(3., 4, 16)
    stress = np.broadcast_to(np.array([[0., 0., 3., 5.], [0., 0., -7., 11.]])[:, None, None], (2, 4, 16, 4))
    sources = {name: np.zeros((4, 16, 2, 4)) for name in producer.SOURCE_NAMES}
    sources["joint_positive"][..., 3] = [2., 4.]
    sources["joint_negative"][..., 3] = [-1., -3.]
    sources["matrix"][..., 2] = [6., 2.]
    arrays = producer.integrate_source_work(stress, sources, grid)
    density = dict(joint_positive=[[10., 20.], [22., 44.]],
                   joint_negative=[[-5., -15.], [-11., -33.]],
                   matrix=[[18., 6.], [-42., -14.]],
                   source_total=[[23., 11.], [-31., -3.]])
    for name, values in density.items():
        expected = np.array(values) * C.U_REF * 1000
        np.testing.assert_allclose(arrays[f"{name}_mm"], expected * 8 * np.pi)
        np.testing.assert_allclose(arrays[f"density_{name}_mm"], np.broadcast_to(expected[..., None, None], (2, 2, 4, 16)))
        np.testing.assert_allclose(arrays[f"radial_density_{name}_mm"], expected[..., None] * grid.r * 2 * np.pi)
        np.testing.assert_allclose(arrays[f"angular_density_{name}_mm"], np.broadcast_to(expected[..., None] * 4, (2, 2, 16)))
        np.testing.assert_allclose(arrays[f"radial_density_{name}_mm"] @ grid.dr, arrays[f"{name}_mm"])
        np.testing.assert_allclose(arrays[f"angular_density_{name}_mm"].sum(-1) * grid.dtheta, arrays[f"{name}_mm"])


@pytest.fixture
def synthetic_checkpoint(monkeypatch, tmp_path):
    case = dict(name="synthetic", eh_over_ev=2.625, bedding_dip_deg=45.,
                opening_axes=[1., 1.], joint_normal_deg=0.)
    material = C.material(case["eh_over_ev"], case["bedding_dip_deg"])
    c = torch.tensor(material.plane_strain_stiffness4() / C.E_REF)
    params = P.WeakPlaneParameters(joint_normal_deg=0.)
    insitu = c.new_tensor([-.5, -1., -.4, 0.])
    total = c.new_tensor([[-.1, -.2, -.8, .4], [-.3, -.1, -.4, -.5]])
    gamma = c.new_tensor([[.2, .5], [.3, .8]])
    checkpoint = tmp_path / "input/synthetic/model.pt"
    calls = dict(fields=0, modes=[], elastic_steps=[])

    class History:
        multiplier_scale = c.new_tensor(.7)
        total_steps = 12
        def __init__(self):
            self.increments = [None] * 3  # Third increment is not completed.
        def __call__(self, xy):
            count = len(self.increments)
            assert count <= 2, "Queried an unfinished increment"
            calls["fields"] += 1
            steps = torch.arange(1, count + 1, dtype=xy.dtype)
            u = -xy[:, None] * xy.new_tensor([.1, .2]) * steps[None, :, None]
            return u, (total[:count] - insitu).expand(len(xy), -1, -1), gamma[:count].expand(len(xy), -1, -1)

    def load(path, device):
        assert path == checkpoint
        assert device == "cpu"
        return (dict(c=c, s0=insitu, params=params),
                dict(case=case, completed_steps=2, settings=dict(radius=3., representation="spectral", projection=dict(angular_modes=5))),
                History())

    class Influence:
        def __init__(self, material, endpoints, modes):
            calls["modes"].append(modes)
            np.testing.assert_allclose(endpoints[0], -endpoints[1])
        def stress(self, xy):
            return np.broadcast_to([0., 0., 3., 5.], (*xy.shape[:-1], 4))

    class Elastic:
        def __init__(self, case, total_steps, like):
            calls["elastic_steps"].append(total_steps)
        def __call__(self, xy):
            return -.02 * xy, None, None

    monkeypatch.setattr(producer, "load_orientation_model", load)
    monkeypatch.setattr(producer, "CircularGaugeInfluence", Influence)
    monkeypatch.setattr(producer, "CircularElasticIncrement", Elastic)
    return checkpoint, calls, params, total, gamma


def test_cli_preserves_step_angle_order_units_and_own_step_signed_sources(synthetic_checkpoint, tmp_path):
    checkpoint, calls, params, total, gamma = synthetic_checkpoint
    output = tmp_path / "output"
    producer.main(["--root", str(checkpoint.parents[1]), "--cases", "synthetic",
                   "--steps", "2", "1", "--angles", "104", "0", "90", "--output", str(output),
                   "--radial-quadrature", "4", "--angular-quadrature", "32", "--device", "cpu"])
    with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(row["load_step"]), float(row["gauge_deg"])) for row in rows] == [(k, a) for k in (2, 1) for a in (104., 0., 90.)]
    scale, area = C.U_REF * 1000, 8 * np.pi
    tau = total[:, 3].numpy()
    signed_joint = gamma[:, 1].numpy() * tau / np.sqrt(tau**2 + params.joint_smoothing**2)
    # The matrix component is evaluated independently at both actual stresses.
    s = total.numpy()
    dev = s[:, :3] - s[:, :3].mean(axis=1, keepdims=True)
    denominator = np.sqrt(.5 * (dev**2).sum(axis=1) + s[:, 3]**2 + params.matrix_smoothing**2)
    tangent = np.tan(np.deg2rad(params.matrix_psi_deg))
    alpha = tangent / np.sqrt(9 + 12 * tangent**2)
    matrix_work = gamma[:, 0].numpy() * (3 * (dev[:, 2] / (2 * denominator) + alpha) + 5 * s[:, 3] / denominator)
    for row in rows:
        step, angle = int(row["load_step"]), np.deg2rad(float(row["gauge_deg"]))
        positive = 5 * signed_joint[0] * area * scale
        negative = (5 * signed_joint[1] if step == 2 else 0.) * area * scale
        matrix = matrix_work[:step].sum() * area * scale
        retained = 2 * step * (.1 * np.cos(angle)**2 + .2 * np.sin(angle)**2) * scale
        elastic = .04 * step / 12 * scale
        expected = dict(joint_positive_mm=positive, joint_negative_mm=negative, matrix_mm=matrix,
                        source_total_mm=positive + negative + matrix, retained_mm=retained,
                        elastic_mm=elastic, imbalance_mm=retained - elastic - positive - negative - matrix)
        for key, value in expected.items():
            np.testing.assert_allclose(float(row[key]), value, rtol=2e-13, atol=2e-13)
    with np.load(output / "synthetic.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["steps"], [2, 1])
        np.testing.assert_array_equal(arrays["gauge_angle_deg"], [104., 0., 90.])
        assert arrays["density_source_total_mm"].shape == (3, 2, 4, 32)
        np.testing.assert_allclose((arrays["density_source_total_mm"] * arrays["area_weights"]).sum((-2, -1)), arrays["source_total_mm"])
        assert not any("imbalance" in key and "density" in key for key in arrays.files)
        metadata = json.loads(str(arrays["metadata_json"]))
        assert metadata["checkpoint"] == str(checkpoint.resolve())
        assert metadata["units"]["response"] == "mm"
        assert metadata["density_axes"] == ["gauge", "step", "radius", "source_angle"]
        assert metadata["strain_components"][-2:] == ["zz", "engineering_xy"]
        assert metadata["common_regions"]["definitions"] == ["|x|>|y|", "|y|>|x|"]
        for name in (*producer.SOURCE_NAMES, "source_total"):
            np.testing.assert_allclose(arrays[f"common_region_{name}_mm"].sum(-1),
                                       arrays[f"{name}_mm"], atol=1e-13)
    assert calls["modes"] == [5, 5, 5]
    assert calls["elastic_steps"] == [1]
    with (output / "common_regions.csv").open(newline="", encoding="utf-8") as stream:
        regions = list(csv.DictReader(stream))
    assert len(regions) == 2 * len(rows)
    for row, pair in zip(rows, zip(regions[::2], regions[1::2])):
        assert [item["region"] for item in pair] == list(producer.COMMON_REGION_NAMES)
        for name in (*producer.SOURCE_NAMES, "source_total"):
            np.testing.assert_allclose([float(item[f"{name}_mm"]) for item in pair],
                                       float(row[f"{name}_mm"]) / 2)
        step = int(row["load_step"])
        expected_activity = np.abs(signed_joint[:step]).sum()*area*(C.U_REF/C.A_TUNNEL)
        np.testing.assert_allclose([float(item["joint_slip_area_integral"]) for item in pair],
                                   expected_activity/2)
    with np.load(output / "synthetic.npz", allow_pickle=False) as arrays:
        expected_activity = np.abs(signed_joint).cumsum()[[1, 0]]*(C.U_REF/C.A_TUNNEL)
        np.testing.assert_allclose(arrays["accumulated_joint_slip_strain"],
                                   np.broadcast_to(expected_activity[:, None, None], (2, 4, 32)))

def test_common_regions_resolve_global_sides_and_preserve_signed_cancellation():
    theta = 2 * np.pi * np.arange(64) / 64
    density = np.stack((2 + 3 * np.cos(2 * theta) - 4 * np.sin(2 * theta),
                        np.cos(2 * theta), np.sin(2 * theta)))
    expected = [[2 * np.pi + 6, 2 * np.pi - 6], [2, -2], [0, 0]]
    np.testing.assert_allclose(producer.integrate_common_regions(density), expected, atol=1e-13)
    np.testing.assert_allclose(producer.integrate_oriented_regions(density, 45.)[..., ::-1],
                               expected, atol=1e-13)


@pytest.mark.parametrize("normal_angle", [0., 21.3, 45., 135.])
def test_fixed_regions_integrate_rotated_harmonics_and_signed_cancellation(normal_angle):
    theta = 2 * np.pi * np.arange(64) / 64
    beta = np.deg2rad(normal_angle)
    density = np.stack((2 + 3 * np.cos(2 * theta) - 4 * np.sin(2 * theta),
                        np.sin(2 * (theta - beta)), np.cos(3 * theta)))
    positive = np.array([2 * np.pi - 6 * np.sin(2 * beta) - 8 * np.cos(2 * beta), 2., 0.])
    total = np.array([4 * np.pi, 0., 0.])
    result = producer.integrate_oriented_regions(density, normal_angle)
    np.testing.assert_allclose(result, np.stack((positive, total - positive), -1), atol=1e-13)
    np.testing.assert_allclose(producer.integrate_oriented_regions(density, normal_angle + 180),
                               result, atol=1e-13)


def test_region_integral_counts_even_grid_nyquist_cosine_once():
    theta = 2 * np.pi * np.arange(12) / 12
    result = producer.integrate_oriented_regions(np.cos(6 * theta), 15.)
    np.testing.assert_allclose(result, [-2 / 3, 2 / 3], atol=1e-13)


def test_unfinished_step_is_rejected_before_any_field_query(synthetic_checkpoint):
    checkpoint, calls, *_ = synthetic_checkpoint
    with pytest.raises(ValueError, match="completed"):
        producer.analyze_checkpoint(checkpoint, [3])
    assert calls["fields"] == 0


def test_angular_grid_must_resolve_checkpoint_modes(synthetic_checkpoint):
    checkpoint, calls, *_ = synthetic_checkpoint
    with pytest.raises(ValueError, match="resolve"):
        producer.analyze_checkpoint(checkpoint, [1], angular_quadrature=10)
    assert calls["fields"] == 0


@pytest.mark.parametrize("normal_angle", [21.3, 45., 135.])
def test_activity_uses_engineering_shear_and_absolute_own_step_flow(synthetic_checkpoint, normal_angle):
    checkpoint, _, params, total, gamma = synthetic_checkpoint
    bundle, _, model = producer.load_orientation_model(checkpoint, "cpu")
    del model.increments[2:]
    bundle["params"] = replace(params, joint_normal_deg=normal_angle)
    grid = producer.quadrature(3., 4, 32)
    _, activity = producer.plastic_sources(bundle, model, grid, [2, 1])
    nx, ny = np.cos(np.deg2rad(normal_angle)), np.sin(np.deg2rad(normal_angle))
    stress = total.numpy()
    tau = nx*ny*(stress[:, 1]-stress[:, 0])+(nx**2-ny**2)*stress[:, 3]
    signed = gamma[:, 1].numpy()*tau/np.sqrt(tau**2+params.joint_smoothing**2)
    expected = np.abs(signed).cumsum()[[1, 0]]
    np.testing.assert_allclose(activity, np.broadcast_to(expected, (4, 32, 2)))
