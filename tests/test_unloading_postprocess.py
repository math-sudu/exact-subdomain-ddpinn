"""Tensor conventions used by the manuscript field export."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from extract_unloading_results import equivalent_increment, plane_shear


def test_equivalent_strain_uses_engineering_shear_and_removes_volume():
    strain = torch.tensor([[0., 0., 0., 2.], [3., 3., 3., 0.]], dtype=torch.float64)
    torch.testing.assert_close(equivalent_increment(strain),
                               torch.tensor([2/np.sqrt(3), 0.], dtype=torch.float64))


def test_plane_shear_matches_tensor_rotation():
    strain = torch.tensor([0.2, -0.1, 0.05, 0.6], dtype=torch.float64)
    tensor = torch.tensor([[0.2, 0.3], [0.3, -0.1]], dtype=torch.float64)
    for angle in (0., 37., 90., 135.):
        beta = np.deg2rad(angle)
        normal = torch.tensor([np.cos(beta), np.sin(beta)], dtype=torch.float64)
        tangent = torch.stack((-normal[1], normal[0]))
        torch.testing.assert_close(plane_shear(strain, angle), 2*tangent@tensor@normal)
