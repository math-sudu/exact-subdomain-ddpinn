"""Time-dependent stress regions must retain signed gauge-source cancellation."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from assess_tensile_history import conditioned_sources
from pilot import config as C


def test_own_step_sources_differ_from_final_region_and_keep_cancellation():
    normal = np.array([[-1., .2, .42], [.2, -.1, .42]])
    plastic = np.zeros((2, 3, 4))
    plastic[..., 3] = [[1., 2., 3.], [2., -1., -2.]]
    auxiliary = np.array([[[0., 0., 0., 1.], [0., 0., 0., -1.]]])
    result = conditioned_sources(normal, plastic, auxiliary, np.array([2., 1.]), .4)
    scale = C.U_REF * 1000
    np.testing.assert_allclose(result["compression"]["history_net_mm"], [[2., 3., 3.]] * np.array(scale))
    np.testing.assert_allclose(result["tension"]["history_net_mm"], [[-2., 2., 2.]] * np.array(scale))
    np.testing.assert_allclose(result["near_apex"]["history_net_mm"], [[0., 0., 8.]] * np.array(scale))
    # Both points enter the final region; that region includes earlier work.
    np.testing.assert_allclose(result["near_apex"]["state_region_net_mm"], [[0., 0., 13.]] * np.array(scale))
    np.testing.assert_allclose(result["near_apex"]["state_region_positive_mm"], [[0., 0., 15.]] * np.array(scale))
    np.testing.assert_allclose(result["near_apex"]["state_region_negative_mm"], [[0., 0., -2.]] * np.array(scale))
    np.testing.assert_allclose(result["tension"]["history_positive_mm"], [[0., 4., 4.]] * np.array(scale))
    np.testing.assert_allclose(result["tension"]["history_negative_mm"], [[-2., -2., -2.]] * np.array(scale))
    total = sum(row["history_net_mm"] for row in result.values())
    np.testing.assert_allclose(total, [[0., 5., 13.]] * np.array(scale))
    for row in result.values():
        np.testing.assert_allclose(row["history_positive_mm"] + row["history_negative_mm"], row["history_net_mm"])
        np.testing.assert_allclose(row["state_region_positive_mm"] + row["state_region_negative_mm"], row["state_region_net_mm"])
