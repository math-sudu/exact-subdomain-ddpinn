"""Widening must preserve the path and expose trainable new neural features."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from widen_pending_increment import widen
from pilot.spectral_mixed import load_orientation_model
from test_returned_orientation import fixture


def test_widening_preserves_history_and_survives_both_loaders(tmp_path):
    before, saved, _ = fixture(6)
    saved.update(evaluations=0, trust_region_state=dict(solves=0, inner={}, last_result={}))
    updated, after = widen(saved, 16)
    for name, value in saved["model"].items():
        if not name.startswith("increments.5."):
            torch.testing.assert_close(updated["model"][name], value, atol=0, rtol=0)
    xy = torch.tensor([[1.05, .15], [-1.2, .4], [.4, 1.8]], dtype=torch.float64)
    for left, right in zip(before.returned_history(xy), after.returned_history(xy)):
        torch.testing.assert_close(left, right, atol=1e-13, rtol=1e-12)
    checkpoint = tmp_path / "model.pt"
    torch.save(updated, checkpoint)
    bundle, _, restored = load_orientation_model(checkpoint)
    for left, right in zip(after.returned_history(xy), restored.returned_history(xy)):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    from publication_model import load_model
    public = tmp_path / "public.pt"
    torch.save(dict(spec=bundle["spec"], c=saved["c"], s0=saved["s0"],
                    load_step=6, model=updated["model"]), public)
    _, _, restored_public = load_model(tmp_path, public)
    torch.testing.assert_close(restored_public(xy)[0], restored(xy)[0], atol=0, rtol=0)
    increment = restored.increments[-1]
    increment.requires_grad_(True)
    increment.coefficients()[..., :2].abs().square().sum().backward()
    assert increment.kinematic_head.weight.grad[:2, 8:].abs().max() > 0
