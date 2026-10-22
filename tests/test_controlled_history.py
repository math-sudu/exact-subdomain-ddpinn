"""Physical acceptance must reject an unchanged but unbalanced iterate."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from resolve_controlled_history import acceptance_reasons, needs_test_enrichment


def test_stagnant_response_does_not_accept_large_physical_residuals():
    targets = dict(response_change_mm=1e-4, gauge_imbalance_mm=.001,
                   wall_rms_mpa=.01, weak_residual_norm=.02)
    state = dict(minimum_prefix_branch_margin=.001, wall_vector_rms_mpa=.328,
                 enriched_residual_dual_norm=.059,
                 source_0_mm=-.03, source_90_mm=.07,
                 nonelastic_0_mm=-.03, nonelastic_90_mm=.0724,
                 gauge_imbalance_0_mm=0., gauge_imbalance_90_mm=.0024)
    reasons = acceptance_reasons(state, state.copy(), targets)
    assert reasons == ["wall_traction", "gauge_90", "enriched_weak_balance"]
    state.update(wall_vector_rms_mpa=.005, enriched_residual_dual_norm=.01,
                 gauge_imbalance_90_mm=.0002, nonelastic_90_mm=.0702)
    assert acceptance_reasons(state, state.copy(), targets) == []
    state["minimum_prefix_branch_margin"] = -.0001
    assert acceptance_reasons(state, state.copy(), targets) == ["unsupported_return"]
    state.update(minimum_prefix_branch_margin=.001, source_0_mm=float("nan"))
    assert acceptance_reasons(state, state.copy(), targets) == ["nonfinite_physical_readout"]


def test_projection_floor_can_trigger_enrichment_before_response_stabilizes():
    targets = dict(weak_residual_norm=.03272)
    state = dict(pending_reasons=["response_change", "enriched_weak_balance"],
                 frozen_source_projection_floor=.03760)
    assert needs_test_enrichment(state, targets)
    state["frozen_source_projection_floor"] = .02
    assert not needs_test_enrichment(state, targets)
    state["pending_reasons"] = ["enriched_weak_balance"]
    assert needs_test_enrichment(state, targets)
    state["pending_reasons"] = ["wall_traction", "enriched_weak_balance"]
    assert not needs_test_enrichment(state, targets)
