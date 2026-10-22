"""Tests for the deterministic :class:`FallbackPolicy` (task 6.2).

Covers the reference decision logic (design.md "Deterministic fallback policy")
and the determinism guarantee (Req 7.4, with Req 23.3 ambiguous->more evidence
and Req 23.4 conflicting->human review):

* same inputs -> same action/args/rationale (determinism);
* a high-uncertainty, not-yet-cropped region -> ``inspect_crop`` of that region;
* an already-cropped region is not re-cropped (loop guard);
* an ambiguous video finding with frames remaining -> ``analyze_another_frame``;
* conflicting evidence -> ``request_human_review``;
* poor image quality -> ``request_better_image``;
* clear / sufficient / nothing-notable -> ``finalize_report``.

Findings are built directly from the typed contracts (no AWS, no cv2) so the
policy is exercised as the pure function it is.
"""

from __future__ import annotations

from typing import Any

from shelfsentry.agent import DecisionModel, FallbackPolicy
from shelfsentry.config.settings import AgentConfig
from shelfsentry.data.contracts import (
    AgentDecision,
    BoundingBox,
    CandidateRegion,
    TraceEntry,
    VisionFinding,
)

# The policy's "notable" threshold mirrors VisionConfig.empty_space_ratio_threshold
# (default 0.6). AgentConfig defaults: sufficient_uncertainty=0.7, conflict_margin=0.25.
_EMPTY_THRESHOLD = 0.6


def _cfg(**overrides: Any) -> AgentConfig:
    return AgentConfig(**overrides)


def _region(
    region_id: str,
    *,
    empty_space_ratio: float | None,
    uncertainty_value: float,
    alt_metric: dict[str, Any] | None = None,
    x: int = 0,
    y: int = 0,
    w: int = 10,
    h: int = 10,
) -> CandidateRegion:
    return CandidateRegion(
        region_id=region_id,
        bounding_box=BoundingBox(x=x, y=y, w=w, h=h),
        empty_space_ratio=empty_space_ratio,
        alt_metric=alt_metric,
        heuristic_score=0.5,
        uncertainty_value=uncertainty_value,
        visual_caveat="estimate only; may be shadow/occlusion/dividers/quality",
    )


def _finding(
    regions: list[CandidateRegion],
    *,
    source: str = "image",
    frame_id: str | None = None,
) -> VisionFinding:
    return VisionFinding(
        finding_id="f1",
        inspection_id="insp-1",
        source=source,  # type: ignore[arg-type]
        frame_id=frame_id,
        opencv_version="5.0.0",
        candidate_regions=regions,
        uncertainty_interpretations={},
        annotated_output_ref="ref://annotated",
        suggested_follow_up="finalize_report",
    )


def _crop_entry(region_id: str, *, iteration: int = 0) -> AgentDecision:
    """A trace entry recording that ``region_id`` was already cropped."""
    return AgentDecision(
        decision_id=f"prev-{iteration}",
        iteration=iteration,
        action="inspect_crop",
        rationale="prior crop",
        args={"region_id": region_id},
        decided_by="fallback",
    )


def _another_frame_entry(iteration: int) -> AgentDecision:
    return AgentDecision(
        decision_id=f"prev-{iteration}",
        iteration=iteration,
        action="analyze_another_frame",
        rationale="prior frame",
        decided_by="fallback",
    )


# ---------------------------------------------------------------------------
# protocol conformance
# ---------------------------------------------------------------------------


def test_fallback_policy_satisfies_decision_model_protocol() -> None:
    assert isinstance(FallbackPolicy(), DecisionModel)


def test_decisions_are_recorded_as_fallback_not_used_fallback() -> None:
    # Used directly, the policy sets decided_by="fallback"; used_fallback stays
    # False (the loop owns that flag when it engages the policy on a model error).
    finding = _finding([_region("r0", empty_space_ratio=0.9, uncertainty_value=0.9)])
    decision = FallbackPolicy().decide(finding, [], _cfg())
    assert decision.decided_by == "fallback"
    assert decision.used_fallback is False
    assert decision.fallback_reason is None


# ---------------------------------------------------------------------------
# determinism (Req 7.4) - same inputs -> same action/args/rationale
# ---------------------------------------------------------------------------


def test_same_inputs_produce_identical_decisions() -> None:
    policy = FallbackPolicy()
    cfg = _cfg()
    regions = [
        _region("r0", empty_space_ratio=0.8, uncertainty_value=0.4, x=1, y=2, w=3, h=4),
        _region("r1", empty_space_ratio=0.7, uncertainty_value=0.5),
    ]
    finding = _finding(regions)

    d1 = policy.decide(finding, [], cfg)
    d2 = policy.decide(finding, [], cfg)
    # Separate policy instance, same inputs -> same result (no hidden state).
    d3 = FallbackPolicy().decide(finding, [], cfg)

    assert d1.action == d2.action == d3.action
    assert d1.args == d2.args == d3.args
    assert d1.rationale == d2.rationale == d3.rationale
    assert d1.canonical_json() == d2.canonical_json() == d3.canonical_json()


def test_region_order_does_not_change_the_chosen_crop() -> None:
    # Determinism must not depend on input region ordering: the most-uncertain
    # region is selected regardless of the order it appears in.
    policy = FallbackPolicy()
    cfg = _cfg()
    a = _region("rA", empty_space_ratio=0.8, uncertainty_value=0.5)
    b = _region("rB", empty_space_ratio=0.8, uncertainty_value=0.2)  # most uncertain

    d_forward = policy.decide(_finding([a, b]), [], cfg)
    d_reversed = policy.decide(_finding([b, a]), [], cfg)

    assert d_forward.action == d_reversed.action == "inspect_crop"
    assert d_forward.args["region_id"] == d_reversed.args["region_id"] == "rB"


# ---------------------------------------------------------------------------
# high uncertainty, uncropped region -> inspect_crop (Req 7.4, 23.2)
# ---------------------------------------------------------------------------


def test_high_uncertainty_uncropped_region_selects_inspect_crop() -> None:
    # Low confidence (0.3 < sufficient_uncertainty 0.7) on a notable region that
    # has not been cropped -> inspect_crop identifying that region.
    region = _region(
        "r0", empty_space_ratio=0.85, uncertainty_value=0.3, x=5, y=6, w=7, h=8
    )
    decision = FallbackPolicy().decide(_finding([region]), [], _cfg())

    assert decision.action == "inspect_crop"
    assert decision.args["region_id"] == "r0"
    assert decision.args["bounding_box"] == {"x": 5, "y": 6, "w": 7, "h": 8}
    assert "r0" in decision.rationale


def test_already_cropped_region_is_not_recropped() -> None:
    # Same uncertain region, but the trace already records a crop of it. The
    # loop guard prevents re-cropping; with no other evidence action for a still
    # image, the policy finalizes instead.
    region = _region("r0", empty_space_ratio=0.85, uncertainty_value=0.3)
    trace: list[TraceEntry] = [_crop_entry("r0")]

    decision = FallbackPolicy().decide(_finding([region]), trace, _cfg())

    assert decision.action != "inspect_crop"
    assert decision.action == "finalize_report"


def test_most_uncertain_uncropped_region_is_chosen_when_one_is_cropped() -> None:
    # r0 already cropped; r1 is still an open, uncertain probe -> crop r1.
    r0 = _region("r0", empty_space_ratio=0.85, uncertainty_value=0.2)
    r1 = _region("r1", empty_space_ratio=0.85, uncertainty_value=0.4)
    trace: list[TraceEntry] = [_crop_entry("r0")]

    decision = FallbackPolicy().decide(_finding([r0, r1]), trace, _cfg())

    assert decision.action == "inspect_crop"
    assert decision.args["region_id"] == "r1"


# ---------------------------------------------------------------------------
# ambiguous video finding -> analyze_another_frame (Req 23.3)
# ---------------------------------------------------------------------------


def test_ambiguous_video_finding_with_frames_remaining_analyzes_another_frame() -> None:
    # The only open uncertain region was already cropped (so inspect_crop is out),
    # the source is video, and frames remain in the budget -> another frame.
    region = _region("r0", empty_space_ratio=0.85, uncertainty_value=0.3)
    trace: list[TraceEntry] = [_crop_entry("r0")]
    finding = _finding([region], source="video_frame", frame_id="frame-0001")

    decision = FallbackPolicy().decide(finding, trace, _cfg())

    assert decision.action == "analyze_another_frame"


def test_ambiguous_video_finds_crop_before_another_frame() -> None:
    # When the uncertain region has NOT been cropped yet, cropping it takes
    # priority over sampling another frame (more targeted evidence first).
    region = _region("r0", empty_space_ratio=0.85, uncertainty_value=0.3)
    finding = _finding([region], source="video_frame", frame_id="frame-0001")

    decision = FallbackPolicy().decide(finding, [], _cfg())

    assert decision.action == "inspect_crop"


def test_video_frame_budget_exhausted_finalizes() -> None:
    # Source is video but the number of recorded analyze_another_frame decisions
    # has reached the iteration budget: no more frames -> finalize.
    region = _region("r0", empty_space_ratio=0.85, uncertainty_value=0.3)
    cfg = _cfg(max_iterations=2)
    trace: list[TraceEntry] = [
        _crop_entry("r0", iteration=0),
        _another_frame_entry(1),
        _another_frame_entry(2),
    ]
    finding = _finding([region], source="video_frame", frame_id="frame-0001")

    decision = FallbackPolicy().decide(finding, trace, cfg)

    assert decision.action == "finalize_report"


# ---------------------------------------------------------------------------
# conflicting evidence -> request_human_review (Req 23.4)
# ---------------------------------------------------------------------------


def test_conflicting_evidence_selects_request_human_review() -> None:
    # One clearly-empty region (0.9 >= 0.6) and one clearly-filled region
    # (0.05 <= 0.4), spread 0.85 >= conflict_margin 0.25 -> human review.
    empty = _region("r0", empty_space_ratio=0.9, uncertainty_value=0.8)
    filled = _region("r1", empty_space_ratio=0.05, uncertainty_value=0.8)

    decision = FallbackPolicy().decide(_finding([empty, filled]), [], _cfg())

    assert decision.action == "request_human_review"
    assert decision.args == {}


def test_conflict_takes_priority_over_an_uncertain_crop() -> None:
    # Even with a low-confidence region present, a genuine empty/filled conflict
    # escalates to a human first (safety-first priority ordering).
    empty = _region("r0", empty_space_ratio=0.9, uncertainty_value=0.3)
    filled = _region("r1", empty_space_ratio=0.05, uncertainty_value=0.9)

    decision = FallbackPolicy().decide(_finding([empty, filled]), [], _cfg())

    assert decision.action == "request_human_review"


# ---------------------------------------------------------------------------
# poor image quality -> request_better_image
# ---------------------------------------------------------------------------


def test_explicit_quality_flag_requests_better_image() -> None:
    region = _region(
        "r0",
        empty_space_ratio=0.8,
        uncertainty_value=0.3,
        alt_metric={"blur": True},
    )
    decision = FallbackPolicy().decide(_finding([region]), [], _cfg())
    assert decision.action == "request_better_image"


def test_uniformly_unreadable_confidence_requests_better_image() -> None:
    # All notable regions at/below the poor-quality confidence floor (0.15).
    region = _region("r0", empty_space_ratio=0.8, uncertainty_value=0.1)
    decision = FallbackPolicy().decide(_finding([region]), [], _cfg())
    assert decision.action == "request_better_image"


# ---------------------------------------------------------------------------
# clear / sufficient / nothing notable -> finalize_report (Req 6.7, 6.8)
# ---------------------------------------------------------------------------


def test_clear_confident_single_region_finalizes() -> None:
    # Notable and confident (0.9 >= sufficient_uncertainty 0.7): nothing to
    # probe -> finalize.
    region = _region("r0", empty_space_ratio=0.9, uncertainty_value=0.9)
    decision = FallbackPolicy().decide(_finding([region]), [], _cfg())
    assert decision.action == "finalize_report"
    assert "sufficient" in decision.rationale.lower()


def test_nothing_notable_finalizes() -> None:
    # No region reaches the empty-space threshold -> nothing notable -> finalize.
    region = _region("r0", empty_space_ratio=0.2, uncertainty_value=0.5)
    decision = FallbackPolicy().decide(_finding([region]), [], _cfg())
    assert decision.action == "finalize_report"
    assert "notable" in decision.rationale.lower()


def test_no_regions_finalizes() -> None:
    decision = FallbackPolicy().decide(_finding([]), [], _cfg())
    assert decision.action == "finalize_report"


def test_iteration_index_tracks_prior_decisions_in_trace() -> None:
    region = _region("r0", empty_space_ratio=0.9, uncertainty_value=0.9)
    trace: list[TraceEntry] = [_crop_entry("x", iteration=0), _another_frame_entry(1)]
    decision = FallbackPolicy().decide(_finding([region]), trace, _cfg())
    # Two prior AgentDecision entries -> this decision is iteration 2.
    assert decision.iteration == 2
    assert decision.decision_id == "fallback-2"
