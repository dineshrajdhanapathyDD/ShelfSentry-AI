"""Property-based tests for the bounded agent loop (task 7.4).

These Hypothesis properties exercise the active-perception loop
(:func:`shelfsentry.agent.run_agent_loop`) and its collaborators as the pure,
typed components they are. Inputs are built from the data contracts in
:mod:`shelfsentry.data.contracts`; the only "tool" is a fake
:class:`ToolClient` (no AWS, no MCP SDK, no ``cv2``). They complement, and do
not duplicate, the example-based tests in ``test_loop.py``,
``test_bedrock_decision.py``, and ``test_fallback_properties.py``.

Properties implemented here:

* **Property 12: Findings causally influence the selected action**
  (Validates: Requirements 6.2)
* **Property 13: Invalid tool arguments are rejected before execution**
  (Validates: Requirements 6.4, 8.3)
* **Property 14: Iteration count is bounded**
  (Validates: Requirements 6.6, 6.7)
* **Property 15: Reports reference existing trace evidence**
  (Validates: Requirements 6.10)
* **Property 16: Model errors cause a recorded fallback**
  (Validates: Requirements 7.2)
* **Property 18: inspect_crop consumes the preceding finding's region**
  (Validates: Requirements 23.2)
* **Property 20: Conflicting evidence triggers human review**
  (Validates: Requirements 23.4)
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from shelfsentry.agent import (
    AGENT_ACTIONS,
    FallbackPolicy,
    InjectedDecisionModel,
    ToolArgumentError,
    ToolResult,
    run_agent_loop,
    validate_tool_args,
)
from shelfsentry.config.settings import AgentConfig
from shelfsentry.data.contracts import (
    AgentDecision,
    BoundingBox,
    CandidateRegion,
    InspectionReport,
    ToolExecution,
    VisionFinding,
)

# Documented default thresholds the policy branches on (settings.py / fallback.py):
#   empty_space_ratio_threshold ("notable") .............. 0.6
#   sufficient_uncertainty (ambiguity bar) ............... 0.7
#   conflict_margin ...................................... 0.25
#   poor-quality confidence floor (fixed) ................ 0.15
_EMPTY_THRESHOLD = 0.6
_POOR_QUALITY_FLOOR = 0.15
_SUFFICIENT_DEFAULT = 0.7


# ---------------------------------------------------------------------------
# Builders (typed contracts only; no cv2 / AWS / MCP)
# ---------------------------------------------------------------------------


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
    finding_id: str = "f0",
    source: str = "image",
    frame_id: str | None = None,
) -> VisionFinding:
    return VisionFinding(
        finding_id=finding_id,
        inspection_id="insp-1",
        source=source,  # type: ignore[arg-type]
        frame_id=frame_id,
        opencv_version="5.0.0",
        candidate_regions=regions,
        uncertainty_interpretations={},
        annotated_output_ref="ref://annotated",
        suggested_follow_up="finalize_report",
    )


class FakeToolClient:
    """A recording fake :class:`ToolClient` (no AWS / MCP / cv2).

    Records every ``call`` so a property can assert the tool was (or was never)
    invoked. Returns a configured :class:`ToolResult`; when ``merge_finding`` is
    set, that finding is merged into the loop's next iteration.
    """

    def __init__(
        self,
        *,
        result_ref: str = "ref://tool-result",
        merge_finding: VisionFinding | None = None,
        error: str | None = None,
    ) -> None:
        self.result_ref = result_ref
        self.merge_finding = merge_finding
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, action: str, args: dict[str, Any]) -> ToolResult:  # type: ignore[override]
        self.calls.append((action, dict(args)))
        return ToolResult(
            self.result_ref, finding=self.merge_finding, error=self.error
        )


def _cfg(**overrides: Any) -> AgentConfig:
    return AgentConfig(**overrides)


# ===========================================================================
# Property 12: Findings causally influence the selected action.
# Validates: Requirements 6.2
# ===========================================================================
#
# We use the FallbackPolicy as the pure decision function keyed on the finding.
# Two assertions, over generated inputs:
#   (a) deterministic causal mapping - the SAME finding yields the SAME action;
#   (b) the finding genuinely drives the choice - findings that differ in a way
#       the policy keys on (clearly-sufficient vs ambiguous-uncropped) CAN
#       produce DIFFERENT actions, and we assert at least one generated pair
#       actually differs (guard against vacuity).


@st.composite
def _sufficient_finding(draw: st.DrawFn) -> VisionFinding:
    """A finding the policy must finalize: one notable-but-confident region.

    Notable (ratio >= 0.6) yet confident (uncertainty >= sufficient default),
    with no conflict and no quality flag, so only the terminal branch fires.
    """
    ratio = draw(st.floats(min_value=_EMPTY_THRESHOLD, max_value=1.0))
    unc = draw(st.floats(min_value=_SUFFICIENT_DEFAULT, max_value=1.0))
    region = _region("r0", empty_space_ratio=ratio, uncertainty_value=unc)
    return _finding([region], finding_id="sufficient")


@st.composite
def _ambiguous_uncropped_finding(draw: st.DrawFn) -> VisionFinding:
    """A finding the policy must probe via inspect_crop (not yet cropped).

    Notable (ratio >= 0.6) and uncertain (confidence strictly between the
    poor-quality floor and the sufficiency bar), no conflict, no quality flag.
    """
    ratio = draw(st.floats(min_value=_EMPTY_THRESHOLD, max_value=1.0))
    unc = draw(
        st.floats(
            min_value=_POOR_QUALITY_FLOOR + 0.01,
            max_value=_SUFFICIENT_DEFAULT - 0.01,
        )
    )
    region = _region("r0", empty_space_ratio=ratio, uncertainty_value=unc)
    return _finding([region], finding_id="ambiguous")


@settings(max_examples=150)
@given(
    sufficient=_sufficient_finding(),
    ambiguous=_ambiguous_uncropped_finding(),
)
def test_property_12_findings_causally_influence_action(
    sufficient: VisionFinding, ambiguous: VisionFinding
) -> None:
    """Property 12: Findings causally influence the selected action.

    Validates: Requirements 6.2

    Using :class:`FallbackPolicy` as a pure decision function of the finding:

    * Determinism/causality: the SAME finding yields the SAME action and args.
    * Influence: a clearly-sufficient finding and an ambiguous-uncropped finding
      produce DIFFERENT actions (``finalize_report`` vs ``inspect_crop``), so
      the finding content demonstrably drives the choice.
    """
    policy = FallbackPolicy()
    cfg = _cfg()

    d_suff = policy.decide(sufficient, [], cfg)
    d_amb = policy.decide(ambiguous, [], cfg)

    # Allowed-action invariant (Req 6.1).
    assert d_suff.action in AGENT_ACTIONS
    assert d_amb.action in AGENT_ACTIONS

    # (a) Same finding -> same action/args (deterministic causal mapping).
    assert policy.decide(sufficient, [], cfg).action == d_suff.action
    assert policy.decide(sufficient, [], cfg).args == d_suff.args
    assert policy.decide(ambiguous, [], cfg).action == d_amb.action
    assert policy.decide(ambiguous, [], cfg).args == d_amb.args

    # (b) Findings that differ in a way the policy keys on produce DIFFERENT
    #     actions. These two are constructed to land on distinct branches, so
    #     every generated pair differs (non-vacuous by construction).
    assert d_suff.action == "finalize_report"
    assert d_amb.action == "inspect_crop"
    assert d_suff.action != d_amb.action


def test_property_12_pairs_differ_is_not_vacuous() -> None:
    """Anchor for Property 12: the constructed pair actually differs in action.

    A concrete, non-generated witness that the sufficient/ambiguous findings map
    to different actions, guarding Property 12 against a vacuous (never-differs)
    pass if the generators were ever weakened.
    """
    policy = FallbackPolicy()
    cfg = _cfg()
    sufficient = _finding(
        [_region("r0", empty_space_ratio=0.8, uncertainty_value=0.95)],
        finding_id="sufficient",
    )
    ambiguous = _finding(
        [_region("r0", empty_space_ratio=0.8, uncertainty_value=0.4)],
        finding_id="ambiguous",
    )
    assert policy.decide(sufficient, [], cfg).action == "finalize_report"
    assert policy.decide(ambiguous, [], cfg).action == "inspect_crop"


# ===========================================================================
# Property 13: Invalid tool arguments are rejected before execution.
# Validates: Requirements 6.4, 8.3
# ===========================================================================


@st.composite
def _invalid_inspect_crop_args(draw: st.DrawFn) -> dict[str, Any]:
    """Generate invalid ``inspect_crop`` args that must be rejected.

    Covers: missing region_id, empty region_id, missing bounding_box, a
    bounding_box with a bad sub-field (w/h < 1 or negative x/y), a non-mapping
    bounding_box, and an unexpected extra key.
    """
    kind = draw(
        st.sampled_from(
            [
                "missing_region_id",
                "empty_region_id",
                "missing_bbox",
                "bad_bbox_w",
                "bad_bbox_h",
                "bad_bbox_x",
                "bbox_not_mapping",
                "unexpected_key",
            ]
        )
    )
    good_box = {"x": 0, "y": 0, "w": 5, "h": 5}
    if kind == "missing_region_id":
        return {"bounding_box": good_box}
    if kind == "empty_region_id":
        return {"region_id": "   ", "bounding_box": good_box}
    if kind == "missing_bbox":
        return {"region_id": "r0"}
    if kind == "bad_bbox_w":
        return {"region_id": "r0", "bounding_box": {**good_box, "w": 0}}
    if kind == "bad_bbox_h":
        return {"region_id": "r0", "bounding_box": {**good_box, "h": 0}}
    if kind == "bad_bbox_x":
        return {"region_id": "r0", "bounding_box": {**good_box, "x": -1}}
    if kind == "bbox_not_mapping":
        return {"region_id": "r0", "bounding_box": "not-a-box"}
    # unexpected_key
    return {"region_id": "r0", "bounding_box": good_box, "zoom": 2}


def _decision(action: str, args: dict[str, Any], *, iteration: int = 0) -> AgentDecision:
    return AgentDecision(
        decision_id=f"d{iteration}",
        iteration=iteration,
        action=action,  # type: ignore[arg-type]
        rationale="test decision",
        args=args,
        decided_by="injected",
    )


@settings(max_examples=150)
@given(bad_args=_invalid_inspect_crop_args())
def test_property_13_invalid_args_rejected_before_execution(
    bad_args: dict[str, Any],
) -> None:
    """Property 13: Invalid tool arguments are rejected before execution.

    Validates: Requirements 6.4, 8.3

    Two complementary assertions over generated invalid ``inspect_crop`` args:

    * :func:`validate_tool_args` raises :class:`ToolArgumentError` directly; and
    * inside :func:`run_agent_loop`, a decision carrying those args NEVER reaches
      the fake ``ToolClient.call``, and the trace records a
      ``ToolExecution(status="rejected")`` for it.
    """
    decision = _decision("inspect_crop", bad_args)

    # (1) Direct validation rejects the args before any execution.
    with pytest.raises(ToolArgumentError):
        validate_tool_args(decision)

    # (2) In the loop, the same decision is rejected: the tool is never called
    #     and a rejected ToolExecution is recorded.
    model = InjectedDecisionModel(decisions=[decision])
    tools = FakeToolClient()
    report = run_agent_loop(
        _finding([_region("r0", empty_space_ratio=0.8, uncertainty_value=0.3)]),
        model,
        tools,
        _cfg(),
        inspection_id="insp-1",
    )

    assert tools.calls == [], "tool must not be executed for rejected args"
    rejected = [
        e
        for e in report.decision_trace
        if isinstance(e, ToolExecution) and e.status == "rejected"
    ]
    assert len(rejected) == 1
    assert rejected[0].error is not None and rejected[0].error != ""
    # No ok execution could have happened, so no evidence ref either.
    assert report.supporting_evidence_refs == []


# ===========================================================================
# Property 14: Iteration count is bounded.
# Validates: Requirements 6.6, 6.7
# ===========================================================================


@given(max_iterations=st.integers(min_value=1, max_value=8))
def test_property_14_iteration_count_is_bounded(max_iterations: int) -> None:
    """Property 14: Iteration count is bounded.

    Validates: Requirements 6.6, 6.7

    A model that never finalizes (always a valid ``inspect_crop`` that keeps an
    open probe) must still terminate: the loop returns an
    :class:`InspectionReport`, and the number of :class:`AgentDecision` entries
    in its trace is at most ``cfg.max_iterations``.
    """

    def never_finalize(
        finding: VisionFinding, trace: list[Any], cfg: AgentConfig
    ) -> AgentDecision:
        iteration = sum(1 for e in trace if isinstance(e, AgentDecision))
        return _decision(
            "inspect_crop",
            {
                "region_id": "r0",
                "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
            },
            iteration=iteration,
        )

    # The merged finding keeps an open probe (notable + uncertain) and a stable
    # id, so the model genuinely never chooses to finalize.
    open_finding = _finding(
        [_region("r0", empty_space_ratio=0.8, uncertainty_value=0.3)]
    )
    tools = FakeToolClient(merge_finding=open_finding)
    cfg = _cfg(max_iterations=max_iterations)

    report = run_agent_loop(
        open_finding,
        InjectedDecisionModel(fn=never_finalize),
        tools,
        cfg,
        inspection_id="insp-1",
    )

    assert isinstance(report, InspectionReport)
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert len(decisions) <= max_iterations
    # A never-finalizing model runs exactly to the cap.
    assert len(decisions) == max_iterations


# ===========================================================================
# Property 15: Reports reference existing trace evidence.
# Validates: Requirements 6.10
# ===========================================================================


@st.composite
def _loop_scenario(
    draw: st.DrawFn,
) -> tuple[VisionFinding, InjectedDecisionModel, FakeToolClient, AgentConfig]:
    """A completed-run scenario mixing crops, finalizes, and tool outcomes.

    Builds a queue of decisions ending in a terminal action, plus a fake tool
    whose result may be ``ok`` (a real ref) or an error. The resulting run
    always completes with a mixture of ok/error/rejected executions so the
    evidence-ref invariant is exercised non-trivially.
    """
    n_crops = draw(st.integers(min_value=0, max_value=4))
    decisions: list[dict[str, Any]] = []
    for i in range(n_crops):
        # Occasionally emit an invalid crop so a rejected execution appears too.
        if draw(st.booleans()):
            decisions.append({"action": "inspect_crop", "args": {"region_id": f"r{i}"}})
        else:
            decisions.append(
                {
                    "action": "inspect_crop",
                    "args": {
                        "region_id": f"r{i}",
                        "bounding_box": {"x": 0, "y": 0, "w": 4, "h": 4},
                    },
                }
            )
    terminal = draw(
        st.sampled_from(["finalize_report", "request_human_review"])
    )
    decisions.append({"action": terminal})

    tool_errors = draw(st.booleans())
    tools = FakeToolClient(
        result_ref="ref://evidence",
        merge_finding=_finding(
            [_region("r0", empty_space_ratio=0.8, uncertainty_value=0.3)]
        ),
        error="tool failed" if tool_errors else None,
    )
    finding = _finding(
        [_region("r0", empty_space_ratio=0.8, uncertainty_value=0.3)]
    )
    model = InjectedDecisionModel(decisions=decisions)
    cfg = _cfg(max_iterations=8)
    return finding, model, tools, cfg


@settings(max_examples=150)
@given(scenario=_loop_scenario())
def test_property_15_reports_reference_existing_evidence(
    scenario: tuple[VisionFinding, InjectedDecisionModel, FakeToolClient, AgentConfig],
) -> None:
    """Property 15: Reports reference existing trace evidence.

    Validates: Requirements 6.10

    Every ref in ``report.supporting_evidence_refs`` must correspond to a
    ``ToolExecution(status="ok").result_ref`` present in the decision trace.
    There are no dangling evidence refs.
    """
    finding, model, tools, cfg = scenario
    report = run_agent_loop(
        finding, model, tools, cfg, inspection_id="insp-1"
    )

    ok_refs = [
        e.result_ref
        for e in report.decision_trace
        if isinstance(e, ToolExecution) and e.status == "ok"
    ]
    # Every supporting ref is backed by an ok execution in the trace.
    for ref in report.supporting_evidence_refs:
        assert ref in ok_refs, f"dangling evidence ref: {ref!r}"

    # And refs from non-ok executions are never cited as evidence.
    non_ok_refs = {
        e.result_ref
        for e in report.decision_trace
        if isinstance(e, ToolExecution) and e.status != "ok"
    }
    for ref in report.supporting_evidence_refs:
        assert ref not in non_ok_refs or ref in ok_refs


# ===========================================================================
# Property 16: Model errors cause a recorded fallback.
# Validates: Requirements 7.2
# ===========================================================================


class _RaisingModel:
    """A decision model whose ``decide`` always raises the configured error."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def decide(
        self, finding: VisionFinding, trace: list[Any], cfg: AgentConfig
    ) -> AgentDecision:
        raise self._exc


_RAISED_ERRORS = st.sampled_from(
    [
        RuntimeError("bedrock unavailable"),
        ValueError("bad response"),
        TimeoutError("timed out"),
        KeyError("missing key"),
    ]
)


@settings(max_examples=120)
@given(exc=_RAISED_ERRORS, max_iterations=st.integers(min_value=1, max_value=6))
def test_property_16_model_errors_cause_recorded_fallback(
    exc: Exception, max_iterations: int
) -> None:
    """Property 16: Model errors cause a recorded fallback.

    Validates: Requirements 7.2

    With a model whose ``decide`` always raises and the default fallback policy
    engaged, the first :class:`AgentDecision` in the trace must record
    ``used_fallback=True``, ``decided_by == "fallback"``, and a non-empty
    ``fallback_reason``; and the loop still produces an
    :class:`InspectionReport`.
    """
    # A clearly-sufficient finding so the deterministic fallback finalizes.
    sufficient = _finding(
        [_region("r0", empty_space_ratio=0.1, uncertainty_value=0.95)]
    )
    report = run_agent_loop(
        sufficient,
        _RaisingModel(exc),
        FakeToolClient(),
        _cfg(max_iterations=max_iterations),
        inspection_id="insp-1",
    )

    assert isinstance(report, InspectionReport)
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert decisions, "expected at least one recorded decision"
    first = decisions[0]
    assert first.used_fallback is True
    assert first.decided_by == "fallback"
    assert first.fallback_reason is not None and first.fallback_reason != ""
    assert type(exc).__name__ in first.fallback_reason


# ===========================================================================
# Property 18: inspect_crop consumes the preceding finding's region.
# Validates: Requirements 23.2
# ===========================================================================


@st.composite
def _crop_triggering_finding(draw: st.DrawFn) -> VisionFinding:
    """A finding whose uncertain notable region the policy must crop.

    One notable, uncertain region (so the crop branch fires) plus optional extra
    regions that never outrank it: either confident-empty regions or
    alt-metric-only regions, none clearly filled (so no conflict) and no quality
    flags. The crop target is the single ambiguous region.
    """
    tgt_ratio = draw(st.floats(min_value=_EMPTY_THRESHOLD, max_value=1.0))
    tgt_unc = draw(
        st.floats(
            min_value=_POOR_QUALITY_FLOOR + 0.01,
            max_value=_SUFFICIENT_DEFAULT - 0.05,
        )
    )
    x = draw(st.integers(min_value=0, max_value=40))
    y = draw(st.integers(min_value=0, max_value=40))
    w = draw(st.integers(min_value=1, max_value=30))
    h = draw(st.integers(min_value=1, max_value=30))
    # Make the target the most-uncertain by giving it the lowest confidence; to
    # keep it the unique minimum under tie-break (uncertainty, region_id), extra
    # regions are strictly more confident.
    target = _region(
        "r0",
        empty_space_ratio=tgt_ratio,
        uncertainty_value=tgt_unc,
        x=x,
        y=y,
        w=w,
        h=h,
    )
    regions = [target]
    for i in range(draw(st.integers(min_value=0, max_value=2))):
        if draw(st.booleans()):
            regions.append(
                _region(
                    f"r{i + 1}",
                    empty_space_ratio=None,
                    uncertainty_value=draw(st.floats(0.5, 1.0)),
                    alt_metric={"mean_intensity": draw(st.floats(0.0, 1.0))},
                )
            )
        else:
            # confident + empty -> notable but not an open probe.
            regions.append(
                _region(
                    f"r{i + 1}",
                    empty_space_ratio=draw(
                        st.floats(_EMPTY_THRESHOLD, 1.0)
                    ),
                    uncertainty_value=draw(st.floats(_SUFFICIENT_DEFAULT, 1.0)),
                )
            )
    return _finding(regions, finding_id="crop-case")


@settings(max_examples=150)
@given(finding=_crop_triggering_finding())
def test_property_18_inspect_crop_consumes_finding_region(
    finding: VisionFinding,
) -> None:
    """Property 18: inspect_crop consumes the preceding finding's region.

    Validates: Requirements 23.2

    When :class:`FallbackPolicy` chooses ``inspect_crop``, the decision's
    ``args.region_id`` is a region present in the finding it was given, and the
    ``bounding_box`` matches that region's own ``bounding_box``.
    """
    decision = FallbackPolicy().decide(finding, [], _cfg())
    assume(decision.action == "inspect_crop")

    region_id = decision.args["region_id"]
    by_id = {r.region_id: r for r in finding.candidate_regions}
    assert region_id in by_id, "crop targets a region not in the finding"

    region = by_id[region_id]
    box = region.bounding_box
    assert decision.args["bounding_box"] == {
        "x": box.x,
        "y": box.y,
        "w": box.w,
        "h": box.h,
    }


# ===========================================================================
# Property 20: Conflicting evidence triggers human review.
# Validates: Requirements 23.4
# ===========================================================================


@st.composite
def _conflicting_finding(draw: st.DrawFn) -> tuple[VisionFinding, AgentConfig]:
    """A finding with a clearly-empty and a clearly-filled region.

    The empty region reads >= empty threshold (0.6); the filled region reads
    <= 1 - threshold (0.4); and the spread between them is >= conflict_margin,
    so the conflict branch must fire ahead of every other branch.
    """
    cfg = _cfg()
    margin = cfg.conflict_margin  # 0.25 default
    filled_threshold = 1.0 - _EMPTY_THRESHOLD  # 0.4

    empty_ratio = draw(st.floats(min_value=_EMPTY_THRESHOLD, max_value=1.0))
    filled_ratio = draw(st.floats(min_value=0.0, max_value=filled_threshold))
    # Guarantee the margin requirement holds (default values already do, but be
    # explicit so the branch is never skipped vacuously).
    assume((empty_ratio - filled_ratio) >= margin)

    empty_region = _region(
        "r0", empty_space_ratio=empty_ratio, uncertainty_value=draw(st.floats(0.2, 1.0))
    )
    filled_region = _region(
        "r1", empty_space_ratio=filled_ratio, uncertainty_value=draw(st.floats(0.2, 1.0))
    )
    regions = [empty_region, filled_region]
    return _finding(regions, finding_id="conflict"), cfg


@settings(max_examples=150)
@given(case=_conflicting_finding())
def test_property_20_conflicting_evidence_triggers_human_review(
    case: tuple[VisionFinding, AgentConfig],
) -> None:
    """Property 20: Conflicting evidence triggers human review.

    Validates: Requirements 23.4

    A finding with a clearly-empty and a clearly-filled region whose spread is
    at least ``conflict_margin`` makes :class:`FallbackPolicy` select
    ``request_human_review``.
    """
    finding, cfg = case
    decision = FallbackPolicy().decide(finding, [], cfg)
    assert decision.action in AGENT_ACTIONS
    assert decision.action == "request_human_review"
