"""Example-based unit tests for agent causality, safety, and trace integrity.

Task 7.5 / Requirement 23. These are explicit, named *scenario* tests (not
property tests) exercising the real agent components end to end with no AWS,
MCP, or cv2 dependency:

* :func:`shelfsentry.agent.loop.run_agent_loop` - the bounded loop;
* :class:`shelfsentry.agent.fallback.FallbackPolicy` - the deterministic policy;
* :class:`shelfsentry.agent.decision.InjectedDecisionModel` - the offline model;
* :class:`shelfsentry.agent.bedrock.BedrockDecisionModel` - driven with a fake
  client to exercise the unavailable/error path;
* :func:`shelfsentry.agent.tools.validate_tool_args` - pre-execution arg check;
* a recording fake :class:`ToolClient`.

Each test targets one Req 23 criterion as a concrete scenario and makes strong
assertions. The property-based tests in ``test_fallback_properties.py`` and the
loop-mechanics tests in ``test_loop.py`` cover the universal/mechanical
guarantees; here we pin the specific causal and safety behaviors the
requirement enumerates:

* Req 23.1 - different findings produce different next actions;
* Req 23.5 - the configured max iteration count is respected;
* Req 23.6 - invalid tool arguments are rejected (recorded, tool not called);
* Req 23.7 - a model error triggers the safe fallback, recorded;
* Req 23.8 - the recorded Decision_Trace matches the tool calls that were made.
"""

from __future__ import annotations

from typing import Any

from shelfsentry.agent import (
    BedrockDecisionModel,
    FallbackPolicy,
    InjectedDecisionModel,
    ToolResult,
    run_agent_loop,
    validate_tool_args,
)
from shelfsentry.agent.loop import ToolClient
from shelfsentry.agent.tools import ToolArgumentError
from shelfsentry.config.settings import AgentConfig
from shelfsentry.data.contracts import (
    AgentDecision,
    BoundingBox,
    CandidateRegion,
    InspectionReport,
    ToolExecution,
    VisionFinding,
)

# ---------------------------------------------------------------------------
# Builders - small, explicit, and distinct from test_loop's so the scenarios
# here read as concrete inputs rather than shared fixtures.
# ---------------------------------------------------------------------------

# The default AgentConfig bars: sufficient_uncertainty=0.7 (confidence at/above
# this is "sufficient"), empty_space_ratio_threshold (vision default 0.6) marks
# a region "notable". These two constants drive the causal branches below.
_SUFFICIENT = AgentConfig().sufficient_uncertainty  # 0.7
_NOTABLE_EMPTY = 0.85  # comfortably >= the 0.6 empty-space threshold


def _region(
    region_id: str,
    *,
    empty_space_ratio: float | None,
    uncertainty_value: float,
    x: int = 10,
    y: int = 20,
    w: int = 30,
    h: int = 40,
) -> CandidateRegion:
    return CandidateRegion(
        region_id=region_id,
        bounding_box=BoundingBox(x=x, y=y, w=w, h=h),
        empty_space_ratio=empty_space_ratio,
        heuristic_score=0.5,
        uncertainty_value=uncertainty_value,
        visual_caveat="estimate only; may be shadow/occlusion/dividers/quality",
    )


def _finding(
    finding_id: str,
    *,
    regions: list[CandidateRegion],
    source: str = "image",
    frame_id: str | None = None,
    suggested_follow_up: str = "finalize_report",
) -> VisionFinding:
    return VisionFinding(
        finding_id=finding_id,
        inspection_id="insp-behavior",
        source=source,  # type: ignore[arg-type]
        frame_id=frame_id,
        opencv_version="5.0.0",
        candidate_regions=regions,
        uncertainty_interpretations={},
        annotated_output_ref="ref://annotated",
        suggested_follow_up=suggested_follow_up,  # type: ignore[arg-type]
    )


def _sufficient_finding() -> VisionFinding:
    """A clearly sufficient finding: no region reaches the empty-space bar.

    With no *notable* region, the deterministic policy has no open probe and
    finalizes (branch 5). This is the "clearly sufficient" input for Req 23.1.
    """
    return _finding(
        "f-clear",
        regions=[_region("r-full", empty_space_ratio=0.05, uncertainty_value=0.95)],
    )


def _ambiguous_uncropped_finding() -> VisionFinding:
    """An ambiguous, not-yet-cropped finding: notable + low confidence.

    empty_space_ratio above the notable threshold and confidence below the
    sufficiency bar, with no prior crop in the (empty) trace -> the policy must
    gather sharper evidence via ``inspect_crop`` (branch 3). This is the
    "ambiguous uncropped" input for Req 23.1.
    """
    return _finding(
        "f-ambiguous",
        regions=[
            _region("r-maybe-empty", empty_space_ratio=_NOTABLE_EMPTY, uncertainty_value=0.3)
        ],
    )


class RecordingToolClient:
    """A recording fake :class:`ToolClient`.

    Records every ``(action, args)`` call in order and returns a configured
    :class:`ToolResult`. Optionally merges a fresh finding so the loop feeds it
    into the next iteration. The recorded :attr:`calls` are what the trace-
    integrity test (Req 23.8) is checked against.
    """

    def __init__(
        self,
        *,
        result_ref: str = "ref://tool",
        merge_finding: VisionFinding | None = None,
    ) -> None:
        self._result_ref = result_ref
        self._merge_finding = merge_finding
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, action: str, args: dict[str, Any]) -> ToolResult:  # type: ignore[override]
        self.calls.append((action, dict(args)))
        return ToolResult(self._result_ref, finding=self._merge_finding)


class RaisingModel:
    """A :class:`DecisionModel` whose ``decide`` always raises (Req 23.7)."""

    def decide(
        self, finding: VisionFinding, trace: list[Any], cfg: AgentConfig
    ) -> AgentDecision:
        raise RuntimeError("primary model exploded")


def _run(
    finding: VisionFinding,
    model: Any,
    tools: ToolClient,
    cfg: AgentConfig | None = None,
    **kwargs: Any,
) -> InspectionReport:
    return run_agent_loop(
        finding,
        model,
        tools,
        cfg if cfg is not None else AgentConfig(),
        inspection_id="insp-behavior",
        **kwargs,
    )


# ===========================================================================
# Req 23.1 - different findings produce different next actions
# ===========================================================================


def test_req_23_1_different_findings_produce_different_actions_fallback() -> None:
    """A clearly-sufficient vs an ambiguous-uncropped finding diverge.

    The deterministic policy is a pure function of the finding, so running it
    on two concrete findings surfaces the causal influence of the initial
    visual findings on the selected action (Req 23.1): the sufficient finding
    finalizes, the ambiguous one inspects a crop.
    """
    policy = FallbackPolicy()
    cfg = AgentConfig()

    sufficient = policy.decide(_sufficient_finding(), [], cfg)
    ambiguous = policy.decide(_ambiguous_uncropped_finding(), [], cfg)

    assert sufficient.action == "finalize_report"
    assert ambiguous.action == "inspect_crop"
    assert sufficient.action != ambiguous.action

    # The crop causally consumes exactly the ambiguous region (args identify it).
    assert ambiguous.args["region_id"] == "r-maybe-empty"
    assert ambiguous.args["bounding_box"] == {"x": 10, "y": 20, "w": 30, "h": 40}
    # And the sufficient decision runs no tool (its action maps to no MCP tool).
    assert sufficient.args == {}


def test_req_23_1_different_findings_drive_different_loop_outcomes() -> None:
    """The divergence holds end to end through the loop, not just the policy.

    Driving the full loop with the fallback policy as the model, the two
    findings produce reports whose first recorded action differs - the
    sufficient one terminates immediately (no tool call) while the ambiguous
    one issues a crop tool call first.
    """
    cfg = AgentConfig()

    clear_tools = RecordingToolClient()
    clear_report = _run(_sufficient_finding(), FallbackPolicy(), clear_tools, cfg)

    # Ambiguous: after the crop, merge a now-confident finding so the loop
    # terminates deterministically on the next decision.
    resolved = _finding(
        "f-resolved",
        regions=[_region("r-maybe-empty", empty_space_ratio=_NOTABLE_EMPTY, uncertainty_value=0.95)],
    )
    amb_tools = RecordingToolClient(result_ref="ref://crop", merge_finding=resolved)
    amb_report = _run(_ambiguous_uncropped_finding(), FallbackPolicy(), amb_tools, cfg)

    clear_first = _decisions(clear_report)[0].action
    amb_first = _decisions(amb_report)[0].action
    assert clear_first == "finalize_report"
    assert amb_first == "inspect_crop"
    assert clear_first != amb_first

    # The causal consequence: the sufficient finding triggered no tool call,
    # the ambiguous one triggered exactly the crop.
    assert clear_tools.calls == []
    assert amb_tools.calls == [
        ("inspect_crop", {"region_id": "r-maybe-empty", "bounding_box": {"x": 10, "y": 20, "w": 30, "h": 40}})
    ]


# ===========================================================================
# Req 23.5 - the configured maximum iteration count is respected
# ===========================================================================


def test_req_23_5_max_iterations_respected_exactly_two() -> None:
    """A model that always issues a valid crop stops at cfg.max_iterations=2.

    With a cap of 2 and a model that never finalizes on its own, the loop must
    record exactly 2 ``AgentDecision`` entries (each a crop with its own tool
    execution), then stop and return a report. No third decision is made.
    """
    cfg = AgentConfig(max_iterations=2)

    def always_crop(
        finding: VisionFinding, trace: list[Any], cfg_: AgentConfig
    ) -> AgentDecision:
        iteration = sum(1 for e in trace if isinstance(e, AgentDecision))
        return AgentDecision(
            decision_id=f"d{iteration}",
            iteration=iteration,
            action="inspect_crop",
            rationale="always crop",
            args={"region_id": "r0", "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}},
            decided_by="injected",
        )

    # Merge the same finding each time so the id stays stable (the model does
    # not depend on it anyway); the cap is what must stop the loop.
    stable = _finding("f0", regions=[_region("r0", empty_space_ratio=_NOTABLE_EMPTY, uncertainty_value=0.3)])
    tools = RecordingToolClient(result_ref="ref://crop", merge_finding=stable)

    report = _run(stable, InjectedDecisionModel(fn=always_crop), tools, cfg)

    decisions = _decisions(report)
    executions = _executions(report)
    assert len(decisions) == 2, "exactly max_iterations decisions"
    assert all(d.action == "inspect_crop" for d in decisions)
    assert len(executions) == 2
    assert len(tools.calls) == 2
    # Iterations are the zero-based 0, 1 - no third iteration occurred.
    assert [d.iteration for d in decisions] == [0, 1]
    # A report is still produced (Req 6.7) and it is not a human-review escalation.
    assert isinstance(report, InspectionReport)
    assert report.status == "complete"
    assert report.requires_human_review is False


def test_req_23_5_max_iterations_single_iteration_cap() -> None:
    """A cap of 1 admits exactly one decision before finalizing."""
    cfg = AgentConfig(max_iterations=1)

    def always_crop(
        finding: VisionFinding, trace: list[Any], cfg_: AgentConfig
    ) -> AgentDecision:
        iteration = sum(1 for e in trace if isinstance(e, AgentDecision))
        return AgentDecision(
            decision_id=f"d{iteration}",
            iteration=iteration,
            action="inspect_crop",
            rationale="always crop",
            args={"region_id": "r0", "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}},
            decided_by="injected",
        )

    stable = _finding("f0", regions=[_region("r0", empty_space_ratio=_NOTABLE_EMPTY, uncertainty_value=0.3)])
    tools = RecordingToolClient(merge_finding=stable)
    report = _run(stable, InjectedDecisionModel(fn=always_crop), tools, cfg)

    assert len(_decisions(report)) == 1
    assert len(tools.calls) == 1


# ===========================================================================
# Req 23.6 - invalid tool arguments are rejected (recorded, tool not called)
# ===========================================================================


def test_req_23_6_invalid_bounding_box_rejected_by_validator() -> None:
    """``validate_tool_args`` itself rejects a crop with a bad bounding box.

    A zero-width box violates the BoundingBox ``w>=1`` rule, so the pre-
    execution validator raises a field-identifying ``ToolArgumentError`` naming
    the offending sub-field - this is the check the loop relies on (Req 23.6).
    """
    bad = AgentDecision(
        decision_id="d-bad",
        iteration=0,
        action="inspect_crop",
        rationale="crop with an invalid box",
        args={"region_id": "r0", "bounding_box": {"x": 1, "y": 2, "w": 0, "h": 4}},
        decided_by="injected",
    )

    try:
        validate_tool_args(bad)
    except ToolArgumentError as err:
        assert err.action == "inspect_crop"
        assert err.field is not None and "bounding_box" in err.field
    else:  # pragma: no cover - the validator must raise
        raise AssertionError("expected validate_tool_args to reject the bad box")


def test_req_23_6_loop_records_rejected_execution_and_skips_tool() -> None:
    """In the loop, a bad-box crop records ``status='rejected'`` and calls no tool.

    The decision is recorded, then a ``ToolExecution(status='rejected')`` whose
    error names the bounding box - and the fake tool client is never called
    (Req 23.6). The loop then finalizes deterministically rather than retrying.
    """
    bad_crop = {
        "action": "inspect_crop",
        "args": {"region_id": "r0", "bounding_box": {"x": 1, "y": 2, "w": 0, "h": 4}},
    }
    model = InjectedDecisionModel(decisions=[bad_crop])
    tools = RecordingToolClient()

    report = _run(
        _finding("f0", regions=[_region("r0", empty_space_ratio=_NOTABLE_EMPTY, uncertainty_value=0.3)]),
        model,
        tools,
    )

    # The tool was never executed.
    assert tools.calls == []

    trace = report.decision_trace
    assert len(trace) == 2
    assert isinstance(trace[0], AgentDecision)
    assert trace[0].action == "inspect_crop"
    assert isinstance(trace[1], ToolExecution)
    assert trace[1].status == "rejected"
    assert trace[1].tool == "inspect_region_crop"
    assert trace[1].error is not None and "bounding_box" in trace[1].error
    # The rejected execution is NOT counted as supporting evidence.
    assert trace[1].result_ref not in report.supporting_evidence_refs
    assert report.status == "complete"


# ===========================================================================
# Req 23.7 - a model error triggers the safe fallback, recorded
# ===========================================================================


def test_req_23_7_model_error_engages_recorded_fallback() -> None:
    """A raising model makes the loop engage the deterministic fallback.

    The recorded first decision is marked ``used_fallback=True`` with a reason
    naming the error type, and ``decided_by='fallback'``; a report is still
    produced (Req 23.7). The finding is clearly sufficient so the engaged
    fallback deterministically finalizes.
    """
    report = _run(_sufficient_finding(), RaisingModel(), RecordingToolClient())

    first = _decisions(report)[0]
    assert first.used_fallback is True
    assert first.decided_by == "fallback"
    assert first.fallback_reason is not None
    assert "RuntimeError" in first.fallback_reason
    assert first.action == "finalize_report"
    assert report.status == "complete"


def test_req_23_7_bedrock_unavailable_triggers_recorded_fallback() -> None:
    """A Bedrock client whose Converse call errors drives the recorded fallback.

    ``BedrockDecisionModel.decide`` raises ``BedrockDecisionError`` when the
    Converse API fails; the loop catches it like any model error and records a
    fallback decision (Req 23.7), with no AWS credentials needed because the
    client is injected.
    """

    class FailingBedrockClient:
        def converse(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("bedrock endpoint down")

    model = BedrockDecisionModel(client=FailingBedrockClient())
    report = _run(_sufficient_finding(), model, RecordingToolClient())

    first = _decisions(report)[0]
    assert first.used_fallback is True
    assert first.decided_by == "fallback"
    assert first.fallback_reason is not None
    # The loop names the originating error type (BedrockDecisionError) in the reason.
    assert "BedrockDecisionError" in first.fallback_reason
    assert report.status == "complete"


# ===========================================================================
# Req 23.8 - the recorded Decision_Trace matches the tool calls that were made
# ===========================================================================


def test_req_23_8_trace_matches_tool_calls_crop_then_finalize() -> None:
    """A multi-step crop-then-finalize run: trace and tool calls correspond.

    Asserts the three correspondences the requirement enumerates:

    (a) the number/type/order of trace entries matches the actions taken -
        decision(crop), execution(crop), decision(finalize);
    (b) every ``ToolExecution`` corresponds to a decision whose action maps to
        that tool, in the same iteration;
    (c) the fake client's recorded calls exactly match the ``status='ok'``
        ToolExecution entries (same tool/args), and no call exists for a non-
        tool action.
    """
    crop = {
        "action": "inspect_crop",
        "args": {"region_id": "r-maybe-empty", "bounding_box": {"x": 10, "y": 20, "w": 30, "h": 40}},
    }
    model = InjectedDecisionModel(decisions=[crop, {"action": "finalize_report"}])
    resolved = _finding(
        "f-resolved",
        regions=[_region("r-maybe-empty", empty_space_ratio=_NOTABLE_EMPTY, uncertainty_value=0.95)],
    )
    tools = RecordingToolClient(result_ref="ref://crop-0", merge_finding=resolved)

    report = _run(_ambiguous_uncropped_finding(), model, tools)
    trace = report.decision_trace

    # (a) number / type / order of trace entries matches the actions taken.
    assert [type(e).__name__ for e in trace] == [
        "AgentDecision",
        "ToolExecution",
        "AgentDecision",
    ]
    assert isinstance(trace[0], AgentDecision) and trace[0].action == "inspect_crop"
    assert isinstance(trace[1], ToolExecution)
    assert isinstance(trace[2], AgentDecision) and trace[2].action == "finalize_report"

    # (b) each ToolExecution maps to a decision whose action backs that tool, in
    #     the same iteration.
    _assert_executions_map_to_decisions(trace)

    # (c) the recorded calls exactly match the ok ToolExecution entries, and no
    #     call exists for the finalize (non-tool) action.
    ok_execs = [e for e in _executions(report) if e.status == "ok"]
    assert len(ok_execs) == 1
    assert tools.calls == [("inspect_crop", ok_execs[0].args)]
    assert tools.calls[0][1] == {"region_id": "r-maybe-empty", "bounding_box": {"x": 10, "y": 20, "w": 30, "h": 40}}
    # finalize_report maps to no MCP tool -> it produced no call and no execution.
    assert not any(action == "finalize_report" for action, _ in tools.calls)
    assert len(tools.calls) == len(ok_execs)


def test_req_23_8_trace_matches_tool_calls_two_crops() -> None:
    """A two-crop run keeps one tool call per ok execution, in order.

    Two distinct crops (different regions) each produce one recorded call and
    one ok execution; the loop then hits the cap. The recorded calls equal the
    ok executions' ``(tool-action, args)`` in the same order, and a non-tool /
    rejected action never appears among the calls.
    """
    cfg = AgentConfig(max_iterations=2)
    crop_a = {
        "action": "inspect_crop",
        "args": {"region_id": "ra", "bounding_box": {"x": 1, "y": 1, "w": 5, "h": 5}},
    }
    crop_b = {
        "action": "inspect_crop",
        "args": {"region_id": "rb", "bounding_box": {"x": 2, "y": 2, "w": 6, "h": 6}},
    }
    model = InjectedDecisionModel(decisions=[crop_a, crop_b])
    start = _finding("f0", regions=[_region("ra", empty_space_ratio=_NOTABLE_EMPTY, uncertainty_value=0.3)])
    tools = RecordingToolClient(result_ref="ref://crop", merge_finding=start)

    report = _run(start, model, tools, cfg)

    _assert_executions_map_to_decisions(report.decision_trace)

    ok_execs = [e for e in _executions(report) if e.status == "ok"]
    assert len(ok_execs) == 2
    # Recorded calls, in order, equal each ok execution's (action, args).
    assert tools.calls == [("inspect_crop", ok_execs[0].args), ("inspect_crop", ok_execs[1].args)]
    assert tools.calls[0][1]["region_id"] == "ra"
    assert tools.calls[1][1]["region_id"] == "rb"


def test_req_23_8_no_tool_call_for_rejected_or_nontool_actions() -> None:
    """A rejected crop followed by nothing: no tool call is ever recorded.

    Pairs with Req 23.6 from the trace-integrity angle: a rejected execution
    exists in the trace but corresponds to no recorded tool call, and there is
    no ok execution. (c) therefore holds vacuously: recorded calls == ok execs
    == none.
    """
    bad_crop = {
        "action": "inspect_crop",
        "args": {"region_id": "r0", "bounding_box": {"x": 1, "y": 2, "w": 0, "h": 4}},
    }
    model = InjectedDecisionModel(decisions=[bad_crop])
    tools = RecordingToolClient()

    report = _run(
        _finding("f0", regions=[_region("r0", empty_space_ratio=_NOTABLE_EMPTY, uncertainty_value=0.3)]),
        model,
        tools,
    )

    rejected = [e for e in _executions(report) if e.status == "rejected"]
    ok_execs = [e for e in _executions(report) if e.status == "ok"]
    assert len(rejected) == 1
    assert ok_execs == []
    # No tool call was recorded for the rejected (or any) action.
    assert tools.calls == []
    assert len(tools.calls) == len(ok_execs)


# ---------------------------------------------------------------------------
# Shared trace helpers
# ---------------------------------------------------------------------------

# Maps a tool-backed agent action to the MCP tool recorded on its ToolExecution.
_ACTION_TO_TOOL = {
    "inspect_crop": "inspect_region_crop",
    "analyze_another_frame": "analyze_shelf_image",
}


def _decisions(report: InspectionReport) -> list[AgentDecision]:
    return [e for e in report.decision_trace if isinstance(e, AgentDecision)]


def _executions(report: InspectionReport) -> list[ToolExecution]:
    return [e for e in report.decision_trace if isinstance(e, ToolExecution)]


def _assert_executions_map_to_decisions(trace: list[Any]) -> None:
    """Every ToolExecution follows a decision whose action backs that tool.

    The loop appends each ``AgentDecision`` immediately followed by its
    ``ToolExecution`` (when the action runs a tool), sharing one iteration
    index. This asserts that structural invariant for Req 23.8(b): for each
    execution at position ``i``, ``trace[i-1]`` is a decision whose action maps
    to ``trace[i].tool`` and whose ``iteration`` equals the execution's.
    """
    for i, entry in enumerate(trace):
        if not isinstance(entry, ToolExecution):
            continue
        assert i > 0, "a ToolExecution cannot be the first trace entry"
        prev = trace[i - 1]
        assert isinstance(prev, AgentDecision), "a ToolExecution must follow a decision"
        assert _ACTION_TO_TOOL.get(prev.action) == entry.tool
        assert prev.iteration == entry.iteration
