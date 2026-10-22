"""Tests for the bounded active-perception agent loop (task 7.2).

Covers the loop behavior in Req 6.3, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10 and the
fallback-on-model-error seam for Req 7.2, driven with
:class:`InjectedDecisionModel` and a fake :class:`ToolClient` so no AWS, MCP, or
cv2 dependency is touched:

* a finalize path -> ``status="complete"``, no human review;
* a human-review path -> ``status="needs_review"``, ``requires_human_review``;
* a crop-then-finalize path whose trace is ``AgentDecision`` then its
  ``ToolExecution`` in order, with the returned evidence merged;
* the max-iterations cap is respected (a model that never finalizes still
  stops at ``cfg.max_iterations``);
* a rejected-args path records a ``ToolExecution(status="rejected")`` and does
  not execute the tool;
* a model that raises engages the fallback (``used_fallback=True``) and records
  a reason.
"""

from __future__ import annotations

from typing import Any

import pytest

from shelfsentry.agent import (
    FallbackPolicy,
    InjectedDecisionModel,
    ToolResult,
    run_agent_loop,
)
from shelfsentry.agent.loop import ToolClient
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
# Builders
# ---------------------------------------------------------------------------


def _region(
    region_id: str = "r0",
    *,
    empty_space_ratio: float | None = 0.8,
    uncertainty_value: float = 0.3,
    x: int = 1,
    y: int = 2,
    w: int = 3,
    h: int = 4,
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
    finding_id: str = "f0",
    *,
    regions: list[CandidateRegion] | None = None,
    source: str = "image",
    frame_id: str | None = None,
) -> VisionFinding:
    return VisionFinding(
        finding_id=finding_id,
        inspection_id="insp-1",
        source=source,  # type: ignore[arg-type]
        frame_id=frame_id,
        opencv_version="5.0.0",
        candidate_regions=regions if regions is not None else [_region()],
        uncertainty_interpretations={},
        annotated_output_ref="ref://annotated",
        suggested_follow_up="finalize_report",
    )


def _cfg(**overrides: Any) -> AgentConfig:
    return AgentConfig(**overrides)


class FakeToolClient:
    """A recording fake :class:`ToolClient`.

    Records each ``call`` and returns a configured :class:`ToolResult`. When
    ``merge_finding`` is set, that finding is returned so the loop merges it
    into the next iteration; otherwise only a reference is returned.
    """

    def __init__(
        self,
        *,
        result_ref: str = "ref://tool-result",
        merge_finding: VisionFinding | None = None,
        error: str | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.result_ref = result_ref
        self.merge_finding = merge_finding
        self.error = error
        self.raises = raises
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, action: str, args: dict[str, Any]) -> ToolResult:  # type: ignore[override]
        self.calls.append((action, dict(args)))
        if self.raises is not None:
            raise self.raises
        return ToolResult(
            self.result_ref, finding=self.merge_finding, error=self.error
        )


def _run(
    model: Any,
    tools: ToolClient,
    *,
    finding: VisionFinding | None = None,
    cfg: AgentConfig | None = None,
    **kwargs: Any,
) -> InspectionReport:
    return run_agent_loop(
        finding if finding is not None else _finding(),
        model,
        tools,
        cfg if cfg is not None else _cfg(),
        inspection_id="insp-1",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# finalize path (Req 6.8)
# ---------------------------------------------------------------------------


def test_finalize_path_produces_complete_report() -> None:
    model = InjectedDecisionModel(decisions=[{"action": "finalize_report"}])
    tools = FakeToolClient()

    report = _run(model, tools)

    assert isinstance(report, InspectionReport)
    assert report.status == "complete"
    assert report.requires_human_review is False
    assert report.inspection_id == "insp-1"
    assert report.initial_finding_id == "f0"
    # One decision recorded, no tool executed.
    assert len(report.decision_trace) == 1
    assert isinstance(report.decision_trace[0], AgentDecision)
    assert report.decision_trace[0].action == "finalize_report"
    assert tools.calls == []


# ---------------------------------------------------------------------------
# human-review path (Req 6.9)
# ---------------------------------------------------------------------------


def test_human_review_path_marks_needs_review() -> None:
    model = InjectedDecisionModel(
        decisions=[{"action": "request_human_review", "args": {}}]
    )
    tools = FakeToolClient()

    report = _run(model, tools)

    assert report.status == "needs_review"
    assert report.requires_human_review is True
    # Stopped immediately; no tool call.
    assert tools.calls == []
    assert len(report.decision_trace) == 1
    assert report.decision_trace[0].action == "request_human_review"


# ---------------------------------------------------------------------------
# crop-then-finalize path with ordered trace (Req 6.3, 6.5, 6.10)
# ---------------------------------------------------------------------------


def test_crop_then_finalize_orders_trace_and_merges_evidence() -> None:
    crop = {
        "action": "inspect_crop",
        "args": {
            "region_id": "r0",
            "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
        },
    }
    model = InjectedDecisionModel(decisions=[crop, {"action": "finalize_report"}])
    merged = _finding("f1-crop", regions=[_region("r0", uncertainty_value=0.9)])
    tools = FakeToolClient(result_ref="ref://crop-0", merge_finding=merged)

    report = _run(model, tools)

    assert report.status == "complete"
    # Trace order: decision(crop), execution(crop), decision(finalize).
    trace = report.decision_trace
    assert len(trace) == 3
    assert isinstance(trace[0], AgentDecision) and trace[0].action == "inspect_crop"
    assert isinstance(trace[1], ToolExecution)
    assert trace[1].tool == "inspect_region_crop"
    assert trace[1].status == "ok"
    assert trace[1].result_ref == "ref://crop-0"
    # Decision and its execution share the same iteration index.
    assert trace[0].iteration == trace[1].iteration == 0
    assert isinstance(trace[2], AgentDecision) and trace[2].action == "finalize_report"
    assert trace[2].iteration == 1
    # The tool was called with the validated args.
    assert tools.calls == [
        ("inspect_crop", {"region_id": "r0", "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}})
    ]
    # The ok execution's ref is referenced as supporting evidence (Req 6.10).
    assert "ref://crop-0" in report.supporting_evidence_refs


def test_merged_finding_feeds_next_iteration() -> None:
    # First decision branches on the finding id; the fake merges a finding with
    # a new id so the second decision can observe the merge.
    seen_ids: list[str] = []

    def fn(finding: VisionFinding, trace: list[Any], cfg: AgentConfig) -> AgentDecision:
        seen_ids.append(finding.finding_id)
        iteration = len(seen_ids) - 1
        if finding.finding_id == "f0":
            return AgentDecision(
                decision_id=f"d{iteration}",
                iteration=iteration,
                action="inspect_crop",
                rationale="crop first",
                args={
                    "region_id": "r0",
                    "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
                },
                decided_by="injected",
            )
        return AgentDecision(
            decision_id=f"d{iteration}",
            iteration=iteration,
            action="finalize_report",
            rationale="done",
            args={},
            decided_by="injected",
        )

    merged = _finding("f1-crop")
    tools = FakeToolClient(merge_finding=merged)
    report = _run(InjectedDecisionModel(fn=fn), tools)

    assert seen_ids == ["f0", "f1-crop"]
    assert report.status == "complete"


# ---------------------------------------------------------------------------
# max-iterations cap (Req 6.6, 6.7)
# ---------------------------------------------------------------------------


def test_max_iterations_cap_is_respected() -> None:
    # A model that always crops never finalizes on its own; the loop must still
    # stop at the cap and produce a report.
    def always_crop(
        finding: VisionFinding, trace: list[Any], cfg: AgentConfig
    ) -> AgentDecision:
        iteration = sum(1 for e in trace if isinstance(e, AgentDecision))
        return AgentDecision(
            decision_id=f"d{iteration}",
            iteration=iteration,
            action="inspect_crop",
            rationale="keep cropping",
            args={
                "region_id": "r0",
                "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
            },
            decided_by="injected",
        )

    tools = FakeToolClient(merge_finding=_finding("f0"))  # merge keeps id stable
    cfg = _cfg(max_iterations=3)
    report = _run(InjectedDecisionModel(fn=always_crop), tools, cfg=cfg)

    assert report.status == "complete"
    assert report.requires_human_review is False
    # Exactly max_iterations decisions, each with a tool execution.
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    executions = [e for e in report.decision_trace if isinstance(e, ToolExecution)]
    assert len(decisions) == 3
    assert len(executions) == 3
    assert len(tools.calls) == 3


def test_default_cap_is_four() -> None:
    def always_crop(
        finding: VisionFinding, trace: list[Any], cfg: AgentConfig
    ) -> AgentDecision:
        iteration = sum(1 for e in trace if isinstance(e, AgentDecision))
        return AgentDecision(
            decision_id=f"d{iteration}",
            iteration=iteration,
            action="inspect_crop",
            rationale="keep cropping",
            args={
                "region_id": "r0",
                "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
            },
            decided_by="injected",
        )

    tools = FakeToolClient(merge_finding=_finding("f0"))
    report = _run(InjectedDecisionModel(fn=always_crop), tools)  # default cfg
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert len(decisions) == 4  # DEFAULT_MAX_ITERATIONS


# ---------------------------------------------------------------------------
# rejected-args path (Req 6.4)
# ---------------------------------------------------------------------------


def test_rejected_args_records_rejected_execution_and_does_not_call_tool() -> None:
    # inspect_crop missing the required bounding_box -> validation rejects it.
    bad_crop = {"action": "inspect_crop", "args": {"region_id": "r0"}}
    model = InjectedDecisionModel(decisions=[bad_crop])
    tools = FakeToolClient()

    report = _run(model, tools)

    # The tool was never executed.
    assert tools.calls == []
    # Trace: decision then a rejected ToolExecution.
    trace = report.decision_trace
    assert len(trace) == 2
    assert isinstance(trace[0], AgentDecision)
    assert isinstance(trace[1], ToolExecution)
    assert trace[1].status == "rejected"
    assert trace[1].tool == "inspect_region_crop"
    assert trace[1].error is not None and "bounding_box" in trace[1].error
    # Finalized deterministically (no infinite loop on repeated rejections).
    assert report.status == "complete"


# ---------------------------------------------------------------------------
# model error -> recorded fallback (Req 7.2)
# ---------------------------------------------------------------------------


class RaisingModel:
    """A decision model whose ``decide`` always raises."""

    def decide(self, finding: VisionFinding, trace: list[Any], cfg: AgentConfig) -> AgentDecision:
        raise RuntimeError("bedrock unavailable")


def test_model_error_engages_fallback_and_records_it() -> None:
    # A clear, sufficient finding so the fallback deterministically finalizes.
    clear = _finding(regions=[_region("r0", empty_space_ratio=0.1, uncertainty_value=0.95)])
    model = RaisingModel()
    tools = FakeToolClient()

    report = _run(model, tools, finding=clear)

    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert decisions, "expected at least one recorded decision"
    first = decisions[0]
    assert first.used_fallback is True
    assert first.decided_by == "fallback"
    assert first.fallback_reason is not None
    assert "RuntimeError" in first.fallback_reason
    assert report.status == "complete"


def test_model_error_without_fallback_propagates() -> None:
    model = RaisingModel()
    tools = FakeToolClient()
    with pytest.raises(RuntimeError, match="bedrock unavailable"):
        _run(model, tools, fallback_on_error=None)


def test_fallback_can_be_supplied_explicitly() -> None:
    clear = _finding(regions=[_region("r0", empty_space_ratio=0.1, uncertainty_value=0.95)])
    report = _run(
        RaisingModel(),
        FakeToolClient(),
        finding=clear,
        fallback_on_error=FallbackPolicy(),
    )
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert decisions[0].used_fallback is True


# ---------------------------------------------------------------------------
# tool error is recorded, not raised (Req 6.5)
# ---------------------------------------------------------------------------


def test_tool_error_is_recorded_as_error_status() -> None:
    crop = {
        "action": "inspect_crop",
        "args": {"region_id": "r0", "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}},
    }
    model = InjectedDecisionModel(decisions=[crop, {"action": "finalize_report"}])
    tools = FakeToolClient(result_ref="none://err", error="tool blew up")

    report = _run(model, tools)

    executions = [e for e in report.decision_trace if isinstance(e, ToolExecution)]
    assert len(executions) == 1
    assert executions[0].status == "error"
    assert executions[0].error == "tool blew up"
    # An errored execution's ref is NOT listed as supporting evidence.
    assert "none://err" not in report.supporting_evidence_refs


def test_tool_raising_is_recorded_as_error_status() -> None:
    crop = {
        "action": "inspect_crop",
        "args": {"region_id": "r0", "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}},
    }
    model = InjectedDecisionModel(decisions=[crop, {"action": "finalize_report"}])
    tools = FakeToolClient(raises=ValueError("boom"))

    report = _run(model, tools)

    executions = [e for e in report.decision_trace if isinstance(e, ToolExecution)]
    assert len(executions) == 1
    assert executions[0].status == "error"
    assert executions[0].error is not None and "ValueError" in executions[0].error
