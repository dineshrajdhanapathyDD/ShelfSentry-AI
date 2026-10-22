"""The bounded active-perception agent loop (Req 6.3, 6.5-6.10; task 7.2).

This is the heart of the active-perception feedback loop (design.md "The
Active-Perception Feedback Loop"). Given an initial :class:`VisionFinding`, the
loop repeatedly:

1. asks a :class:`DecisionModel` for the next action and records the resulting
   :class:`AgentDecision` in an ordered trace (Req 6.3);
2. terminates immediately on ``finalize_report`` (sufficiency, Req 6.8) or on
   ``request_human_review`` (escalation, Req 6.9);
3. otherwise validates the chosen action's arguments *before* execution
   (Req 6.4) and, when valid, executes the backing tool via a
   :class:`ToolClient`, recording a :class:`ToolExecution` (Req 6.5);
4. merges any evidence the tool returned into the ``finding`` so it feeds the
   next iteration (Req 6.5);
5. stops once the configured ``max_iterations`` cap is reached (Req 6.6, 6.7).

In every exit path the loop finalizes an :class:`InspectionReport` that
references the decision trace and the supporting evidence (Req 6.10).

Design decisions made explicit (the task called these out):

* **Loop independence from MCP.** The loop never imports the MCP SDK. It calls
  a tiny :class:`ToolClient` protocol (:meth:`ToolClient.call`). The real MCP
  client (task 9.x) and the local wiring (task 10.2) supply a concrete
  ``ToolClient``; tests supply a fake. The loop only knows "give me a result
  that may carry a reference and/or a new finding to merge".

* **Rejected arguments do not spin forever.** When
  :func:`validate_tool_args` rejects a decision's arguments, the loop records a
  ``ToolExecution(status="rejected")`` (Req 6.4) and then **finalizes** the
  report rather than re-deciding. Re-deciding on the same unchanged finding
  would let a model that keeps emitting the same invalid call loop up to the
  cap producing nothing but rejections; finalizing deterministically on the
  first rejection is the safe, bounded choice and still records the rejection
  in the trace. (The cap independently guarantees termination regardless.)

* **Model error -> recorded fallback (the 7.2 seam for Req 7.2).** If
  ``model.decide`` raises, the loop engages the deterministic
  :class:`FallbackPolicy` for that iteration, marks the resulting decision
  ``used_fallback=True`` with a ``fallback_reason``, and continues. This is the
  generic "the loop engaged the fallback because the model errored" behavior
  that Req 7.2 describes; task 7.3 only needs to supply a
  ``BedrockDecisionModel`` as the primary ``model`` - no loop change required.
  A caller may disable this (``fallback_on_error=None``) to let a model error
  surface, but by default the loop never fails silently.

Ordering guarantee (Req 6.10). The trace is appended in causal order: each
``AgentDecision`` is appended first, then (when the action runs a tool) its
``ToolExecution`` immediately after. The two share the same ``iteration`` index.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable

from shelfsentry.agent.decision import DecisionModel
from shelfsentry.agent.fallback import FallbackPolicy
from shelfsentry.agent.tools import (
    ToolArgumentError,
    rejected_tool_execution,
    validate_tool_args,
)
from shelfsentry.config.settings import AgentConfig
from shelfsentry.data.contracts import (
    AgentAction,
    AgentDecision,
    InspectionReport,
    ToolExecution,
    TraceEntry,
    VisionFinding,
)

__all__ = [
    "ToolClient",
    "ToolResult",
    "run_agent_loop",
]


#: Maps an agent action to the restricted MCP tool that backs it (design.md
#: "Agent Loop" tool table). Only these two actions run a tool; the ``request_*``
#: and ``finalize_report`` actions are handled by the loop directly and never
#: reach :meth:`ToolClient.call`.
_ACTION_TO_TOOL: dict[str, str] = {
    "inspect_crop": "inspect_region_crop",
    "analyze_another_frame": "analyze_shelf_image",
}


class ToolResult:
    """The result of one :meth:`ToolClient.call`.

    Deliberately small and transport-agnostic: it carries a ``result_ref`` (a
    reference to the produced evidence - a finding JSON, an annotated image,
    etc., stored via the ``StorageAdapter`` by the concrete client) and an
    optional ``finding`` that the loop merges into the working finding for the
    next iteration (Req 6.5). A concrete MCP client returns references; a test
    fake can return a fresh :class:`VisionFinding` directly.

    :param result_ref: non-empty reference to the evidence the tool produced.
    :param finding: optional new finding to feed the next iteration. When
        ``None``, the working finding is left unchanged (the reference is still
        recorded on the ``ToolExecution``).
    :param error: optional error detail; when set, the execution is recorded
        with ``status="error"`` and the finding is not merged.
    """

    __slots__ = ("result_ref", "finding", "error")

    def __init__(
        self,
        result_ref: str,
        *,
        finding: VisionFinding | None = None,
        error: str | None = None,
    ) -> None:
        if not result_ref:
            raise ValueError("ToolResult.result_ref must be a non-empty string.")
        self.result_ref = result_ref
        self.finding = finding
        self.error = error


@runtime_checkable
class ToolClient(Protocol):
    """The minimal tool surface the loop calls to execute a chosen action.

    The loop stays independent of MCP concretely (design.md separation of
    concerns): it only needs to hand an action plus validated args to a client
    and get back a :class:`ToolResult`. The MCP client (task 9.x) and the local
    wiring (task 10.2) implement this; tests provide a fake.
    """

    def call(self, action: AgentAction, args: dict[str, Any]) -> ToolResult:
        """Execute the tool backing ``action`` with validated ``args``."""
        ...


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_agent_loop(
    finding: VisionFinding,
    model: DecisionModel,
    tools: ToolClient,
    cfg: AgentConfig,
    *,
    inspection_id: str,
    mode: Literal["local", "aws"] = "local",
    fallback_on_error: FallbackPolicy | None = ...,  # type: ignore[assignment]
) -> InspectionReport:
    """Run the bounded active-perception loop and return an ``InspectionReport``.

    :param finding: the initial :class:`VisionFinding` (the loop's first input).
    :param model: the :class:`DecisionModel` consulted each iteration
        (injected/fallback in local mode, Bedrock in AWS mode).
    :param tools: a :class:`ToolClient` used to execute ``inspect_crop`` /
        ``analyze_another_frame`` actions. Never called for the ``request_*`` or
        ``finalize_report`` actions.
    :param cfg: the :class:`AgentConfig`; ``cfg.max_iterations`` (default 4)
        bounds the loop (Req 6.6, 6.7).
    :param inspection_id: id stamped on the produced report.
    :param mode: operating mode recorded on the report (Req 17).
    :param fallback_on_error: the deterministic policy engaged when
        ``model.decide`` raises (Req 7.2). Defaults to a fresh
        :class:`FallbackPolicy`; pass ``None`` to let a model error propagate.
    :returns: an :class:`InspectionReport` referencing the decision trace and
        supporting evidence (Req 6.10).

    The loop is guaranteed to terminate: it runs at most ``cfg.max_iterations``
    iterations, and every non-iterating action (``finalize_report``,
    ``request_human_review``, and an argument rejection) exits immediately.
    """
    # Resolve the default fallback policy lazily so the sentinel default keeps
    # "pass None to disable" distinct from "not supplied -> use the default".
    if fallback_on_error is ...:
        fallback_on_error = FallbackPolicy()

    initial_finding_id = finding.finding_id
    trace: list[TraceEntry] = []
    iteration = 0

    while iteration < cfg.max_iterations:  # cap, default 4 (Req 6.6)
        decision = _decide(finding, trace, cfg, model, fallback_on_error)
        trace.append(decision)  # Agent_Decision recorded first (Req 6.3, 6.10)

        if decision.action == "finalize_report":  # sufficiency (Req 6.8)
            return _finalize(
                inspection_id=inspection_id,
                mode=mode,
                initial_finding_id=initial_finding_id,
                trace=trace,
                requires_human_review=False,
                status="complete",
            )

        if decision.action == "request_human_review":  # escalation (Req 6.9)
            return _finalize(
                inspection_id=inspection_id,
                mode=mode,
                initial_finding_id=initial_finding_id,
                trace=trace,
                requires_human_review=True,
                status="needs_review",
            )

        # Validate args before execution (Req 6.4). On rejection, record a
        # rejected ToolExecution and finalize deterministically (documented
        # above): never execute, never spin.
        try:
            validated_args = validate_tool_args(decision)
        except ToolArgumentError as err:
            trace.append(rejected_tool_execution(decision, err))  # Req 6.4
            return _finalize(
                inspection_id=inspection_id,
                mode=mode,
                initial_finding_id=initial_finding_id,
                trace=trace,
                requires_human_review=False,
                status="complete",
            )

        # Execute the backing tool and record the ToolExecution (Req 6.5).
        execution, new_finding = _execute(decision, validated_args, tools)
        trace.append(execution)  # immediately after its decision (Req 6.10)

        # Feed returned evidence into the next iteration (Req 6.5).
        if new_finding is not None:
            finding = new_finding

        iteration += 1

    # Reached the iteration cap -> finalize (not an error) (Req 6.7).
    return _finalize(
        inspection_id=inspection_id,
        mode=mode,
        initial_finding_id=initial_finding_id,
        trace=trace,
        requires_human_review=False,
        status="complete",
    )


# ---------------------------------------------------------------------------
# Helpers (pure except for the ToolClient call)
# ---------------------------------------------------------------------------


def _decide(
    finding: VisionFinding,
    trace: list[TraceEntry],
    cfg: AgentConfig,
    model: DecisionModel,
    fallback_on_error: FallbackPolicy | None,
) -> AgentDecision:
    """Consult ``model``; on error engage the fallback and record it (Req 7.2).

    When ``model.decide`` raises and a ``fallback_on_error`` policy is present,
    the deterministic policy produces the decision for this iteration and the
    returned decision is marked ``used_fallback=True`` with a
    ``fallback_reason`` naming the error (design.md error-handling table,
    Req 7.2). When no policy is supplied, the error propagates.
    """
    try:
        return model.decide(finding, trace, cfg)
    except Exception as exc:  # noqa: BLE001 - any model error engages fallback
        if fallback_on_error is None:
            raise
        reason = (
            f"primary decision model raised {type(exc).__name__}: {exc}; "
            "engaged deterministic fallback policy"
        )
        decision = fallback_on_error.decide(finding, trace, cfg)
        # Record that the loop engaged the fallback because the model errored.
        # The policy itself only sets decided_by="fallback"; the used_fallback
        # flag + reason are owned by the loop (Req 7.2).
        return decision.model_copy(
            update={"used_fallback": True, "fallback_reason": reason}
        )


def _execute(
    decision: AgentDecision,
    validated_args: dict[str, Any],
    tools: ToolClient,
) -> tuple[ToolExecution, VisionFinding | None]:
    """Execute the tool backing ``decision`` and build its ``ToolExecution``.

    Returns the trace entry plus any new finding to merge. A tool that raises,
    or that returns a :class:`ToolResult` carrying an ``error``, is recorded
    with ``status="error"`` and contributes no new finding (Req 6.5 records the
    tool, args, and result either way).
    """
    tool_name = _ACTION_TO_TOOL[decision.action]

    try:
        result = tools.call(decision.action, validated_args)
    except Exception as exc:  # noqa: BLE001 - a tool failure is recorded, not raised
        execution = ToolExecution(
            execution_id=f"exec-{decision.decision_id}",
            iteration=decision.iteration,
            tool=tool_name,  # type: ignore[arg-type]  # constrained by _ACTION_TO_TOOL
            args=validated_args,
            result_ref="none://error",
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )
        return execution, None

    status: Literal["ok", "error"] = "error" if result.error else "ok"
    execution = ToolExecution(
        execution_id=f"exec-{decision.decision_id}",
        iteration=decision.iteration,
        tool=tool_name,  # type: ignore[arg-type]  # constrained by _ACTION_TO_TOOL
        args=validated_args,
        result_ref=result.result_ref,
        status=status,
        error=result.error,
    )
    new_finding = result.finding if result.error is None else None
    return execution, new_finding


def _finalize(
    *,
    inspection_id: str,
    mode: Literal["local", "aws"],
    initial_finding_id: str,
    trace: list[TraceEntry],
    requires_human_review: bool,
    status: Literal["complete", "failed", "needs_review"],
) -> InspectionReport:
    """Build the final :class:`InspectionReport` referencing the trace (Req 6.10).

    ``supporting_evidence_refs`` collects the references of every recorded
    ``ToolExecution`` so the report points at the concrete evidence that backs
    the recommendation (Req 6.10). The ``final_recommendation`` is an honest,
    non-fabricated summary of how the loop terminated (Req 27.2): it states the
    terminal condition and defers the stock judgement, never asserting
    out-of-stock as proven (Req 27.4).
    """
    supporting_evidence_refs = [
        entry.result_ref
        for entry in trace
        if isinstance(entry, ToolExecution) and entry.status == "ok"
    ]

    recommendation = _recommendation(trace, requires_human_review)

    return InspectionReport(
        inspection_id=inspection_id,
        mode=mode,
        final_recommendation=recommendation,
        requires_human_review=requires_human_review,
        initial_finding_id=initial_finding_id,
        decision_trace=list(trace),
        supporting_evidence_refs=supporting_evidence_refs,
        created_at=datetime.now(timezone.utc).isoformat(),
        status=status,
    )


def _recommendation(trace: list[TraceEntry], requires_human_review: bool) -> str:
    """An honest one-line recommendation describing the terminal condition.

    Never fabricates a confidence score or asserts out-of-stock as proven
    (Req 27.2, 27.4); it reports what the loop did and what the evidence
    supports, deferring a definitive stock call to human judgement when flagged.
    """
    decisions = [e for e in trace if isinstance(e, AgentDecision)]
    executions = [e for e in trace if isinstance(e, ToolExecution)]
    last_action = decisions[-1].action if decisions else "finalize_report"

    if requires_human_review:
        return (
            "Flagged for human review: the agent found conflicting or "
            "insufficiently resolved evidence and is not asserting an "
            "out-of-stock condition. A reviewer should confirm."
        )

    rejected = any(e.status == "rejected" for e in executions)
    if rejected:
        return (
            "Finalized after an invalid tool invocation was rejected before "
            f"execution; {len(decisions)} decision(s) recorded. The finding is "
            "an estimate and empty-looking regions carry visual caveats; no "
            "out-of-stock condition is asserted as proven."
        )

    return (
        f"Finalized after {len(decisions)} decision(s) and "
        f"{sum(1 for e in executions if e.status == 'ok')} tool execution(s) "
        f"(terminal action: {last_action}). Candidate regions are estimates "
        "with visual caveats; any empty-looking region is reported with its "
        "caveats and is not asserted as a proven out-of-stock condition."
    )
