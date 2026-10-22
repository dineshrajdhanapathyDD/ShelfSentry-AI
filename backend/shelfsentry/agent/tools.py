"""Agent-side tool-argument validation (Req 6.4; task 7.1).

Before the bounded loop (task 7.2) executes the MCP tool that backs a chosen
action, it validates that action's arguments here. An invalid invocation is
*rejected before execution* with a structured, field-identifying error, so the
loop can record a ``ToolExecution`` with ``status="rejected"`` and the error
detail instead of calling the tool (design.md error-handling table: "Invalid
tool arguments (agent or MCP) -> reject before execution; record
``ToolExecution.status='rejected'``", Req 6.4; supports the agent boundary of
Req 8.3 and the "invalid tool arguments are rejected" agent test, Req 23.6).

This is the *agent-side* pre-execution check. It is conceptually shared with
the MCP tool-schema validation (task 9.2), but kept here so the loop can reject
a bad invocation without ever touching the tool layer. The module is
deliberately import-light: it pulls in only the typed contracts, never
``boto3``, ``strands``, or the MCP SDK, so the loop and its tests run with no
cloud dependencies.

The action -> required-argument contract (design.md "Agent Loop" tool table):

===========================  =========================================================
Action                       Required / accepted arguments
===========================  =========================================================
``inspect_crop``             a ``region_id`` (non-empty str) AND a ``bounding_box``
                             mapping ``{x>=0, y>=0, w>=1, h>=1}`` (ints), validated
                             via the :class:`BoundingBox` contract. Consumes the
                             region identified by the preceding finding (Req 23.2).
``analyze_another_frame``    no required args; an optional ``frame_id`` hint, which
                             must be a non-empty str when supplied.
``request_better_image``     no required args.
``request_human_review``     no required args; an optional ``reason`` str.
``finalize_report``          no required args.
===========================  =========================================================

Unknown keys for an action are rejected (closed argument sets keep the agent
boundary tight). Unknown actions are rejected too, though an ``AgentDecision``
already constrains ``action`` to the fixed five via its ``AgentAction`` literal.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from shelfsentry.data.contracts import (
    AGENT_ACTIONS,
    AgentDecision,
    BoundingBox,
    ToolExecution,
)

__all__ = [
    "ToolArgumentError",
    "validate_tool_args",
    "rejected_tool_execution",
]


class ToolArgumentError(ValueError):
    """A tool invocation was rejected because its arguments are invalid.

    Carries a machine-readable :attr:`code` and a human-readable message that
    names the offending ``action`` and ``field`` where applicable, so the loop
    can record a structured ``ToolExecution.status="rejected"`` with a useful
    ``error`` detail (Req 6.4).

    :param code: a stable, machine-readable classifier, one of
        ``"unknown_action"``, ``"missing_field"``, ``"invalid_type"``,
        ``"invalid_value"``, or ``"unexpected_field"``.
    :param action: the action whose arguments failed validation.
    :param field: the offending argument field, when the failure is tied to a
        specific field (``None`` for whole-invocation failures such as an
        unknown action).
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        action: str | None = None,
        field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.action = action
        self.field = field

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        base = super().__str__()
        parts = [f"[{self.code}]", base]
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Per-action validators. Each returns a normalized args dict or raises
# ToolArgumentError. Required args are enforced; a closed set of allowed keys
# keeps the agent boundary tight.
# ---------------------------------------------------------------------------


def _reject(
    message: str, *, code: str, action: str, field: str | None = None
) -> "ToolArgumentError":
    return ToolArgumentError(message, code=code, action=action, field=field)


def _check_no_unexpected_keys(
    args: dict[str, Any], *, action: str, allowed: frozenset[str]
) -> None:
    extra = sorted(set(args) - allowed)
    if extra:
        raise _reject(
            f"action {action!r} does not accept argument(s) "
            f"{', '.join(repr(k) for k in extra)}; "
            f"allowed: {sorted(allowed) or 'none'}.",
            code="unexpected_field",
            action=action,
            field=extra[0],
        )


def _validate_inspect_crop(args: dict[str, Any]) -> dict[str, Any]:
    """Validate ``inspect_crop``: non-empty ``region_id`` + valid ``bounding_box``.

    The bounding box is validated through the :class:`BoundingBox` contract so
    the ``x>=0, y>=0, w>=1, h>=1`` integer rules (and their field-identifying
    errors) are enforced in exactly one place.
    """
    action = "inspect_crop"
    allowed = frozenset({"region_id", "bounding_box"})
    _check_no_unexpected_keys(args, action=action, allowed=allowed)

    # region_id: required, non-empty string.
    if "region_id" not in args:
        raise _reject(
            "action 'inspect_crop' requires a 'region_id' identifying the "
            "region to crop.",
            code="missing_field",
            action=action,
            field="region_id",
        )
    region_id = args["region_id"]
    if not isinstance(region_id, str) or isinstance(region_id, bool):
        raise _reject(
            f"'region_id' must be a string, got {type(region_id).__name__}.",
            code="invalid_type",
            action=action,
            field="region_id",
        )
    region_id = region_id.strip()
    if not region_id:
        raise _reject(
            "'region_id' must be a non-empty string.",
            code="invalid_value",
            action=action,
            field="region_id",
        )

    # bounding_box: required mapping validated via the BoundingBox contract.
    if "bounding_box" not in args:
        raise _reject(
            "action 'inspect_crop' requires a 'bounding_box' "
            "{x, y, w, h} identifying the region to crop.",
            code="missing_field",
            action=action,
            field="bounding_box",
        )
    raw_box = args["bounding_box"]
    if isinstance(raw_box, BoundingBox):
        box = raw_box
    elif isinstance(raw_box, dict):
        try:
            box = BoundingBox.validate_boundary(raw_box)
        except ValidationError as exc:
            # Surface the specific offending sub-field (e.g. 'w') in the error.
            first = exc.errors()[0] if exc.errors() else {}
            loc = first.get("loc", ())
            sub = str(loc[-1]) if loc else None
            field = f"bounding_box.{sub}" if sub else "bounding_box"
            raise _reject(
                f"'bounding_box' is invalid: {exc.error_count()} error(s); "
                f"first: {first.get('msg', 'invalid value')} at "
                f"{field!r}.",
                code="invalid_value",
                action=action,
                field=field,
            ) from exc
    else:
        raise _reject(
            f"'bounding_box' must be a mapping with integer x, y, w, h, got "
            f"{type(raw_box).__name__}.",
            code="invalid_type",
            action=action,
            field="bounding_box",
        )

    return {
        "region_id": region_id,
        "bounding_box": {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
    }


def _validate_analyze_another_frame(args: dict[str, Any]) -> dict[str, Any]:
    """Validate ``analyze_another_frame``: optional ``frame_id`` hint.

    Permissive but typed: no argument is required, but when a ``frame_id`` hint
    is supplied it must be a non-empty string.
    """
    action = "analyze_another_frame"
    allowed = frozenset({"frame_id"})
    _check_no_unexpected_keys(args, action=action, allowed=allowed)

    normalized: dict[str, Any] = {}
    if "frame_id" in args:
        frame_id = args["frame_id"]
        if not isinstance(frame_id, str) or isinstance(frame_id, bool):
            raise _reject(
                f"'frame_id' hint must be a string when supplied, got "
                f"{type(frame_id).__name__}.",
                code="invalid_type",
                action=action,
                field="frame_id",
            )
        frame_id = frame_id.strip()
        if not frame_id:
            raise _reject(
                "'frame_id' hint must be a non-empty string when supplied.",
                code="invalid_value",
                action=action,
                field="frame_id",
            )
        normalized["frame_id"] = frame_id
    return normalized


def _validate_request_better_image(args: dict[str, Any]) -> dict[str, Any]:
    """Validate ``request_better_image``: no arguments accepted."""
    action = "request_better_image"
    _check_no_unexpected_keys(args, action=action, allowed=frozenset())
    return {}


def _validate_request_human_review(args: dict[str, Any]) -> dict[str, Any]:
    """Validate ``request_human_review``: optional ``reason`` string."""
    action = "request_human_review"
    allowed = frozenset({"reason"})
    _check_no_unexpected_keys(args, action=action, allowed=allowed)

    normalized: dict[str, Any] = {}
    if "reason" in args:
        reason = args["reason"]
        if not isinstance(reason, str) or isinstance(reason, bool):
            raise _reject(
                f"'reason' must be a string when supplied, got "
                f"{type(reason).__name__}.",
                code="invalid_type",
                action=action,
                field="reason",
            )
        reason = reason.strip()
        if reason:
            normalized["reason"] = reason
    return normalized


def _validate_finalize_report(args: dict[str, Any]) -> dict[str, Any]:
    """Validate ``finalize_report``: no arguments accepted."""
    action = "finalize_report"
    _check_no_unexpected_keys(args, action=action, allowed=frozenset())
    return {}


#: Dispatch table: action -> validator returning a normalized args dict.
_VALIDATORS: dict[str, Any] = {
    "inspect_crop": _validate_inspect_crop,
    "analyze_another_frame": _validate_analyze_another_frame,
    "request_better_image": _validate_request_better_image,
    "request_human_review": _validate_request_human_review,
    "finalize_report": _validate_finalize_report,
}


def validate_tool_args(decision: AgentDecision) -> dict[str, Any]:
    """Validate a decision's tool arguments before execution (Req 6.4).

    Returns a normalized, validated ``args`` dict for the decision's action, or
    raises :class:`ToolArgumentError` naming the offending field. Pure and
    side-effect free: it validates only and never executes a tool. The loop
    calls this before dispatching the MCP tool; on a raised error it records a
    ``ToolExecution`` with ``status="rejected"`` (see
    :func:`rejected_tool_execution`) rather than calling the tool.

    :param decision: the selected :class:`AgentDecision` whose ``action`` and
        ``args`` are checked.
    :returns: a normalized copy of the arguments valid for ``decision.action``.
    :raises ToolArgumentError: if the action is unknown or the arguments are
        missing, mistyped, out of range, or contain unexpected keys.
    """
    action = decision.action
    validator = _VALIDATORS.get(action)
    if validator is None:
        raise ToolArgumentError(
            f"unknown action {action!r}; expected one of {sorted(AGENT_ACTIONS)}.",
            code="unknown_action",
            action=action,
            field=None,
        )
    # decision.args defaults to {} on the contract; guard against a caller that
    # passed something non-dict-like via an untyped path.
    args = decision.args if isinstance(decision.args, dict) else {}
    return validator(args)


def rejected_tool_execution(
    decision: AgentDecision,
    error: "ToolArgumentError | str",
    *,
    execution_id: str | None = None,
    result_ref: str = "none://rejected",
) -> ToolExecution:
    """Build a ``ToolExecution`` recording that an invocation was rejected.

    Convenience for the loop (task 7.2): turn a :func:`validate_tool_args`
    rejection into the trace entry the design prescribes, a ``ToolExecution``
    with ``status="rejected"`` and the structured error detail (Req 6.4). No
    tool is executed, so ``result_ref`` is a sentinel reference by default.

    Note: ``ToolExecution.tool`` is one of the four restricted MCP tool names,
    not an agent action. This helper only applies to the two actions that map
    to a real MCP tool - ``inspect_crop`` (-> ``inspect_region_crop``) and
    ``analyze_another_frame`` (-> ``analyze_shelf_image``). For the
    non-tool actions (``request_*`` / ``finalize_report``) there is no tool call
    to record; a rejection there is surfaced via the raised error directly.

    :param decision: the rejected decision.
    :param error: the :class:`ToolArgumentError` (or a message) describing why.
    :param execution_id: optional id for the execution record; defaults to one
        derived from the decision.
    :param result_ref: reference stored on the record; defaults to a sentinel
        since no tool ran.
    :raises ValueError: if ``decision.action`` does not map to an MCP tool, so
        the caller does not silently mis-record a non-tool rejection.
    """
    action_to_tool = {
        "inspect_crop": "inspect_region_crop",
        "analyze_another_frame": "analyze_shelf_image",
    }
    tool = action_to_tool.get(decision.action)
    if tool is None:
        raise ValueError(
            f"action {decision.action!r} does not map to an MCP tool, so a "
            "rejected ToolExecution cannot be recorded for it; surface the "
            "ToolArgumentError directly instead."
        )

    detail = str(error)
    return ToolExecution(
        execution_id=execution_id or f"rejected-{decision.decision_id}",
        iteration=decision.iteration,
        tool=tool,  # type: ignore[arg-type]  # constrained by action_to_tool
        args=decision.args if isinstance(decision.args, dict) else {},
        result_ref=result_ref,
        status="rejected",
        error=detail,
    )
