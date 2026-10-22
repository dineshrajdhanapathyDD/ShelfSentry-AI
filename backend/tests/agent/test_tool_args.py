"""Tests for agent-side tool-argument validation (task 7.1).

Covers Req 6.4 (validate tool arguments before execution and reject invalid
invocations with a structured error), which also backs the agent boundary of
Req 8.3 and the "invalid tool arguments are rejected" agent test (Req 23.6):

* ``inspect_crop`` with a valid ``region_id`` + ``bounding_box`` passes and
  normalizes;
* a missing ``region_id`` is rejected, naming ``region_id``;
* a ``bounding_box`` with ``w=0`` is rejected (via the BoundingBox contract);
* a non-dict ``bounding_box`` is rejected;
* the no-arg actions pass;
* a bad ``frame_id`` hint type is rejected;
* a rejection can be turned into a ``ToolExecution(status="rejected")``.

Decisions are built directly from the typed contracts (no AWS, no MCP, no cv2)
so validation is exercised as the pure, import-light check it is.
"""

from __future__ import annotations

from typing import Any

import pytest

from shelfsentry.agent import (
    ToolArgumentError,
    rejected_tool_execution,
    validate_tool_args,
)
from shelfsentry.data.contracts import AgentDecision, ToolExecution


def _decision(action: str, args: dict[str, Any] | None = None) -> AgentDecision:
    return AgentDecision(
        decision_id="d0",
        iteration=0,
        action=action,  # type: ignore[arg-type]
        rationale="test decision",
        args=args or {},
        decided_by="injected",
    )


# ---------------------------------------------------------------------------
# inspect_crop - valid
# ---------------------------------------------------------------------------


def test_inspect_crop_valid_region_and_bbox_passes() -> None:
    decision = _decision(
        "inspect_crop",
        {"region_id": "r0", "bounding_box": {"x": 5, "y": 6, "w": 7, "h": 8}},
    )
    normalized = validate_tool_args(decision)
    assert normalized == {
        "region_id": "r0",
        "bounding_box": {"x": 5, "y": 6, "w": 7, "h": 8},
    }


def test_inspect_crop_strips_region_id_whitespace() -> None:
    decision = _decision(
        "inspect_crop",
        {"region_id": "  r1  ", "bounding_box": {"x": 0, "y": 0, "w": 1, "h": 1}},
    )
    assert validate_tool_args(decision)["region_id"] == "r1"


# ---------------------------------------------------------------------------
# inspect_crop - rejections
# ---------------------------------------------------------------------------


def test_inspect_crop_missing_region_id_is_rejected_naming_region_id() -> None:
    decision = _decision(
        "inspect_crop", {"bounding_box": {"x": 0, "y": 0, "w": 1, "h": 1}}
    )
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    err = exc_info.value
    assert err.field == "region_id"
    assert err.code == "missing_field"
    assert "region_id" in str(err)


def test_inspect_crop_empty_region_id_is_rejected() -> None:
    decision = _decision(
        "inspect_crop",
        {"region_id": "   ", "bounding_box": {"x": 0, "y": 0, "w": 1, "h": 1}},
    )
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    assert exc_info.value.field == "region_id"


def test_inspect_crop_bbox_width_zero_is_rejected() -> None:
    # w=0 violates the BoundingBox contract (w>=1).
    decision = _decision(
        "inspect_crop",
        {"region_id": "r0", "bounding_box": {"x": 0, "y": 0, "w": 0, "h": 8}},
    )
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    err = exc_info.value
    assert err.code == "invalid_value"
    assert err.field is not None and err.field.startswith("bounding_box")
    assert "bounding_box" in err.field


def test_inspect_crop_non_dict_bbox_is_rejected() -> None:
    decision = _decision(
        "inspect_crop", {"region_id": "r0", "bounding_box": "not-a-box"}
    )
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    err = exc_info.value
    assert err.code == "invalid_type"
    assert err.field == "bounding_box"


def test_inspect_crop_missing_bbox_is_rejected() -> None:
    decision = _decision("inspect_crop", {"region_id": "r0"})
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    assert exc_info.value.field == "bounding_box"
    assert exc_info.value.code == "missing_field"


def test_inspect_crop_unexpected_key_is_rejected() -> None:
    decision = _decision(
        "inspect_crop",
        {
            "region_id": "r0",
            "bounding_box": {"x": 0, "y": 0, "w": 1, "h": 1},
            "path": "/etc/passwd",
        },
    )
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    assert exc_info.value.code == "unexpected_field"
    assert exc_info.value.field == "path"


# ---------------------------------------------------------------------------
# analyze_another_frame - optional frame hint
# ---------------------------------------------------------------------------


def test_analyze_another_frame_no_args_passes() -> None:
    assert validate_tool_args(_decision("analyze_another_frame")) == {}


def test_analyze_another_frame_valid_frame_hint_passes() -> None:
    decision = _decision("analyze_another_frame", {"frame_id": "frame-0002"})
    assert validate_tool_args(decision) == {"frame_id": "frame-0002"}


def test_analyze_another_frame_bad_frame_hint_type_is_rejected() -> None:
    decision = _decision("analyze_another_frame", {"frame_id": 123})
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    err = exc_info.value
    assert err.code == "invalid_type"
    assert err.field == "frame_id"


def test_analyze_another_frame_empty_frame_hint_is_rejected() -> None:
    decision = _decision("analyze_another_frame", {"frame_id": "   "})
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    assert exc_info.value.code == "invalid_value"
    assert exc_info.value.field == "frame_id"


# ---------------------------------------------------------------------------
# no-arg actions pass
# ---------------------------------------------------------------------------


def test_request_better_image_passes() -> None:
    assert validate_tool_args(_decision("request_better_image")) == {}


def test_request_human_review_passes_with_no_args() -> None:
    assert validate_tool_args(_decision("request_human_review")) == {}


def test_request_human_review_accepts_optional_reason() -> None:
    decision = _decision("request_human_review", {"reason": "conflicting evidence"})
    assert validate_tool_args(decision) == {"reason": "conflicting evidence"}


def test_request_human_review_bad_reason_type_is_rejected() -> None:
    decision = _decision("request_human_review", {"reason": 5})
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    assert exc_info.value.field == "reason"


def test_finalize_report_passes() -> None:
    assert validate_tool_args(_decision("finalize_report")) == {}


def test_no_arg_action_rejects_unexpected_key() -> None:
    decision = _decision("request_better_image", {"url": "http://evil"})
    with pytest.raises(ToolArgumentError) as exc_info:
        validate_tool_args(decision)
    assert exc_info.value.code == "unexpected_field"


# ---------------------------------------------------------------------------
# rejection -> ToolExecution(status="rejected")
# ---------------------------------------------------------------------------


def test_rejection_becomes_rejected_tool_execution() -> None:
    decision = _decision(
        "inspect_crop", {"bounding_box": {"x": 0, "y": 0, "w": 1, "h": 1}}
    )
    try:
        validate_tool_args(decision)
        raise AssertionError("expected rejection")
    except ToolArgumentError as err:
        execution = rejected_tool_execution(decision, err)

    assert isinstance(execution, ToolExecution)
    assert execution.status == "rejected"
    assert execution.tool == "inspect_region_crop"
    assert execution.iteration == decision.iteration
    assert execution.error is not None and "region_id" in execution.error


def test_rejected_tool_execution_for_analyze_another_frame() -> None:
    decision = _decision("analyze_another_frame", {"frame_id": 1})
    try:
        validate_tool_args(decision)
        raise AssertionError("expected rejection")
    except ToolArgumentError as err:
        execution = rejected_tool_execution(decision, err)
    assert execution.tool == "analyze_shelf_image"
    assert execution.status == "rejected"


def test_rejected_tool_execution_refuses_non_tool_action() -> None:
    # request_human_review / finalize_report do not map to an MCP tool, so a
    # rejected ToolExecution cannot be recorded for them.
    decision = _decision("finalize_report")
    with pytest.raises(ValueError):
        rejected_tool_execution(decision, "boom")
