"""MCP tool-schema argument validation tests (task 9.2; Req 8.3).

Task 9.1's ``test_server.py`` already covers the happy path, the security
boundary, and a few live-transport rejections. These tests focus narrowly on
the *validation contract* task 9.2 hardens: every invocation is validated
against the tool's JSON schema **before execution**, invalid arguments are
**rejected with a structured error**, and **no resolver or vision side effect
occurs** on rejection.

Two entry points are exercised for each rejection case:

* the direct :class:`VisionToolDispatcher` method (the explicit validation entry
  point added so direct callers cannot bypass schema validation), which raises a
  structured :class:`McpToolError`; and
* the live ``FastMCP`` ``call_tool`` path, which rejects the same inputs before
  dispatch (FastMCP schema validation plus the dispatcher's re-validation).

A :class:`RecordingResolver` counts every resolver method call, so each test can
assert zero calls happened when an invocation is rejected - proving nothing ran
downstream of validation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import cv2
import numpy as np
import pytest

from shelfsentry.data.contracts import BoundingBox, VisionFinding
from shelfsentry.mcp import (
    McpToolError,
    VisionToolDispatcher,
    build_mcp_server,
    validate_tool_args,
)


# ---------------------------------------------------------------------------
# Synthetic media + a resolver that records every call (for side-effect checks)
# ---------------------------------------------------------------------------


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok, "failed to encode synthetic image"
    return buf.tobytes()


def _shelf_image() -> np.ndarray:
    """A deterministic synthetic shelf with one dark empty gap (valid input)."""
    h, w = 240, 320
    img = np.full((h, w), 220, dtype=np.uint8)
    img[78:82, :] = 20
    img[158:162, :] = 20
    img[100:158, 10:150] = 15
    img[100:158, 170:300] = 130
    return img


class RecordingResolver:
    """A :class:`MediaResolver` that records every call it receives.

    ``calls`` captures (method, *args-identifying-ids) for *all* methods, so a
    test can assert the list is empty after a rejection: if validation rejected
    the invocation before execution, the resolver was never touched and no
    vision function ran (vision work is only reachable *through* the resolver).
    """

    def __init__(self) -> None:
        self._image = _encode_png(_shelf_image())
        self._frames = {
            "frame_000000": _encode_png(_shelf_image()),
            "frame_000001": _encode_png(_shelf_image()),
        }
        self.calls: list[tuple[str, ...]] = []

    def get_image(self, inspection_id: str) -> bytes:
        self.calls.append(("get_image", inspection_id))
        return self._image

    def get_frame(self, inspection_id: str, frame_id: str) -> bytes:
        self.calls.append(("get_frame", inspection_id, frame_id))
        return self._frames[frame_id]

    def put_evidence(self, inspection_id: str, finding_id: str, data: bytes) -> str:
        self.calls.append(("put_evidence", inspection_id, finding_id))
        return f"evref://{inspection_id}/{finding_id}"

    def get_evidence_ref(self, inspection_id: str, finding_id: str) -> str | None:
        self.calls.append(("get_evidence_ref", inspection_id, finding_id))
        return None


@pytest.fixture()
def resolver() -> RecordingResolver:
    return RecordingResolver()


@pytest.fixture()
def dispatcher(resolver: RecordingResolver) -> VisionToolDispatcher:
    return VisionToolDispatcher(resolver)


def _call_tool(server: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    result = asyncio.run(server.call_tool(name, args))
    if isinstance(result, tuple):
        _content, structured = result
        return structured
    import json

    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# validate_tool_args: the single schema-validation entry point (Req 8.3)
# ---------------------------------------------------------------------------


def test_validate_tool_args_unknown_tool_rejected() -> None:
    with pytest.raises(McpToolError) as exc:
        validate_tool_args("not_a_tool", {"inspection_id": "x"})
    assert exc.value.tool == "not_a_tool"
    assert exc.value.code == "invalid_value"


@pytest.mark.parametrize(
    ("tool", "args", "code", "field"),
    [
        # empty inspection_id (min_length=1 on the schema)
        ("analyze_shelf_image", {"inspection_id": ""}, "invalid_value", "inspection_id"),
        # missing required field
        (
            "compare_shelf_frames",
            {"inspection_id": "x", "frame_id_a": "a"},
            "missing_field",
            "frame_id_b",
        ),
        # unknown extra field (extra="forbid")
        (
            "analyze_shelf_image",
            {"inspection_id": "x", "rogue": 1},
            "unexpected_field",
            "rogue",
        ),
        # wrong type for a string id
        (
            "get_annotated_evidence",
            {"inspection_id": "x", "finding_id": 123},
            "invalid_type",
            "finding_id",
        ),
        # zero-size bounding box -> nested field identified
        (
            "inspect_region_crop",
            {"inspection_id": "x", "bounding_box": {"x": 0, "y": 0, "w": 0, "h": 5}},
            "invalid_value",
            "bounding_box.w",
        ),
        # negative box dimension -> nested field identified
        (
            "inspect_region_crop",
            {"inspection_id": "x", "bounding_box": {"x": 0, "y": 0, "w": 10, "h": -1}},
            "invalid_value",
            "bounding_box.h",
        ),
    ],
)
def test_validate_tool_args_rejects_with_structured_error(
    tool: str, args: dict[str, Any], code: str, field: str
) -> None:
    """Each invalid shape is rejected with a machine-readable error naming the field."""
    with pytest.raises(McpToolError) as exc:
        validate_tool_args(tool, args)
    err = exc.value
    assert err.tool == tool
    assert err.code == code
    assert err.field == field
    assert err.reason  # human-readable reason is populated


def test_validate_tool_args_accepts_valid_args() -> None:
    """Valid args validate and normalize (e.g. a dict box becomes a BoundingBox)."""
    model = validate_tool_args(
        "inspect_region_crop",
        {"inspection_id": "insp", "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4}},
    )
    assert isinstance(model.bounding_box, BoundingBox)  # type: ignore[attr-defined]
    assert model.inspection_id == "insp"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Direct dispatcher path: rejects BEFORE any resolver/vision side effect
# ---------------------------------------------------------------------------


def test_dispatcher_analyze_rejects_empty_id_no_side_effect(
    dispatcher: VisionToolDispatcher, resolver: RecordingResolver
) -> None:
    with pytest.raises(McpToolError) as exc:
        dispatcher.analyze_shelf_image("")
    assert exc.value.field == "inspection_id"
    assert resolver.calls == [], "no resolver/vision work may run on rejection"


def test_dispatcher_inspect_crop_rejects_zero_box_no_side_effect(
    dispatcher: VisionToolDispatcher, resolver: RecordingResolver
) -> None:
    with pytest.raises(McpToolError) as exc:
        dispatcher.inspect_region_crop("insp", {"x": 0, "y": 0, "w": 0, "h": 5})
    assert exc.value.field == "bounding_box.w"
    assert resolver.calls == []


def test_dispatcher_inspect_crop_rejects_negative_box_no_side_effect(
    dispatcher: VisionToolDispatcher, resolver: RecordingResolver
) -> None:
    with pytest.raises(McpToolError):
        dispatcher.inspect_region_crop("insp", {"x": -1, "y": 0, "w": 5, "h": 5})
    assert resolver.calls == []


def test_dispatcher_compare_rejects_empty_frame_id_no_side_effect(
    dispatcher: VisionToolDispatcher, resolver: RecordingResolver
) -> None:
    with pytest.raises(McpToolError) as exc:
        dispatcher.compare_shelf_frames("insp", "frame_000000", "")
    assert exc.value.field == "frame_id_b"
    assert resolver.calls == []


def test_dispatcher_get_evidence_rejects_empty_finding_id_no_side_effect(
    dispatcher: VisionToolDispatcher, resolver: RecordingResolver
) -> None:
    with pytest.raises(McpToolError) as exc:
        dispatcher.get_annotated_evidence("insp", "")
    assert exc.value.field == "finding_id"
    assert resolver.calls == []


# ---------------------------------------------------------------------------
# Direct dispatcher path: valid args still pass and dispatch (no regression)
# ---------------------------------------------------------------------------


def test_dispatcher_valid_args_still_dispatch(
    dispatcher: VisionToolDispatcher, resolver: RecordingResolver
) -> None:
    out = dispatcher.analyze_shelf_image("insp-ok")
    finding = VisionFinding.model_validate(out)
    assert finding.inspection_id == "insp-ok"
    # The resolver WAS used for valid input (get_image + put_evidence at least).
    methods = {c[0] for c in resolver.calls}
    assert "get_image" in methods and "put_evidence" in methods


# ---------------------------------------------------------------------------
# Live FastMCP call_tool path: same inputs rejected before execution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("analyze_shelf_image", {"inspection_id": ""}),
        ("compare_shelf_frames", {"inspection_id": "x", "frame_id_a": "a"}),  # missing b
        ("analyze_shelf_image", {"inspection_id": "x", "rogue": 1}),  # unknown field
        ("get_annotated_evidence", {"inspection_id": "x", "finding_id": 123}),  # wrong type
        (
            "inspect_region_crop",
            {"inspection_id": "x", "bounding_box": {"x": 0, "y": 0, "w": 0, "h": 5}},
        ),
        (
            "inspect_region_crop",
            {"inspection_id": "x", "bounding_box": {"x": 0, "y": 0, "w": 10, "h": -1}},
        ),
    ],
)
def test_live_call_tool_rejects_invalid_args_no_side_effect(
    resolver: RecordingResolver, tool: str, args: dict[str, Any]
) -> None:
    """The live transport rejects invalid args before any resolver/vision call."""
    server = build_mcp_server(resolver)
    with pytest.raises(Exception):  # FastMCP surfaces validation as ToolError
        _call_tool(server, tool, args)
    assert resolver.calls == [], f"{tool} must not run downstream work on rejection"


def test_live_call_tool_valid_args_still_dispatch(resolver: RecordingResolver) -> None:
    """A valid invocation still passes validation and dispatches on the live path."""
    server = build_mcp_server(resolver)
    out = _call_tool(server, "analyze_shelf_image", {"inspection_id": "live-ok"})
    assert VisionFinding.model_validate(out).inspection_id == "live-ok"
    assert {"get_image", "put_evidence"} <= {c[0] for c in resolver.calls}
