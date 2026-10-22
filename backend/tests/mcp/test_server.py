"""Tests for the restricted MCP vision-tool server (task 9.1).

Covers ``shelfsentry.mcp.server`` against the real vision functions it reuses
and the security boundary the design mandates (Req 8.1, 8.2, 8.4; supports 9):

* a fake :class:`MediaResolver` returns synthetic PNG bytes (generated with
  ``cv2.imencode``), so no real storage adapter is needed and the server stays
  testable without a live transport;
* each of the four tools, driven BOTH through the pure
  :class:`VisionToolDispatcher` and through the live ``FastMCP`` ``call_tool``
  path, dispatches to the real vision functions and returns a valid
  :class:`VisionFinding` (or an evidence reference);
* ``analyze_shelf_image`` on a synthetic shelf returns at least one candidate
  region;
* no tool exposes a filesystem path, URL, credential, or shell argument in its
  JSON schema - the schemas take only ``inspection_id`` + typed ids/boxes;
* invalid arguments are rejected before any vision function runs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import cv2
import numpy as np
import pytest

from shelfsentry.data.contracts import BoundingBox, VisionFinding
from shelfsentry.mcp import (
    VisionToolDispatcher,
    build_mcp_server,
)


# ---------------------------------------------------------------------------
# Synthetic media + fake resolver
# ---------------------------------------------------------------------------


def _encode_png(img: np.ndarray) -> bytes:
    """PNG-encode a synthetic image via cv2.imencode (no file I/O)."""
    ok, buf = cv2.imencode(".png", img)
    assert ok, "failed to encode synthetic image"
    return buf.tobytes()


def _shelf_image(*, gaps: int = 1) -> np.ndarray:
    """A deterministic synthetic shelf image with ``gaps`` dark empty regions.

    Bright background with two dark shelf lips. The bottom shelf's left half is
    always a dark empty gap. With ``gaps >= 2`` the top shelf is also a dark
    gap, so a two-gap frame measures clearly MORE aggregate empty space than a
    one-gap frame (used by the frame-comparison test).
    """
    h, w = 240, 320
    img = np.full((h, w), 220, dtype=np.uint8)
    img[78:82, :] = 20
    img[158:162, :] = 20
    if gaps >= 2:
        img[40:78, 10:150] = 15
    else:
        img[40:78, 30:150] = 120
        img[40:78, 150:154] = 10
    img[100:158, 10:150] = 15
    img[100:158, 170:300] = 130
    return img


class FakeMediaResolver:
    """An in-memory :class:`MediaResolver` returning synthetic PNG bytes.

    Satisfies the structural Protocol the server depends on without any concrete
    storage adapter. Image bytes are a one-gap shelf; frames are keyed so the
    comparison test can hand back a one-gap and a two-gap frame. Stored evidence
    is kept in a dict and echoed back as an opaque ``evref://`` reference.
    """

    def __init__(self) -> None:
        self._image = _encode_png(_shelf_image(gaps=1))
        self._frames: dict[str, bytes] = {
            "frame_000000": _encode_png(_shelf_image(gaps=1)),
            "frame_000001": _encode_png(_shelf_image(gaps=2)),
        }
        self.evidence: dict[tuple[str, str], bytes] = {}
        self.put_calls: list[tuple[str, str]] = []

    def get_image(self, inspection_id: str) -> bytes:
        return self._image

    def get_frame(self, inspection_id: str, frame_id: str) -> bytes:
        return self._frames[frame_id]

    def put_evidence(self, inspection_id: str, finding_id: str, data: bytes) -> str:
        self.evidence[(inspection_id, finding_id)] = data
        self.put_calls.append((inspection_id, finding_id))
        return f"evref://{inspection_id}/{finding_id}"

    def get_evidence_ref(self, inspection_id: str, finding_id: str) -> str | None:
        if (inspection_id, finding_id) in self.evidence:
            return f"evref://{inspection_id}/{finding_id}"
        return None


@pytest.fixture()
def resolver() -> FakeMediaResolver:
    return FakeMediaResolver()


@pytest.fixture()
def dispatcher(resolver: FakeMediaResolver) -> VisionToolDispatcher:
    return VisionToolDispatcher(resolver)


# ---------------------------------------------------------------------------
# Live MCP call_tool helper
# ---------------------------------------------------------------------------


def _call_tool(server: Any, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Invoke a tool through the live FastMCP ``call_tool`` path.

    ``call_tool`` validates the arguments against the tool's JSON schema and
    returns ``(content_blocks, structured_result)``; we return the structured
    dict (the tool's actual return value) so the test asserts against the same
    data the agent would receive.
    """
    result = asyncio.run(server.call_tool(name, args))
    if isinstance(result, tuple):
        _content, structured = result
        return structured
    # Fallback: parse the first text content block as JSON.
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# Tool behavior: delegates to real vision functions, returns valid findings
# ---------------------------------------------------------------------------


def test_analyze_shelf_image_returns_valid_finding_with_regions(
    dispatcher: VisionToolDispatcher, resolver: FakeMediaResolver
) -> None:
    """analyze_shelf_image dispatches to the real pipeline and finds >= 1 region."""
    out = dispatcher.analyze_shelf_image("insp-1")

    finding = VisionFinding.model_validate(out)  # round-trips through the contract
    assert finding.source == "image"
    assert finding.inspection_id == "insp-1"
    assert finding.is_estimate is True
    assert len(finding.candidate_regions) >= 1, "synthetic shelf should yield a region"
    # Evidence was stored and its reference recorded on the finding.
    assert ("insp-1", finding.finding_id) in resolver.evidence
    assert finding.annotated_output_ref == f"evref://insp-1/{finding.finding_id}"


def test_inspect_region_crop_returns_crop_finding(
    dispatcher: VisionToolDispatcher, resolver: FakeMediaResolver
) -> None:
    """inspect_region_crop dispatches to inspect_region and returns a crop finding."""
    box = BoundingBox(x=10, y=100, w=140, h=58)  # the bottom-left empty gap
    out = dispatcher.inspect_region_crop("insp-2", box, region_id="region_0000")

    finding = VisionFinding.model_validate(out)
    assert finding.source == "crop"
    assert finding.inspection_id == "insp-2"
    assert finding.finding_id == "insp-2-region_0000"
    assert ("insp-2", finding.finding_id) in resolver.evidence


def test_inspect_region_crop_accepts_mapping_box(
    dispatcher: VisionToolDispatcher,
) -> None:
    """A plain {x, y, w, h} mapping is validated via the BoundingBox contract."""
    out = dispatcher.inspect_region_crop(
        "insp-2b", {"x": 0, "y": 0, "w": 100, "h": 100}
    )
    finding = VisionFinding.model_validate(out)
    assert finding.source == "crop"


def test_compare_shelf_frames_returns_diff_finding(
    dispatcher: VisionToolDispatcher, resolver: FakeMediaResolver
) -> None:
    """compare_shelf_frames dispatches to compare_frames and returns a diff finding."""
    out = dispatcher.compare_shelf_frames("insp-3", "frame_000000", "frame_000001")

    finding = VisionFinding.model_validate(out)
    assert finding.source == "frame_compare"
    assert finding.inspection_id == "insp-3"
    # The synthesized comparison summary region rides along first.
    assert finding.candidate_regions[0].region_id == "frame_compare_summary"
    summary = finding.candidate_regions[0].alt_metric
    assert summary is not None and summary["metric"] == "frame_compare"
    # Frame B (two gaps) is emptier than frame A (one gap): positive delta.
    assert summary["empty_space_ratio_delta"] > 0.0
    assert ("insp-3", finding.finding_id) in resolver.evidence


def test_get_annotated_evidence_returns_reference_not_bytes(
    dispatcher: VisionToolDispatcher, resolver: FakeMediaResolver
) -> None:
    """get_annotated_evidence returns an opaque reference, never raw bytes."""
    # No evidence stored yet -> render on demand and return the minted ref.
    out = dispatcher.get_annotated_evidence("insp-4", "finding-x")
    assert set(out) == {"inspection_id", "finding_id", "annotated_output_ref"}
    assert out["annotated_output_ref"] == "evref://insp-4/finding-x"
    assert "bytes" not in out and "data" not in out
    assert ("insp-4", "finding-x") in resolver.evidence

    # Second call returns the already-stored reference (no re-render needed).
    resolver.put_calls.clear()
    out2 = dispatcher.get_annotated_evidence("insp-4", "finding-x")
    assert out2["annotated_output_ref"] == "evref://insp-4/finding-x"
    assert resolver.put_calls == [], "a stored ref must not be re-stored"


# ---------------------------------------------------------------------------
# Live transport path (FastMCP call_tool) exercises the same delegation
# ---------------------------------------------------------------------------


def test_live_call_tool_all_four_tools(resolver: FakeMediaResolver) -> None:
    """Each tool dispatches correctly through the live FastMCP call_tool path."""
    server = build_mcp_server(resolver)

    analyze = _call_tool(server, "analyze_shelf_image", {"inspection_id": "live-1"})
    assert len(VisionFinding.model_validate(analyze).candidate_regions) >= 1

    crop = _call_tool(
        server,
        "inspect_region_crop",
        {"inspection_id": "live-1", "bounding_box": {"x": 10, "y": 100, "w": 140, "h": 58}},
    )
    assert VisionFinding.model_validate(crop).source == "crop"

    cmp_out = _call_tool(
        server,
        "compare_shelf_frames",
        {"inspection_id": "live-1", "frame_id_a": "frame_000000", "frame_id_b": "frame_000001"},
    )
    assert VisionFinding.model_validate(cmp_out).source == "frame_compare"

    evidence = _call_tool(
        server,
        "get_annotated_evidence",
        {"inspection_id": "live-1", "finding_id": "live-1-image"},
    )
    assert evidence["annotated_output_ref"] == "evref://live-1/live-1-image"


# ---------------------------------------------------------------------------
# Security boundary: schemas expose no path / url / credential / shell arg
# ---------------------------------------------------------------------------

_FORBIDDEN_ARG_SUBSTRINGS = (
    "path",
    "url",
    "uri",
    "file",
    "dir",
    "cred",
    "secret",
    "token",
    "key",
    "command",
    "cmd",
    "shell",
    "exec",
    "bytes",
    "data",
)


def test_tool_schemas_expose_only_ids_and_boxes(resolver: FakeMediaResolver) -> None:
    """No tool schema takes a path, URL, credential, or shell argument (Req 9)."""
    server = build_mcp_server(resolver)
    tools = {t.name: t for t in asyncio.run(server.list_tools())}

    assert set(tools) == {
        "analyze_shelf_image",
        "inspect_region_crop",
        "compare_shelf_frames",
        "get_annotated_evidence",
    }

    allowed_top_level = {
        "analyze_shelf_image": {"inspection_id"},
        "inspect_region_crop": {"inspection_id", "bounding_box", "region_id"},
        "compare_shelf_frames": {"inspection_id", "frame_id_a", "frame_id_b"},
        "get_annotated_evidence": {"inspection_id", "finding_id"},
    }

    for name, tool in tools.items():
        props = set(tool.inputSchema.get("properties", {}))
        assert props == allowed_top_level[name], f"{name} exposes unexpected args: {props}"
        # Defense in depth: no top-level property name hints at a path/url/etc.
        for prop in props:
            lowered = prop.lower()
            for bad in _FORBIDDEN_ARG_SUBSTRINGS:
                assert bad not in lowered, f"{name}.{prop} looks like a {bad} argument"


# ---------------------------------------------------------------------------
# Invalid arguments are rejected before any vision function runs (Req 8.3)
# ---------------------------------------------------------------------------


def test_live_call_tool_rejects_empty_inspection_id(
    resolver: FakeMediaResolver,
) -> None:
    """An empty inspection_id is rejected by schema validation, before dispatch."""
    server = build_mcp_server(resolver)
    with pytest.raises(Exception):  # FastMCP wraps validation errors in ToolError
        asyncio.run(server.call_tool("analyze_shelf_image", {"inspection_id": ""}))
    assert resolver.put_calls == [], "no vision work should run on invalid input"


def test_live_call_tool_rejects_missing_required_arg(
    resolver: FakeMediaResolver,
) -> None:
    """A missing required argument (frame_id_b) is rejected before dispatch."""
    server = build_mcp_server(resolver)
    with pytest.raises(Exception):
        asyncio.run(
            server.call_tool(
                "compare_shelf_frames",
                {"inspection_id": "x", "frame_id_a": "frame_000000"},
            )
        )


def test_live_call_tool_rejects_zero_size_box(resolver: FakeMediaResolver) -> None:
    """A degenerate bounding box (w=0) is rejected by the BoundingBox schema."""
    server = build_mcp_server(resolver)
    with pytest.raises(Exception):
        asyncio.run(
            server.call_tool(
                "inspect_region_crop",
                {"inspection_id": "x", "bounding_box": {"x": 0, "y": 0, "w": 0, "h": 5}},
            )
        )


# ---------------------------------------------------------------------------
# Importing the package must not require a running server / the SDK eagerly
# ---------------------------------------------------------------------------


def test_importing_package_does_not_eagerly_import_sdk() -> None:
    """shelfsentry.mcp imports its public surface without building a server."""
    import shelfsentry.mcp as mcp_pkg

    assert hasattr(mcp_pkg, "build_mcp_server")
    assert hasattr(mcp_pkg, "VisionToolDispatcher")
    assert hasattr(mcp_pkg, "MediaResolver")
