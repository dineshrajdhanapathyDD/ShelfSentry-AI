"""Tests for Annotated_Output rendering (task 3.3; Req 2.4, 4.5, 5.3).

Covers ``shelfsentry.vision.annotate``:

* :func:`render_annotation` draws onto the original image and returns decodable
  bytes whose shape matches the input (Req 2.4, 4.5).
* Annotations land at *exactly* the recorded bounding-box coordinates: the
  drawn rectangle color appears along the recorded box border (Req 5.3).
* Rendering is deterministic: two renders of the same ``(bytes, finding)`` are
  byte-for-byte identical (Req 5.1).
* :func:`build_annotated_finding` returns a consistent ``(finding, bytes)``
  pair, reusing ``build_finding`` as the assembler (not duplicating it).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import (
    BoundingBox,
    CandidateRegion,
    VisionFinding,
)
from shelfsentry.vision.analyze import analyze_image_bytes
from shelfsentry.vision.annotate import (
    ANNOTATION_COLOR_BGR,
    build_annotated_finding,
    render_annotation,
)
from shelfsentry.vision.scoring import build_finding


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def _blank_png(h: int = 120, w: int = 160) -> bytes:
    """Encode a plain mid-gray image as PNG bytes (no green anywhere)."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _synthetic_shelf_png() -> bytes:
    """A two-shelf image with a clear empty gap (reused from the scoring tests)."""
    h, w = 240, 320
    img = np.full((h, w), 220, dtype=np.uint8)
    img[78:82, :] = 20
    img[158:162, :] = 20
    img[40:78, 30:150] = 120
    img[40:78, 150:154] = 10
    img[100:158, 10:150] = 15
    img[100:158, 170:300] = 130
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _finding_with_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> VisionFinding:
    """Build a minimal valid VisionFinding with known bounding boxes."""
    regions = [
        CandidateRegion(
            region_id=f"region_{i:04d}",
            bounding_box=BoundingBox(x=x, y=y, w=w, h=h),
            empty_space_ratio=0.5,
            heuristic_score=0.5,
            uncertainty_value=0.5,
            visual_caveat="estimate only; shadow, occlusion, dividers, image quality.",
        )
        for i, (x, y, w, h) in enumerate(boxes)
    ]
    return VisionFinding(
        finding_id="find-1",
        inspection_id="insp-1",
        source="image",
        opencv_version="5.0.0",
        candidate_regions=regions,
        uncertainty_interpretations={"uncertainty_value": "confidence in [0,1]."},
        annotated_output_ref="ref://annotated/find-1.png",
        suggested_follow_up="finalize_report",
    )


@pytest.fixture()
def cfg() -> VisionConfig:
    return VisionConfig()


# ---------------------------------------------------------------------------
# Output shape + decodability (Req 2.4, 4.5)
# ---------------------------------------------------------------------------


def test_output_decodes_and_shape_matches_input() -> None:
    """render_annotation returns decodable bytes with the input's shape."""
    data = _blank_png(120, 160)
    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    finding = _finding_with_boxes([(20, 30, 40, 25)])
    out = render_annotation(data, finding)

    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == original.shape


def test_no_regions_leaves_image_unchanged() -> None:
    """A finding with zero regions annotates nothing (image content unchanged)."""
    data = _blank_png(100, 100)
    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    finding = _finding_with_boxes([])
    out = render_annotation(data, finding)
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)

    assert np.array_equal(decoded, original)


# ---------------------------------------------------------------------------
# Annotations match recorded bounding boxes (Req 5.3)
# ---------------------------------------------------------------------------


def _is_annotation_color(pixel: np.ndarray) -> bool:
    """True if a BGR pixel equals the fixed annotation color."""
    return bool(
        pixel[0] == ANNOTATION_COLOR_BGR[0]
        and pixel[1] == ANNOTATION_COLOR_BGR[1]
        and pixel[2] == ANNOTATION_COLOR_BGR[2]
    )


def test_rectangle_drawn_at_recorded_coordinates() -> None:
    """The drawn color appears along the recorded box border (Req 5.3).

    The source image is uniform gray with no green, so any annotation-colored
    pixel on the recorded box edges must come from the rectangle drawn at those
    exact coordinates.
    """
    data = _blank_png(200, 200)
    box = (50, 60, 70, 40)  # x, y, w, h
    finding = _finding_with_boxes([box])
    out = render_annotation(data, finding)
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)

    x, y, w, h = box
    x1, y1, x2, y2 = x, y, x + w - 1, y + h - 1

    # Midpoints of each of the four recorded edges must carry the drawn color.
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    assert _is_annotation_color(decoded[y1, mid_x]), "top edge not at recorded y"
    assert _is_annotation_color(decoded[y2, mid_x]), "bottom edge not at recorded y"
    assert _is_annotation_color(decoded[mid_y, x1]), "left edge not at recorded x"
    assert _is_annotation_color(decoded[mid_y, x2]), "right edge not at recorded x"

    # The four recorded corners must also carry the drawn color.
    for cy, cx in ((y1, x1), (y1, x2), (y2, x1), (y2, x2)):
        assert _is_annotation_color(decoded[cy, cx]), f"corner ({cx},{cy}) missing"


def test_interior_far_from_border_is_untouched() -> None:
    """Pixels well inside the box (away from label/border) stay the source color."""
    data = _blank_png(200, 200)
    box = (40, 50, 90, 90)
    finding = _finding_with_boxes([box])
    out = render_annotation(data, finding)
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)

    x, y, w, h = box
    # Deep interior center, far from the 2px border and the label.
    cy, cx = y + h // 2, x + w // 2
    assert not _is_annotation_color(decoded[cy, cx])
    assert tuple(int(v) for v in decoded[cy, cx]) == (128, 128, 128)


# ---------------------------------------------------------------------------
# Determinism (Req 5.1)
# ---------------------------------------------------------------------------


def test_render_is_byte_identical_across_runs() -> None:
    """Two renders of the same (bytes, finding) are byte-for-byte identical."""
    data = _blank_png(150, 170)
    finding = _finding_with_boxes([(10, 10, 50, 40), (90, 70, 60, 50)])
    first = render_annotation(data, finding)
    second = render_annotation(data, finding)
    assert first == second


def test_render_does_not_mutate_source_on_repeat() -> None:
    """Re-decoding the original bytes after rendering yields the original image."""
    data = _blank_png(100, 100)
    before = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).copy()
    finding = _finding_with_boxes([(10, 10, 30, 30)])
    render_annotation(data, finding)
    after = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(before, after)


# ---------------------------------------------------------------------------
# build_annotated_finding convenience (reuses build_finding; Req 4.5)
# ---------------------------------------------------------------------------


def test_build_annotated_finding_matches_build_finding(cfg: VisionConfig) -> None:
    """The convenience returns the same finding build_finding would, plus bytes."""
    data = _synthetic_shelf_png()
    analysis = analyze_image_bytes(data, cfg)

    kwargs = dict(
        inspection_id="insp-1",
        finding_id="find-1",
        annotated_output_ref="ref://annotated/find-1.png",
        source="image",
        cfg=cfg,
    )
    expected = build_finding(analysis, **kwargs)
    finding, annotated = build_annotated_finding(analysis, data, **kwargs)

    # Finding assembly is delegated, not duplicated: identical canonical JSON.
    assert finding.canonical_json() == expected.canonical_json()
    assert finding.annotated_output_ref == "ref://annotated/find-1.png"

    # Annotated bytes decode to the original image shape.
    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    decoded = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == original.shape

    # Rendering via the convenience equals rendering the built finding directly.
    assert annotated == render_annotation(data, finding)


def test_build_annotated_finding_boxes_drawn_on_shelf_image(cfg: VisionConfig) -> None:
    """Each recorded region's border carries the annotation color on a real analysis."""
    data = _synthetic_shelf_png()
    analysis = analyze_image_bytes(data, cfg)
    finding, annotated = build_annotated_finding(
        analysis,
        data,
        inspection_id="insp-1",
        finding_id="find-1",
        annotated_output_ref="ref://annotated/find-1.png",
        cfg=cfg,
    )
    assert len(finding.candidate_regions) >= 1

    decoded = cv2.imdecode(np.frombuffer(annotated, np.uint8), cv2.IMREAD_COLOR)
    for region in finding.candidate_regions:
        b = region.bounding_box
        mid_x = b.x + b.w // 2
        top = decoded[b.y, mid_x]
        assert _is_annotation_color(top), (
            f"{region.region_id}: top edge not drawn at recorded y={b.y}"
        )
