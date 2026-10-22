"""Property-based tests for coordinate reproducibility + annotation match (task 4.2).

Covers two properties over *generated* synthetic shelf-like images (not the
fixed example assertions already in ``test_annotate.py``):

* **Property 8: Annotation coordinates are reproducible** (Req 5.1, and the
  repeated-run coordinate consistency test, Req 22.4). Running
  :func:`shelfsentry.vision.analyze.analyze_image_bytes` and
  :func:`shelfsentry.vision.scoring.build_finding` twice over the *same* bytes
  and config yields byte-identical candidate bounding boxes and byte-identical
  ``canonical_json`` for the assembled finding.

* **Property 9: Annotations match recorded bounding boxes** (Req 5.3). For a
  finding rendered by :func:`shelfsentry.vision.annotate.render_annotation`, the
  fixed ``ANNOTATION_COLOR_BGR`` appears along every recorded box border - i.e.
  the drawn rectangle lands at exactly the recorded ``(x, y, w, h)``
  coordinates. The source images are built from grayscale content (gray -> BGR
  decode), so they contain no green anywhere; any annotation-colored pixel on a
  recorded edge must therefore come from the rectangle drawn at that edge.

Synthetic images are generated deterministically from Hypothesis-drawn
parameters (size, shelf gaps, product blocks, lighting offset). Images are kept
small so >= 100 examples per property stay fast.
"""

from __future__ import annotations

import cv2
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from shelfsentry.config.settings import VisionConfig
from shelfsentry.vision.analyze import analyze_image_bytes
from shelfsentry.vision.annotate import ANNOTATION_COLOR_BGR, render_annotation
from shelfsentry.vision.scoring import build_finding


# ---------------------------------------------------------------------------
# Deterministic synthetic shelf-image generator
# ---------------------------------------------------------------------------


def _render_shelf_png(
    *,
    height: int,
    width: int,
    n_shelves: int,
    gap_fraction: float,
    lighting: int,
) -> bytes:
    """Build a small shelf-like grayscale image deterministically and PNG-encode it.

    The image has a bright background, ``n_shelves`` dark horizontal shelf lips,
    and on each shelf a product block followed by a dark empty gap sized by
    ``gap_fraction``. ``lighting`` shifts the whole frame's brightness to
    exercise the lighting-normalization path. Everything is a pure function of
    the arguments (no randomness), so the same arguments produce identical bytes
    - which is what the reproducibility property relies on.

    The content is single-channel grayscale; decoding it to BGR yields equal
    channels, so the rendered image contains no green (the annotation color) and
    Property 9 can attribute any green pixel to the drawn rectangle.

    Returns:
        PNG-encoded bytes of the synthetic image.
    """
    base = int(np.clip(210 + lighting, 0, 255))
    img = np.full((height, width), base, dtype=np.uint8)

    # Horizontal shelf lips evenly spaced; the band under each lip holds a
    # product block on the left and an empty (dark) gap on the right.
    band_h = height // (n_shelves + 1)
    for s in range(n_shelves):
        lip_y = (s + 1) * band_h
        y0 = max(0, lip_y - 2)
        y1 = min(height, lip_y + 2)
        img[y0:y1, :] = int(np.clip(20 + lighting, 0, 255))

        # Band contents sit just below the lip.
        content_top = min(height - 1, lip_y + 3)
        content_bot = min(height, lip_y + band_h - 3)
        if content_bot - content_top < 3:
            continue

        split = int(width * (1.0 - gap_fraction))
        split = max(2, min(width - 2, split))
        # Product block: mid-gray filled.
        img[content_top:content_bot, 1:split] = int(np.clip(120 + lighting, 0, 255))
        # Empty gap: dark.
        img[content_top:content_bot, split : width - 1] = int(
            np.clip(12 + lighting, 0, 255)
        )

    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode failed to encode the synthetic shelf image"
    return buf.tobytes()


# --- Strategies -------------------------------------------------------------

# Small dimensions keep decode/analyze/annotate fast at >= 100 examples.
_shelf_image = st.builds(
    _render_shelf_png,
    height=st.integers(min_value=80, max_value=200),
    width=st.integers(min_value=80, max_value=240),
    n_shelves=st.integers(min_value=1, max_value=4),
    gap_fraction=st.floats(min_value=0.15, max_value=0.6),
    lighting=st.integers(min_value=-30, max_value=30),
)


def _cfg() -> VisionConfig:
    """Default vision config (fixed kernels/thresholds -> deterministic)."""
    return VisionConfig()


def _is_annotation_color(pixel: np.ndarray) -> bool:
    """True if a decoded BGR pixel equals the fixed annotation color."""
    return bool(
        pixel[0] == ANNOTATION_COLOR_BGR[0]
        and pixel[1] == ANNOTATION_COLOR_BGR[1]
        and pixel[2] == ANNOTATION_COLOR_BGR[2]
    )


# ---------------------------------------------------------------------------
# Property 8: Annotation coordinates are reproducible (Req 5.1; also Req 22.4)
# ---------------------------------------------------------------------------


@given(data=_shelf_image)
@settings(max_examples=120)
def test_property8_candidate_boxes_reproducible(data: bytes) -> None:
    """Property 8: repeated analysis over the same bytes yields identical boxes.

    Running ``analyze_image_bytes`` twice over identical ``(bytes, config)``
    produces the same ordered candidate bounding boxes, box-for-box. This is the
    repeated-run coordinate consistency check (Req 22.4) at the analysis layer.

    Validates: Requirements 5.1
    """
    cfg = _cfg()
    first = analyze_image_bytes(data, cfg)
    second = analyze_image_bytes(data, cfg)

    boxes_first = [c.bounding_box for c in first.candidates]
    boxes_second = [c.bounding_box for c in second.candidates]
    assert boxes_first == boxes_second

    # Image dimensions are also stable across runs.
    assert (first.height, first.width) == (second.height, second.width)


@given(data=_shelf_image)
@settings(max_examples=120)
def test_property8_finding_canonical_json_reproducible(data: bytes) -> None:
    """Property 8: the assembled finding is byte-identical across repeated runs.

    Building a ``VisionFinding`` from a fresh analysis of the same bytes/config
    twice yields byte-for-byte identical ``canonical_json`` - the emitted
    coordinates and every derived value are reproducible (Req 5.1, Req 22.4).

    Validates: Requirements 5.1
    """
    cfg = _cfg()

    def build() -> str:
        analysis = analyze_image_bytes(data, cfg)
        finding = build_finding(
            analysis,
            inspection_id="insp-repro",
            finding_id="find-repro",
            annotated_output_ref="ref://annotated/find-repro.png",
            source="image",
            cfg=cfg,
        )
        return finding.canonical_json()

    assert build() == build()


@given(data=_shelf_image)
@settings(max_examples=100)
def test_property8_annotation_bytes_reproducible(data: bytes) -> None:
    """Property 8: rendering the annotation twice is byte-for-byte identical.

    The recorded boxes drive the drawn rectangles, so if coordinates are
    reproducible the encoded Annotated_Output must be identical across runs too
    (Req 5.1).

    Validates: Requirements 5.1
    """
    cfg = _cfg()
    analysis = analyze_image_bytes(data, cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-repro",
        finding_id="find-repro",
        annotated_output_ref="ref://annotated/find-repro.png",
        source="image",
        cfg=cfg,
    )
    first = render_annotation(data, finding)
    second = render_annotation(data, finding)
    assert first == second


# ---------------------------------------------------------------------------
# Property 9: Annotations match recorded bounding boxes (Req 5.3)
# ---------------------------------------------------------------------------


@given(data=_shelf_image)
@settings(max_examples=120)
def test_property9_annotations_match_recorded_boxes(data: bytes) -> None:
    """Property 9: the drawn color lands on every recorded box border (Req 5.3).

    Build a finding, render the Annotated_Output, decode it, and for each
    recorded region assert the fixed annotation color appears along each of the
    box's four edges at exactly the recorded coordinates. The source image is
    grayscale content (no green anywhere), so any annotation-colored pixel on a
    recorded edge is attributable to the rectangle drawn there - i.e. the
    annotation matches the recorded bounding box.

    Validates: Requirements 5.3
    """
    cfg = _cfg()
    analysis = analyze_image_bytes(data, cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-annot",
        finding_id="find-annot",
        annotated_output_ref="ref://annotated/find-annot.png",
        source="image",
        cfg=cfg,
    )

    # The property is vacuous without at least one region; only meaningful
    # examples should exercise the edge check.
    assume(len(finding.candidate_regions) >= 1)

    out = render_annotation(data, finding)
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    img_h, img_w = decoded.shape[:2]

    for region in finding.candidate_regions:
        b = region.bounding_box
        x1, y1 = b.x, b.y
        x2, y2 = b.x + b.w - 1, b.y + b.h - 1

        # render_annotation clips the drawn corners to the frame; mirror that
        # here so we check where the rectangle can actually land.
        x1c = max(0, min(x1, img_w - 1))
        y1c = max(0, min(y1, img_h - 1))
        x2c = max(0, min(x2, img_w - 1))
        y2c = max(0, min(y2, img_h - 1))

        mid_x = (x1c + x2c) // 2
        mid_y = (y1c + y2c) // 2

        # Each recorded edge midpoint must carry the drawn annotation color,
        # proving the rectangle sits at the recorded coordinates (Req 5.3).
        assert _is_annotation_color(decoded[y1c, mid_x]), (
            f"{region.region_id}: top edge not drawn at recorded y={y1c}"
        )
        assert _is_annotation_color(decoded[y2c, mid_x]), (
            f"{region.region_id}: bottom edge not drawn at recorded y={y2c}"
        )
        assert _is_annotation_color(decoded[mid_y, x1c]), (
            f"{region.region_id}: left edge not drawn at recorded x={x1c}"
        )
        assert _is_annotation_color(decoded[mid_y, x2c]), (
            f"{region.region_id}: right edge not drawn at recorded x={x2c}"
        )


@given(data=_shelf_image)
@settings(max_examples=100)
def test_property9_border_pixels_change_from_source(data: bytes) -> None:
    """Property 9: recorded box borders differ from the original source there.

    A complementary check that the annotation actually *changed* the pixels at
    the recorded border coordinates: the source region is known not to contain
    the annotation color (grayscale content), so each recorded edge midpoint
    goes from a non-annotation color in the original to the annotation color in
    the rendered output (Req 5.3).

    Validates: Requirements 5.3
    """
    cfg = _cfg()
    analysis = analyze_image_bytes(data, cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-annot",
        finding_id="find-annot",
        annotated_output_ref="ref://annotated/find-annot.png",
        source="image",
        cfg=cfg,
    )
    assume(len(finding.candidate_regions) >= 1)

    original = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    out = render_annotation(data, finding)
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert original is not None and decoded is not None
    img_h, img_w = decoded.shape[:2]

    # Sanity: the grayscale source contains no annotation-colored pixel.
    assert not np.any(
        (original[:, :, 0] == ANNOTATION_COLOR_BGR[0])
        & (original[:, :, 1] == ANNOTATION_COLOR_BGR[1])
        & (original[:, :, 2] == ANNOTATION_COLOR_BGR[2])
    ), "source image unexpectedly already contains the annotation color"

    for region in finding.candidate_regions:
        b = region.bounding_box
        x1c = max(0, min(b.x, img_w - 1))
        y1c = max(0, min(b.y, img_h - 1))
        x2c = max(0, min(b.x + b.w - 1, img_w - 1))
        y2c = max(0, min(b.y + b.h - 1, img_h - 1))
        mid_x = (x1c + x2c) // 2

        # Top edge midpoint: not annotation color before, annotation color after.
        assert not _is_annotation_color(original[y1c, mid_x])
        assert _is_annotation_color(decoded[y1c, mid_x]), (
            f"{region.region_id}: top edge did not change to annotation color"
        )
