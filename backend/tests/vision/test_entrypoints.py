"""Tests for region inspection and frame comparison entry points (task 3.5).

Covers ``shelfsentry.vision.entrypoints`` against the data contracts and the
shared analysis/scoring path (Req 2.2, 4.1):

* :func:`inspect_region` crops a synthetic image and returns a pydantic-valid
  :class:`~shelfsentry.data.contracts.VisionFinding` with ``source == "crop"``,
  whose region coordinates are relative to the crop origin.
* :func:`compare_frames` over two synthetic frames that differ in one gap
  returns a pydantic-valid finding with ``source == "frame_compare"`` whose
  comparison summary reflects the difference (frame B emptier -> positive
  delta).
* Both entry points are deterministic: repeating a call yields byte-for-byte
  identical ``canonical_json`` (Req 5.1).
* An out-of-bounds / malformed box raises a clear error.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import BoundingBox, VisionFinding
from shelfsentry.vision.entrypoints import (
    RegionInspectionError,
    compare_frames,
    inspect_region,
)


# ---------------------------------------------------------------------------
# Synthetic inputs
# ---------------------------------------------------------------------------


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok, "failed to encode synthetic image"
    return buf.tobytes()


def _shelf_image(*, gaps: int) -> np.ndarray:
    """A deterministic shelf image with ``gaps`` dark empty regions.

    A bright background with two dark shelf lips. The bottom shelf's left half
    is always a dark empty gap. When ``gaps == 2`` the top shelf's product
    block is replaced with a second dark empty gap, so a two-gap frame measures
    clearly MORE aggregate empty space than a one-gap frame.
    """
    h, w = 240, 320
    img = np.full((h, w), 220, dtype=np.uint8)  # bright background

    # Two dark horizontal shelf lips.
    img[78:82, :] = 20
    img[158:162, :] = 20

    if gaps >= 2:
        # Top shelf becomes a dark empty gap (the "extra" gap vs. the 1-gap frame).
        img[40:78, 10:150] = 15
    else:
        # Top shelf holds product (bright block with an internal edge).
        img[40:78, 30:150] = 120
        img[40:78, 150:154] = 10

    # Bottom shelf: left half dark empty gap, right half product (both frames).
    img[100:158, 10:150] = 15
    img[100:158, 170:300] = 130
    return img


def _one_gap_png() -> bytes:
    return _encode_png(_shelf_image(gaps=1))


def _two_gap_png() -> bytes:
    return _encode_png(_shelf_image(gaps=2))


@pytest.fixture()
def cfg() -> VisionConfig:
    return VisionConfig()


# ---------------------------------------------------------------------------
# inspect_region
# ---------------------------------------------------------------------------


def test_inspect_region_returns_valid_crop_finding(cfg: VisionConfig) -> None:
    """inspect_region yields a valid VisionFinding with source == 'crop' (Req 4.1)."""
    data = _one_gap_png()
    # Crop around the bottom-left empty gap (x=10..150, y=100..158).
    box = BoundingBox(x=5, y=95, w=160, h=70)

    finding = inspect_region(
        data,
        box,
        cfg,
        inspection_id="insp-crop",
        finding_id="find-crop",
        annotated_output_ref="ref://annotated/crop.png",
    )

    assert isinstance(finding, VisionFinding)
    assert finding.source == "crop"
    assert finding.is_estimate is True
    assert finding.opencv_version == "5.0.0"
    # Round-trips through strict boundary validation.
    VisionFinding.validate_boundary(finding.model_dump(mode="json"))


def test_inspect_region_accepts_tuple_box(cfg: VisionConfig) -> None:
    """A raw (x, y, w, h) tuple is accepted and equals the BoundingBox result."""
    data = _one_gap_png()
    from_tuple = inspect_region(
        data,
        (5, 95, 160, 70),
        cfg,
        inspection_id="insp-crop",
        finding_id="find-crop",
        annotated_output_ref="ref://annotated/crop.png",
    )
    from_box = inspect_region(
        data,
        BoundingBox(x=5, y=95, w=160, h=70),
        cfg,
        inspection_id="insp-crop",
        finding_id="find-crop",
        annotated_output_ref="ref://annotated/crop.png",
    )
    assert from_tuple.canonical_json() == from_box.canonical_json()


def test_inspect_region_coordinates_are_crop_local(cfg: VisionConfig) -> None:
    """Returned region boxes are relative to the crop origin (documented)."""
    data = _one_gap_png()
    crop_w, crop_h = 160, 70
    box = BoundingBox(x=5, y=95, w=crop_w, h=crop_h)

    finding = inspect_region(
        data,
        box,
        cfg,
        inspection_id="insp-crop",
        finding_id="find-crop",
        annotated_output_ref="ref://annotated/crop.png",
    )

    # Every region must fit entirely inside the crop's own dimensions, which it
    # can only do if coordinates are measured from the crop origin.
    for region in finding.candidate_regions:
        bb = region.bounding_box
        assert bb.x + bb.w <= crop_w
        assert bb.y + bb.h <= crop_h


def test_inspect_region_is_deterministic(cfg: VisionConfig) -> None:
    """Repeating inspect_region is byte-for-byte identical (Req 5.1)."""
    data = _one_gap_png()
    box = BoundingBox(x=5, y=95, w=160, h=70)
    kwargs = dict(
        inspection_id="insp-crop",
        finding_id="find-crop",
        annotated_output_ref="ref://annotated/crop.png",
    )
    first = inspect_region(data, box, cfg, **kwargs)
    second = inspect_region(data, box, cfg, **kwargs)
    assert first.canonical_json() == second.canonical_json()


def test_inspect_region_rejects_out_of_bounds_box(cfg: VisionConfig) -> None:
    """A box past the image edge raises a clear RegionInspectionError."""
    data = _one_gap_png()  # 320x240
    with pytest.raises(RegionInspectionError, match="does not lie within"):
        inspect_region(
            data,
            BoundingBox(x=300, y=200, w=100, h=100),
            cfg,
            inspection_id="insp-crop",
            finding_id="find-crop",
            annotated_output_ref="ref://annotated/crop.png",
        )


def test_inspect_region_rejects_malformed_box(cfg: VisionConfig) -> None:
    """A non-4-tuple box raises a clear RegionInspectionError."""
    data = _one_gap_png()
    with pytest.raises(RegionInspectionError):
        inspect_region(
            data,
            (1, 2, 3),  # type: ignore[arg-type]
            cfg,
            inspection_id="insp-crop",
            finding_id="find-crop",
            annotated_output_ref="ref://annotated/crop.png",
        )


# ---------------------------------------------------------------------------
# compare_frames
# ---------------------------------------------------------------------------


def _frame_compare_summary(finding: VisionFinding) -> dict:
    """Extract the synthesized frame-comparison summary region's alt_metric."""
    for region in finding.candidate_regions:
        if region.region_id == "frame_compare_summary":
            assert region.alt_metric is not None
            return region.alt_metric
    raise AssertionError("no frame_compare_summary region present")


def test_compare_frames_returns_valid_frame_compare_finding(cfg: VisionConfig) -> None:
    """compare_frames yields a valid VisionFinding, source == 'frame_compare'."""
    finding = compare_frames(
        _one_gap_png(),
        _two_gap_png(),
        cfg,
        inspection_id="insp-cmp",
        finding_id="find-cmp",
        annotated_output_ref="ref://annotated/cmp.png",
    )

    assert isinstance(finding, VisionFinding)
    assert finding.source == "frame_compare"
    assert finding.is_estimate is True
    # Round-trips through strict boundary validation (comparison region included).
    VisionFinding.validate_boundary(finding.model_dump(mode="json"))


def test_compare_frames_reflects_added_gap(cfg: VisionConfig) -> None:
    """Frame B (two gaps) is emptier than frame A (one gap) -> positive delta."""
    finding = compare_frames(
        _one_gap_png(),
        _two_gap_png(),
        cfg,
        inspection_id="insp-cmp",
        finding_id="find-cmp",
        annotated_output_ref="ref://annotated/cmp.png",
    )

    summary = _frame_compare_summary(finding)
    assert summary["metric"] == "frame_compare"
    assert summary["is_estimate"] is True
    # B has strictly more empty area than A.
    assert summary["frame_b_empty_fraction"] > summary["frame_a_empty_fraction"]
    assert summary["empty_space_ratio_delta"] > 0.0
    assert summary["empty_space_ratio_delta"] == pytest.approx(
        summary["frame_b_empty_fraction"] - summary["frame_a_empty_fraction"]
    )
    # Comparison interpretations are documented on the finding (Req 4.3).
    assert "frame_compare.empty_space_ratio_delta" in finding.uncertainty_interpretations


def test_compare_frames_identical_frames_zero_delta(cfg: VisionConfig) -> None:
    """Comparing a frame with itself yields a zero empty-space delta."""
    data = _one_gap_png()
    finding = compare_frames(
        data,
        data,
        cfg,
        inspection_id="insp-cmp",
        finding_id="find-cmp",
        annotated_output_ref="ref://annotated/cmp.png",
    )
    summary = _frame_compare_summary(finding)
    assert summary["empty_space_ratio_delta"] == pytest.approx(0.0)
    assert summary["region_count_delta"] == 0


def test_compare_frames_is_deterministic(cfg: VisionConfig) -> None:
    """Repeating compare_frames is byte-for-byte identical (Req 5.1)."""
    a, b = _one_gap_png(), _two_gap_png()
    kwargs = dict(
        inspection_id="insp-cmp",
        finding_id="find-cmp",
        annotated_output_ref="ref://annotated/cmp.png",
    )
    first = compare_frames(a, b, cfg, **kwargs)
    second = compare_frames(a, b, cfg, **kwargs)
    assert first.canonical_json() == second.canonical_json()
