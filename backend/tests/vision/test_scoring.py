"""Tests for scoring, uncertainty, caveats, and VisionFinding assembly.

Covers ``shelfsentry.vision.scoring`` (task 3.2) against the data contracts:

* ``build_finding`` over a synthetic shelf image yields a pydantic-valid
  :class:`~shelfsentry.data.contracts.VisionFinding` (Req 4.1).
* Every region's ``empty_space_ratio``, ``heuristic_score``, and
  ``uncertainty_value`` land in ``[0, 1]`` and every region carries a non-empty
  ``visual_caveat`` (Req 4.2, 4.7, 27.4).
* The finding is labeled an estimate (``is_estimate=True``, Req 2.5), records
  the exact OpenCV 5 runtime version (Req 2.1), and includes the documented
  per-finding ``uncertainty_interpretations`` (Req 4.3).
* ``empty_space_ratio == empty_pixels / box_area`` for each region (the
  documented definition).
* Building the finding twice from the same analysis yields byte-for-byte
  identical ``canonical_json`` (deterministic; Req 5.1).

Unit tests additionally pin the scoring helpers' documented behavior directly
on synthetic :class:`~shelfsentry.vision.analyze.RegionCandidate` values.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import VisionFinding
from shelfsentry.vision.analyze import RegionCandidate, analyze_image_bytes
from shelfsentry.vision.scoring import (
    UNCERTAINTY_INTERPRETATIONS,
    build_finding,
    compute_empty_space_ratio,
    compute_heuristic_score,
    compute_uncertainty_value,
    suggested_follow_up,
)


# ---------------------------------------------------------------------------
# Synthetic input
# ---------------------------------------------------------------------------


def _synthetic_shelf_png() -> bytes:
    """Encode a synthetic two-shelf image with one gap as PNG bytes.

    A light background with two dark horizontal shelf lips and a block of
    "product" on the top shelf, leaving a dark empty gap on the bottom shelf.
    Deterministic (fixed pixels, no randomness) so the whole pipeline is
    reproducible.
    """
    h, w = 240, 320
    img = np.full((h, w), 220, dtype=np.uint8)  # bright background

    # Two dark horizontal shelf lips.
    img[78:82, :] = 20
    img[158:162, :] = 20

    # Product block on the top shelf (bright, textured edges).
    img[40:78, 30:150] = 120
    img[40:78, 150:154] = 10  # a vertical edge between products

    # Bottom shelf left half: a dark EMPTY gap; right half: product.
    img[100:158, 10:150] = 15  # empty gap (dark)
    img[100:158, 170:300] = 130  # product

    ok, buf = cv2.imencode(".png", img)
    assert ok, "failed to encode synthetic shelf image"
    return buf.tobytes()


@pytest.fixture()
def cfg() -> VisionConfig:
    return VisionConfig()


# ---------------------------------------------------------------------------
# End-to-end: analyze_image_bytes -> build_finding
# ---------------------------------------------------------------------------


def test_build_finding_is_valid_vision_finding(cfg: VisionConfig) -> None:
    """build_finding produces a pydantic-valid VisionFinding (Req 4.1)."""
    analysis = analyze_image_bytes(_synthetic_shelf_png(), cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-1",
        finding_id="find-1",
        annotated_output_ref="ref://annotated/find-1.png",
        source="image",
        cfg=cfg,
    )

    assert isinstance(finding, VisionFinding)
    # Round-trips through strict boundary validation without error.
    VisionFinding.validate_boundary(finding.model_dump(mode="json"))


def test_finding_is_estimate_and_records_opencv_5(cfg: VisionConfig) -> None:
    """Finding is labeled an estimate and records OpenCV 5.0.0 (Req 2.1, 2.5)."""
    analysis = analyze_image_bytes(_synthetic_shelf_png(), cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-1",
        finding_id="find-1",
        annotated_output_ref="ref://annotated/find-1.png",
        cfg=cfg,
    )

    assert finding.is_estimate is True
    assert finding.opencv_version == "5.0.0"


def test_regions_have_unit_interval_metrics_and_caveats(cfg: VisionConfig) -> None:
    """Every region has in-range metrics and a non-empty caveat (Req 4.2, 4.7, 27.4)."""
    analysis = analyze_image_bytes(_synthetic_shelf_png(), cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-1",
        finding_id="find-1",
        annotated_output_ref="ref://annotated/find-1.png",
        cfg=cfg,
    )

    # The synthetic image has a clear gap, so analysis should find >= 1 region.
    assert len(finding.candidate_regions) >= 1

    for region in finding.candidate_regions:
        assert region.empty_space_ratio is not None
        assert 0.0 <= region.empty_space_ratio <= 1.0
        assert 0.0 <= region.heuristic_score <= 1.0
        assert 0.0 <= region.uncertainty_value <= 1.0
        assert region.visual_caveat.strip() != ""
        # Caveat always names the honesty confounders (Req 4.7, 27.4).
        caveat = region.visual_caveat.lower()
        assert "shadow" in caveat
        assert "occlusion" in caveat
        assert "divider" in caveat


def test_empty_space_ratio_matches_definition(cfg: VisionConfig) -> None:
    """empty_space_ratio == empty_pixels / box_area for each region (documented)."""
    analysis = analyze_image_bytes(_synthetic_shelf_png(), cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-1",
        finding_id="find-1",
        annotated_output_ref="ref://annotated/find-1.png",
        cfg=cfg,
    )

    for candidate, region in zip(analysis.candidates, finding.candidate_regions):
        expected = candidate.empty_pixels / candidate.box_area
        assert region.empty_space_ratio == pytest.approx(expected)


def test_uncertainty_interpretations_present_and_documented(cfg: VisionConfig) -> None:
    """The finding carries the documented uncertainty interpretations (Req 4.3)."""
    analysis = analyze_image_bytes(_synthetic_shelf_png(), cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-1",
        finding_id="find-1",
        annotated_output_ref="ref://annotated/find-1.png",
        cfg=cfg,
    )

    assert finding.uncertainty_interpretations == UNCERTAINTY_INTERPRETATIONS
    assert "uncertainty_value" in finding.uncertainty_interpretations
    assert finding.uncertainty_interpretations["uncertainty_value"].strip() != ""


def test_building_twice_is_identical_canonical_json(cfg: VisionConfig) -> None:
    """Building a finding twice from the same analysis is byte-identical (Req 5.1)."""
    analysis = analyze_image_bytes(_synthetic_shelf_png(), cfg)
    kwargs = dict(
        inspection_id="insp-1",
        finding_id="find-1",
        annotated_output_ref="ref://annotated/find-1.png",
        source="image",
        cfg=cfg,
    )
    first = build_finding(analysis, **kwargs)
    second = build_finding(analysis, **kwargs)

    assert first.canonical_json() == second.canonical_json()


def test_suggested_follow_up_is_a_valid_action(cfg: VisionConfig) -> None:
    """The suggested follow-up is one of the fixed five actions (Req 4.6)."""
    analysis = analyze_image_bytes(_synthetic_shelf_png(), cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-1",
        finding_id="find-1",
        annotated_output_ref="ref://annotated/find-1.png",
        cfg=cfg,
    )
    assert finding.suggested_follow_up in {
        "inspect_crop",
        "analyze_another_frame",
        "request_better_image",
        "request_human_review",
        "finalize_report",
    }


# ---------------------------------------------------------------------------
# Unit tests for the scoring helpers (documented behavior)
# ---------------------------------------------------------------------------


def _candidate(
    *,
    box: tuple[int, int, int, int],
    filled: int,
    edges: int,
    in_band: bool = True,
) -> RegionCandidate:
    """Build a synthetic RegionCandidate with a given box/fill/edge profile."""
    x, y, w, h = box
    box_area = w * h
    return RegionCandidate(
        bounding_box=box,
        contour_area=float(box_area),
        box_area=box_area,
        filled_pixels=filled,
        empty_pixels=box_area - filled,
        edge_pixels=edges,
        in_shelf_band=in_band,
    )


def test_empty_space_ratio_is_fraction_empty() -> None:
    """ratio = empty_pixels / box_area, clamped to [0, 1]."""
    cand = _candidate(box=(0, 0, 10, 10), filled=25, edges=0)
    # 100 - 25 = 75 empty of 100 -> 0.75
    assert compute_empty_space_ratio(cand) == pytest.approx(0.75)


def test_heuristic_score_distinct_from_probability_and_ordinal() -> None:
    """A fuller empty region outranks a sparser one; score stays in [0, 1]."""
    image_area = 320 * 240
    emptier = _candidate(box=(0, 0, 40, 40), filled=0, edges=0)  # ratio 1.0
    fuller = _candidate(box=(0, 0, 40, 40), filled=1600, edges=0)  # ratio 0.0

    s_emptier = compute_heuristic_score(emptier, image_area=image_area)
    s_fuller = compute_heuristic_score(fuller, image_area=image_area)

    assert 0.0 <= s_fuller <= 1.0
    assert 0.0 <= s_emptier <= 1.0
    assert s_emptier > s_fuller  # ordinal: emptier ranks higher


def test_uncertainty_lower_for_tiny_regions() -> None:
    """A very small region has lower measurement confidence than a large one."""
    image_area = 320 * 240
    tiny = _candidate(box=(0, 0, 5, 5), filled=0, edges=0)  # ratio 1.0, tiny
    large = _candidate(box=(0, 0, 120, 120), filled=0, edges=0)  # ratio 1.0, big

    assert compute_uncertainty_value(tiny, image_area=image_area) < (
        compute_uncertainty_value(large, image_area=image_area)
    )


def test_uncertainty_lower_for_high_edge_density() -> None:
    """High edge density inside a box lowers confidence (clutter/occlusion)."""
    image_area = 320 * 240
    box = (0, 0, 100, 100)  # area 10000
    clean = _candidate(box=box, filled=0, edges=0)
    cluttered = _candidate(box=box, filled=0, edges=9000)  # density 0.9

    assert compute_uncertainty_value(cluttered, image_area=image_area) < (
        compute_uncertainty_value(clean, image_area=image_area)
    )


def test_uncertainty_in_unit_interval() -> None:
    """Confidence is always a bounded [0, 1] value, even in extremes."""
    image_area = 320 * 240
    for cand in (
        _candidate(box=(0, 0, 1, 1), filled=0, edges=0),
        _candidate(box=(0, 0, 200, 200), filled=20000, edges=20000),
        _candidate(box=(0, 0, 100, 100), filled=5000, edges=0),
    ):
        u = compute_uncertainty_value(cand, image_area=image_area)
        assert 0.0 <= u <= 1.0


def test_follow_up_finalize_when_nothing_notable() -> None:
    """No regions -> finalize_report (nothing to probe)."""
    assert (
        suggested_follow_up([], empty_ratio_threshold=0.6, source="image")
        == "finalize_report"
    )
