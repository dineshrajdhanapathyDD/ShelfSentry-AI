"""Region inspection and frame comparison entry points (task 3.5; Req 2.2, 4.1).

The agent's active-perception loop needs two *derived* vision operations beyond
the single-image analysis path:

* :func:`inspect_region` - zoom in on one bounding box the agent chose and
  re-run the full classical analysis on just that crop (the ``inspect_crop``
  action). This is how the loop "looks closer" at an uncertain region.
* :func:`compare_frames` - analyze two frames and summarize how the measured
  empty shelf space changed between them (the ``analyze_another_frame`` action's
  cross-frame comparison). This is how the loop reasons about *change* over time.

Both reuse the **shared** analysis + scoring path and never re-implement any
image processing or finding assembly:

* decode/preprocess/analyze via :func:`shelfsentry.vision.analyze.analyze_image_bytes`
  / :func:`~shelfsentry.vision.analyze.analyze_image_arrays`;
* finding assembly via :func:`shelfsentry.vision.scoring.build_finding`, exactly
  as the image and video paths do.

Determinism (Req 5.1): every operation here is a pure, config-driven function of
the input bytes plus the supplied identifiers. There is no randomness, wall
clock, or iteration-order dependence, so repeated calls with the same inputs
produce byte-for-byte identical ``canonical_json``.

Coordinate system for crops (documented, Req 5.2): :func:`inspect_region`
analyzes *within the crop*, so every bounding box on the returned finding's
regions is ``(x, y, w, h)`` in integer pixels relative to the **crop's own
origin** (the top-left of the requested box), NOT the original image. To map a
crop-local box back to the original image, add the crop origin ``(box.x,
box.y)`` to the region's ``(x, y)``. This mirrors how the single-image path
reports coordinates relative to whatever array it analyzed.
"""

from __future__ import annotations

from typing import Union

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import (
    BoundingBox,
    CandidateRegion,
    VisionFinding,
)
from shelfsentry.vision.analyze import analyze_image_arrays, analyze_image_bytes
from shelfsentry.vision.preprocess import PreprocessResult, preprocess_image
from shelfsentry.vision.scoring import build_finding

__all__ = [
    "RegionInspectionError",
    "BoxLike",
    "inspect_region",
    "compare_frames",
]


class RegionInspectionError(ValueError):
    """Raised when a requested inspection region is invalid for the image.

    Covers a malformed box (non-integer, non-positive width/height) and a box
    that does not lie fully within the decoded image bounds. The message names
    the specific condition so the failure is actionable (Req 10.2 spirit).
    """


#: A bounding box accepted by :func:`inspect_region`: either a
#: :class:`~shelfsentry.data.contracts.BoundingBox` or a raw
#: ``(x, y, w, h)`` integer 4-tuple.
BoxLike = Union[BoundingBox, "tuple[int, int, int, int]"]


def _coerce_box(box: BoxLike) -> tuple[int, int, int, int]:
    """Normalize a :class:`BoundingBox` or ``(x, y, w, h)`` tuple to a tuple.

    Args:
        box: A :class:`BoundingBox` or a 4-element ``(x, y, w, h)`` sequence.

    Returns:
        The box as a plain ``(x, y, w, h)`` integer tuple.

    Raises:
        RegionInspectionError: if the box is not a BoundingBox and not a
            4-element sequence of integers.
    """
    if isinstance(box, BoundingBox):
        return (int(box.x), int(box.y), int(box.w), int(box.h))

    try:
        x, y, w, h = box  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise RegionInspectionError(
            "box must be a BoundingBox or an (x, y, w, h) 4-tuple; "
            f"got {box!r}"
        ) from exc

    try:
        return (int(x), int(y), int(w), int(h))
    except (TypeError, ValueError) as exc:
        raise RegionInspectionError(
            f"box components must be integers; got {box!r}"
        ) from exc


def _validate_box_within(
    box: tuple[int, int, int, int], *, height: int, width: int
) -> None:
    """Validate that ``box`` has positive size and lies within the image bounds.

    Args:
        box: ``(x, y, w, h)`` integer-pixel box, top-left origin.
        height: Decoded image height in pixels.
        width: Decoded image width in pixels.

    Raises:
        RegionInspectionError: if the box has non-positive width/height, a
            negative origin, or extends past the image's right/bottom edge.
    """
    x, y, w, h = box
    if w < 1 or h < 1:
        raise RegionInspectionError(
            f"box width and height must each be >= 1 pixel; got w={w}, h={h}"
        )
    if x < 0 or y < 0:
        raise RegionInspectionError(
            f"box origin must be non-negative; got x={x}, y={y}"
        )
    if x + w > width or y + h > height:
        raise RegionInspectionError(
            f"box {box} does not lie within the {width}x{height} image: "
            f"its right edge {x + w} / bottom edge {y + h} exceeds the image "
            f"width {width} / height {height}"
        )


def _crop_preprocess(pre: PreprocessResult, box: tuple[int, int, int, int]) -> PreprocessResult:
    """Crop every preprocessing intermediate to ``box`` (shared origin).

    Slicing the already-computed arrays (rather than re-decoding/re-filtering a
    cropped image) keeps the crop's pixels byte-identical to the corresponding
    window of the full-image preprocessing, so analysis of the crop is a true
    zoom-in and stays deterministic (Req 5.1).

    Args:
        pre: Full-image preprocessing intermediates.
        box: ``(x, y, w, h)`` crop window, already validated to lie in bounds.

    Returns:
        A :class:`PreprocessResult` whose arrays are the ``box`` window of the
        inputs. Coordinates in the result are relative to the crop origin.
    """
    x, y, w, h = box
    y0, y1, x0, x1 = y, y + h, x, x + w
    return PreprocessResult(
        bgr=pre.bgr[y0:y1, x0:x1],
        gray=pre.gray[y0:y1, x0:x1],
        hsv=pre.hsv[y0:y1, x0:x1],
        blurred=pre.blurred[y0:y1, x0:x1],
        equalized=pre.equalized[y0:y1, x0:x1],
    )


def inspect_region(
    data: bytes,
    box: BoxLike,
    cfg: VisionConfig,
    *,
    inspection_id: str,
    finding_id: str,
    annotated_output_ref: str,
    frame_id: str | None = None,
) -> VisionFinding:
    """Re-run the shared analysis on a crop of the image (the ``inspect_crop`` path).

    Decodes and preprocesses ``data`` once, crops every preprocessing
    intermediate to ``box``, re-runs :func:`analyze_image_arrays` on the crop,
    and assembles the result with the shared
    :func:`shelfsentry.vision.scoring.build_finding` under ``source="crop"``. No
    image-processing or finding-assembly logic is duplicated here.

    Coordinate system (documented, Req 5.2): the returned finding's region
    bounding boxes are ``(x, y, w, h)`` in integer pixels relative to the
    **crop's own origin** (the top-left of ``box``), not the original image. Add
    the crop origin ``(box.x, box.y)`` to map a region back to original-image
    coordinates.

    Args:
        data: Raw encoded image bytes (already validated upstream).
        box: The region to inspect, as a
            :class:`~shelfsentry.data.contracts.BoundingBox` or an
            ``(x, y, w, h)`` integer tuple, in original-image coordinates.
        cfg: Vision configuration driving the deterministic pipeline.
        inspection_id: Owning inspection identifier (non-empty).
        finding_id: Identifier for the produced finding (non-empty).
        annotated_output_ref: Storage reference to the crop's Annotated_Output
            (Req 4.5); recorded on the finding.
        frame_id: Source frame id when the crop derives from a specific video
            frame; ``None`` for image inputs. Recorded on the finding.

    Returns:
        A validated :class:`VisionFinding` with ``source="crop"``.

    Raises:
        RegionInspectionError: if ``box`` is malformed or does not lie within
            the decoded image bounds.
        shelfsentry.vision.preprocess.PreprocessError: if ``data`` cannot be
            decoded into an image.
    """
    coerced = _coerce_box(box)
    pre = preprocess_image(data, cfg)
    height, width = pre.gray.shape[:2]
    _validate_box_within(coerced, height=height, width=width)

    crop_pre = _crop_preprocess(pre, coerced)
    analysis = analyze_image_arrays(crop_pre, cfg)

    return build_finding(
        analysis,
        inspection_id=inspection_id,
        finding_id=finding_id,
        annotated_output_ref=annotated_output_ref,
        source="crop",
        frame_id=frame_id,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Frame comparison
# ---------------------------------------------------------------------------

#: Documented interpretations attached to a frame-comparison finding so a
#: consumer never has to guess what the comparison numbers mean. Merged into the
#: finding's ``uncertainty_interpretations`` (Req 4.3) alongside the per-region
#: ones that :func:`build_finding` already supplies.
FRAME_COMPARE_INTERPRETATIONS: dict[str, str] = {
    "frame_compare.empty_space_ratio_delta": (
        "ESTIMATE of change in aggregate empty shelf space from frame A to "
        "frame B, defined as (frame B empty-area fraction) minus (frame A "
        "empty-area fraction), each measured as the sum of candidate empty "
        "pixels over the frame area. Positive means B looks emptier than A. "
        "This is a measured visual estimate, NOT a calibrated out-of-stock or "
        "restock probability, and it compares whole frames, not aligned "
        "products; apparent change may be shadow, occlusion, dividers, "
        "parallax, or image-quality differences between the two frames."
    ),
    "frame_compare.region_count_delta": (
        "ESTIMATE of change in the number of candidate empty-shelf regions "
        "from frame A to frame B (B count minus A count). A rough structural "
        "cue only; regions are not matched one-to-one across frames."
    ),
}


def _aggregate_empty_fraction(candidates, *, frame_area: int) -> float:
    """Sum candidate empty pixels over the frame area, clamped to ``[0, 1]``.

    A deterministic, whole-frame empty-space estimate used only to compare two
    frames. Overlapping candidate boxes are not deduplicated (candidates come
    from external contours on the empty mask, which do not overlap), so this is
    a stable sum of per-candidate ``empty_pixels``.

    Args:
        candidates: The analysis candidates for one frame.
        frame_area: ``height * width`` of that frame; non-positive yields 0.0.

    Returns:
        Aggregate empty-area fraction in ``[0, 1]``.
    """
    if frame_area <= 0:
        return 0.0
    total_empty = sum(int(c.empty_pixels) for c in candidates)
    fraction = total_empty / frame_area
    if fraction < 0.0:
        return 0.0
    if fraction > 1.0:
        return 1.0
    return float(fraction)


def compare_frames(
    a: bytes,
    b: bytes,
    cfg: VisionConfig,
    *,
    inspection_id: str,
    finding_id: str,
    annotated_output_ref: str,
) -> VisionFinding:
    """Analyze two frames and summarize the change in measured empty space.

    Runs the shared analysis path on both frames
    (:func:`analyze_image_bytes`), assembles the finding for frame **B** with
    the shared :func:`shelfsentry.vision.scoring.build_finding` under
    ``source="frame_compare"``, then records an HONEST comparison summary on the
    finding without duplicating any finding-assembly logic:

    * the aggregate ``empty_space_ratio`` delta (B minus A) and the per-frame
      fractions and region counts are attached as ``alt_metric`` on a single
      synthesized full-frame "changed area" region (labeled as an estimate via
      its ``visual_caveat``), so the comparison rides along as a normal region;
    * the documented meaning of each comparison number is merged into the
      finding's ``uncertainty_interpretations`` (Req 4.3) via
      :data:`FRAME_COMPARE_INTERPRETATIONS`.

    The comparison is a whole-frame estimate (not a per-aligned-product match)
    and is explicitly labeled as such, keeping the result honest (Req 2.5,
    27.4). Deterministic for the same ``a``, ``b``, ``cfg`` (Req 5.1).

    Args:
        a: Raw encoded bytes of the first (earlier) frame.
        b: Raw encoded bytes of the second (later) frame.
        cfg: Vision configuration driving the deterministic pipeline.
        inspection_id: Owning inspection identifier (non-empty).
        finding_id: Identifier for the produced finding (non-empty).
        annotated_output_ref: Storage reference to frame B's Annotated_Output
            (Req 4.5); recorded on the finding.

    Returns:
        A validated :class:`VisionFinding` with ``source="frame_compare"`` whose
        comparison summary reflects the difference between the two frames.

    Raises:
        shelfsentry.vision.preprocess.PreprocessError: if either frame's bytes
            cannot be decoded into an image.
    """
    analysis_a = analyze_image_bytes(a, cfg)
    analysis_b = analyze_image_bytes(b, cfg)

    frame_area_a = int(analysis_a.height) * int(analysis_a.width)
    frame_area_b = int(analysis_b.height) * int(analysis_b.width)

    empty_fraction_a = _aggregate_empty_fraction(
        analysis_a.candidates, frame_area=frame_area_a
    )
    empty_fraction_b = _aggregate_empty_fraction(
        analysis_b.candidates, frame_area=frame_area_b
    )
    empty_space_ratio_delta = empty_fraction_b - empty_fraction_a
    region_count_delta = len(analysis_b.candidates) - len(analysis_a.candidates)

    # Reuse the shared assembly for frame B. Do not re-implement any of it.
    finding = build_finding(
        analysis_b,
        inspection_id=inspection_id,
        finding_id=finding_id,
        annotated_output_ref=annotated_output_ref,
        source="frame_compare",
        cfg=cfg,
    )

    # Synthesized full-frame "changed area" region carrying the comparison
    # summary as a documented alternative metric. It spans frame B and uses
    # alt_metric (empty_space_ratio left None), exactly the contract's
    # documented escape hatch for a non-ratio visual metric.
    comparison_region = CandidateRegion(
        region_id="frame_compare_summary",
        bounding_box=BoundingBox(
            x=0, y=0, w=int(analysis_b.width), h=int(analysis_b.height)
        ),
        empty_space_ratio=None,
        alt_metric={
            "metric": "frame_compare",
            "frame_a_empty_fraction": empty_fraction_a,
            "frame_b_empty_fraction": empty_fraction_b,
            "empty_space_ratio_delta": empty_space_ratio_delta,
            "frame_a_region_count": len(analysis_a.candidates),
            "frame_b_region_count": len(analysis_b.candidates),
            "region_count_delta": region_count_delta,
            "is_estimate": True,
        },
        heuristic_score=0.0,
        uncertainty_value=0.0,
        frame_id=None,
        visual_caveat=(
            "Frame-comparison summary: this is an ESTIMATE of the change in "
            "aggregate empty shelf space between two whole frames (B minus A), "
            "not a per-product match and not a proven restock or out-of-stock "
            "event. Apparent change may instead be shadow, occlusion, shelf "
            "dividers, parallax, or differing image quality between the frames."
        ),
    )

    # Prepend the summary region and merge the comparison interpretations,
    # reassigning the validated fields (the model re-validates on assignment).
    finding.candidate_regions = [comparison_region, *finding.candidate_regions]
    merged_interpretations = dict(finding.uncertainty_interpretations)
    merged_interpretations.update(FRAME_COMPARE_INTERPRETATIONS)
    finding.uncertainty_interpretations = merged_interpretations

    return finding
