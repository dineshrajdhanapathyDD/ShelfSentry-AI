"""Scoring, uncertainty, caveats, and VisionFinding assembly (Req 2.3-2.6, 4).

This stage turns the raw, *measured* geometry and pixel counts produced by
:func:`shelfsentry.vision.analyze.analyze_image_arrays` (an
:class:`~shelfsentry.vision.analyze.AnalysisResult` of
:class:`~shelfsentry.vision.analyze.RegionCandidate` objects) into a fully
assembled :class:`~shelfsentry.data.contracts.VisionFinding`.

For every candidate region it computes three distinct numbers, each with a
deliberately narrow, documented meaning:

* ``empty_space_ratio`` - a **measured** visual metric: the fraction of the
  region's bounding box judged empty, defined exactly as
  ``empty_pixels / box_area`` and clamped into ``[0, 1]`` (Req 4.2). It is an
  *estimate* of empty shelf space, NOT a calibrated out-of-stock probability.

* ``heuristic_score`` - a rule-derived **ordinal** score in ``[0, 1]`` used only
  to *rank* regions (Req 2.3, 2.6). It is a fixed weighted blend of the
  empty-space ratio, the region's relative size, and its edge sparsity. It is
  explicitly NOT a probability and is never calibrated; it exists so a caller
  can order regions "most worth a closer look first". See
  :func:`compute_heuristic_score`.

* ``uncertainty_value`` - a **measurement-confidence** in ``[0, 1]`` (higher ==
  more confident the measurement is trustworthy, Req 4.2, 4.3). Confidence is
  *reduced* for very small regions (few pixels -> noisy ratio), for borderline
  ratios near the empty/filled decision boundary (ambiguous), and for high edge
  density inside the box (clutter/occlusion makes "empty" uncertain). The
  meaning of this number is spelled out per finding in
  ``uncertainty_interpretations`` (Req 4.3).

Every region also carries a non-empty ``visual_caveat`` (Req 4.7, 27.4): a
human-readable note that an empty-looking region may be shadow, occlusion,
shelf dividers, or poor image quality rather than a proven out-of-stock
condition. The structural guarantee lives in the contract (``visual_caveat`` is
required and ``min_length=1``); this module always supplies a meaningful one.

The assembled finding is labeled an estimate (``is_estimate=True``, Req 2.5),
records the exact verified OpenCV runtime version (Req 2.1 via
:func:`shelfsentry.vision.version.opencv_version`), and includes a
``suggested_follow_up`` next-action *hint* for the agent (Req 4.6). The hint is
only a suggestion; the agent decides the real action later.

Determinism (Req 5.1): every computation here is a pure function of the
``AnalysisResult`` plus the supplied identifiers and config. There is no
randomness, no wall-clock, and no dependence on dict/set iteration order, so
building a finding twice from the same inputs yields byte-for-byte identical
``canonical_json``.
"""

from __future__ import annotations

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import (
    AgentAction,
    BoundingBox,
    CandidateRegion,
    VisionFinding,
)
from shelfsentry.vision.analyze import AnalysisResult, RegionCandidate
from shelfsentry.vision.version import opencv_version

__all__ = [
    "HEURISTIC_WEIGHT_EMPTY",
    "HEURISTIC_WEIGHT_SIZE",
    "HEURISTIC_WEIGHT_EDGE_SPARSITY",
    "UNCERTAINTY_INTERPRETATIONS",
    "compute_empty_space_ratio",
    "compute_edge_density",
    "compute_heuristic_score",
    "compute_uncertainty_value",
    "build_visual_caveat",
    "suggested_follow_up",
    "build_candidate_region",
    "build_finding",
]


# ---------------------------------------------------------------------------
# Heuristic-score weights (fixed, documented; Req 2.6)
# ---------------------------------------------------------------------------
#
# The heuristic score is a FIXED weighted blend of three measured cues. The
# weights are constants (not learned, not tuned to a dataset) precisely so the
# score stays an honest, explainable ORDINAL and is never mistaken for a
# calibrated probability. They sum to 1.0 so the blend already lands in [0, 1]
# when each input cue is in [0, 1]; the result is clamped defensively anyway.

#: Weight on the measured empty-space ratio (the dominant empty-shelf cue).
HEURISTIC_WEIGHT_EMPTY: float = 0.6
#: Weight on the region's relative size (bigger gaps rank higher, all else equal).
HEURISTIC_WEIGHT_SIZE: float = 0.2
#: Weight on edge sparsity (few edges inside the box is an empty-space cue).
HEURISTIC_WEIGHT_EDGE_SPARSITY: float = 0.2


#: The documented meaning of each uncertainty-related value carried by a
#: finding (Req 4.3). This exact mapping is attached to every assembled
#: :class:`VisionFinding` so a consumer never has to guess what the numbers
#: mean. Keys are stable identifiers; values are human-readable explanations.
UNCERTAINTY_INTERPRETATIONS: dict[str, str] = {
    "uncertainty_value": (
        "Measurement confidence in [0, 1]: higher means the per-region visual "
        "measurement is more trustworthy. It is reduced for very small regions "
        "(few pixels make the ratio noisy), for empty_space_ratio values near "
        "the empty/filled decision boundary (ambiguous), and for high edge "
        "density inside the box (clutter or occlusion). It is NOT a calibrated "
        "probability that the shelf is out of stock."
    ),
    "empty_space_ratio": (
        "Measured fraction of a region's bounding box judged empty, defined as "
        "empty_pixels / box_area and clamped to [0, 1]. An estimate of empty "
        "shelf space, reported as an estimate and never as a proven "
        "out-of-stock condition."
    ),
    "heuristic_score": (
        "Rule-derived ordinal score in [0, 1] for ranking regions only (a fixed "
        "weighted blend of empty-space ratio, relative size, and edge "
        "sparsity). Explicitly NOT a probability and not calibrated."
    ),
}


# ---------------------------------------------------------------------------
# Small deterministic numeric helpers
# ---------------------------------------------------------------------------


def _clamp_unit(value: float) -> float:
    """Clamp ``value`` into the closed unit interval ``[0.0, 1.0]``.

    Used so every reported ratio/score/confidence is a well-formed ``[0, 1]``
    number even in degenerate inputs, which also keeps the pydantic
    ``ge=0.0, le=1.0`` boundary validations satisfied.
    """
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


# ---------------------------------------------------------------------------
# Per-region measured metric: empty_space_ratio
# ---------------------------------------------------------------------------


def compute_empty_space_ratio(candidate: RegionCandidate) -> float:
    """Measured empty-space ratio for a region: ``empty_pixels / box_area``.

    Definition (documented, Req 4.2): the fraction of the region's bounding box
    whose pixels were judged *empty* (not product) by the analysis stage's
    combined filled mask. ``empty_pixels`` already equals
    ``box_area - filled_pixels`` (see
    :class:`~shelfsentry.vision.analyze.RegionCandidate`), so this is a direct
    area fraction. The result is clamped to ``[0, 1]``.

    This is a **measured estimate** of empty shelf space, not a calibrated
    out-of-stock probability (Req 2.5, 27.4).

    Args:
        candidate: The measured region from analysis.

    Returns:
        The empty-space ratio in ``[0, 1]``. Returns ``0.0`` for a degenerate
        zero-area box (which analysis does not emit, but is handled defensively).
    """
    if candidate.box_area <= 0:
        return 0.0
    return _clamp_unit(candidate.empty_pixels / candidate.box_area)


def compute_edge_density(candidate: RegionCandidate) -> float:
    """Fraction of the box that is Canny edge pixels: ``edge_pixels / box_area``.

    High edge density inside a box means visual clutter (product detail,
    dividers, occluding objects), which both lowers confidence that the box is
    genuinely empty and, via its *sparsity* complement, feeds the heuristic
    ranking score. Clamped to ``[0, 1]``.

    Args:
        candidate: The measured region from analysis.

    Returns:
        Edge density in ``[0, 1]`` (``0.0`` for a degenerate zero-area box).
    """
    if candidate.box_area <= 0:
        return 0.0
    return _clamp_unit(candidate.edge_pixels / candidate.box_area)


# ---------------------------------------------------------------------------
# Per-region ordinal ranking: heuristic_score (NOT a probability; Req 2.6)
# ---------------------------------------------------------------------------


def compute_heuristic_score(
    candidate: RegionCandidate,
    *,
    image_area: int,
    empty_space_ratio: float | None = None,
    edge_density: float | None = None,
) -> float:
    """Rule-derived ORDINAL score in ``[0, 1]`` for ranking regions (Req 2.3, 2.6).

    This is a fixed, explainable weighted blend of three measured cues, each
    already in ``[0, 1]``:

    * **empty-space ratio** (weight :data:`HEURISTIC_WEIGHT_EMPTY`) - the
      dominant empty-shelf signal;
    * **relative size** (weight :data:`HEURISTIC_WEIGHT_SIZE`) - the box area as
      a fraction of the whole image, so larger gaps rank above specks;
    * **edge sparsity** (weight :data:`HEURISTIC_WEIGHT_EDGE_SPARSITY`) -
      ``1 - edge_density``, since few internal edges is an empty-space cue.

    The weights are constants that sum to ``1.0``; the output is clamped to
    ``[0, 1]`` for safety. It is deterministic given the same inputs.

    IMPORTANT (Req 2.6): this value is an **ordinal ranking aid only**. It is
    explicitly NOT a calibrated probability and must never be presented as the
    likelihood that a shelf is out of stock. It exists solely to order regions
    "most worth a closer look first".

    Args:
        candidate: The measured region from analysis.
        image_area: ``height * width`` of the analyzed input, used to normalize
            region size. Non-positive values are treated as "unknown" and the
            size cue contributes ``0``.
        empty_space_ratio: Optional precomputed ratio to avoid recomputation;
            computed from ``candidate`` when ``None``.
        edge_density: Optional precomputed edge density; computed from
            ``candidate`` when ``None``.

    Returns:
        The ordinal heuristic score in ``[0, 1]``.
    """
    ratio = (
        compute_empty_space_ratio(candidate)
        if empty_space_ratio is None
        else _clamp_unit(empty_space_ratio)
    )
    density = (
        compute_edge_density(candidate)
        if edge_density is None
        else _clamp_unit(edge_density)
    )
    edge_sparsity = 1.0 - density

    if image_area > 0:
        relative_size = _clamp_unit(candidate.box_area / image_area)
    else:
        relative_size = 0.0

    score = (
        HEURISTIC_WEIGHT_EMPTY * ratio
        + HEURISTIC_WEIGHT_SIZE * relative_size
        + HEURISTIC_WEIGHT_EDGE_SPARSITY * edge_sparsity
    )
    return _clamp_unit(score)


# ---------------------------------------------------------------------------
# Per-region measurement confidence: uncertainty_value (Req 4.2, 4.3)
# ---------------------------------------------------------------------------

#: Below this fraction of the whole image, a region is "very small" and its
#: ratio is treated as pixel-noisy, reducing confidence. Fixed for determinism.
_SMALL_REGION_AREA_FRACTION: float = 0.01

#: Edge density at/above which a box is considered heavily cluttered/occluded,
#: driving confidence toward its floor. Fixed for determinism.
_HIGH_EDGE_DENSITY: float = 0.5

#: Confidence can never be reported as absolute certainty: an empty-looking
#: region is never proven out of stock (Req 27.4), so confidence is capped
#: below 1.0 even in the most favorable measurement.
_MAX_CONFIDENCE: float = 0.95

#: And it never collapses to exactly zero; a measurement still carries some
#: signal. Keeps the value a usable, bounded confidence.
_MIN_CONFIDENCE: float = 0.05


def compute_uncertainty_value(
    candidate: RegionCandidate,
    *,
    image_area: int,
    empty_space_ratio: float | None = None,
    edge_density: float | None = None,
) -> float:
    """Measurement confidence in ``[0, 1]`` for a region (Req 4.2, 4.3).

    Higher means the measurement is more trustworthy. The value starts at
    :data:`_MAX_CONFIDENCE` and is multiplied down by three independent,
    documented penalty factors, each in ``[0, 1]``:

    * **Size penalty** - very small regions (area below
      :data:`_SMALL_REGION_AREA_FRACTION` of the image) have few pixels, so
      their ratio is noisy; confidence scales up with relative size until the
      small-region threshold is reached.
    * **Borderline penalty** - a ratio near ``0.5`` (equal empty/filled) sits on
      the ambiguous decision boundary; confidence is highest when the ratio is
      decisively near ``0`` or ``1`` and lowest at ``0.5``.
    * **Edge/clutter penalty** - high edge density inside the box signals
      clutter or occlusion, which makes an "empty" judgment less certain;
      confidence falls as density rises toward :data:`_HIGH_EDGE_DENSITY`.

    The product is clamped into ``[_MIN_CONFIDENCE, _MAX_CONFIDENCE]`` so the
    system never reports absolute certainty (an empty-looking region is never
    *proven* out of stock, Req 27.4) nor total ignorance. Deterministic given
    the same inputs; its meaning is documented in
    :data:`UNCERTAINTY_INTERPRETATIONS`.

    Args:
        candidate: The measured region from analysis.
        image_area: ``height * width`` of the analyzed input for size scaling.
        empty_space_ratio: Optional precomputed ratio; computed when ``None``.
        edge_density: Optional precomputed edge density; computed when ``None``.

    Returns:
        A measurement confidence in ``[_MIN_CONFIDENCE, _MAX_CONFIDENCE]``.
    """
    ratio = (
        compute_empty_space_ratio(candidate)
        if empty_space_ratio is None
        else _clamp_unit(empty_space_ratio)
    )
    density = (
        compute_edge_density(candidate)
        if edge_density is None
        else _clamp_unit(edge_density)
    )

    # Size factor: 0 at zero area, rising linearly to 1.0 once the region
    # reaches the small-region threshold fraction of the image. Larger regions
    # stay at full confidence on this axis.
    if image_area > 0 and _SMALL_REGION_AREA_FRACTION > 0.0:
        relative_size = candidate.box_area / image_area
        size_factor = _clamp_unit(relative_size / _SMALL_REGION_AREA_FRACTION)
    else:
        size_factor = 0.0

    # Borderline factor: 1.0 when the ratio is decisive (near 0 or 1), dropping
    # to 0 at the fully ambiguous midpoint 0.5. |ratio - 0.5| is in [0, 0.5];
    # doubling maps it to [0, 1].
    decisiveness = abs(ratio - 0.5) * 2.0
    borderline_factor = _clamp_unit(decisiveness)

    # Edge factor: 1.0 with no edges, falling to 0 once density reaches the
    # high-edge-density threshold (heavy clutter/occlusion).
    if _HIGH_EDGE_DENSITY > 0.0:
        edge_factor = _clamp_unit(1.0 - (density / _HIGH_EDGE_DENSITY))
    else:
        edge_factor = 1.0

    confidence = _MAX_CONFIDENCE * size_factor * borderline_factor * edge_factor

    if confidence < _MIN_CONFIDENCE:
        return _MIN_CONFIDENCE
    if confidence > _MAX_CONFIDENCE:
        return _MAX_CONFIDENCE
    return float(confidence)


# ---------------------------------------------------------------------------
# Per-region visual caveat (Req 4.7, 27.4)
# ---------------------------------------------------------------------------

#: The base honesty caveat attached to every region: the four documented
#: confounders an empty-looking region could actually be (Req 4.7, 27.4). This
#: phrasing never asserts a proven out-of-stock condition.
_BASE_CAVEAT: str = (
    "This region is an estimate of empty shelf space, not a confirmed "
    "out-of-stock condition. An empty-looking region may instead be shadow, "
    "occlusion, shelf dividers, or poor image quality."
)


def build_visual_caveat(
    candidate: RegionCandidate,
    *,
    empty_space_ratio: float,
    edge_density: float,
    uncertainty_value: float,
) -> str:
    """Build a non-empty, human-readable visual caveat for a region (Req 4.7, 27.4).

    Always includes the base caveat naming the four documented confounders
    (shadow, occlusion, dividers, image quality) so an empty-looking region is
    never asserted as a proven out-of-stock condition. Deterministically appends
    the most relevant specific note for this region when its measurements point
    at a particular confounder:

    * high edge density -> clutter/occlusion warning;
    * region outside any detected shelf band -> off-shelf position warning;
    * borderline ratio -> ambiguous-boundary warning;
    * low measurement confidence -> general low-confidence warning.

    The returned string is always non-empty, satisfying the contract's
    ``visual_caveat`` requirement structurally as well as in spirit.

    Args:
        candidate: The measured region from analysis.
        empty_space_ratio: The region's computed empty-space ratio.
        edge_density: The region's computed edge density.
        uncertainty_value: The region's computed measurement confidence.

    Returns:
        A non-empty caveat string.
    """
    notes: list[str] = [_BASE_CAVEAT]

    if edge_density >= _HIGH_EDGE_DENSITY:
        notes.append(
            "High edge density inside this region suggests clutter or an "
            "occluding object, so the empty estimate is especially uncertain."
        )
    if not candidate.in_shelf_band:
        notes.append(
            "This region falls outside a detected shelf band, so it may not "
            "correspond to shelf stocking space at all."
        )
    if abs(empty_space_ratio - 0.5) * 2.0 < 0.2:
        notes.append(
            "The empty-space ratio is near the ambiguous midpoint, so whether "
            "the region is empty or filled is not clear-cut."
        )
    if uncertainty_value <= 0.3:
        notes.append(
            "Overall measurement confidence for this region is low; treat the "
            "estimate with added caution."
        )

    return " ".join(notes)


# ---------------------------------------------------------------------------
# Follow-up action hint (Req 4.6)
# ---------------------------------------------------------------------------

#: A region is a "clear" empty-shelf candidate when its empty-space ratio is at
#: or above this value. Keyed off the config's empty_space_ratio_threshold.
#: (Threshold passed in; this is only a documentation anchor.)

#: Measurement confidence at/above which a region's measurement is treated as
#: reliable enough to act on directly. Fixed and documented for determinism.
_CONFIDENT_UNCERTAINTY: float = 0.6

#: Confidence at/below which a region is "ambiguous" (not trustworthy enough to
#: act on, but notable enough to gather more evidence).
_AMBIGUOUS_UNCERTAINTY: float = 0.35


def suggested_follow_up(
    candidate_regions: list[CandidateRegion],
    *,
    empty_ratio_threshold: float,
    source: str,
) -> AgentAction:
    """Derive a suggested next-action HINT for the agent (Req 4.6).

    This is only a suggestion; the agent makes the real decision later. The hint
    is derived deterministically from the assembled regions:

    * **No notable regions** (none at/above the empty-ratio threshold) ->
      ``finalize_report``: nothing stands out, so there is nothing more to probe.
    * **A clear, confident empty region** (ratio >= threshold and confidence >=
      :data:`_CONFIDENT_UNCERTAINTY`) -> ``inspect_crop``: zoom in to confirm.
    * **Conflicting evidence** (at least one clear-empty region AND at least one
      clearly-filled region) -> ``request_human_review``: the frame disagrees
      with itself.
    * **Ambiguous evidence** (notable regions but low confidence) -> gather more:
      ``analyze_another_frame`` for a video-derived finding (another frame is
      available), otherwise ``request_better_image`` for a single still.

    Args:
        candidate_regions: The assembled regions for this finding.
        empty_ratio_threshold: The config's empty-space ratio flag threshold.
        source: The finding source; ``"video_frame"`` enables the
            another-frame suggestion.

    Returns:
        One of the fixed five :data:`~shelfsentry.data.contracts.AgentAction`
        values as a suggestion only.
    """
    if not candidate_regions:
        return "finalize_report"

    clear_empty = [
        r
        for r in candidate_regions
        if r.empty_space_ratio is not None
        and r.empty_space_ratio >= empty_ratio_threshold
    ]
    clearly_filled = [
        r
        for r in candidate_regions
        if r.empty_space_ratio is not None
        and r.empty_space_ratio <= (1.0 - empty_ratio_threshold)
    ]

    if not clear_empty:
        # Nothing rises to the "looks empty" bar: nothing notable to probe.
        return "finalize_report"

    # Conflicting evidence within one input: some region looks clearly empty
    # while another looks clearly filled. Escalate to a human.
    if clearly_filled:
        return "request_human_review"

    # There is at least one clear-empty region. Act on it if we are confident.
    best_confidence = max(r.uncertainty_value for r in clear_empty)
    if best_confidence >= _CONFIDENT_UNCERTAINTY:
        return "inspect_crop"

    # Notable but not confident -> gather more evidence.
    if source == "video_frame":
        return "analyze_another_frame"
    return "request_better_image"


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_candidate_region(
    candidate: RegionCandidate,
    *,
    region_id: str,
    image_area: int,
    frame_id: str | None = None,
) -> CandidateRegion:
    """Build one validated :class:`CandidateRegion` from a measured candidate.

    Computes the region's ``empty_space_ratio`` (measured), ``heuristic_score``
    (ordinal), ``uncertainty_value`` (measurement confidence), and a non-empty
    ``visual_caveat``, then returns the pydantic-validated contract object. All
    four derived values are pure deterministic functions of ``candidate`` and
    ``image_area``.

    Args:
        candidate: The measured region from analysis.
        region_id: Stable identifier for this region within the finding.
        image_area: ``height * width`` of the analyzed input.
        frame_id: Source frame id when the region derives from a video frame
            (Req 4.4); ``None`` for image inputs.

    Returns:
        A validated :class:`CandidateRegion`.
    """
    ratio = compute_empty_space_ratio(candidate)
    density = compute_edge_density(candidate)
    heuristic = compute_heuristic_score(
        candidate,
        image_area=image_area,
        empty_space_ratio=ratio,
        edge_density=density,
    )
    uncertainty = compute_uncertainty_value(
        candidate,
        image_area=image_area,
        empty_space_ratio=ratio,
        edge_density=density,
    )
    caveat = build_visual_caveat(
        candidate,
        empty_space_ratio=ratio,
        edge_density=density,
        uncertainty_value=uncertainty,
    )

    x, y, w, h = candidate.bounding_box
    return CandidateRegion(
        region_id=region_id,
        bounding_box=BoundingBox(x=x, y=y, w=w, h=h),
        empty_space_ratio=ratio,
        alt_metric=None,
        heuristic_score=heuristic,
        uncertainty_value=uncertainty,
        frame_id=frame_id,
        visual_caveat=caveat,
    )


def build_finding(
    analysis: AnalysisResult,
    *,
    inspection_id: str,
    finding_id: str,
    annotated_output_ref: str,
    source: str = "image",
    frame_id: str | None = None,
    cfg: VisionConfig,
) -> VisionFinding:
    """Assemble a validated :class:`VisionFinding` from an ``AnalysisResult``.

    Converts each measured :class:`~shelfsentry.vision.analyze.RegionCandidate`
    into a :class:`CandidateRegion` (ratio + ordinal score + confidence +
    caveat), attaches the documented ``uncertainty_interpretations`` map
    (Req 4.3), labels the result an estimate (``is_estimate=True``, Req 2.5),
    records the exact verified OpenCV runtime version (Req 2.1), and derives a
    ``suggested_follow_up`` next-action hint (Req 4.6).

    Region ids are assigned deterministically in the analysis stage's stable
    candidate order as ``region_{index:04d}``, so repeated builds from the same
    ``AnalysisResult`` produce byte-for-byte identical ``canonical_json``
    (Req 5.1). The caller supplies ``annotated_output_ref`` (annotation
    rendering is task 3.3); this function only records it.

    Args:
        analysis: The deterministic analysis output to score.
        inspection_id: Owning inspection identifier (non-empty).
        finding_id: Identifier for this finding (non-empty).
        annotated_output_ref: Storage reference to the Annotated_Output
            (Req 4.5); supplied by the caller.
        source: Finding source; one of ``"image"``, ``"video_frame"``,
            ``"crop"``, ``"frame_compare"``. Defaults to ``"image"``.
        frame_id: Present for video-derived findings (Req 4.4); also applied to
            each region when ``source == "video_frame"``.
        cfg: Vision configuration (supplies the empty-space ratio threshold used
            by the follow-up hint).

    Returns:
        A validated :class:`VisionFinding`.

    Raises:
        pydantic.ValidationError: if any assembled value violates a contract
            boundary (names the failing field; Req 10.2, 10.5).
    """
    image_area = int(analysis.height) * int(analysis.width)

    # A region's frame_id is only meaningful for video-derived findings.
    region_frame_id = frame_id if source == "video_frame" else None

    regions: list[CandidateRegion] = []
    for index, candidate in enumerate(analysis.candidates):
        regions.append(
            build_candidate_region(
                candidate,
                region_id=f"region_{index:04d}",
                image_area=image_area,
                frame_id=region_frame_id,
            )
        )

    follow_up = suggested_follow_up(
        regions,
        empty_ratio_threshold=cfg.empty_space_ratio_threshold,
        source=source,
    )

    return VisionFinding(
        finding_id=finding_id,
        inspection_id=inspection_id,
        source=source,  # validated against the Literal set by pydantic
        frame_id=frame_id,
        opencv_version=opencv_version(),
        candidate_regions=regions,
        uncertainty_interpretations=dict(UNCERTAINTY_INTERPRETATIONS),
        annotated_output_ref=annotated_output_ref,
        suggested_follow_up=follow_up,
        is_estimate=True,
    )
