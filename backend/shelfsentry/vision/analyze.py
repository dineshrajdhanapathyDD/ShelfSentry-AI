"""Substantive OpenCV 5 classical analysis of shelf images (Req 2.2, 5.1, 5.2).

This stage turns the deterministic preprocessing intermediates produced by
:func:`shelfsentry.vision.preprocess.preprocess_image` into *candidate
empty-shelf regions* plus the intermediate masks and per-region measurements
that the scoring stage (``scoring.py``, task 3.2) consumes to compute
``empty_space_ratio`` / ``heuristic_score`` / ``uncertainty_value``.

It does **not** assemble a :class:`~shelfsentry.data.contracts.VisionFinding`;
that is scoring's job. The output of this module is a typed, geometry-plus-masks
holder (:class:`AnalysisResult`) carrying a stable, deterministic list of
:class:`RegionCandidate` objects.

Pipeline (every operation is a fixed, config-driven OpenCV 5 classical
primitive with no randomness, so repeated runs over the same bytes + the same
:class:`~shelfsentry.config.settings.VisionConfig` are byte-for-byte identical,
Req 5.1):

1. **Shelf-region analysis** - ``cv2.Sobel`` vertical gradient magnitude plus a
   horizontal projection (row-wise sum via ``cv2.reduce``) locates the
   horizontal shelf *bands* that constrain where product should be. An optional
   ``cv2.HoughLinesP`` pass (``detect_shelf_lines``) corroborates the strongest
   horizontal edges and is recorded for downstream use/annotation.
2. **Thresholding** - Otsu (``cv2.threshold`` with ``THRESH_OTSU``) separates
   filled (bright/product) from empty (dark gap) pixels under even lighting; an
   adaptive threshold (``cv2.adaptiveThreshold``) is also computed for uneven
   lighting. The two are combined into one deterministic *filled* mask.
3. **Morphology** - ``cv2.morphologyEx`` open-then-close with a
   ``morphology_ksize`` square structuring element removes speckle and
   consolidates gap/product regions so measurement is stable.
4. **Edge analysis** - ``cv2.Canny`` (``canny_threshold_low`` /
   ``canny_threshold_high``) yields an edge map; low edge density inside a shelf
   band is a strong empty-space cue and is handed to scoring.
5. **Contour analysis** - ``cv2.findContours`` + ``cv2.boundingRect`` +
   ``cv2.contourArea`` on the *empty* (inverted filled) mask delineate gap
   blobs. Contours below ``min_contour_area`` are dropped, and the survivors are
   sorted by a stable key (top-left ``(y, x)`` then area) so the emitted
   ``(x, y, w, h)`` integer-pixel boxes are reproducible across runs (Req 5.1,
   5.2).

Coordinate system (single, documented; Req 5.2): every bounding box is
``(x, y, w, h)`` in **integer pixels**, origin **top-left**, measured relative
to the analyzed input's own ``(height, width)`` as returned by preprocessing.
Downstream annotation draws at exactly these coordinates (Req 5.3).

OpenCV 5 note: ``cv2.findContours`` returns a **2-tuple**
``(contours, hierarchy)`` in OpenCV 5.0.0 (verified against the installed
runtime), unlike the 3-tuple of OpenCV 3.x. :func:`_find_contours` normalizes
the arity defensively so the module keeps working if a future 5.x point release
shifts it.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from shelfsentry.config.settings import VisionConfig
from shelfsentry.vision.preprocess import PreprocessResult, preprocess_image

__all__ = [
    "ShelfBand",
    "RegionCandidate",
    "AnalysisResult",
    "analyze_image_arrays",
    "analyze_image_bytes",
]


class ShelfBand(NamedTuple):
    """A horizontal shelf band located by gradient + horizontal projection.

    Bands constrain where product (and therefore empty space) is expected. The
    band spans the full image width; ``y_top``/``y_bottom`` are inclusive pixel
    rows in the analyzed input's coordinate system (origin top-left).

    Attributes:
        y_top: Top row of the band (pixels from the top edge).
        y_bottom: Bottom row of the band (inclusive, pixels from the top edge).
        strength: Mean vertical-gradient projection strength over the band's
            boundary rows; a relative, deterministic measure used only for
            ranking/interpretation (never a probability).
    """

    y_top: int
    y_bottom: int
    strength: float


class RegionCandidate(NamedTuple):
    """One candidate empty-shelf region and the raw measurements scoring needs.

    This is the typed intermediate the scoring stage (task 3.2) consumes. It
    intentionally carries *measured* quantities only - no ``empty_space_ratio``,
    ``heuristic_score``, or ``uncertainty_value`` are computed here; scoring
    derives those from these fields and the shared masks on
    :class:`AnalysisResult`.

    Coordinate system: ``bounding_box`` is ``(x, y, w, h)`` integer pixels,
    origin top-left, relative to the analyzed input dimensions (Req 5.2).

    Attributes:
        bounding_box: ``(x, y, w, h)`` integer-pixel box, top-left origin.
        contour_area: ``cv2.contourArea`` of the source contour (pixels^2).
        box_area: ``w * h`` of the bounding box (pixels^2); never zero.
        filled_pixels: Count of *filled* (product) pixels inside the box, from
            the combined filled mask.
        empty_pixels: Count of *empty* (gap) pixels inside the box
            (``box_area - filled_pixels``).
        edge_pixels: Count of Canny edge pixels inside the box; low edge density
            is an empty-space cue.
        in_shelf_band: ``True`` if the box center row falls within a detected
            shelf band (or no bands were found, in which case the whole image is
            treated as one implicit band).
    """

    bounding_box: tuple[int, int, int, int]
    contour_area: float
    box_area: int
    filled_pixels: int
    empty_pixels: int
    edge_pixels: int
    in_shelf_band: bool


class AnalysisResult(NamedTuple):
    """Raw geometry plus intermediate masks/measurements for scoring (task 3.2).

    Every array shares the analyzed input's ``(height, width)``. Masks are
    ``uint8`` with values in ``{0, 255}``. Nothing here is a
    :class:`~shelfsentry.data.contracts.VisionFinding`; scoring assembles that.

    Attributes:
        height: Analyzed input height in pixels.
        width: Analyzed input width in pixels.
        candidates: Deterministically ordered candidate empty-shelf regions
            (sorted by top-left ``(y, x)`` then area).
        shelf_bands: Horizontal shelf bands located by the shelf-region stage.
        shelf_lines: Optional ``cv2.HoughLinesP`` horizontal line segments as
            ``(x1, y1, x2, y2)`` integer tuples (corroborating evidence); may be
            empty.
        filled_mask: ``uint8`` mask, 255 where a pixel is judged *filled*
            (product) by the combined Otsu/adaptive + morphology step.
        empty_mask: ``uint8`` mask, the inversion of ``filled_mask`` (255 where
            *empty*); contours are found on this mask.
        edge_map: ``uint8`` ``cv2.Canny`` edge map (255 on edges).
        otsu_threshold: The Otsu threshold value OpenCV selected (for the
            interpretation record; deterministic for a given input).
    """

    height: int
    width: int
    candidates: tuple[RegionCandidate, ...]
    shelf_bands: tuple[ShelfBand, ...]
    shelf_lines: tuple[tuple[int, int, int, int], ...]
    filled_mask: np.ndarray
    empty_mask: np.ndarray
    edge_map: np.ndarray
    otsu_threshold: float


# ---------------------------------------------------------------------------
# Stage 1: shelf-region analysis (Sobel gradient + horizontal projection)
# ---------------------------------------------------------------------------


def _detect_shelf_bands(gray: np.ndarray) -> tuple[ShelfBand, ...]:
    """Locate horizontal shelf bands via vertical-gradient horizontal projection.

    Shelves and their contents produce strong horizontal edges (the shelf lips
    and product tops). We take the vertical Sobel gradient magnitude, project it
    across each row (``cv2.reduce`` row-sum), and treat contiguous runs of rows
    whose projection is at/above the mean projection as band *boundaries*. The
    regions *between* consecutive strong boundaries are the shelf bands where
    product is expected.

    The computation is fully deterministic (fixed Sobel kernel, mean-based
    threshold). It is intentionally conservative: if no clear structure is
    found, it returns an empty tuple and the caller treats the whole image as
    one implicit band.

    Args:
        gray: Single-channel ``uint8`` grayscale image (the raw, non-equalized
            grayscale is preferred so flat-background texture does not inflate
            the gradient projection).

    Returns:
        A tuple of :class:`ShelfBand`, top-to-bottom, possibly empty.
    """
    height = gray.shape[0]
    if height < 3:
        return ()

    # Vertical gradient magnitude: |d/dy|. Strong where horizontal structure
    # (shelf lips / product tops) sits.
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1, ksize=3)
    grad = np.abs(sobel_y)

    # Horizontal projection: sum the gradient across each row -> one value/row.
    projection = cv2.reduce(grad, 1, cv2.REDUCE_SUM).reshape(-1)
    if not np.any(projection):
        return ()

    # Deterministic threshold scaled into the projection's dynamic range: a row
    # is a candidate shelf-edge row only when its gradient projection rises a
    # fixed fraction of the way from the mean toward the peak. Keying off
    # (max - mean) rather than the mean alone keeps a handful of genuinely
    # strong shelf lips and rejects background texture, which otherwise fires
    # on a bare mean threshold.
    mean_proj = float(projection.mean())
    max_proj = float(projection.max())
    edge_threshold = mean_proj + 0.5 * (max_proj - mean_proj)
    strong = projection >= edge_threshold

    # Collapse contiguous runs of strong rows into boundary markers at each
    # run's center; the gaps between boundaries are the shelf bands.
    boundaries: list[tuple[int, float]] = []
    row = 0
    while row < height:
        if strong[row]:
            start = row
            while row < height and strong[row]:
                row += 1
            end = row - 1
            center = (start + end) // 2
            boundaries.append((center, float(projection[start : end + 1].mean())))
        else:
            row += 1

    if len(boundaries) < 2:
        return ()

    bands: list[ShelfBand] = []
    for (top_center, top_strength), (bot_center, bot_strength) in zip(
        boundaries, boundaries[1:]
    ):
        y_top = top_center
        y_bottom = bot_center
        if y_bottom - y_top < 1:
            continue
        bands.append(
            ShelfBand(
                y_top=int(y_top),
                y_bottom=int(y_bottom),
                strength=(top_strength + bot_strength) / 2.0,
            )
        )
    return tuple(bands)


def detect_shelf_lines(gray: np.ndarray, cfg: VisionConfig) -> tuple[
    tuple[int, int, int, int], ...
]:
    """Corroborate shelf edges with ``cv2.HoughLinesP`` (optional evidence).

    Runs probabilistic Hough on the Canny edge map and keeps only near-horizontal
    segments (the shelf lips). Parameters are fixed/config-derived so the result
    is deterministic. The segments are supporting evidence for shelf structure
    and for annotation; the core band detection does not depend on them.

    Args:
        gray: Single-channel ``uint8`` image to find edges in.
        cfg: Vision config supplying the Canny thresholds.

    Returns:
        A tuple of ``(x1, y1, x2, y2)`` integer segments, ordered by top edge
        then left edge; possibly empty.
    """
    edges = cv2.Canny(gray, cfg.canny_threshold_low, cfg.canny_threshold_high)
    width = gray.shape[1]
    # Require a segment to span a meaningful fraction of the width to count as a
    # shelf line; thresholds fixed relative to image width for determinism.
    min_length = max(10, width // 4)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(10, width // 4),
        minLineLength=min_length,
        maxLineGap=max(5, width // 40),
    )
    if lines is None:
        return ()

    horizontal: list[tuple[int, int, int, int]] = []
    for seg in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(v) for v in seg)
        # Near-horizontal: vertical span small relative to horizontal span.
        if abs(y2 - y1) <= max(2, abs(x2 - x1) // 10):
            horizontal.append((x1, y1, x2, y2))

    # Stable ordering so repeated runs emit identical sequences (Req 5.1).
    horizontal.sort(key=lambda s: (min(s[1], s[3]), min(s[0], s[2])))
    return tuple(horizontal)


# ---------------------------------------------------------------------------
# Stage 2 + 3: thresholding (Otsu / adaptive) + morphology -> filled mask
# ---------------------------------------------------------------------------


def _filled_mask(gray: np.ndarray, cfg: VisionConfig) -> tuple[np.ndarray, float]:
    """Build a deterministic *filled* (product) mask via Otsu + adaptive + morphology.

    Bright pixels are treated as product/filled and dark pixels as empty gap.
    Global Otsu (``cv2.threshold`` + ``THRESH_OTSU``) separates filled from empty
    under even lighting; an adaptive Gaussian threshold
    (``cv2.adaptiveThreshold``) recovers locally-bright product under *uneven*
    lighting that a single global cut would miss. A pixel is judged *filled* when
    **either** marks it bright (logical OR): this keeps real product on the mask
    across lighting variation while leaving only genuinely dark gaps empty. (The
    stricter AND variant tended to erode flat-lit product and merge the whole
    frame into one empty blob.) ``cv2.morphologyEx`` then opens (remove speckle)
    and closes (fill pinholes / join product blocks) with a ``morphology_ksize``
    square element so the mask is stable to measure.

    Runs on the raw grayscale image rather than the equalized one: histogram
    equalization rescales a flat, well-lit background toward the threshold and
    can split it spuriously, whereas raw-gray Otsu lands cleanly between dark
    gaps and bright product.

    Args:
        gray: Single-channel ``uint8`` grayscale image.
        cfg: Vision config supplying ``morphology_ksize``.

    Returns:
        ``(filled_mask, otsu_threshold)`` where ``filled_mask`` is ``uint8`` in
        ``{0, 255}`` (255 == filled) and ``otsu_threshold`` is the value OpenCV
        selected.
    """
    # Otsu: THRESH_BINARY so bright (product) -> 255. Threshold arg is ignored
    # when THRESH_OTSU is set; OpenCV computes it and returns it.
    otsu_t, otsu_bin = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Adaptive Gaussian: block size must be odd and >= 3; derive deterministically
    # from the configured morphology kernel so it scales with config, stays odd.
    block = cfg.morphology_ksize * 4 + 1  # odd, >= 5 for ksize >= 1
    adaptive_bin = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block,
        2,
    )

    # Either global or local thresholding marking a pixel bright -> filled.
    filled = cv2.bitwise_or(otsu_bin, adaptive_bin)

    ksize = cfg.morphology_ksize
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    opened = cv2.morphologyEx(filled, cv2.MORPH_OPEN, element)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, element)
    return closed, float(otsu_t)


# ---------------------------------------------------------------------------
# Stage 5: contour analysis (findContours / boundingRect / contourArea)
# ---------------------------------------------------------------------------


def _find_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Call ``cv2.findContours`` and return the contour list, arity-safe.

    OpenCV 5.0.0 returns ``(contours, hierarchy)`` (2-tuple); OpenCV 3.x
    returned ``(image, contours, hierarchy)`` (3-tuple). We select the
    second-to-last element so the code is correct under either arity, which
    guards against a future 5.x point release shifting the return shape.

    Args:
        mask: ``uint8`` binary mask to trace external contours on.

    Returns:
        A list of contour point arrays (possibly empty).
    """
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours is always the element immediately before hierarchy (the last).
    contours = result[-2]
    return list(contours)


def _extract_candidates(
    empty_mask: np.ndarray,
    filled_mask: np.ndarray,
    edge_map: np.ndarray,
    shelf_bands: tuple[ShelfBand, ...],
    cfg: VisionConfig,
) -> tuple[RegionCandidate, ...]:
    """Delineate gap blobs into deterministically ordered candidate regions.

    Finds external contours on the *empty* mask, drops any whose
    ``cv2.contourArea`` is below ``min_contour_area``, and for each survivor
    measures the filled/empty/edge pixel counts inside its ``cv2.boundingRect``.
    Candidates are sorted by a stable key - top-left ``(y, x)`` then box area -
    so the emitted boxes are byte-for-byte reproducible (Req 5.1, 5.2).

    Args:
        empty_mask: ``uint8`` mask with 255 on empty (gap) pixels.
        filled_mask: ``uint8`` mask with 255 on filled (product) pixels.
        edge_map: ``uint8`` Canny edge map (255 on edges).
        shelf_bands: Detected shelf bands used to flag ``in_shelf_band``.
        cfg: Vision config supplying ``min_contour_area``.

    Returns:
        A tuple of :class:`RegionCandidate` in stable order.
    """
    contours = _find_contours(empty_mask)

    candidates: list[RegionCandidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < cfg.min_contour_area:
            continue
        x, y, w, h = (int(v) for v in cv2.boundingRect(contour))
        if w < 1 or h < 1:
            continue

        filled_roi = filled_mask[y : y + h, x : x + w]
        edge_roi = edge_map[y : y + h, x : x + w]
        box_area = int(w * h)
        filled_pixels = int(np.count_nonzero(filled_roi))
        empty_pixels = int(box_area - filled_pixels)
        edge_pixels = int(np.count_nonzero(edge_roi))

        center_row = y + h // 2
        if shelf_bands:
            in_band = any(
                band.y_top <= center_row <= band.y_bottom for band in shelf_bands
            )
        else:
            # No bands detected -> whole image is one implicit band.
            in_band = True

        candidates.append(
            RegionCandidate(
                bounding_box=(x, y, w, h),
                contour_area=area,
                box_area=box_area,
                filled_pixels=filled_pixels,
                empty_pixels=empty_pixels,
                edge_pixels=edge_pixels,
                in_shelf_band=in_band,
            )
        )

    # Deterministic emission order: top-left (y, x) then area (Req 5.1, 5.2).
    candidates.sort(
        key=lambda c: (
            c.bounding_box[1],
            c.bounding_box[0],
            c.box_area,
            c.bounding_box[2],
            c.bounding_box[3],
        )
    )
    return tuple(candidates)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def analyze_image_arrays(pre: PreprocessResult, cfg: VisionConfig) -> AnalysisResult:
    """Run the full classical analysis over preprocessing intermediates.

    Deterministic for a given ``pre`` + ``cfg`` (Req 5.1). Produces the raw
    candidate geometry and the intermediate masks/measurements the scoring stage
    consumes; it does not build a ``VisionFinding``.

    Args:
        pre: Deterministic preprocessing intermediates from
            :func:`shelfsentry.vision.preprocess.preprocess_image`.
        cfg: Vision configuration (kernel sizes, thresholds, min area).

    Returns:
        An :class:`AnalysisResult` with ordered candidates, shelf bands/lines,
        and the filled/empty/edge masks.
    """
    gray = pre.gray
    equalized = pre.equalized
    height, width = gray.shape[:2]

    # Stage 1: shelf regions (gradient + horizontal projection, optional Hough).
    # Band/line localization runs on the raw grayscale image: CLAHE/equalization
    # amplifies flat-background texture into spurious gradient, which would blur
    # where the real shelf lips are. Thresholding below still uses the
    # lighting-normalized image where that normalization genuinely helps.
    shelf_bands = _detect_shelf_bands(gray)
    shelf_lines = detect_shelf_lines(gray, cfg)

    # Stage 2 + 3: thresholding + morphology -> filled mask (on raw gray).
    filled_mask, otsu_threshold = _filled_mask(gray, cfg)
    empty_mask = cv2.bitwise_not(filled_mask)

    # Stage 4: edge analysis.
    edge_map = cv2.Canny(equalized, cfg.canny_threshold_low, cfg.canny_threshold_high)

    # Stage 5: contour analysis -> deterministically ordered candidates.
    candidates = _extract_candidates(
        empty_mask=empty_mask,
        filled_mask=filled_mask,
        edge_map=edge_map,
        shelf_bands=shelf_bands,
        cfg=cfg,
    )

    return AnalysisResult(
        height=int(height),
        width=int(width),
        candidates=candidates,
        shelf_bands=shelf_bands,
        shelf_lines=shelf_lines,
        filled_mask=filled_mask,
        empty_mask=empty_mask,
        edge_map=edge_map,
        otsu_threshold=otsu_threshold,
    )


def analyze_image_bytes(data: bytes, cfg: VisionConfig) -> AnalysisResult:
    """Decode + preprocess + analyze raw image bytes in one deterministic call.

    Convenience wrapper that runs
    :func:`shelfsentry.vision.preprocess.preprocess_image` and then
    :func:`analyze_image_arrays`. Repeated calls with the same ``data`` + ``cfg``
    yield identical candidate bounding boxes (Req 5.1).

    Args:
        data: Raw encoded image bytes (already validated upstream).
        cfg: Vision configuration.

    Returns:
        The :class:`AnalysisResult` for the image.

    Raises:
        shelfsentry.vision.preprocess.PreprocessError: if the bytes cannot be
            decoded into an image.
    """
    pre = preprocess_image(data, cfg)
    return analyze_image_arrays(pre, cfg)
