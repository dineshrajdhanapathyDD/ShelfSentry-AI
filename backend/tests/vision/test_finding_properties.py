"""Property-based tests for VisionFinding candidate-region fields and caveats.

Task 4.1. These properties exercise the *real* vision pipeline - synthetic
shelf-like images are generated deterministically from Hypothesis-drawn
parameters, decoded + analyzed by
:func:`shelfsentry.vision.analyze.analyze_image_bytes`, and assembled into a
:class:`~shelfsentry.data.contracts.VisionFinding` by
:func:`shelfsentry.vision.scoring.build_finding`. Each property then holds
across every generated finding (>= 100 examples via the ``ci`` Hypothesis
profile in ``conftest.py``).

Unlike ``test_scoring.py`` - which pins the documented behavior on a single
fixed image and on hand-built ``RegionCandidate`` values - these tests assert
the field/caveat *invariants hold universally* over a varied population of
images (different sizes, gap/product placement and brightness, and lighting
offsets). The intent is to avoid re-stating the example-based assertions in
``test_scoring.py`` and instead stress the invariants across the input space.

Properties implemented (from the design's Correctness Properties):

* **Property 3: Every finding records the exact OpenCV version.** (Req 2.1)
* **Property 4: All vision results are labeled as estimates.** (Req 2.5)
* **Property 5: Every candidate region carries required, well-typed fields.**
  (Req 2.6, 4.2)
* **Property 6: Every uncertainty value has a documented interpretation.**
  (Req 4.3)
* **Property 7: Empty-looking regions always carry a visual caveat.**
  (Req 4.7, 27.4)

Image generation is kept small/fast (<= ~200 px per side, grayscale PNG). The
generator paints dark gaps on a bright background inside explicit shelf bands,
so the overwhelming majority of examples yield >= 1 candidate region; a shared
"at least one region was produced" counter plus a session-end check guarantees
the region-dependent properties are genuinely exercised, and ``assume`` skips
only the rare degenerate image that produces none.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import BoundingBox, CandidateRegion, VisionFinding
from shelfsentry.vision.analyze import analyze_image_bytes
from shelfsentry.vision.scoring import build_finding
from shelfsentry.vision.version import opencv_version

# The exact OpenCV major.minor.patch this system is pinned to and validated
# against (Req 2.1, 26.4). Property 3 asserts the finding records precisely it.
_EXPECTED_OPENCV_VERSION = "5.0.0"

# Confounders every visual caveat must name so an empty-looking region is never
# asserted as a proven out-of-stock condition (Req 4.7, 27.4). "divider" is a
# stem matching both "divider" and "dividers".
_REQUIRED_CONFOUNDERS = ("shadow", "occlusion", "divider", "image quality")

# Tracks, across the whole generated population, whether at least one finding
# carried >= 1 candidate region. Region-level properties (5 and 7) use
# ``assume`` to skip the rare region-less image, so this guards against the
# degenerate case where the generator somehow never produced a region at all.
_regions_seen = {"count": 0}


# ---------------------------------------------------------------------------
# Deterministic synthetic shelf-image generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ShelfSpec:
    """A deterministic description of a synthetic shelf image.

    Drawn entirely from Hypothesis integers so the resulting image bytes are a
    pure function of the spec (no randomness), keeping the pipeline reproducible
    per example.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        n_bands: Number of horizontal shelf bands.
        n_gaps: Number of dark empty gaps to paint across the bands.
        gap_brightness: Gray value used for the dark gaps (low == clearly empty).
        product_brightness: Gray value used for bright product blocks.
        lighting_offset: A signed brightness offset added across the image to
            simulate uneven/overall lighting changes.
        seed: Placement seed mixed into deterministic gap/product x-positions.
    """

    width: int
    height: int
    n_bands: int
    n_gaps: int
    gap_brightness: int
    product_brightness: int
    lighting_offset: int
    seed: int


def _render_shelf(spec: _ShelfSpec) -> np.ndarray:
    """Render a deterministic grayscale shelf image from a spec.

    Paints a bright background, a set of dark horizontal shelf lips dividing the
    frame into bands, then alternating dark gaps and bright product blocks
    inside those bands. A uniform lighting offset is added last and the result
    is clipped to ``uint8``. Entirely deterministic given ``spec``.
    """
    w, h = spec.width, spec.height
    img = np.full((h, w), 215, dtype=np.int16)  # bright background (headroom)

    # Horizontal shelf lips split the frame into n_bands evenly spaced bands.
    band_edges = [int(round(i * h / (spec.n_bands + 1))) for i in range(spec.n_bands + 2)]
    lip = max(1, h // 60)
    for edge in band_edges[1:-1]:
        y0 = min(max(edge - lip, 0), h - 1)
        y1 = min(edge + lip, h)
        img[y0:y1, :] = 20

    # Fill each band with alternating dark gaps / bright product blocks.
    gaps_left = spec.n_gaps
    for b in range(spec.n_bands + 1):
        top = band_edges[b] + lip
        bottom = band_edges[b + 1] - lip
        if bottom - top < 4:
            continue
        # Deterministic horizontal partition of this band into cells.
        n_cells = 2 + ((spec.seed + b) % 3)  # 2..4 cells
        for c in range(n_cells):
            x0 = int(round(c * w / n_cells)) + 2
            x1 = int(round((c + 1) * w / n_cells)) - 2
            if x1 - x0 < 3:
                continue
            # Alternate: even cells -> product, odd cells -> gap, biased so we
            # still paint the requested number of dark gaps first.
            make_gap = (c % 2 == 1) or (gaps_left > 0 and (spec.seed + b + c) % 2 == 0)
            if make_gap and gaps_left > 0:
                img[top:bottom, x0:x1] = spec.gap_brightness
                gaps_left -= 1
            else:
                img[top:bottom, x0:x1] = spec.product_brightness
                # A vertical edge inside the product block (texture).
                mid = (x0 + x1) // 2
                img[top:bottom, mid : mid + 1] = max(0, spec.product_brightness - 90)

    img = img + spec.lighting_offset
    return np.clip(img, 0, 255).astype(np.uint8)


def _encode_png(img: np.ndarray) -> bytes:
    """Encode a grayscale array to lossless PNG bytes (deterministic)."""
    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode failed to encode the synthetic shelf image"
    return buf.tobytes()


# Hypothesis strategy producing a deterministic shelf spec. Ranges are kept
# small so each decode+analyze stays fast, while varying size, band/gap counts,
# gap/product brightness, lighting offset, and placement.
_shelf_specs = st.builds(
    _ShelfSpec,
    width=st.integers(min_value=64, max_value=200),
    height=st.integers(min_value=64, max_value=200),
    n_bands=st.integers(min_value=1, max_value=4),
    n_gaps=st.integers(min_value=1, max_value=5),
    gap_brightness=st.integers(min_value=0, max_value=45),  # clearly dark gap
    product_brightness=st.integers(min_value=150, max_value=255),  # clearly bright
    lighting_offset=st.integers(min_value=-40, max_value=40),
    seed=st.integers(min_value=0, max_value=1000),
)


def _finding_for(spec: _ShelfSpec, cfg: VisionConfig) -> VisionFinding:
    """Run the real pipeline end to end for a spec and return the finding.

    ``analyze_image_bytes`` (decode + preprocess + OpenCV 5 analysis) feeds
    ``build_finding`` (scoring + assembly), mirroring production use.
    """
    data = _encode_png(_render_shelf(spec))
    analysis = analyze_image_bytes(data, cfg)
    return build_finding(
        analysis,
        inspection_id="insp-prop",
        finding_id="find-prop",
        annotated_output_ref="ref://annotated/find-prop.png",
        source="image",
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Property 3: Every finding records the exact OpenCV version.
# Validates: Requirements 2.1
# ---------------------------------------------------------------------------


@given(spec=_shelf_specs)
def test_property3_finding_records_exact_opencv_version(spec: _ShelfSpec) -> None:
    """Property 3: Every finding records the exact OpenCV version.

    Across every generated image the assembled finding's recorded
    ``opencv_version`` equals the verified runtime version reported by
    :func:`shelfsentry.vision.version.opencv_version`, which is exactly
    ``"5.0.0"`` (Req 2.1). This holds regardless of whether any region was
    found, so no ``assume`` is needed.

    Validates: Requirements 2.1
    """
    cfg = VisionConfig()
    finding = _finding_for(spec, cfg)

    assert finding.opencv_version == opencv_version()
    assert finding.opencv_version == _EXPECTED_OPENCV_VERSION


# ---------------------------------------------------------------------------
# Property 4: All vision results are labeled as estimates.
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------


@given(spec=_shelf_specs)
def test_property4_all_results_labeled_estimates(spec: _ShelfSpec) -> None:
    """Property 4: All vision results are labeled as estimates.

    Every assembled finding, for any generated image, carries
    ``is_estimate is True`` so downstream consumers can never mistake a visual
    measurement for a proven out-of-stock condition (Req 2.5). Independent of
    region count, so no ``assume`` is needed.

    Validates: Requirements 2.5
    """
    cfg = VisionConfig()
    finding = _finding_for(spec, cfg)

    assert finding.is_estimate is True


# ---------------------------------------------------------------------------
# Property 5: Every candidate region carries required, well-typed fields.
# Validates: Requirements 2.6, 4.2
# ---------------------------------------------------------------------------


@given(spec=_shelf_specs)
def test_property5_regions_carry_required_well_typed_fields(spec: _ShelfSpec) -> None:
    """Property 5: Every candidate region carries required, well-typed fields.

    For every region of every generated finding:

    * ``bounding_box`` is a :class:`BoundingBox` whose integer ``(x, y, w, h)``
      lie inside the analyzed image (``w``/``h`` positive, box within bounds);
    * either ``empty_space_ratio`` is a float in ``[0, 1]`` OR a documented
      ``alt_metric`` mapping is supplied in its place (exactly one of the two);
    * ``heuristic_score`` is present and a float in ``[0, 1]`` (an ordinal
      ranking score, Req 2.6);
    * ``uncertainty_value`` is present and a float in ``[0, 1]`` (Req 4.2).

    The image dimensions used for the bounds check are read back from the same
    analysis that produced the finding.

    Validates: Requirements 2.6, 4.2
    """
    cfg = VisionConfig()
    data = _encode_png(_render_shelf(spec))
    analysis = analyze_image_bytes(data, cfg)
    finding = build_finding(
        analysis,
        inspection_id="insp-prop",
        finding_id="find-prop",
        annotated_output_ref="ref://annotated/find-prop.png",
        source="image",
        cfg=cfg,
    )

    # Keep the region-dependent property meaningful: skip the rare image that
    # yields no region, but record when regions were seen.
    assume(len(finding.candidate_regions) >= 1)
    _regions_seen["count"] += 1

    img_w, img_h = analysis.width, analysis.height

    for region in finding.candidate_regions:
        assert isinstance(region, CandidateRegion)

        # bounding_box: well-typed integer box inside the analyzed image.
        box = region.bounding_box
        assert isinstance(box, BoundingBox)
        assert isinstance(box.x, int) and isinstance(box.y, int)
        assert isinstance(box.w, int) and isinstance(box.h, int)
        assert box.w > 0 and box.h > 0
        assert box.x >= 0 and box.y >= 0
        assert box.x + box.w <= img_w
        assert box.y + box.h <= img_h

        # empty_space_ratio in [0, 1] OR a documented alt_metric (exactly one).
        has_ratio = region.empty_space_ratio is not None
        has_alt = region.alt_metric is not None
        assert has_ratio ^ has_alt, (
            "a region must report exactly one of empty_space_ratio / alt_metric"
        )
        if has_ratio:
            assert isinstance(region.empty_space_ratio, float)
            assert 0.0 <= region.empty_space_ratio <= 1.0
        else:
            assert isinstance(region.alt_metric, dict)
            assert region.alt_metric  # documented, non-empty

        # heuristic_score present and in [0, 1] (ordinal ranking, Req 2.6).
        assert isinstance(region.heuristic_score, float)
        assert 0.0 <= region.heuristic_score <= 1.0

        # uncertainty_value present and in [0, 1] (Req 4.2).
        assert isinstance(region.uncertainty_value, float)
        assert 0.0 <= region.uncertainty_value <= 1.0


# ---------------------------------------------------------------------------
# Property 6: Every uncertainty value has a documented interpretation.
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------


@given(spec=_shelf_specs)
def test_property6_uncertainty_values_documented(spec: _ShelfSpec) -> None:
    """Property 6: Every uncertainty value has a documented interpretation.

    Every finding carries a non-empty ``uncertainty_interpretations`` map that
    contains a non-empty entry explaining ``uncertainty_value`` (Req 4.3), so a
    consumer never has to guess what the per-region ``uncertainty_value`` means.
    This is a finding-level guarantee and holds whether or not any region was
    produced.

    Validates: Requirements 4.3
    """
    cfg = VisionConfig()
    finding = _finding_for(spec, cfg)

    interpretations = finding.uncertainty_interpretations
    assert isinstance(interpretations, dict)
    assert interpretations, "uncertainty_interpretations must be non-empty"

    # An entry must explain the per-region uncertainty_value specifically.
    assert "uncertainty_value" in interpretations
    explanation = interpretations["uncertainty_value"]
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


# ---------------------------------------------------------------------------
# Property 7: Empty-looking regions always carry a visual caveat.
# Validates: Requirements 4.7, 27.4
# ---------------------------------------------------------------------------


@given(spec=_shelf_specs)
def test_property7_regions_carry_visual_caveat(spec: _ShelfSpec) -> None:
    """Property 7: Empty-looking regions always carry a visual caveat.

    Every candidate region of every generated finding has a non-empty
    ``visual_caveat`` that names each documented confounder - shadow,
    occlusion, dividers, and image quality - so an empty-looking region is
    never presented as a proven out-of-stock condition (Req 4.7, 27.4).

    Validates: Requirements 4.7, 27.4
    """
    cfg = VisionConfig()
    finding = _finding_for(spec, cfg)

    # Region-dependent: skip the rare region-less image, record otherwise.
    assume(len(finding.candidate_regions) >= 1)
    _regions_seen["count"] += 1

    for region in finding.candidate_regions:
        caveat = region.visual_caveat
        assert isinstance(caveat, str)
        assert caveat.strip() != ""
        lowered = caveat.lower()
        for confounder in _REQUIRED_CONFOUNDERS:
            assert confounder in lowered, (
                f"visual caveat must name the {confounder!r} confounder"
            )


# ---------------------------------------------------------------------------
# Guard: the region-dependent properties were actually exercised.
# ---------------------------------------------------------------------------


def test_zz_region_dependent_properties_were_exercised() -> None:
    """At least one generated image produced >= 1 candidate region.

    Properties 5 and 7 use ``assume`` to skip the rare degenerate image that
    yields no region. This sentinel (named ``zz_*`` so it runs after the
    property tests in file order) fails loudly if the generator never produced
    a single region, which would otherwise let those properties pass vacuously.
    """
    assert _regions_seen["count"] > 0, (
        "no generated image produced a candidate region; Properties 5 and 7 "
        "would have passed vacuously - the shelf-image generator needs review"
    )
