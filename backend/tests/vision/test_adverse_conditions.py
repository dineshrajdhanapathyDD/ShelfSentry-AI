"""Example-based vision tests for normal and adverse conditions (task 4.4).

Covers the vision pipeline across the conditions Requirement 22 calls out:

* **Req 22.1** - valid, invalid, empty, and corrupt inputs: a normal shelf with
  a clear gap flows through validation -> analysis -> ``build_finding`` and
  yields at least one well-formed region; empty / corrupt / unsupported inputs
  are rejected by :func:`shelfsentry.vision.validate.validate_input` with the
  matching structured condition, and :func:`preprocess_image` raises
  :class:`PreprocessError` on garbage bytes.
* **Req 22.2** - varied lighting, shadows, dividers, occlusions, and multiple
  regions: each confounder is rendered as a deterministic synthetic image and
  run through the real OpenCV 5 pipeline. For the confounders, the assertions
  target the system's HONESTY contract (``is_estimate`` is True, every region
  carries a non-empty ``visual_caveat`` naming shadow / occlusion / dividers /
  image quality, and all reported metrics sit in ``[0, 1]``) rather than
  brittle exact-detection claims. The multiple-regions case uses two clearly
  separated dark gaps and asserts the analyzer yields at least two candidate
  regions.

Every image is constructed from NumPy arrays and encoded with ``cv2.imencode``,
so the inputs are deterministic and the pipeline (which is itself deterministic,
Req 5.1) produces stable results run to run. No mocks or fakes are used: the
tests drive real ``cv2`` analysis end to end.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import CandidateRegion, VisionFinding
from shelfsentry.vision.analyze import analyze_image_bytes
from shelfsentry.vision.errors import InputValidationError
from shelfsentry.vision.preprocess import PreprocessError, preprocess_image
from shelfsentry.vision.scoring import build_finding
from shelfsentry.vision.validate import validate_input


# ---------------------------------------------------------------------------
# Synthetic image construction (deterministic; numpy + cv2.imencode)
# ---------------------------------------------------------------------------

#: The four documented confounders a region's caveat must acknowledge, so an
#: empty-looking region is never asserted as a proven out-of-stock condition
#: (Req 4.7, 27.4). The caveat text uses these exact words.
_CAVEAT_CONFOUNDERS = ("shadow", "occlusion", "divider", "image quality")


def _encode_png(img: np.ndarray) -> bytes:
    """Encode a grayscale or BGR array as deterministic PNG bytes."""
    ok, buf = cv2.imencode(".png", img)
    assert ok, "failed to encode synthetic image"
    return buf.tobytes()


def _base_shelf(*, brightness: int = 220) -> np.ndarray:
    """A single-gap shelf: bright background, two shelf lips, one dark gap.

    The bottom shelf's left half is a dark empty gap; its right half holds a
    bright product block. The top shelf holds product. ``brightness`` scales the
    background/product levels so the same geometry can be rendered dark or
    bright for the lighting tests.
    """
    h, w = 240, 320
    product = min(255, int(brightness * 0.55) + 40)
    gap = max(0, int(brightness * 0.07))

    img = np.full((h, w), brightness, dtype=np.uint8)
    # Two dark horizontal shelf lips.
    img[78:82, :] = 20
    img[158:162, :] = 20
    # Top shelf: product block with an internal edge.
    img[40:78, 30:150] = product
    img[40:78, 150:154] = 10
    # Bottom shelf: left half dark empty gap, right half product.
    img[100:158, 10:150] = gap
    img[100:158, 170:300] = product
    return img


def _two_gap_shelf() -> np.ndarray:
    """A shelf with two clearly separated dark gaps (for the multi-region case).

    Both shelf rows carry a dark empty gap on their left, with a bright product
    block between and to the right. A wide bright band separates the two gaps
    vertically (the two shelf rows plus the lip between them), so the empty-mask
    contours do not merge into a single blob.
    """
    h, w = 240, 320
    img = np.full((h, w), 225, dtype=np.uint8)

    # Two dark horizontal shelf lips bound the two product rows.
    img[78:82, :] = 20
    img[158:162, :] = 20

    # Top shelf row: left dark gap, right product.
    img[20:78, 20:140] = 12
    img[20:78, 170:300] = 135

    # Bottom shelf row: left dark gap, right product. Well separated from the
    # top gap by the bright lip band around y=78..82.
    img[100:158, 20:140] = 12
    img[100:158, 170:300] = 135
    return img


def _shadow_shelf() -> np.ndarray:
    """A filled shelf with a dark diagonal shadow band across the products.

    The shelf is fully stocked (bright product on both rows). A graded dark band
    sweeps diagonally across the frame to imitate a cast shadow, a classic
    confounder that can read as empty space. The test asserts honesty, not a
    specific detection outcome.
    """
    img = _base_shelf(brightness=215)
    # Re-fill the bottom-left gap so the shelf is fully stocked before shadowing.
    img[100:158, 10:150] = 150

    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    # Diagonal gradient in [0, 1], strongest toward the top-left.
    grad = 1.0 - np.clip((xx + yy) / float(w + h), 0.0, 1.0)
    # Darken by up to ~55% along the gradient to simulate a shadow.
    shadow = (img.astype(np.float32) * (1.0 - 0.55 * grad)).astype(np.uint8)
    return shadow


def _divider_shelf() -> np.ndarray:
    """A stocked shelf with vertical dark divider lines between products.

    Evenly spaced dark vertical bars imitate shelf dividers, which can carve a
    filled shelf into dark slivers. The test asserts caveats are present, not
    that dividers are perfectly ignored.
    """
    img = _base_shelf(brightness=215)
    img[100:158, 10:150] = 150  # fill the gap: shelf is stocked

    # Vertical dark dividers every 40px across the product rows. The 8px width
    # is deliberate: thin dividers carve dark slivers the analyzer surfaces as
    # candidate regions, so the caveat assertion below has regions to check
    # (the honesty point: dividers must be caveated, not silently reported as
    # empty shelf).
    for x in range(20, 300, 40):
        img[40:78, x : x + 8] = 10
        img[100:158, x : x + 8] = 10
    return img


def _occlusion_shelf() -> np.ndarray:
    """A stocked shelf partially covered by a dark blob (an occluding object).

    A filled shelf with a large dark irregular blob drawn over part of the
    product, imitating something (a hand, a cart, a price sign) occluding the
    view. The test asserts honesty caveats are present.
    """
    img = _base_shelf(brightness=215)
    img[100:158, 10:150] = 150  # fill the gap: shelf is stocked

    # A dark filled blob (circle + rectangle) occluding part of the shelf.
    cv2.circle(img, (120, 120), 35, 8, thickness=-1)
    cv2.rectangle(img, (95, 95), (160, 150), 8, thickness=-1)
    return img


@pytest.fixture()
def cfg() -> VisionConfig:
    return VisionConfig()


# ---------------------------------------------------------------------------
# Shared honesty assertions
# ---------------------------------------------------------------------------


def _assert_region_is_honest(region: CandidateRegion) -> None:
    """Every region must be a well-formed, honestly-caveated estimate."""
    # Metrics in range (pydantic enforces [0, 1] on empty_space_ratio; the
    # ordinal/confidence values are documented [0, 1] too).
    if region.empty_space_ratio is not None:
        assert 0.0 <= region.empty_space_ratio <= 1.0
    assert 0.0 <= region.heuristic_score <= 1.0
    assert 0.0 <= region.uncertainty_value <= 1.0

    # Caveat is non-empty and names the documented confounders (Req 4.7, 27.4).
    caveat = region.visual_caveat.lower()
    assert caveat.strip(), "visual_caveat must be non-empty"
    for word in _CAVEAT_CONFOUNDERS:
        assert word in caveat, f"caveat must mention {word!r}; got {region.visual_caveat!r}"


def _assert_finding_is_honest(finding: VisionFinding) -> None:
    """A finding must be labeled an estimate and round-trip boundary validation."""
    assert finding.is_estimate is True
    assert finding.opencv_version  # exact recorded runtime version (Req 2.1)
    for region in finding.candidate_regions:
        _assert_region_is_honest(region)
    # Strict boundary validation accepts the assembled finding.
    VisionFinding.validate_boundary(finding.model_dump(mode="json"))


def _build(data: bytes, cfg: VisionConfig, *, finding_id: str) -> VisionFinding:
    """Run the shared analysis + assembly path on image bytes."""
    analysis = analyze_image_bytes(data, cfg)
    return build_finding(
        analysis,
        inspection_id="insp-adverse",
        finding_id=finding_id,
        annotated_output_ref=f"ref://annotated/{finding_id}.png",
        source="image",
        cfg=cfg,
    )


# ===========================================================================
# Req 22.1 - valid / invalid / empty / corrupt inputs
# ===========================================================================


def test_valid_image_analyzes_and_builds_finding(cfg: VisionConfig) -> None:
    """A normal shelf with a clear gap passes validation and yields a finding."""
    data = _encode_png(_base_shelf())

    # Validation accepts it (returns None, Req 1.7).
    assert validate_input(data, "image/png", cfg) is None

    finding = _build(data, cfg, finding_id="valid")
    assert isinstance(finding, VisionFinding)
    assert len(finding.candidate_regions) >= 1
    _assert_finding_is_honest(finding)


def test_empty_file_rejected(cfg: VisionConfig) -> None:
    """Zero bytes -> empty_file condition (Req 1.3)."""
    with pytest.raises(InputValidationError) as exc:
        validate_input(b"", "image/png", cfg)
    assert exc.value.condition == "empty_file"


def test_corrupt_bytes_rejected_as_undecodable(cfg: VisionConfig) -> None:
    """Garbage declared as an image -> undecodable condition (Req 1.4)."""
    garbage = b"\x89PNG\r\n\x1a\n" + b"not a real image" * 8
    with pytest.raises(InputValidationError) as exc:
        validate_input(garbage, "image/png", cfg)
    assert exc.value.condition == "undecodable"


def test_unsupported_content_type_rejected(cfg: VisionConfig) -> None:
    """A valid PNG declared with an unaccepted content type -> unsupported (Req 1.5)."""
    data = _encode_png(_base_shelf())
    with pytest.raises(InputValidationError) as exc:
        validate_input(data, "application/pdf", cfg)
    assert exc.value.condition == "unsupported_content_type"


def test_oversize_file_rejected(cfg: VisionConfig) -> None:
    """A file above the configured ceiling -> size_exceeded condition (Req 1.6)."""
    small_cfg = VisionConfig(max_file_size_bytes=16)
    data = _encode_png(_base_shelf())
    assert len(data) > small_cfg.max_file_size_bytes
    with pytest.raises(InputValidationError) as exc:
        validate_input(data, "image/png", small_cfg)
    assert exc.value.condition == "size_exceeded"


def test_preprocess_raises_on_empty_bytes(cfg: VisionConfig) -> None:
    """preprocess_image fails loudly on an empty buffer (not a silent None)."""
    with pytest.raises(PreprocessError):
        preprocess_image(b"", cfg)


def test_preprocess_raises_on_garbage_bytes(cfg: VisionConfig) -> None:
    """preprocess_image fails loudly on undecodable garbage bytes."""
    with pytest.raises(PreprocessError):
        preprocess_image(b"this is not an image at all" * 4, cfg)


# ===========================================================================
# Req 22.2 - varied lighting
# ===========================================================================


def test_dark_lighting_pipeline_runs_with_caveats(cfg: VisionConfig) -> None:
    """A low-brightness version of the shelf still produces caveated findings."""
    data = _encode_png(_base_shelf(brightness=70))
    assert validate_input(data, "image/png", cfg) is None

    finding = _build(data, cfg, finding_id="dark")
    assert len(finding.candidate_regions) >= 1
    _assert_finding_is_honest(finding)


def test_bright_lighting_pipeline_runs_with_caveats(cfg: VisionConfig) -> None:
    """A high-brightness version of the shelf still produces caveated findings."""
    data = _encode_png(_base_shelf(brightness=250))
    assert validate_input(data, "image/png", cfg) is None

    finding = _build(data, cfg, finding_id="bright")
    assert len(finding.candidate_regions) >= 1
    _assert_finding_is_honest(finding)


# ===========================================================================
# Req 22.2 - shadow confounder
# ===========================================================================


def test_shadow_pipeline_runs_with_caveats(cfg: VisionConfig) -> None:
    """A shadow band across a stocked shelf still yields honest estimates.

    The point is honesty under a confounder, not a specific detection outcome:
    whatever regions the analyzer reports must be labeled estimates and carry a
    caveat naming shadow / occlusion / dividers / image quality.
    """
    data = _encode_png(_shadow_shelf())
    assert validate_input(data, "image/png", cfg) is None

    finding = _build(data, cfg, finding_id="shadow")
    # The shadow band reads as dark area the analyzer surfaces as regions;
    # assert at least one so the honesty/caveat check is exercised.
    assert len(finding.candidate_regions) >= 1
    _assert_finding_is_honest(finding)


# ===========================================================================
# Req 22.2 - shelf dividers confounder
# ===========================================================================


def test_dividers_pipeline_runs_with_caveats(cfg: VisionConfig) -> None:
    """Vertical dividers on a stocked shelf still yield honest, caveated estimates."""
    data = _encode_png(_divider_shelf())
    assert validate_input(data, "image/png", cfg) is None

    finding = _build(data, cfg, finding_id="dividers")
    # The dividers carve dark slivers the analyzer surfaces as regions; assert
    # at least one so the caveat check is actually exercised (not vacuous).
    assert len(finding.candidate_regions) >= 1
    _assert_finding_is_honest(finding)


# ===========================================================================
# Req 22.2 - occlusion confounder
# ===========================================================================


def test_occlusion_pipeline_runs_with_caveats(cfg: VisionConfig) -> None:
    """A dark blob occluding product still yields honest, caveated estimates."""
    data = _encode_png(_occlusion_shelf())
    assert validate_input(data, "image/png", cfg) is None

    finding = _build(data, cfg, finding_id="occlusion")
    # The occluding blob reads as dark area the analyzer surfaces as a region;
    # assert at least one so the honesty/caveat check is exercised.
    assert len(finding.candidate_regions) >= 1
    _assert_finding_is_honest(finding)


# ===========================================================================
# Req 22.2 - multiple regions
# ===========================================================================


def test_multiple_regions_detected(cfg: VisionConfig) -> None:
    """Two clearly separated dark gaps yield at least two candidate regions.

    The image is designed so the two gaps are unambiguously separated by a
    bright shelf-lip band, so their empty-mask contours cannot merge into one
    blob. We assert >= 2 regions.
    """
    data = _encode_png(_two_gap_shelf())
    assert validate_input(data, "image/png", cfg) is None

    analysis = analyze_image_bytes(data, cfg)
    assert len(analysis.candidates) >= 2, (
        f"expected >= 2 separated candidate regions, got {len(analysis.candidates)}"
    )

    finding = _build(data, cfg, finding_id="multi")
    assert len(finding.candidate_regions) >= 2
    _assert_finding_is_honest(finding)
