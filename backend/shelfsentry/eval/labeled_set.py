"""A small, reproducible, labeled evaluation set (task 14.1; Req 25.1).

This module is the **labeled evaluation set itself**, stored *as code* so it is
fully reproducible and needs no external assets. Every sample is a synthetic
shelf image drawn programmatically with only NumPy + OpenCV 5 (the same
own-work / public-domain approach used by :mod:`shelfsentry.samples`), paired
with a small, human-authored GROUND-TRUTH label.

Labeling methodology (documented, Req 25.1)
-------------------------------------------
Because the images are *drawn by us*, we know their content exactly, so the
labels are not guesses - they are a direct statement of what we drew:

* ``expected_has_gap`` (bool) - the PRIMARY detection label. ``True`` when we
  deliberately drew one or more clearly empty gaps into an otherwise stocked
  shelf, ``False`` when we drew either a fully stocked shelf or a scene with no
  stockable shelf space at all. This is the ground truth the runner scores the
  vision pipeline's empty-region detection against (false positives / false
  negatives; Req 25.2).

* ``expected_gap_count`` (int) - how many empty gaps we drew (``0`` for the
  no-gap samples). Recorded for documentation and for the ``actual`` vs
  ``expected`` text on collected examples. It is intentionally NOT used as a
  strict equality target for scoring, because the classical CV contour pass may
  legitimately split or merge adjacent gaps; forcing exact-count equality would
  measure drawing geometry, not detection quality. The detection metric is the
  coarser, well-defined ``has_gap`` boolean above.

* ``expected_action`` (one of the five :data:`AgentAction` values) - the agent
  terminal-action label used for the decision-correctness metric. See
  :mod:`shelfsentry.eval.runner` for exactly what that metric measures and its
  documented limitations. We label the *terminal* action the deterministic
  active-perception loop is expected to settle on for each scene. Samples where
  the expected terminal action is genuinely scene-dependent are labeled
  ``None`` and are simply excluded from the decision-correctness denominator
  (never guessed).

These are SYNTHETIC labels, not real retail ground truth. That limitation is
reported with every result (see :data:`SYNTHETIC_LABEL_LIMITATIONS` and the
runner's ``documented_limitations``), per Req 25.4 / 27.2.

Determinism (Req 25.1). Every image is drawn with fixed geometry, fixed colors,
and no randomness or wall-clock input, so :func:`labeled_set` returns the same
samples - with byte-identical image bytes - on every call and every machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from shelfsentry.data.contracts import AgentAction

__all__ = [
    "EvalSample",
    "SYNTHETIC_LABEL_LIMITATIONS",
    "labeled_set",
    "render_sample_png",
]


# ---------------------------------------------------------------------------
# The labeled-sample record (label + reproducible generator parameters)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSample:
    """One labeled evaluation sample: a synthetic scene plus its ground truth.

    The image is not stored as bytes on the record; it is *drawn on demand* by
    :func:`render_sample_png` from the fixed ``gaps`` geometry and the scene
    ``kind``, so the whole set stays reproducible as code with no bundled binary
    assets. Equal samples therefore produce byte-identical PNG bytes.

    Attributes:
        sample_id: Stable identifier (used as the evaluation example id and the
            synthetic inspection id).
        kind: Scene archetype that controls how the image is drawn:
            ``"stocked_with_gaps"`` (products plus deliberate empty gaps),
            ``"fully_stocked"`` (products filling every slot; no gap),
            ``"empty_scene"`` (a flat background with no shelf/product at all).
        content_type: Declared content type for the pipeline (always an image).
        expected_has_gap: Ground-truth detection label (see module docstring).
        expected_gap_count: Number of gaps deliberately drawn (documentation).
        expected_action: Expected agent terminal action, or ``None`` to exclude
            this sample from the decision-correctness metric.
        gaps: ``(x, y, w, h)`` boxes left empty when drawing a stocked shelf.
        description: Short human-readable note shown alongside examples.
    """

    sample_id: str
    kind: str
    content_type: str
    expected_has_gap: bool
    expected_gap_count: int
    expected_action: AgentAction | None
    gaps: tuple[tuple[int, int, int, int], ...] = field(default=())
    description: str = ""


# ---------------------------------------------------------------------------
# Documented limitations of the synthetic labels (Req 25.3, 25.4, 27.2)
# ---------------------------------------------------------------------------

#: Limitations that MUST travel with every evaluation result produced against
#: this set. They state plainly that the labels are synthetic own-work, that the
#: metrics are heuristic, and that latency is environment-dependent, so a reader
#: never mistakes these numbers for validated real-retail performance.
SYNTHETIC_LABEL_LIMITATIONS: tuple[str, ...] = (
    "The evaluation images are SYNTHETIC: they are drawn programmatically with "
    "NumPy and OpenCV, not photographed from real shelves. Their labels are the "
    "project authors' own ground truth for drawn content and are NOT real "
    "retail ground truth; measured performance here does not predict real-world "
    "accuracy.",
    "The detection metric scores a coarse per-image 'has_gap' boolean (does the "
    "vision pipeline flag any empty region at or above the configured "
    "empty-space-ratio threshold) against whether a gap was deliberately drawn. "
    "It is a heuristic sanity check, not a bounding-box-level accuracy measure, "
    "and it inherits the classical-CV heuristic's limitations.",
    "Agent decision correctness compares the active-perception loop's TERMINAL "
    "action against a hand-labeled expected terminal action, only for the "
    "subset of samples where a single expected terminal action is well defined. "
    "It measures whether the deterministic fallback policy routes each scene to "
    "the intended action; it does NOT measure the correctness of a learned/LLM "
    "policy and does not apply to samples left unlabeled.",
    "The set is deliberately small and reproducible (a handful of scenes) to "
    "stay fast and asset-free; it is not a statistically powered benchmark.",
    "Latency percentiles are wall-clock measurements taken on the machine that "
    "ran the evaluation (time.perf_counter around the real pipeline) and are "
    "therefore environment-dependent; absolute values will vary by hardware, "
    "load, and OpenCV build.",
)


# ---------------------------------------------------------------------------
# Fixed drawing geometry (no randomness, no wall clock -> deterministic)
# ---------------------------------------------------------------------------

_WIDTH = 640
_HEIGHT = 480
_SHELF_ROWS = 3
_PRODUCT_W = 70
_PRODUCT_H = 70
_GAP = 20
_BG_GRAY = 200


def _overlaps_any(
    box: tuple[int, int, int, int], gaps: tuple[tuple[int, int, int, int], ...]
) -> bool:
    """True if ``box`` overlaps any gap rectangle (both as ``(x, y, w, h)``)."""
    bx, by, bw, bh = box
    for gx, gy, gw, gh in gaps:
        if bx < gx + gw and bx + bw > gx and by < gy + gh and by + bh > gy:
            return True
    return False


def _draw_stocked_shelf(gaps: tuple[tuple[int, int, int, int], ...]) -> np.ndarray:
    """Draw a three-row shelf of product boxes, leaving ``gaps`` empty.

    Identical drawing rules to :func:`shelfsentry.samples.generate_sample_shelf_png`
    so the evaluation scenes are visually consistent with the bundled demo
    sample. With ``gaps`` empty, the pipeline has clearly empty regions to find;
    with ``gaps`` empty tuple, every slot is filled (a fully stocked shelf).
    """
    img = np.full((_HEIGHT, _WIDTH, 3), _BG_GRAY, dtype=np.uint8)
    row_height = _HEIGHT // _SHELF_ROWS

    for row in range(_SHELF_ROWS):
        shelf_y = row * row_height + row_height - 10
        # Shelf board.
        cv2.rectangle(img, (0, shelf_y), (_WIDTH, shelf_y + 8), (120, 120, 120), -1)
        box_top = shelf_y - _PRODUCT_H - 5
        x = _GAP
        while x + _PRODUCT_W <= _WIDTH:
            box = (x, box_top, _PRODUCT_W, _PRODUCT_H)
            if not _overlaps_any(box, gaps):
                shade = 60 + (x // (_PRODUCT_W + _GAP)) * 15
                shade = min(shade, 180)
                cv2.rectangle(
                    img,
                    (x, box_top),
                    (x + _PRODUCT_W, box_top + _PRODUCT_H),
                    (shade, shade, shade),
                    -1,
                )
                cv2.rectangle(
                    img,
                    (x, box_top),
                    (x + _PRODUCT_W, box_top + _PRODUCT_H),
                    (40, 40, 40),
                    2,
                )
            x += _PRODUCT_W + _GAP
    return img


def _draw_empty_scene() -> np.ndarray:
    """Draw a flat, near-uniform background with no shelf or product.

    Used as a negative sample: there is no stockable shelf space, so the correct
    detection label is ``has_gap = False`` (there is nothing that should be
    flagged as an empty *shelf* gap).
    """
    return np.full((_HEIGHT, _WIDTH, 3), _BG_GRAY, dtype=np.uint8)


def render_sample_png(sample: EvalSample) -> bytes:
    """Render a labeled sample to deterministic PNG bytes.

    Draws the scene for ``sample.kind`` and encodes it as PNG. The function is
    pure and deterministic: the same sample always yields byte-identical bytes.

    Raises:
        ValueError: if ``sample.kind`` is not a known archetype.
        RuntimeError: if PNG encoding fails (does not happen for a valid array).
    """
    if sample.kind in ("stocked_with_gaps", "fully_stocked"):
        img = _draw_stocked_shelf(sample.gaps)
    elif sample.kind == "empty_scene":
        img = _draw_empty_scene()
    else:  # pragma: no cover - guarded by the fixed labeled set below
        raise ValueError(f"unknown eval sample kind: {sample.kind!r}")

    ok, buf = cv2.imencode(".png", img)
    if not ok:  # pragma: no cover - encoding a valid array does not fail
        raise RuntimeError(f"failed to encode eval sample {sample.sample_id!r}")
    return buf.tobytes()


# ---------------------------------------------------------------------------
# The labeled set (the ground truth, authored here as code)
# ---------------------------------------------------------------------------

# Expected terminal-action labels, explained (grounded in the DESIGNED routing
# of the deterministic FallbackPolicy, not in aspiration).
#
# The local evaluation runs the DETERMINISTIC active-perception loop. The
# fallback policy routes a scene by what the vision pass measures, so the honest
# expected terminal action is the action the DESIGNED policy is meant to reach
# for each archetype:
#
#   * A clearly stocked shelf with large/multiple empty gaps shows BOTH clearly
#     empty regions AND clearly filled regions in one frame. That is the
#     policy's definition of CONFLICTING evidence, which it is designed to
#     escalate with ``request_human_review`` - the right call for a shelf that
#     is partly empty and partly stocked. So clear-gap scenes are labeled
#     ``request_human_review``.
#   * A fully stocked shelf (no gap) and the flat empty scene have nothing
#     flagged as empty, so the policy finalizes directly -> ``finalize_report``.
#   * Borderline gap scenes, where the heuristic may or may not flag the drawn
#     gap at the configured threshold, have a terminal action that depends on
#     the heuristic's sensitivity rather than on a single intended route. Their
#     expected action is genuinely not well defined, so they are labeled
#     ``None`` and EXCLUDED from the decision-correctness metric rather than
#     given a guessed expectation.
#
# These labels were set to match the policy's documented routing design; the
# runner then MEASURES whether the loop actually routes there. They are not
# tuned to make the number look good - a borderline scene is excluded, not
# relabeled, precisely to keep the metric honest.

_SAMPLES: tuple[EvalSample, ...] = (
    # --- stocked shelves WITH deliberate empty gaps (expected_has_gap=True) ---
    EvalSample(
        sample_id="eval-gap-single-left",
        kind="stocked_with_gaps",
        content_type="image/png",
        expected_has_gap=True,
        expected_gap_count=1,
        expected_action="request_human_review",
        gaps=((40, 70, 120, 90),),
        description="Stocked shelf with one empty gap on the top-left row.",
    ),
    EvalSample(
        sample_id="eval-gap-single-right",
        kind="stocked_with_gaps",
        content_type="image/png",
        expected_has_gap=True,
        expected_gap_count=1,
        expected_action=None,
        gaps=((430, 250, 150, 90),),
        description=(
            "Stocked shelf with one empty gap on the middle-right row; a "
            "borderline case for the empty-ratio threshold, so its terminal "
            "action is excluded from the decision-correctness metric."
        ),
    ),
    EvalSample(
        sample_id="eval-gap-two",
        kind="stocked_with_gaps",
        content_type="image/png",
        expected_has_gap=True,
        expected_gap_count=2,
        expected_action="request_human_review",
        gaps=((40, 70, 120, 90), (430, 250, 150, 90)),
        description="Stocked shelf with two empty gaps (top-left and mid-right).",
    ),
    EvalSample(
        sample_id="eval-gap-bottom",
        kind="stocked_with_gaps",
        content_type="image/png",
        expected_has_gap=True,
        expected_gap_count=1,
        expected_action=None,
        gaps=((200, 380, 180, 90),),
        description=(
            "Stocked shelf with one wide empty gap on the bottom row; a "
            "borderline case for the empty-ratio threshold, so its terminal "
            "action is excluded from the decision-correctness metric."
        ),
    ),
    EvalSample(
        sample_id="eval-gap-three",
        kind="stocked_with_gaps",
        content_type="image/png",
        expected_has_gap=True,
        expected_gap_count=3,
        expected_action="request_human_review",
        gaps=((40, 70, 120, 90), (430, 250, 150, 90), (200, 380, 180, 90)),
        description="Stocked shelf with three empty gaps across all rows.",
    ),
    # --- fully stocked shelves, NO gap (expected_has_gap=False) ---
    EvalSample(
        sample_id="eval-full-1",
        kind="fully_stocked",
        content_type="image/png",
        expected_has_gap=False,
        expected_gap_count=0,
        expected_action="finalize_report",
        gaps=(),
        description="Fully stocked shelf; every slot filled, no empty gap.",
    ),
    EvalSample(
        sample_id="eval-full-2",
        kind="fully_stocked",
        content_type="image/png",
        expected_has_gap=False,
        expected_gap_count=0,
        expected_action="finalize_report",
        gaps=(),
        description="Fully stocked shelf (duplicate scene) as a second negative.",
    ),
    # --- flat empty scene, NO stockable shelf space (expected_has_gap=False) ---
    EvalSample(
        sample_id="eval-empty-scene",
        kind="empty_scene",
        content_type="image/png",
        expected_has_gap=False,
        expected_gap_count=0,
        expected_action="finalize_report",
        gaps=(),
        description="Flat background with no shelf or product; nothing to flag.",
    ),
)


def labeled_set() -> tuple[EvalSample, ...]:
    """Return the deterministic, non-empty labeled evaluation set (Req 25.1).

    The returned tuple is stable across calls and machines, and each sample's
    image bytes (via :func:`render_sample_png`) are reproducible. Mixing scenes
    with and without gaps gives the detection metric both positive and negative
    cases so false positives and false negatives are both observable.
    """
    return _SAMPLES
