"""Bundled, licensed sample inputs for ShelfSentry AI example mode (Req 15.4).

The sample shelf image shipped in this package is **synthetic**: it is drawn
programmatically by :func:`generate_sample_shelf_png` using only NumPy and
OpenCV 5, with no external asset downloaded or copied. Because the project
authors it from scratch, its provenance is own-work / public-domain and there
is no third-party license to honor (see ``samples/PROVENANCE.md``).

:data:`SAMPLE_SHELF_PNG` is the path to the generated ``sample_shelf.png`` that
the local example mode can feed through the pipeline. :func:`ensure_samples`
(re)generates the file deterministically if it is missing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = [
    "SAMPLES_DIR",
    "SAMPLE_SHELF_PNG",
    "generate_sample_shelf_png",
    "ensure_samples",
]

#: Directory containing the bundled samples and their provenance note.
SAMPLES_DIR: Path = Path(__file__).resolve().parent

#: Path to the generated synthetic shelf sample image.
SAMPLE_SHELF_PNG: Path = SAMPLES_DIR / "sample_shelf.png"

# Fixed geometry so the drawing is fully deterministic (no randomness).
_WIDTH = 640
_HEIGHT = 480
_SHELF_ROWS = 3
# (x, y, w, h) boxes, in top-left-origin pixel coordinates, that are left empty
# so the vision pipeline has clearly empty gaps to find.
_EMPTY_GAPS = ((40, 70, 120, 90), (430, 250, 150, 90))


def generate_sample_shelf_png() -> bytes:
    """Draw a synthetic shelf image and return deterministic PNG bytes.

    The image is a plain gray background with three horizontal shelf rows of
    evenly spaced "product" boxes, minus a couple of intentional empty gaps.
    Everything is drawn with fixed coordinates and colors (no randomness, no
    wall clock), so the function returns byte-identical PNG bytes on every call.
    The result is owned by this project (own-work / public domain).
    """
    img = np.full((_HEIGHT, _WIDTH, 3), 200, dtype=np.uint8)

    row_height = _HEIGHT // _SHELF_ROWS
    product_w, product_h = 70, 70
    gap = 20

    for row in range(_SHELF_ROWS):
        shelf_y = row * row_height + row_height - 10
        # Draw the shelf board.
        cv2.rectangle(img, (0, shelf_y), (_WIDTH, shelf_y + 8), (120, 120, 120), -1)
        # Place product boxes along the row above the board.
        box_top = shelf_y - product_h - 5
        x = gap
        while x + product_w <= _WIDTH:
            box = (x, box_top, product_w, product_h)
            if not _overlaps_any(box, _EMPTY_GAPS):
                # A distinct-but-fixed shade per column keeps edges detectable.
                shade = 60 + (x // (product_w + gap)) * 15
                shade = min(shade, 180)
                cv2.rectangle(
                    img,
                    (x, box_top),
                    (x + product_w, box_top + product_h),
                    (shade, shade, shade),
                    -1,
                )
                cv2.rectangle(
                    img,
                    (x, box_top),
                    (x + product_w, box_top + product_h),
                    (40, 40, 40),
                    2,
                )
            x += product_w + gap

    ok, buf = cv2.imencode(".png", img)
    if not ok:  # pragma: no cover - PNG encoding of a valid array does not fail
        raise RuntimeError("failed to encode synthetic sample PNG")
    return buf.tobytes()


def _overlaps_any(
    box: tuple[int, int, int, int], gaps: tuple[tuple[int, int, int, int], ...]
) -> bool:
    """True if ``box`` overlaps any gap rectangle (both as (x, y, w, h))."""
    bx, by, bw, bh = box
    for gx, gy, gw, gh in gaps:
        if bx < gx + gw and bx + bw > gx and by < gy + gh and by + bh > gy:
            return True
    return False


def ensure_samples() -> Path:
    """Ensure :data:`SAMPLE_SHELF_PNG` exists on disk, generating it if missing.

    Returns the path to the sample image. Regeneration is deterministic, so a
    regenerated file matches the bundled one byte-for-byte.
    """
    if not SAMPLE_SHELF_PNG.exists():
        SAMPLE_SHELF_PNG.write_bytes(generate_sample_shelf_png())
    return SAMPLE_SHELF_PNG
