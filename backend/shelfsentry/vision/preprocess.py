"""Deterministic image preprocessing for the OpenCV 5 vision pipeline (Req 2.2).

This module turns raw image bytes into the normalized intermediate arrays the
downstream analysis stage (``analyze.py``) reuses. Every operation is a fixed,
config-driven classical OpenCV 5 primitive with no randomness, so the same
bytes and the same :class:`~shelfsentry.config.settings.VisionConfig` always
produce byte-for-byte identical arrays. That determinism underpins the
reproducible-coordinate guarantee (Req 5.1).

Pipeline (all parameters come from ``VisionConfig``):

1. ``cv2.imdecode`` - decode the bytes to a 3-channel BGR image.
2. ``cv2.cvtColor`` - BGR -> grayscale and BGR -> HSV.
3. ``cv2.GaussianBlur`` - denoise the grayscale image with an odd kernel
   (``gaussian_blur_ksize``).
4. Lighting normalization - CLAHE (``cv2.createCLAHE`` + ``apply``) followed by
   global histogram equalization (``cv2.equalizeHist``) on the blurred
   grayscale image, producing a contrast-normalized image robust to uneven
   lighting and shadow before measurement.

The decoded BGR, grayscale, HSV, blurred, and equalized arrays are returned
together in a small typed :class:`PreprocessResult` so ``analyze.py`` can pick
whichever representation each stage needs without recomputing it.
"""

from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

from shelfsentry.config.settings import VisionConfig


class PreprocessError(ValueError):
    """Raised when raw bytes cannot be decoded into a usable BGR image.

    Validation of content type and size lives in ``validate.py``; this error
    covers the narrower decode step so a caller that preprocesses already
    validated bytes still fails loudly (naming the decode condition) rather
    than propagating a ``None`` from ``cv2.imdecode`` into analysis.
    """


class PreprocessResult(NamedTuple):
    """Deterministic preprocessing intermediates shared with ``analyze.py``.

    All arrays describe the same image at the same (height, width); the
    multi-channel arrays add a trailing channel axis.

    Attributes:
        bgr: Decoded 3-channel ``uint8`` image, shape ``(H, W, 3)``, in
            OpenCV's native BGR channel order.
        gray: Single-channel ``uint8`` grayscale image, shape ``(H, W)``.
        hsv: 3-channel ``uint8`` HSV image, shape ``(H, W, 3)``.
        blurred: Gaussian-blurred grayscale image, shape ``(H, W)``,
            ``uint8``.
        equalized: Lighting-normalized grayscale image (CLAHE then global
            histogram equalization applied to ``blurred``), shape ``(H, W)``,
            ``uint8``.
    """

    bgr: np.ndarray
    gray: np.ndarray
    hsv: np.ndarray
    blurred: np.ndarray
    equalized: np.ndarray


def decode_bgr(data: bytes) -> np.ndarray:
    """Decode raw image bytes into a 3-channel BGR ``uint8`` image.

    Uses ``cv2.imdecode`` with ``IMREAD_COLOR``, which always yields a
    3-channel BGR image regardless of the source's channel count.

    Args:
        data: Raw encoded image bytes (for example PNG or JPEG).

    Returns:
        The decoded image as a ``(H, W, 3)`` ``uint8`` BGR array.

    Raises:
        PreprocessError: if ``data`` is empty or cannot be decoded as an
            image (``cv2.imdecode`` returns ``None``).
    """
    if not data:
        raise PreprocessError("cannot decode empty byte buffer as an image")
    # np.frombuffer gives a read-only view over the bytes; imdecode copies out
    # of it, so the decoded array is safe to use and mutate downstream.
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise PreprocessError(
            "cv2.imdecode could not decode the provided bytes as an image"
        )
    return image


def preprocess_image(data: bytes, cfg: VisionConfig) -> PreprocessResult:
    """Decode and deterministically preprocess image bytes (Req 2.2).

    Runs the fixed, config-driven preprocessing pipeline and returns every
    intermediate array for reuse by downstream analysis. No step uses
    randomness, so repeated calls with the same ``data`` and ``cfg`` produce
    identical arrays (``np.array_equal`` holds across runs), supporting the
    reproducible-coordinate guarantee (Req 5.1).

    Args:
        data: Raw encoded image bytes (already validated upstream).
        cfg: Vision configuration supplying the fixed kernel size
            (``gaussian_blur_ksize``) and CLAHE parameters
            (``clahe_clip_limit``, ``clahe_tile_grid_size``).

    Returns:
        A :class:`PreprocessResult` with the decoded BGR image and the
        grayscale, HSV, blurred, and lighting-normalized (equalized) arrays.

    Raises:
        PreprocessError: if the bytes cannot be decoded into an image.
    """
    bgr = decode_bgr(data)

    # Color-space conversions for downstream shelf-region / color analysis.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Denoise with an odd, config-driven kernel. VisionConfig already enforces
    # that gaussian_blur_ksize is an odd positive integer.
    ksize = cfg.gaussian_blur_ksize
    # sigmaX=0 asks OpenCV to derive sigma from the kernel size deterministically.
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=0)

    # Lighting normalization: CLAHE first (local, limits shadow/highlight
    # blowout) then a global equalizeHist pass, both config-driven and fixed.
    tile = cfg.clahe_tile_grid_size
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(tile, tile),
    )
    clahe_applied = clahe.apply(blurred)
    equalized = cv2.equalizeHist(clahe_applied)

    return PreprocessResult(
        bgr=bgr,
        gray=gray,
        hsv=hsv,
        blurred=blurred,
        equalized=equalized,
    )
