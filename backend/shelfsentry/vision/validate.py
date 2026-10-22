"""Input validation for the vision pipeline (Req 1).

:func:`validate_input` is the single gate every inspection input passes through
before any analysis. It enforces, in order, the four rejectable conditions of
Requirement 1 and accepts anything that passes all of them (Req 1.7):

1. **Empty file** (Req 1.3) -> ``empty_file``.
2. **Unsupported content type** (Req 1.5) -> ``unsupported_content_type``.
3. **Size over the configured maximum** (Req 1.6) -> ``size_exceeded``.
4. **Undecodable as the expected media kind** (Req 1.1, 1.2, 1.4) ->
   ``undecodable``.

On any failure it raises :class:`~shelfsentry.vision.errors.InputValidationError`
naming the specific condition; on success it returns ``None`` (Req 1.7).

The media kind (image vs video) is inferred from ``content_type`` so the right
decoder is used: ``image/*`` is probed with :func:`cv2.imdecode` over an
in-memory NumPy buffer, and ``video/*`` is probed with :class:`cv2.VideoCapture`.
Because ``VideoCapture`` needs a filesystem path, the video bytes are written to
a :class:`tempfile.NamedTemporaryFile`, probed, and then removed. The content
type is validated against the accepted set *before* decoding, so an unsupported
type is rejected without ever touching a decoder.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np

from shelfsentry.config.settings import VisionConfig
from shelfsentry.vision.errors import InputValidationError

__all__ = ["validate_input", "media_kind_for_content_type"]


def media_kind_for_content_type(content_type: str) -> str | None:
    """Infer the media kind from a content type.

    Returns ``"image"`` for an ``image/*`` type, ``"video"`` for a ``video/*``
    type, and ``None`` for anything else. The comparison is case-insensitive
    and ignores any parameters after a ``;`` (for example ``image/png;
    charset=binary``).
    """
    primary = content_type.split(";", 1)[0].strip().lower()
    if primary.startswith("image/"):
        return "image"
    if primary.startswith("video/"):
        return "video"
    return None


def _decodable_as_image(data: bytes) -> bool:
    """Return True if ``data`` decodes as an image via ``cv2.imdecode``."""
    buffer = np.frombuffer(data, dtype=np.uint8)
    if buffer.size == 0:
        return False
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image is not None and image.size > 0


def _decodable_as_video(data: bytes) -> bool:
    """Return True if ``data`` opens and yields a frame via ``cv2.VideoCapture``.

    ``VideoCapture`` requires a filesystem path, so the bytes are written to a
    :class:`tempfile.NamedTemporaryFile` (created with ``delete=False`` so it
    can be reopened by OpenCV on Windows), probed, and then removed in a
    ``finally`` block regardless of the outcome.
    """
    tmp_path: str | None = None
    capture: cv2.VideoCapture | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="shelfsentry_probe_", suffix=".bin", delete=False
        ) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        capture = cv2.VideoCapture(tmp_path)
        if not capture.isOpened():
            return False
        # Opening can succeed on garbage; require at least one readable frame.
        ok, frame = capture.read()
        return bool(ok) and frame is not None and getattr(frame, "size", 0) > 0
    finally:
        if capture is not None:
            capture.release()
        if tmp_path is not None:
            try:
                os.remove(tmp_path)
            except OSError:
                # Best-effort cleanup; a leftover temp file must never mask the
                # validation result or raise over a successful validation.
                pass


def validate_input(data: bytes, content_type: str, cfg: VisionConfig) -> None:
    """Validate an inspection input, raising on the first failing condition.

    Checks are ordered cheapest-first so a clearly bad input is rejected
    without doing decode work: empty-file, content-type, and size checks run
    before the (more expensive) decodability probe.

    Args:
        data: The raw uploaded file bytes.
        content_type: The declared content type (for example ``"image/png"``).
        cfg: Resolved :class:`VisionConfig` providing
            ``accepted_content_types`` and ``max_file_size_bytes``.

    Returns:
        ``None`` when every check passes and the input is accepted for
        analysis (Req 1.7).

    Raises:
        InputValidationError: naming the specific failing condition:
            ``empty_file`` (Req 1.3), ``unsupported_content_type`` (Req 1.5),
            ``size_exceeded`` (Req 1.6), or ``undecodable`` (Req 1.1, 1.2,
            1.4).
    """
    # 1. Empty file (Req 1.3).
    if len(data) == 0:
        raise InputValidationError(
            "empty_file",
            "The submitted file is empty (0 bytes) and cannot be inspected.",
            details={"size_bytes": 0},
        )

    # 2. Content type must be in the configured accepted set (Req 1.5). Checked
    #    before decoding so an unsupported type never reaches a decoder.
    normalized = content_type.split(";", 1)[0].strip().lower()
    accepted = {ct.strip().lower() for ct in cfg.accepted_content_types}
    if normalized not in accepted:
        raise InputValidationError(
            "unsupported_content_type",
            (
                f"Content type {content_type!r} is not accepted. Accepted types: "
                f"{', '.join(cfg.accepted_content_types)}."
            ),
            details={
                "content_type": content_type,
                "accepted_content_types": list(cfg.accepted_content_types),
            },
        )

    # 3. Size must not exceed the configured maximum (Req 1.6).
    size_bytes = len(data)
    if size_bytes > cfg.max_file_size_bytes:
        raise InputValidationError(
            "size_exceeded",
            (
                f"File size {size_bytes} bytes exceeds the configured maximum of "
                f"{cfg.max_file_size_bytes} bytes."
            ),
            details={
                "size_bytes": size_bytes,
                "max_file_size_bytes": cfg.max_file_size_bytes,
            },
        )

    # 4. Decodability as the expected media kind (Req 1.1, 1.2, 1.4). The kind
    #    is inferred from the (already accepted) content type so the right
    #    decoder is used.
    kind = media_kind_for_content_type(normalized)
    if kind == "image":
        decodable = _decodable_as_image(data)
    elif kind == "video":
        decodable = _decodable_as_video(data)
    else:  # pragma: no cover - unreachable: accepted types are image/* or video/*
        # An accepted content type that is neither image/* nor video/* is a
        # configuration contradiction; treat it as undecodable rather than
        # guessing a decoder.
        decodable = False

    if not decodable:
        raise InputValidationError(
            "undecodable",
            (
                f"The submitted file could not be decoded as a valid "
                f"{kind or 'image or video'} (corrupt or invalid)."
            ),
            details={"content_type": content_type, "media_kind": kind},
        )

    # 5. All checks passed: accept for analysis (Req 1.7).
    return None
