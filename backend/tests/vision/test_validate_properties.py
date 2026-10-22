"""Property-based tests for input validation (Req 1.3-1.7).

Covers ``shelfsentry.vision.validate.validate_input`` against a
:class:`~shelfsentry.config.settings.VisionConfig`. The validator applies four
checks in a fixed cheapest-first order and raises
:class:`~shelfsentry.vision.errors.InputValidationError` naming the first
failing condition, or returns ``None`` on accept:

1. empty file      -> ``empty_file``                 (Req 1.3)
2. content type     -> ``unsupported_content_type``   (Req 1.5)
3. size             -> ``size_exceeded``              (Req 1.6)
4. decodability     -> ``undecodable``                (Req 1.1, 1.2, 1.4)

Two properties are implemented:

* **Property 1: Bad inputs are always rejected with the matching condition.**
  Each bad-input generator targets exactly one condition and, because the
  check order is known, every other check is kept passing so the targeted
  condition is the one that fires.
* **Property 2: Valid inputs are accepted.** A genuinely decodable image, with
  an accepted content type and within the configured size, returns ``None``.

Image inputs carry the generated bulk (cheap, in-memory ``cv2.imdecode``);
a couple of explicit video cases are included separately because probing a
video goes through a temp file and ``cv2.VideoCapture`` and is comparatively
slow.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from shelfsentry.config.settings import VisionConfig
from shelfsentry.vision.errors import InputValidationError
from shelfsentry.vision.validate import validate_input

# Accepted image content types from the default config; used so the
# content-type check passes while another condition is the one under test.
_ACCEPTED_IMAGE_TYPES = ("image/jpeg", "image/png", "image/webp", "image/bmp")


def _encode_image(width: int, height: int, channels: int) -> bytes:
    """Encode a small random array to PNG bytes via ``cv2.imencode``.

    PNG is lossless and always encodes a non-empty array, so the result is a
    genuinely decodable image. ``channels`` is 1 (grayscale) or 3 (BGR).
    """
    if channels == 1:
        arr = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    else:
        arr = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", arr)
    assert ok, "cv2.imencode failed to encode a valid array"
    return buf.tobytes()


# --- Strategies -----------------------------------------------------------

# Small image dimensions keep encode/decode cheap while varying shape.
_dims = st.integers(min_value=1, max_value=48)
_channels = st.sampled_from([1, 3])
_accepted_image_type = st.sampled_from(_ACCEPTED_IMAGE_TYPES)

# Random non-image bytes: keep the length well below any valid-image size and
# avoid the empty case so the ``undecodable`` check is the one exercised. A
# short run of arbitrary bytes is not a decodable image; the length guard plus
# ``assume`` below makes an accidental valid-image header astronomically
# unlikely (and explicitly skipped if it ever occurs).
_junk_bytes = st.binary(min_size=1, max_size=64)

# A content type that is NOT in the default accepted set.
_unsupported_type = st.sampled_from(
    [
        "application/pdf",
        "text/plain",
        "application/octet-stream",
        "image/gif",  # image/* but not in the accepted set
        "image/tiff",
        "video/x-matroska",  # video/* but not accepted
        "",
    ]
)


# --- Property 1: Bad inputs are always rejected with the matching condition ---
# Validates: Requirements 1.3, 1.4, 1.5, 1.6


@given(content_type=_accepted_image_type)
@settings(max_examples=100)
def test_property1_empty_file_rejected(content_type: str) -> None:
    """Property 1: empty bytes are rejected with ``empty_file`` (Req 1.3).

    Validates: Requirements 1.3
    """
    cfg = VisionConfig()
    with pytest.raises(InputValidationError) as exc:
        validate_input(b"", content_type, cfg)
    assert exc.value.condition == "empty_file"


@given(data=_junk_bytes, content_type=_accepted_image_type)
@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
def test_property1_undecodable_rejected(data: bytes, content_type: str) -> None:
    """Property 1: non-image bytes with an accepted image type -> ``undecodable``.

    The bytes are non-empty, the content type is accepted, and the length is
    far under the configured max, so the empty/content-type/size checks all
    pass and the decodability check is the one that must fire (Req 1.4). Guard
    against the astronomically unlikely case that random bytes happen to be a
    valid image.

    Validates: Requirements 1.4
    """
    cfg = VisionConfig()
    # Defensive: skip the vanishingly rare case of accidentally-valid bytes.
    buffer = np.frombuffer(data, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    assume(decoded is None or getattr(decoded, "size", 0) == 0)

    with pytest.raises(InputValidationError) as exc:
        validate_input(data, content_type, cfg)
    assert exc.value.condition == "undecodable"


@given(data=st.binary(min_size=1, max_size=64), content_type=_unsupported_type)
@settings(max_examples=100)
def test_property1_unsupported_content_type_rejected(
    data: bytes, content_type: str
) -> None:
    """Property 1: a content type not in the accepted set -> ``unsupported_content_type``.

    The empty-file check runs first, so the bytes are non-empty; the
    content-type check is then the first to fail (Req 1.5).

    Validates: Requirements 1.5
    """
    cfg = VisionConfig()
    # Sanity: the generated type is genuinely not accepted.
    normalized = content_type.split(";", 1)[0].strip().lower()
    assume(normalized not in {ct.lower() for ct in cfg.accepted_content_types})

    with pytest.raises(InputValidationError) as exc:
        validate_input(data, content_type, cfg)
    assert exc.value.condition == "unsupported_content_type"


@given(
    over_by=st.integers(min_value=1, max_value=512),
    content_type=_accepted_image_type,
    max_size=st.integers(min_value=1, max_value=256),
)
@settings(max_examples=100)
def test_property1_size_exceeded_rejected(
    over_by: int, content_type: str, max_size: int
) -> None:
    """Property 1: data larger than the configured max -> ``size_exceeded``.

    A small configured ``max_file_size_bytes`` keeps the generated data tiny.
    The bytes are non-empty and the content type is accepted, so the size
    check is the first to fail (Req 1.6). The data is arbitrary bytes but
    would also be ``undecodable``; since the size check runs before the
    decodability check, the reported condition must be ``size_exceeded``.

    Validates: Requirements 1.6
    """
    cfg = VisionConfig(max_file_size_bytes=max_size)
    data = b"\x00" * (max_size + over_by)
    assert len(data) > cfg.max_file_size_bytes

    with pytest.raises(InputValidationError) as exc:
        validate_input(data, content_type, cfg)
    assert exc.value.condition == "size_exceeded"


# --- Property 2: Valid inputs are accepted ---------------------------------
# Validates: Requirements 1.7


@given(
    width=_dims,
    height=_dims,
    channels=_channels,
    content_type=_accepted_image_type,
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property2_valid_image_accepted(
    width: int, height: int, channels: int, content_type: str
) -> None:
    """Property 2: a decodable image, accepted type, within size -> accepted.

    ``validate_input`` returns ``None`` when every check passes (Req 1.7). The
    image is produced with ``cv2.imencode`` so it genuinely decodes, the type
    is in the accepted set, and the default max size comfortably contains the
    small PNG.

    Validates: Requirements 1.7
    """
    cfg = VisionConfig()
    data = _encode_image(width, height, channels)
    assume(0 < len(data) <= cfg.max_file_size_bytes)

    result = validate_input(data, content_type, cfg)
    assert result is None


# --- A few explicit video cases (slow: temp file + VideoCapture) -----------


def test_valid_video_accepted() -> None:
    """A genuinely decodable short video with an accepted type is accepted (Req 1.7).

    Written as an explicit example rather than a generated case because
    encoding a video and probing it via ``cv2.VideoCapture`` is comparatively
    slow. Validates the image/video branch of acceptance.
    """
    import os
    import tempfile

    cfg = VisionConfig()
    tmp_dir = tempfile.mkdtemp(prefix="shelfsentry_vidtest_")
    path = os.path.join(tmp_dir, "clip.mp4")
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 10.0, (32, 32))
        if not writer.isOpened():
            pytest.skip("mp4v VideoWriter backend unavailable in this environment")
        for _ in range(5):
            frame = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

        with open(path, "rb") as fh:
            data = fh.read()
        if not data:
            pytest.skip("VideoWriter produced no bytes in this environment")

        result = validate_input(data, "video/mp4", cfg)
        assert result is None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass


def test_undecodable_video_rejected() -> None:
    """Junk bytes with a video content type -> ``undecodable`` (Req 1.4).

    Explicit example (temp file + VideoCapture probe is slow).
    """
    cfg = VisionConfig()
    data = b"not a real video" * 4
    with pytest.raises(InputValidationError) as exc:
        validate_input(data, "video/mp4", cfg)
    assert exc.value.condition == "undecodable"
