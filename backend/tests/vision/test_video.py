"""Tests for video frame sampling and per-frame analysis (task 3.4).

Covers ``shelfsentry.vision.video`` against Req 3.1-3.4 and 4.4:

* :func:`sample_frame_indices` is a pure, unit-tested function: it returns at
  most ``M`` unique, sorted indices within ``[0, N)``; uses all frames when
  ``N <= M``; and its count never exceeds ``M`` (or ``N``) for any inputs
  (Req 3.1, 3.4).
* Frame ids are the stable, unique ``frame_{index:06d}`` form (Req 3.2).
* :func:`analyze_video_bytes` runs a short synthetic video through the SAME
  image analysis path and yields ``<= M`` per-frame
  :class:`~shelfsentry.data.contracts.VisionFinding` objects, each with
  ``source == "video_frame"`` and a unique ``frame_id`` (Req 3.3, 4.4). The
  synthetic-video test skips gracefully when the OpenCV ``VideoWriter`` backend
  is unavailable in the environment.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import VisionFinding
from shelfsentry.vision.video import (
    VideoDecodeError,
    analyze_video_bytes,
    frame_id_for_index,
    sample_frame_indices,
)


@pytest.fixture()
def cfg() -> VisionConfig:
    return VisionConfig()


# ---------------------------------------------------------------------------
# sample_frame_indices (pure, Req 3.1, 3.4)
# ---------------------------------------------------------------------------


def test_sample_indices_subsamples_large_video() -> None:
    """total=100, max=16 -> 16 unique sorted indices within [0, 100)."""
    indices = sample_frame_indices(100, 16)

    assert len(indices) == 16
    assert len(set(indices)) == 16  # unique
    assert indices == sorted(indices)  # ascending
    assert all(0 <= i < 100 for i in indices)


def test_sample_indices_uses_all_when_fewer_than_max() -> None:
    """total=5, max=16 -> every frame [0, 1, 2, 3, 4] (N <= M)."""
    assert sample_frame_indices(5, 16) == [0, 1, 2, 3, 4]


def test_sample_indices_equal_to_max_uses_all() -> None:
    """total == max -> all frames, exactly M of them."""
    assert sample_frame_indices(16, 16) == list(range(16))


def test_sample_indices_empty_video() -> None:
    """total=0 -> no indices."""
    assert sample_frame_indices(0, 16) == []


def test_sample_indices_count_never_exceeds_max_or_total() -> None:
    """For a range of (N, M), the count never exceeds M (or N), and stays valid."""
    for total in (0, 1, 3, 7, 16, 17, 50, 100, 999):
        for max_frames in (1, 2, 4, 16, 32):
            indices = sample_frame_indices(total, max_frames)
            assert len(indices) <= max_frames
            assert len(indices) <= total
            assert len(set(indices)) == len(indices)  # unique
            assert indices == sorted(indices)  # ascending
            assert all(0 <= i < total for i in indices)


def test_sample_indices_rejects_bad_args() -> None:
    """Negative total / non-positive max are rejected with ValueError."""
    with pytest.raises(ValueError):
        sample_frame_indices(-1, 16)
    with pytest.raises(ValueError):
        sample_frame_indices(10, 0)


# ---------------------------------------------------------------------------
# frame_id_for_index (Req 3.2)
# ---------------------------------------------------------------------------


def test_frame_id_is_stable_zero_padded_form() -> None:
    assert frame_id_for_index(0) == "frame_000000"
    assert frame_id_for_index(42) == "frame_000042"
    assert frame_id_for_index(123456) == "frame_123456"


def test_frame_id_rejects_negative_index() -> None:
    with pytest.raises(ValueError):
        frame_id_for_index(-1)


# ---------------------------------------------------------------------------
# analyze_video_bytes (Req 3.1-3.4, 4.4)
# ---------------------------------------------------------------------------


def _shelf_frame(variant: int) -> np.ndarray:
    """Build a synthetic BGR shelf frame with a dark gap (varies by index).

    Deterministic per ``variant`` (no randomness) so the pipeline stays
    reproducible. Returns a 3-channel BGR ``uint8`` array matching what a video
    decoder yields.
    """
    h, w = 120, 160
    gray = np.full((h, w), 210, dtype=np.uint8)  # bright background
    gray[38:42, :] = 20  # shelf lip
    gray[78:82, :] = 20  # shelf lip
    # A dark empty gap whose left edge shifts with the variant.
    left = 10 + (variant % 3) * 15
    gray[50:78, left : left + 60] = 15
    gray[50:78, left + 80 : w - 5] = 130  # product block
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _synthetic_video_bytes(tmp_path, *, frames: int, fps: int = 10) -> bytes | None:
    """Write a short mp4v video to ``tmp_path`` and return its bytes.

    Returns ``None`` when the OpenCV ``VideoWriter`` backend cannot encode an
    mp4v file in this environment (no FFmpeg/codec), so callers can skip.
    """
    h, w = 120, 160
    path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        writer.release()
        return None
    try:
        for i in range(frames):
            writer.write(_shelf_frame(i))
    finally:
        writer.release()

    if not path.exists() or path.stat().st_size == 0:
        return None
    return path.read_bytes()


def test_analyze_video_bytes_produces_bounded_frame_findings(tmp_path, cfg) -> None:
    """A synthetic video yields <= M video_frame findings with unique frame ids."""
    total_frames = 30
    data = _synthetic_video_bytes(tmp_path, frames=total_frames)
    if data is None:
        pytest.skip("OpenCV VideoWriter mp4v backend unavailable in this environment")

    findings = analyze_video_bytes(data, cfg, inspection_id="insp-vid-1")

    # At most M findings (Req 3.1, 3.4); at least one from a 30-frame clip.
    assert 1 <= len(findings) <= cfg.max_frame_count

    frame_ids = [f.frame_id for f in findings]
    for f in findings:
        assert isinstance(f, VisionFinding)
        assert f.source == "video_frame"  # Req 3.3
        assert f.frame_id is not None and f.frame_id.startswith("frame_")  # Req 3.2
        assert f.inspection_id == "insp-vid-1"
        # Each region on a video finding carries the frame id (Req 4.4).
        for region in f.candidate_regions:
            assert region.frame_id == f.frame_id
        # Round-trips through strict boundary validation.
        VisionFinding.validate_boundary(f.model_dump(mode="json"))

    # Unique frame identifiers (Req 3.2).
    assert len(set(frame_ids)) == len(frame_ids)


def test_analyze_video_uses_default_ref_template(tmp_path, cfg) -> None:
    """The default annotated_ref_template is formatted with the frame id (Req 4.5)."""
    data = _synthetic_video_bytes(tmp_path, frames=8)
    if data is None:
        pytest.skip("OpenCV VideoWriter mp4v backend unavailable in this environment")

    findings = analyze_video_bytes(data, cfg, inspection_id="insp-vid-2")
    for f in findings:
        assert f.annotated_output_ref == f"ref://{f.frame_id}.png"


def test_analyze_video_ref_callback_is_used(tmp_path, cfg) -> None:
    """A supplied annotated_ref callback resolves the ref and sees the frame."""
    data = _synthetic_video_bytes(tmp_path, frames=6)
    if data is None:
        pytest.skip("OpenCV VideoWriter mp4v backend unavailable in this environment")

    seen: list[str] = []

    def _ref(frame_id: str, frame: np.ndarray) -> str:
        assert frame.ndim == 3 and frame.shape[2] == 3  # decoded BGR frame
        seen.append(frame_id)
        return f"stored://{frame_id}"

    findings = analyze_video_bytes(
        data, cfg, inspection_id="insp-vid-3", annotated_ref=_ref
    )
    assert seen  # callback was invoked
    for f in findings:
        assert f.annotated_output_ref == f"stored://{f.frame_id}"


def test_analyze_video_rejects_empty_bytes(cfg) -> None:
    """Empty video bytes fail loudly (Req: fail loud, not silent empty result)."""
    with pytest.raises(VideoDecodeError):
        analyze_video_bytes(b"", cfg, inspection_id="insp-vid-4")


def test_analyze_video_rejects_undecodable_bytes(cfg) -> None:
    """Non-video bytes cannot be opened and raise VideoDecodeError."""
    with pytest.raises(VideoDecodeError):
        analyze_video_bytes(b"not a video at all", cfg, inspection_id="insp-vid-5")
