"""Video frame sampling and per-frame analysis (Req 3.1-3.4, 4.4).

A short shelf *video* is analyzed by reducing it to a bounded set of still
frames and running each one through the exact same OpenCV 5 image path used for
single images (:func:`shelfsentry.vision.analyze.analyze_image_arrays` via
:func:`shelfsentry.vision.preprocess.preprocess_image`), then assembling a
per-frame :class:`~shelfsentry.data.contracts.VisionFinding` with
:func:`shelfsentry.vision.scoring.build_finding` tagged ``source="video_frame"``
and the frame's stable identifier (Req 3.3, 4.4). Reusing the image path keeps
video findings directly comparable to image findings and avoids any duplicate
image-processing logic.

Sampling strategy (documented, Req 3.1, 3.4): given a video with ``N`` total
frames and a configured cap ``M = cfg.max_frame_count``, we sample **at most
``M`` evenly spaced frame indices** across the half-open range ``[0, N)``:

* If ``N <= M`` we keep every frame (indices ``0 .. N-1``).
* Otherwise we place ``M`` indices uniformly across ``[0, N)`` using the
  midpoints of ``M`` equal-width buckets: ``floor((i + 0.5) * N / M)`` for
  ``i`` in ``0 .. M-1``. This spreads the samples across the whole clip
  (including near the start and end) without ever selecting index ``N`` or
  beyond, and yields a strictly increasing, de-duplicated, sorted list whose
  length never exceeds ``M`` (Req 3.4).

:func:`sample_frame_indices` is a pure function of ``(N, M)`` and is unit
tested in isolation. The count is a hard guarantee: ``len(result) <= M`` and
``len(result) <= N`` always.

Frame identifiers (Req 3.2): each sampled frame gets a stable, unique id
``"frame_{index:06d}"`` where ``index`` is the frame's position in the *source*
video (not its position within the sampled subset). Because the sampled indices
are unique, the ids are unique too.

Annotated-output references (Req 4.5): a video frame still needs an
``annotated_output_ref`` on its finding, but this module deliberately does
**not** hard-depend on storage or on the annotation renderer (task 3.3). The
caller supplies either an ``annotated_ref_template`` string formatted with the
frame id (default ``"ref://{frame_id}.png"``) or an ``annotated_ref`` callback
``(frame_id, bgr_frame) -> str`` that may render/store annotated bytes and
return the reference. The callback receives the decoded BGR frame so a caller
that wants real annotated evidence can produce it without this module importing
the renderer.

Decoding (``cv2.VideoCapture`` needs a file path): OpenCV cannot open a video
from an in-memory buffer, so the raw bytes are written to a
``NamedTemporaryFile``, opened with ``cv2.VideoCapture(path)``, and the temp
file is removed in a ``finally`` regardless of outcome. Each sampled frame is
seeked to with ``CAP_PROP_POS_FRAMES`` and read; the decoded BGR frame is
re-encoded to a PNG byte buffer in memory and handed to the same
``preprocess``/``analyze`` path as an image.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable

import cv2
import numpy as np

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import VisionFinding
from shelfsentry.vision.analyze import analyze_image_arrays
from shelfsentry.vision.preprocess import PreprocessError, preprocess_image
from shelfsentry.vision.scoring import build_finding

__all__ = [
    "VideoDecodeError",
    "AnnotatedRefCallback",
    "frame_id_for_index",
    "sample_frame_indices",
    "analyze_video_bytes",
]


class VideoDecodeError(ValueError):
    """Raised when video bytes cannot be opened or no frames can be read.

    Input *validation* (content type, size, decodability) lives in
    ``validate.py``; this narrower error covers the decode/read step so a caller
    that analyzes already-validated bytes still fails loudly - naming the
    specific condition (could not open, zero frames, unreadable frame) - rather
    than silently returning an empty result when the backend is missing or the
    container is corrupt.
    """


#: A caller-supplied hook that turns a sampled frame into an
#: ``annotated_output_ref`` string. It receives the frame's stable id and the
#: decoded BGR frame (so it may render/store annotated bytes) and returns the
#: reference to record on the finding. Kept as a callback so this module never
#: hard-depends on storage or the annotation renderer.
AnnotatedRefCallback = Callable[[str, np.ndarray], str]


def frame_id_for_index(index: int) -> str:
    """Return the stable, unique frame id for a source-video frame index (Req 3.2).

    The id is ``"frame_{index:06d}"`` where ``index`` is the frame's position in
    the *source* video. Distinct indices therefore map to distinct ids, so the
    ids of a set of uniquely sampled indices are themselves unique.

    Args:
        index: Zero-based frame position in the source video (non-negative).

    Returns:
        The zero-padded frame id, for example ``"frame_000042"``.

    Raises:
        ValueError: if ``index`` is negative.
    """
    if index < 0:
        raise ValueError(f"frame index must be non-negative, got {index}")
    return f"frame_{index:06d}"


def sample_frame_indices(total_frames: int, max_frames: int) -> list[int]:
    """Choose at most ``max_frames`` evenly spaced indices across ``[0, N)`` (Req 3.4).

    Pure function implementing the documented uniform sampling strategy:

    * ``N = total_frames``, ``M = max_frames``.
    * If ``N <= M``, return every index ``0 .. N-1`` (use all frames).
    * Otherwise return the midpoints of ``M`` equal-width buckets spanning
      ``[0, N)``: ``floor((i + 0.5) * N / M)`` for ``i in 0 .. M-1``. This is
      strictly increasing for ``N > M`` (bucket width ``N / M > 1``), so the
      result is already sorted, unique, and of length exactly ``M``; it is
      de-duplicated defensively anyway.

    Guarantees (Req 3.4): the result is sorted ascending, contains only unique
    indices in ``[0, N)``, and ``len(result) <= M`` and ``len(result) <= N``.

    Args:
        total_frames: Number of frames ``N`` in the source video (``>= 0``).
        max_frames: Configured cap ``M`` on sampled frames (``> 0``).

    Returns:
        A sorted list of unique source-frame indices to sample.

    Raises:
        ValueError: if ``total_frames`` is negative or ``max_frames`` is not
            positive.
    """
    if total_frames < 0:
        raise ValueError(f"total_frames must be non-negative, got {total_frames}")
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    if total_frames == 0:
        return []

    if total_frames <= max_frames:
        return list(range(total_frames))

    # N > M: midpoints of M equal-width buckets across [0, N). Bucket width is
    # N / M > 1, so the midpoints are strictly increasing and all stay < N.
    indices: list[int] = []
    for i in range(max_frames):
        idx = int((i + 0.5) * total_frames / max_frames)
        # Clamp defensively so floating-point never pushes the last index to N.
        if idx >= total_frames:
            idx = total_frames - 1
        indices.append(idx)

    # De-duplicate while preserving ascending order (defensive; the midpoints
    # are already unique for N > M).
    unique_sorted = sorted(set(indices))
    return unique_sorted


def _read_frame_at(capture: "cv2.VideoCapture", index: int) -> np.ndarray | None:
    """Seek to ``index`` and read one BGR frame, or ``None`` if unreadable.

    Uses ``CAP_PROP_POS_FRAMES`` to position the decoder, then ``read()``. Some
    backends/containers do not honor exact seeks; a ``None`` return is treated
    by the caller as an unreadable frame for that index.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
    ok, frame = capture.read()
    if not ok or frame is None:
        return None
    return frame


def analyze_video_bytes(
    data: bytes,
    cfg: VisionConfig,
    *,
    inspection_id: str,
    annotated_ref_template: str = "ref://{frame_id}.png",
    annotated_ref: AnnotatedRefCallback | None = None,
) -> list[VisionFinding]:
    """Sample frames from video bytes and analyze each with the image path.

    Writes ``data`` to a temporary file (``cv2.VideoCapture`` requires a path),
    reads the total frame count, picks at most ``cfg.max_frame_count`` evenly
    spaced frames via :func:`sample_frame_indices`, and for each sampled frame:

    1. assigns the stable id ``frame_{index:06d}`` (Req 3.2);
    2. re-encodes the decoded BGR frame to an in-memory PNG buffer and runs the
       SAME :func:`shelfsentry.vision.preprocess.preprocess_image` ->
       :func:`shelfsentry.vision.analyze.analyze_image_arrays` path used for
       images (Req 3.3);
    3. resolves an ``annotated_output_ref`` - via the ``annotated_ref`` callback
       when given, otherwise by formatting ``annotated_ref_template`` with the
       frame id (Req 4.5) - and assembles a per-frame
       :class:`~shelfsentry.data.contracts.VisionFinding` with
       ``source="video_frame"`` and ``frame_id`` set (Req 4.4) through
       :func:`shelfsentry.vision.scoring.build_finding`.

    The temporary file is always removed in a ``finally`` block.

    Args:
        data: Raw encoded video bytes (already validated upstream).
        cfg: Vision configuration; ``max_frame_count`` caps the sample size.
        inspection_id: Owning inspection identifier recorded on each finding
            (non-empty).
        annotated_ref_template: Format string for the per-frame annotated-output
            reference, formatted with ``frame_id`` (default
            ``"ref://{frame_id}.png"``). Used when ``annotated_ref`` is ``None``.
        annotated_ref: Optional callback ``(frame_id, bgr_frame) -> str`` that
            renders/stores annotated bytes and returns the reference. Takes
            precedence over ``annotated_ref_template`` when provided.

    Returns:
        A list of per-frame :class:`VisionFinding`, one per successfully read
        sampled frame, ordered by ascending source-frame index. Its length is
        at most ``cfg.max_frame_count``.

    Raises:
        VideoDecodeError: if the bytes are empty, the video cannot be opened, or
            its frame count cannot be determined / no frames can be read.
    """
    if not data:
        raise VideoDecodeError("cannot analyze an empty video byte buffer")

    # VideoCapture needs a real path, so persist the bytes to a temp file. We
    # close the handle before opening with OpenCV (Windows will not let a second
    # handle read a file that is still open for writing), and delete it in the
    # finally regardless of what happens in between.
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()

        capture = cv2.VideoCapture(tmp_path)
        try:
            if not capture.isOpened():
                raise VideoDecodeError(
                    "cv2.VideoCapture could not open the provided video bytes"
                )

            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise VideoDecodeError(
                    "could not determine a positive frame count for the video"
                )

            indices = sample_frame_indices(total_frames, cfg.max_frame_count)

            findings: list[VisionFinding] = []
            for index in indices:
                frame = _read_frame_at(capture, index)
                if frame is None:
                    # Skip an unreadable frame rather than failing the whole
                    # video; other sampled frames may still be analyzable.
                    continue

                frame_id = frame_id_for_index(index)

                # Re-encode the decoded frame to an in-memory PNG and run the
                # exact same image path (preprocess -> analyze). PNG is lossless
                # so the analyzed pixels match the decoded frame.
                ok, buf = cv2.imencode(".png", frame)
                if not ok:
                    continue
                try:
                    pre = preprocess_image(buf.tobytes(), cfg)
                except PreprocessError:
                    # A frame that round-trips to undecodable bytes is skipped.
                    continue
                analysis = analyze_image_arrays(pre, cfg)

                if annotated_ref is not None:
                    output_ref = annotated_ref(frame_id, frame)
                else:
                    output_ref = annotated_ref_template.format(frame_id=frame_id)

                finding = build_finding(
                    analysis,
                    inspection_id=inspection_id,
                    finding_id=frame_id,
                    annotated_output_ref=output_ref,
                    source="video_frame",
                    frame_id=frame_id,
                    cfg=cfg,
                )
                findings.append(finding)

            if not findings:
                raise VideoDecodeError(
                    "no frames could be read from the provided video bytes"
                )

            return findings
        finally:
            capture.release()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            # Best-effort cleanup; a leftover temp file must not mask the result.
            pass
