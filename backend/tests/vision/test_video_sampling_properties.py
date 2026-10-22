"""Property-based tests for pure video frame sampling (task 4.3).

Covers the pure, side-effect-free sampling functions in
``shelfsentry.vision.video`` across wide input ranges with Hypothesis:

* :func:`shelfsentry.vision.video.sample_frame_indices` -- choosing at most
  ``M`` evenly spaced source-frame indices across ``[0, N)``.
* :func:`shelfsentry.vision.video.frame_id_for_index` -- the stable
  ``frame_{index:06d}`` identifier for a source-frame index.

The example-based coverage of these functions (specific ``N``/``M`` pairs,
error cases, id formatting) already lives in ``test_video.py``; this module
deliberately does NOT duplicate those. It instead asserts the universal
sampling properties hold for Hypothesis-drawn inputs over a broad range.

Two properties are implemented:

* **Property 10: Video sampling stays within the configured limit** -- the
  result never exceeds ``M`` or ``N``, every index is within ``[0, N)``, the
  list is sorted ascending, and when ``N <= M`` it is exactly
  ``list(range(N))`` (Req 3.1, 3.3, 3.4; sampling level of Req 22.3).
* **Property 11: Sampled frames have unique identifiers** -- mapping the
  sampled indices through :func:`frame_id_for_index` yields unique ids of the
  ``frame_{index:06d}`` form, and the indices themselves are unique (Req 3.2).

Both functions are cheap pure functions, so ``max_examples`` is pushed well
above the 100-example floor.
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from shelfsentry.vision.video import frame_id_for_index, sample_frame_indices

# total_frames N: 0 up to a few thousand (includes the empty-video edge).
_total_frames = st.integers(min_value=0, max_value=3000)
# max_frames M: a valid positive cap, matching the configurable range (1..64).
_max_frames = st.integers(min_value=1, max_value=64)

# A frame id is exactly "frame_" followed by >= 6 zero-padded digits.
_FRAME_ID_RE = re.compile(r"^frame_\d{6,}$")


# --- Property 10: Video sampling stays within the configured limit ---------
# Validates: Requirements 3.1, 3.3, 3.4


@given(total_frames=_total_frames, max_frames=_max_frames)
@settings(max_examples=500)
def test_property10_sampling_stays_within_limit(
    total_frames: int, max_frames: int
) -> None:
    """Property 10: Video sampling stays within the configured limit.

    For any ``N`` in ``[0, 3000]`` and ``M`` in ``[1, 64]``,
    :func:`sample_frame_indices` returns a sorted list of unique indices, each
    within ``[0, N)``, whose length never exceeds ``M`` or ``N``; and when
    ``N <= M`` it keeps every frame (``list(range(N))``). This is the sampling
    level of video frame extraction (Req 22.3).

    Validates: Requirements 3.1, 3.3, 3.4
    """
    result = sample_frame_indices(total_frames, max_frames)

    # Count never exceeds the configured cap or the available frames (Req 3.4).
    assert len(result) <= max_frames
    assert len(result) <= total_frames

    # Every index is a valid position in the source video (Req 3.1).
    assert all(0 <= i < total_frames for i in result)

    # Sorted ascending and unique (Req 3.4).
    assert result == sorted(result)
    assert len(set(result)) == len(result)

    # When the clip fits under the cap, every frame is kept (Req 3.1, 3.4).
    if total_frames <= max_frames:
        assert result == list(range(total_frames))


# --- Property 11: Sampled frames have unique identifiers -------------------
# Validates: Requirements 3.2


@given(total_frames=_total_frames, max_frames=_max_frames)
@settings(max_examples=500)
def test_property11_sampled_frames_have_unique_identifiers(
    total_frames: int, max_frames: int
) -> None:
    """Property 11: Sampled frames have unique identifiers.

    Mapping the sampled indices through :func:`frame_id_for_index` yields ids
    that are all unique and all match the stable ``frame_{index:06d}`` form;
    the sampled indices are themselves unique (Req 3.2).

    Validates: Requirements 3.2
    """
    indices = sample_frame_indices(total_frames, max_frames)

    # The indices feeding the ids are unique (so ids cannot collide).
    assert len(set(indices)) == len(indices)

    frame_ids = [frame_id_for_index(i) for i in indices]

    # Every id has the stable zero-padded form "frame_{index:06d}".
    assert all(_FRAME_ID_RE.match(fid) for fid in frame_ids)
    # And matches exactly what the formatting rule produces for its index.
    assert all(fid == f"frame_{i:06d}" for fid, i in zip(frame_ids, indices))

    # Ids are unique across the sampled set (Req 3.2).
    assert len(set(frame_ids)) == len(frame_ids)
