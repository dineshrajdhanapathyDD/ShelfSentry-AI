"""Annotated_Output rendering for the OpenCV 5 vision pipeline (Req 2.4, 4.5, 5.3).

This stage turns an already-assembled
:class:`~shelfsentry.data.contracts.VisionFinding` plus the *original* input
bytes into an **Annotated_Output** image: the original frame with every
:class:`~shelfsentry.data.contracts.CandidateRegion` drawn on top at *exactly*
the bounding-box coordinates recorded in the finding (Req 2.4, 5.3).

Separation of concerns:

* The finding (ratios, scores, ids, boxes) is assembled upstream by
  :func:`shelfsentry.vision.scoring.build_finding` (task 3.2). This module does
  **not** duplicate that assembly - it only *renders* what the finding already
  recorded. :func:`render_annotation` takes a finished finding and the original
  bytes and returns encoded image bytes; it never recomputes geometry.
* Wiring the resulting bytes into persistent storage and setting the finding's
  ``annotated_output_ref`` is the caller's job (the pipeline stores the bytes
  and already supplied the ref when it built the finding). To make the common
  path ergonomic, :func:`build_annotated_finding` is a thin convenience that
  calls ``build_finding`` once (it does not re-implement it) and returns both
  the finding and its rendered annotation bytes together.

Coordinate fidelity (Req 5.3): each region is drawn with ``cv2.rectangle`` whose
opposite corners are ``(x, y)`` and ``(x + w - 1, y + h - 1)`` - i.e. the exact
integer-pixel box recorded in ``region.bounding_box`` in the single documented
top-left-origin ``(x, y, w, h)`` coordinate system (Req 5.2). No scaling, no
rounding, no coordinate transform is applied, so a drawn border pixel lands on
the recorded box edge.

Determinism (Req 5.1): colors, thickness, font, and label text are fixed
functions of the finding's recorded values with no randomness, wall-clock, or
dict/set iteration order dependence. Regions are drawn in the finding's recorded
region order, and the output is PNG-encoded (lossless, deterministic), so
rendering the same finding over the same bytes twice yields byte-for-byte
identical output.
"""

from __future__ import annotations

import cv2
import numpy as np

from shelfsentry.config.settings import VisionConfig
from shelfsentry.data.contracts import CandidateRegion, VisionFinding
from shelfsentry.vision.analyze import AnalysisResult
from shelfsentry.vision.preprocess import decode_bgr
from shelfsentry.vision.scoring import build_finding

__all__ = [
    "ANNOTATION_COLOR_BGR",
    "ANNOTATION_THICKNESS",
    "LABEL_COLOR_BGR",
    "LABEL_FONT",
    "LABEL_FONT_SCALE",
    "LABEL_THICKNESS",
    "render_annotation",
    "build_annotated_finding",
]


# ---------------------------------------------------------------------------
# Fixed, deterministic drawing style (Req 5.1)
# ---------------------------------------------------------------------------

#: Rectangle color for a candidate-region box, in OpenCV BGR order. A saturated
#: green reads clearly over typical shelf imagery. Fixed for determinism.
ANNOTATION_COLOR_BGR: tuple[int, int, int] = (0, 255, 0)

#: Rectangle line thickness in pixels. Fixed for determinism. The border is
#: drawn centered on the box edge; see the coordinate note in the module docs.
ANNOTATION_THICKNESS: int = 2

#: Text color for the per-region label, in OpenCV BGR order. Fixed.
LABEL_COLOR_BGR: tuple[int, int, int] = (0, 255, 0)

#: Hershey font used for the label. Fixed for determinism.
LABEL_FONT: int = cv2.FONT_HERSHEY_SIMPLEX

#: Label font scale and stroke thickness. Fixed for determinism.
LABEL_FONT_SCALE: float = 0.4
LABEL_THICKNESS: int = 1


def _label_text(region: CandidateRegion) -> str:
    """Build the short, deterministic per-region label string.

    Shows the region id and its measured ``empty_space_ratio`` (to two decimal
    places) so the Annotated_Output marks *which* region and *how empty* it was
    measured (Req 2.4). When a region uses a documented alternative metric
    instead of a ratio (``empty_space_ratio is None``), the id alone is shown.

    Args:
        region: The candidate region whose label to render.

    Returns:
        A short label such as ``"region_0000 0.82"``.
    """
    if region.empty_space_ratio is None:
        return region.region_id
    return f"{region.region_id} {region.empty_space_ratio:.2f}"


def render_annotation(data: bytes, finding: VisionFinding) -> bytes:
    """Render the Annotated_Output for ``finding`` onto the original image bytes.

    Decodes the *original* input bytes, then draws each candidate region in
    ``finding.candidate_regions`` onto a copy using ``cv2.rectangle`` at exactly
    the recorded ``region.bounding_box`` ``(x, y, w, h)`` coordinates, plus a
    small ``cv2.putText`` label naming the region and its empty-space ratio
    (Req 2.4). The annotated image is PNG-encoded and returned as bytes (Req
    4.5): the caller stores those bytes and sets ``annotated_output_ref`` on the
    finding.

    The drawn rectangle's opposite corners are ``(x, y)`` and
    ``(x + w - 1, y + h - 1)``, so annotation coordinates match the bounding
    boxes recorded in the finding with no transform (Req 5.3). Boxes are clipped
    defensively to the decoded image bounds so a box that (mis)reports beyond the
    frame still draws without error.

    Determinism (Req 5.1): fixed style, recorded region order, and lossless PNG
    encoding make two renders of the same ``(data, finding)`` byte-identical.

    Args:
        data: The original, encoded input image bytes the finding was derived
            from (PNG/JPEG/etc.).
        finding: The assembled finding whose recorded regions to draw.

    Returns:
        PNG-encoded annotated image bytes.

    Raises:
        shelfsentry.vision.preprocess.PreprocessError: if ``data`` is empty or
            cannot be decoded as an image.
        RuntimeError: if ``cv2.imencode`` fails to encode the annotated image.
    """
    # Decode the ORIGINAL bytes (same decode path the analysis used) and draw on
    # a copy so the decoded source array is never mutated out from under a
    # caller that might reuse it.
    image = decode_bgr(data).copy()
    height, width = image.shape[:2]

    for region in finding.candidate_regions:
        box = region.bounding_box
        x, y, w, h = box.x, box.y, box.w, box.h

        # Exact recorded corners. w, h >= 1 (enforced by BoundingBox), so the
        # far corner is the last pixel inside the box, not one past it.
        x1 = x
        y1 = y
        x2 = x + w - 1
        y2 = y + h - 1

        # Defensive clip to the decoded frame so an out-of-frame box still draws
        # (its visible edges land on the recorded coordinates).
        x1c = max(0, min(x1, width - 1))
        y1c = max(0, min(y1, height - 1))
        x2c = max(0, min(x2, width - 1))
        y2c = max(0, min(y2, height - 1))

        cv2.rectangle(
            image,
            (x1c, y1c),
            (x2c, y2c),
            ANNOTATION_COLOR_BGR,
            ANNOTATION_THICKNESS,
            lineType=cv2.LINE_8,
        )

        # Place the label just above the box top-left; nudge it below the top
        # edge when the box sits flush against the top of the frame so the text
        # stays on-image.
        label = _label_text(region)
        label_org_y = y1c - 4 if y1c - 4 >= 8 else min(y1c + 14, height - 2)
        label_org_x = max(0, min(x1c, width - 1))
        cv2.putText(
            image,
            label,
            (label_org_x, label_org_y),
            LABEL_FONT,
            LABEL_FONT_SCALE,
            LABEL_COLOR_BGR,
            LABEL_THICKNESS,
            lineType=cv2.LINE_8,
        )

    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("cv2.imencode failed to encode the annotated image")
    return buf.tobytes()


def build_annotated_finding(
    analysis: AnalysisResult,
    original_bytes: bytes,
    *,
    inspection_id: str,
    finding_id: str,
    annotated_output_ref: str,
    source: str = "image",
    frame_id: str | None = None,
    cfg: VisionConfig,
) -> tuple[VisionFinding, bytes]:
    """Assemble a finding and render its Annotated_Output in one convenience call.

    A thin ergonomic wrapper for the pipeline: it calls
    :func:`shelfsentry.vision.scoring.build_finding` (the single finding
    assembler; this wrapper does **not** duplicate that logic) to build the
    finding, then :func:`render_annotation` to draw that finding's recorded
    regions onto ``original_bytes``. The finding carries the
    ``annotated_output_ref`` the caller supplies (the ref the caller will store
    the returned bytes under), so the two stay consistent.

    Args:
        analysis: The deterministic analysis output to score.
        original_bytes: The original encoded input bytes to annotate.
        inspection_id: Owning inspection identifier (non-empty).
        finding_id: Identifier for this finding (non-empty).
        annotated_output_ref: Storage reference the caller will store the
            returned annotation bytes under (recorded on the finding; Req 4.5).
        source: Finding source; one of ``"image"``, ``"video_frame"``,
            ``"crop"``, ``"frame_compare"``. Defaults to ``"image"``.
        frame_id: Present for video-derived findings (Req 4.4).
        cfg: Vision configuration.

    Returns:
        A ``(finding, annotated_bytes)`` tuple: the validated
        :class:`VisionFinding` and its PNG-encoded Annotated_Output bytes.
    """
    finding = build_finding(
        analysis,
        inspection_id=inspection_id,
        finding_id=finding_id,
        annotated_output_ref=annotated_output_ref,
        source=source,
        frame_id=frame_id,
        cfg=cfg,
    )
    annotated_bytes = render_annotation(original_bytes, finding)
    return finding, annotated_bytes
