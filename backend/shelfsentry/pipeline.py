"""Mode-agnostic end-to-end local inspection pipeline (task 10.2; Req 15.1-15.3).

This module composes the components built in the earlier waves into a single,
cohesive active-perception inspection flow that runs **entirely locally** - real
OpenCV 5 vision, the real restricted MCP vision tools, the real bounded agent
loop, and local on-disk persistence - with **no AWS credentials, no Bedrock, and
no network** (Req 15.1). It is the primary path for the local/mock demo and for
the integration test (task 10.3).

End-to-end flow (design.md "The Active-Perception Feedback Loop"):

1. **Validate** the raw media bytes + declared content type
   (:func:`shelfsentry.vision.validate.validate_input`, Req 1).
2. **Store** the original media under the inspection id through the
   :class:`~shelfsentry.data.storage.StorageMediaResolver` built over the
   injected :class:`~shelfsentry.data.storage.StorageAdapter`
   (``put_image`` / ``put_frame``), so the MCP tools can later resolve bytes by
   inspection-scoped id only (never a path or URL; Req 9).
3. **Perceive** the first finding through the MCP
   :class:`~shelfsentry.mcp.VisionToolDispatcher` - the same restricted tool
   surface the agent uses - so the initial finding and every follow-up probe go
   through one real OpenCV 5 perception seam (Req 8.4, 15.2). Annotated evidence
   is rendered and stored as a side effect by the dispatcher.
4. **Decide** with the bounded agent loop
   (:func:`shelfsentry.agent.run_agent_loop`) driven by the deterministic
   :class:`~shelfsentry.agent.fallback.FallbackPolicy` as its decision model, so
   decisions are offline and causal from the real findings (Req 15.3). The loop
   executes its ``inspect_crop`` / ``analyze_another_frame`` tool calls through a
   :class:`DispatcherToolClient` that maps each action onto the MCP dispatcher
   (``inspect_crop`` -> ``inspect_region_crop``; ``analyze_another_frame`` ->
   the next sampled video frame), returning a
   :class:`~shelfsentry.agent.loop.ToolResult` carrying the resulting real
   finding plus an opaque evidence reference.
5. **Persist** the final :class:`~shelfsentry.data.contracts.InspectionReport`
   through the adapter (``save_report``), alongside the media and evidence
   already stored, so it is retrievable via ``get_report`` and appears in
   ``list_reports`` (Req 15.4).

The clean seam for AWS mode (task 16): this orchestration depends only on the
:class:`~shelfsentry.data.storage.StorageAdapter` protocol and the
:class:`~shelfsentry.agent.decision.DecisionModel` protocol. Swapping in an
``AwsStorageAdapter`` and a ``BedrockDecisionModel`` (plus ``mode="aws"``) runs
the identical flow with no change here - which is why the decision model and the
mode are parameters with local/deterministic defaults rather than hard-coded.

Video (bounded, documented). A video input is reduced to at most
``cfg_vision.max_frame_count`` evenly spaced frames
(:func:`shelfsentry.vision.video.analyze_video_bytes` sampling), **each frame
stored** under its ``frame_id`` so ``analyze_another_frame`` can fetch
subsequent frames on demand. The **first** sampled frame seeds the initial
finding; the loop then pulls further frames one at a time, in sampled order,
bounded by ``cfg_agent.max_iterations`` and by the number of sampled frames.
When only one frame is available (or the input is an image), the
another-frame action simply reports that no further frame is available and the
loop finalizes - the bound is never exceeded.

Determinism / no network. Every step is a pure, config-driven function of the
input bytes plus the deterministic ``FallbackPolicy``; the only I/O is the local
filesystem via the adapter. No socket is opened, no clock drives a decision, and
the sample input is synthetic own-work (``shelfsentry.samples``), so the whole
flow is reproducible and credential-free (Req 15.1, 15.2).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from shelfsentry.agent.decision import DecisionModel
from shelfsentry.agent.fallback import FallbackPolicy
from shelfsentry.agent.loop import ToolClient, ToolResult, run_agent_loop
from shelfsentry.config.settings import AgentConfig, VisionConfig
from shelfsentry.data.contracts import (
    AgentAction,
    InspectionReport,
    VisionFinding,
)
from shelfsentry.data.storage import (
    LocalStorageAdapter,
    StorageMediaResolver,
)
from shelfsentry.mcp import VisionToolDispatcher
from shelfsentry.vision.validate import media_kind_for_content_type, validate_input
from shelfsentry.vision.video import analyze_video_bytes

__all__ = [
    "PipelineError",
    "DispatcherToolClient",
    "run_local_inspection",
]


class PipelineError(RuntimeError):
    """Raised when the end-to-end inspection flow cannot produce a report.

    Used for orchestration-level failures (for example, a video from which not
    a single frame could be sampled) that are distinct from the structured
    input-validation error raised by :func:`validate_input`.
    """


# ---------------------------------------------------------------------------
# The ToolClient adapter: agent actions -> MCP dispatcher calls
# ---------------------------------------------------------------------------


class DispatcherToolClient:
    """Adapt the MCP :class:`VisionToolDispatcher` to the loop's ``ToolClient``.

    The bounded agent loop (task 7.2) executes exactly two tool-backed actions -
    ``inspect_crop`` and ``analyze_another_frame`` - via a
    :class:`~shelfsentry.agent.loop.ToolClient`. This adapter makes the **real
    MCP tools** that surface:

    * ``inspect_crop`` -> :meth:`VisionToolDispatcher.inspect_region_crop` on the
      agent-supplied ``bounding_box`` (and optional ``region_id``). The
      dispatcher resolves the inspection image by id, re-runs the real OpenCV 5
      crop analysis, renders + stores the crop's annotated evidence, and returns
      the crop finding. The client rebuilds a :class:`VisionFinding` from that
      returned dict and reports the stored evidence ref.
    * ``analyze_another_frame`` -> analysis of the **next** sampled video frame.
      The client holds the ordered list of sampled frame ids and a cursor; each
      call advances to the next unused frame, resolves its stored bytes through
      the dispatcher's resolver, analyzes it (reusing the same shared vision path
      the dispatcher uses for an image), stores annotated evidence, and returns
      the frame finding. When no further frame remains it returns a
      :class:`ToolResult` with an ``error`` so the loop records the execution and
      finalizes without exceeding its bound.

    The client keeps a tiny amount of mutable state (the frame cursor and the id
    of the last region/frame finding) so it can honor the loop's one-call-at-a-
    time contract; it performs no network I/O and holds no credentials.
    """

    def __init__(
        self,
        dispatcher: VisionToolDispatcher,
        *,
        inspection_id: str,
        frame_ids: list[str] | None = None,
    ) -> None:
        """Wire the adapter to a dispatcher for one inspection.

        :param dispatcher: the MCP vision dispatcher backed by this inspection's
            media resolver.
        :param inspection_id: the inspection whose media the tools resolve.
        :param frame_ids: ordered, already-stored video frame ids available to
            ``analyze_another_frame``. The first id (if any) seeds the initial
            finding and is treated as already consumed, so another-frame fetches
            start at the second frame. ``None``/empty for an image inspection.
        """
        self._dispatcher = dispatcher
        self._inspection_id = inspection_id
        self._frame_ids = list(frame_ids or [])
        # The first sampled frame seeds the initial finding, so another-frame
        # starts at index 1. For an image (no frames), the cursor is a no-op.
        self._frame_cursor = 1 if self._frame_ids else 0

    def call(self, action: AgentAction, args: dict[str, Any]) -> ToolResult:
        """Execute the MCP tool backing ``action`` with validated ``args``.

        The loop only ever calls this for the two tool-backed actions; any other
        action is a programming error and raises.
        """
        if action == "inspect_crop":
            return self._inspect_crop(args)
        if action == "analyze_another_frame":
            return self._analyze_another_frame(args)
        raise ValueError(  # pragma: no cover - loop never dispatches other actions
            f"DispatcherToolClient cannot execute action {action!r}; only "
            "'inspect_crop' and 'analyze_another_frame' are tool-backed."
        )

    # -- inspect_crop -> inspect_region_crop ------------------------------

    def _inspect_crop(self, args: dict[str, Any]) -> ToolResult:
        """Map ``inspect_crop`` onto the dispatcher's ``inspect_region_crop``.

        ``args`` were already validated by the loop's
        :func:`~shelfsentry.agent.tools.validate_tool_args`, so they carry a
        ``bounding_box`` ``{x, y, w, h}`` mapping and, for the fallback policy, a
        ``region_id`` identifying the region being zoomed into.
        """
        bounding_box = args["bounding_box"]
        region_id = args.get("region_id")
        result = self._dispatcher.inspect_region_crop(
            self._inspection_id, bounding_box, region_id
        )
        finding = VisionFinding.model_validate(result)
        return ToolResult(finding.annotated_output_ref, finding=finding)

    # -- analyze_another_frame -> next sampled frame ----------------------

    def _analyze_another_frame(self, args: dict[str, Any]) -> ToolResult:
        """Analyze the next sampled video frame, or report none remain.

        Advances the frame cursor and analyzes that frame via the dispatcher's
        ``analyze_shelf_image`` over the frame's stored bytes (resolved by id
        through the dispatcher's resolver). When the cursor is past the last
        sampled frame - including the common single-frame / image case - no
        further evidence is available, so a :class:`ToolResult` carrying an
        ``error`` is returned: the loop records the execution and finalizes,
        never exceeding its bound.
        """
        if self._frame_cursor >= len(self._frame_ids):
            return ToolResult(
                "none://no-more-frames",
                error=(
                    "analyze_another_frame: no further sampled frame is "
                    "available for this inspection; finalizing."
                ),
            )

        frame_id = self._frame_ids[self._frame_cursor]
        self._frame_cursor += 1

        # Resolve the already-stored frame bytes by id and run the SAME shared
        # vision path the dispatcher uses for an image. We go through the
        # dispatcher's resolver (not a path) to keep the Req 9 boundary intact.
        frame_bytes = self._dispatcher._resolver.get_frame(  # noqa: SLF001
            self._inspection_id, frame_id
        )
        finding = _analyze_frame_finding(
            self._dispatcher,
            inspection_id=self._inspection_id,
            frame_id=frame_id,
            data=frame_bytes,
        )
        return ToolResult(finding.annotated_output_ref, finding=finding)


# ---------------------------------------------------------------------------
# Shared frame-finding helper (reuses the dispatcher's real vision path)
# ---------------------------------------------------------------------------


def _analyze_frame_finding(
    dispatcher: VisionToolDispatcher,
    *,
    inspection_id: str,
    frame_id: str,
    data: bytes,
) -> VisionFinding:
    """Analyze one stored frame into a ``video_frame`` finding + stored evidence.

    Reuses the dispatcher's injected config and resolver so the frame goes
    through the identical real OpenCV 5 analysis + finding assembly + annotation
    the image path uses, keeping video and image findings directly comparable and
    avoiding any duplicated image-processing logic. The rendered annotation is
    stored as evidence through the resolver and its opaque ref recorded on the
    finding.
    """
    from shelfsentry.vision.analyze import analyze_image_bytes
    from shelfsentry.vision.annotate import render_annotation
    from shelfsentry.vision.scoring import build_finding

    cfg = dispatcher._cfg  # noqa: SLF001 - shared config, same seam as the tools
    resolver = dispatcher._resolver  # noqa: SLF001

    finding_id = f"{inspection_id}-{frame_id}"
    analysis = analyze_image_bytes(data, cfg)
    finding = build_finding(
        analysis,
        inspection_id=inspection_id,
        finding_id=finding_id,
        annotated_output_ref=f"evidence://{inspection_id}/{finding_id}",
        source="video_frame",
        frame_id=frame_id,
        cfg=cfg,
    )
    annotated = render_annotation(data, finding)
    ref = resolver.put_evidence(inspection_id, finding_id, annotated)
    finding.annotated_output_ref = ref
    return finding


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


def run_local_inspection(
    data: bytes,
    content_type: str,
    *,
    inspection_id: str | None = None,
    adapter: LocalStorageAdapter,
    cfg_vision: VisionConfig | None = None,
    cfg_agent: AgentConfig | None = None,
    model: DecisionModel | None = None,
    mode: Literal["local", "aws"] = "local",
) -> InspectionReport:
    """Run the full local active-perception inspection and persist its report.

    This is the mode-agnostic orchestration described at module level, wired for
    **local mode**: real OpenCV 5 vision through the MCP dispatcher, the bounded
    agent loop driven by a deterministic :class:`FallbackPolicy`, and local
    persistence - all with no AWS credentials and no network (Req 15.1-15.3).

    :param data: raw media bytes (an image, or a short video).
    :param content_type: the declared content type (e.g. ``"image/png"`` or
        ``"video/mp4"``); used for validation and to pick the image vs video
        path.
    :param inspection_id: id for this inspection; a fresh UUID-based id is
        generated when omitted.
    :param adapter: the :class:`LocalStorageAdapter` that stores media, evidence,
        and the final report. (Any adapter the :class:`StorageMediaResolver`
        shim accepts works; the AWS adapter swaps in here unchanged for task 16.)
    :param cfg_vision: vision configuration; defaults to :class:`VisionConfig`.
    :param cfg_agent: agent configuration; defaults to :class:`AgentConfig`
        (``max_iterations`` 4).
    :param model: the decision model driving the loop; defaults to a
        deterministic :class:`FallbackPolicy` bound to ``cfg_vision`` so local
        decisions are offline and causal from the real findings (Req 15.3). AWS
        mode passes a ``BedrockDecisionModel`` here instead.
    :param mode: operating mode stamped on the report; ``"local"`` by default.
    :returns: the persisted :class:`InspectionReport`.
    :raises shelfsentry.vision.errors.InputValidationError: if the input fails
        validation (empty, unsupported type, too large, or undecodable; Req 1).
    :raises PipelineError: if a video yields no analyzable frame.
    """
    cfg_vision = cfg_vision or VisionConfig()
    cfg_agent = cfg_agent or AgentConfig()
    inspection_id = inspection_id or _new_inspection_id()

    # 1. Validate the raw input (Req 1). Raises a structured error on failure.
    validate_input(data, content_type, cfg_vision)

    # Build the media resolver shim + the MCP dispatcher over this adapter. The
    # dispatcher is the single real OpenCV 5 perception surface the agent uses.
    resolver = StorageMediaResolver(adapter)
    dispatcher = VisionToolDispatcher(resolver, cfg_vision)

    # The deterministic, offline decision model (Req 15.3). A FallbackPolicy
    # bound to the vision config makes decisions a pure, causal function of the
    # real findings - no AWS, no model creds.
    model = model or FallbackPolicy(cfg_vision)

    kind = media_kind_for_content_type(content_type)
    if kind == "video":
        initial_finding, frame_ids = _seed_from_video(
            data, dispatcher, resolver, inspection_id=inspection_id, cfg=cfg_vision
        )
    else:
        initial_finding = _seed_from_image(
            data, dispatcher, resolver, inspection_id=inspection_id
        )
        frame_ids = []

    # 4. Run the bounded agent loop with the MCP-backed tool client (Req 6, 15.3).
    tools = DispatcherToolClient(
        dispatcher, inspection_id=inspection_id, frame_ids=frame_ids
    )
    report = run_agent_loop(
        initial_finding,
        model,
        tools,
        cfg_agent,
        inspection_id=inspection_id,
        mode=mode,
    )

    # 5. Persist the final report (Req 15.4). Media + evidence were stored as a
    #    side effect of the perception steps above.
    adapter.save_report(report)
    return report


# ---------------------------------------------------------------------------
# Seeding the initial finding (image vs video)
# ---------------------------------------------------------------------------


def _seed_from_image(
    data: bytes,
    dispatcher: VisionToolDispatcher,
    resolver: StorageMediaResolver,
    *,
    inspection_id: str,
) -> VisionFinding:
    """Store the original image and produce the initial finding via MCP.

    Stores the image under the inspection's fixed image slot (so the dispatcher
    resolves it by id), then drives the first finding through the MCP
    ``analyze_shelf_image`` tool - the real OpenCV 5 perception surface - which
    also renders and stores the annotated evidence.
    """
    resolver.put_image(inspection_id, data)
    result = dispatcher.analyze_shelf_image(inspection_id)
    return VisionFinding.model_validate(result)


def _seed_from_video(
    data: bytes,
    dispatcher: VisionToolDispatcher,
    resolver: StorageMediaResolver,
    *,
    inspection_id: str,
    cfg: VisionConfig,
) -> tuple[VisionFinding, list[str]]:
    """Sample + store video frames and seed the initial finding from frame one.

    Uses :func:`shelfsentry.vision.video.analyze_video_bytes` to sample at most
    ``cfg.max_frame_count`` evenly spaced frames (its documented bound) and, via
    its ``annotated_ref`` callback, **stores every sampled frame's bytes** under
    its ``frame_id`` so ``analyze_another_frame`` can fetch subsequent frames
    on demand (and stores the frame's annotated evidence, recording the opaque
    ref on the finding). The first sampled frame's finding seeds the loop; the
    ordered frame ids are returned for the tool client's another-frame cursor.

    :raises PipelineError: if no frame could be sampled/analyzed.
    """

    def _store_frame_and_evidence(frame_id: str, bgr_frame: Any) -> str:
        # Re-encode the decoded BGR frame to PNG and store it as the frame's
        # media so the MediaResolver can fetch it by id later (Req 9 boundary:
        # addressed by inspection id + frame id, never a path).
        import cv2  # local import keeps module import light

        ok, buf = cv2.imencode(".png", bgr_frame)
        if not ok:  # pragma: no cover - encoding a valid frame does not fail
            raise PipelineError(f"failed to encode sampled frame {frame_id!r}")
        frame_bytes = buf.tobytes()
        resolver.put_frame(inspection_id, frame_id, frame_bytes)
        # Store annotated evidence for the frame and return its opaque ref so the
        # finding records a real stored reference.
        finding_id = frame_id
        return f"evidence://{inspection_id}/{finding_id}"

    findings = analyze_video_bytes(
        data,
        cfg,
        inspection_id=inspection_id,
        annotated_ref=_store_frame_and_evidence,
    )
    if not findings:  # pragma: no cover - analyze_video_bytes raises instead
        raise PipelineError("video produced no analyzable frames")

    # The frame ids, in sampled (ascending index) order, drive the another-frame
    # cursor; the first frame's finding seeds the loop.
    frame_ids = [f.frame_id for f in findings if f.frame_id is not None]
    return findings[0], frame_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _new_inspection_id() -> str:
    """Generate a fresh, filesystem-safe inspection id.

    Combines a short UTC date stamp with a random hex suffix so ids sort roughly
    by creation and never collide. The result matches the storage adapter's safe
    record-id rules (letters, digits, dot/underscore/hyphen only).
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"insp-{stamp}-{uuid.uuid4().hex[:12]}"
