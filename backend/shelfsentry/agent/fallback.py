"""The deterministic :class:`FallbackPolicy` (Req 7.2, 7.4; task 6.2).

This is the thin, SAFE fallback that slots behind the :class:`DecisionModel`
protocol (``shelfsentry.agent.decision.DecisionModel``). When the Bedrock path
is unavailable or errors, the agent loop switches to this policy to keep an
inspection moving without silent failure (Req 7.2). Here it is also usable
*directly* as a decision model.

Determinism (Req 7.4). :meth:`FallbackPolicy.decide` is a **pure** function of
``(finding, trace, cfg)``: no clock, no randomness, and no dependence on
dict/set iteration order. The same inputs always select the same action with
the same arguments and the same rationale. The property that backs this is
"Fallback policy is deterministic" (design.md Property 17).

Provenance. Decisions this policy produces set ``decided_by="fallback"``. It
does NOT set ``used_fallback=True`` - that flag is owned by the loop and means
"the loop engaged the fallback because the model errored" (Req 7.2). A
``FallbackPolicy`` used directly simply records that *it* made the call.

Reference decision logic (design.md "Deterministic fallback policy"), evaluated
against the current ``finding`` plus the accumulated ``trace``. The branches are
checked in a fixed priority order so exactly one action is selected:

1. **Conflicting evidence -> ``request_human_review``** (Req 23.4). The frame (or
   the accumulated trace) disagrees with itself: at least one region reads as
   clearly empty while another reads as clearly filled, by a margin of at least
   ``cfg.conflict_margin``. A self-contradicting input is escalated to a human
   rather than guessed at.
2. **Poor image quality -> ``request_better_image``**. A quality problem (very
   low contrast / blur) is signalled either by an explicit quality flag on the
   finding/region ``alt_metric`` or by *every* notable region being measured at
   very low confidence (``uncertainty_value`` at/below
   :data:`_POOR_QUALITY_CONFIDENCE`). A better image is requested before
   spending the evidence budget on an unreadable one. For a still image with no
   further frames to sample, this is the evidence-gathering action an ambiguous
   finding falls back to.
3. **High uncertainty on a specific, not-yet-cropped region -> ``inspect_crop``**
   (Req 7.4, 23.2). "High uncertainty" means *low measurement confidence*: a
   notable region (empty-space ratio at/above
   ``cfg ... empty_space_ratio_threshold``, read from the vision config) whose
   ``uncertainty_value`` is below ``cfg.sufficient_uncertainty``. The region
   chosen is the most-uncertain such region that has **not** already been
   cropped earlier in the trace (loop guard), and the crop ``args`` identify it
   by ``region_id`` and ``bounding_box`` so the next perception step acts on
   exactly that region.
4. **Ambiguous finding with frames still to sample (video) ->
   ``analyze_another_frame``** (Req 23.3). When the finding is ambiguous (a
   notable-but-not-confident region remains) and the source is video with
   unsampled frames left within the iteration budget, gather another frame
   rather than finalize.
5. **Evidence sufficient, nothing notable, or the iteration budget is spent ->
   ``finalize_report``** (Req 6.7, 6.8). This is the terminal default.

"Sufficiency" is defined concretely (the review flagged it as underspecified):
a finding is *sufficient to finalize* when it has no **open** probe left - that
is, no notable region whose confidence is below ``cfg.sufficient_uncertainty``
and that has not already been cropped - and it is not conflicting and not of
poor quality. Equivalently, every notable region is either already confident
(``uncertainty_value >= cfg.sufficient_uncertainty``) or has already been
inspected via a crop recorded in the trace. See :func:`_is_sufficient`.
"""

from __future__ import annotations

from typing import Any

from shelfsentry.agent.decision import is_allowed_action
from shelfsentry.config.settings import AgentConfig, VisionConfig
from shelfsentry.data.contracts import (
    AgentDecision,
    CandidateRegion,
    TraceEntry,
    VisionFinding,
)

__all__ = ["FallbackPolicy"]


# ---------------------------------------------------------------------------
# Fixed, documented thresholds (no magic numbers in the branch logic).
# ---------------------------------------------------------------------------

#: Default empty-space ratio at/above which a region counts as "notable"
#: (looks empty enough to be worth probing). This mirrors
#: :attr:`VisionConfig.empty_space_ratio_threshold` and is used as the fallback
#: when no ``VisionConfig`` is supplied. Fixed for determinism (Req 5.1).
_DEFAULT_EMPTY_RATIO_THRESHOLD: float = VisionConfig().empty_space_ratio_threshold

#: Measurement confidence at/below which a region is treated as "poor quality"
#: (so unreadable that the right move is to ask for a better image rather than
#: to crop or finalize). Distinct from, and stricter than, the general
#: ambiguity bar: this is the low-contrast / blur floor. Fixed for determinism.
_POOR_QUALITY_CONFIDENCE: float = 0.15

#: Keys on a region/finding ``alt_metric`` mapping that, when truthy, are an
#: explicit "poor image quality" signal from the vision stage (e.g. a blur or
#: low-contrast flag). Checked in a fixed, sorted order for determinism.
_QUALITY_FLAG_KEYS: tuple[str, ...] = (
    "blur",
    "blurry",
    "low_contrast",
    "poor_quality",
    "quality_poor",
)


class FallbackPolicy:
    """Deterministic, pure rule policy implementing ``DecisionModel`` (Req 7.4).

    Construct with an optional :class:`VisionConfig` so the "notable region"
    threshold matches the vision pipeline that produced the finding; when
    omitted, the documented default
    (:data:`_DEFAULT_EMPTY_RATIO_THRESHOLD`) is used. The policy holds no
    mutable state and reads no clock or RNG, so it is safe to reuse across
    inspections and its :meth:`decide` is referentially transparent.
    """

    def __init__(self, vision_cfg: VisionConfig | None = None) -> None:
        self._empty_ratio_threshold: float = (
            vision_cfg.empty_space_ratio_threshold
            if vision_cfg is not None
            else _DEFAULT_EMPTY_RATIO_THRESHOLD
        )

    # -- public API -------------------------------------------------------

    def decide(
        self,
        finding: VisionFinding,
        trace: list[TraceEntry],
        cfg: AgentConfig,
    ) -> AgentDecision:
        """Select the next action deterministically from ``(finding, trace, cfg)``.

        Pure function (Req 7.4): same inputs -> same :class:`AgentDecision`,
        including identical ``action``, ``args``, and ``rationale``. The
        returned decision records ``decided_by="fallback"``; the loop sets
        ``used_fallback``/``fallback_reason`` separately when it engages this
        policy due to a model error (Req 7.2).
        """
        iteration = self._next_iteration(trace)
        action, args, rationale = self._select(finding, trace, cfg)

        # Internal consistency guard: the branch logic must only ever emit one
        # of the fixed five actions (Req 6.1). This never fires in practice; it
        # turns any future bug into a loud failure instead of a silent one.
        if not is_allowed_action(action):  # pragma: no cover - defensive
            raise ValueError(
                f"FallbackPolicy produced out-of-set action {action!r}."
            )

        return AgentDecision(
            decision_id=f"fallback-{iteration}",
            iteration=iteration,
            action=action,  # type: ignore[arg-type]  # guarded above
            rationale=rationale,
            args=args,
            decided_by="fallback",
        )

    # -- branch selection (the reference logic) ---------------------------

    def _select(
        self,
        finding: VisionFinding,
        trace: list[TraceEntry],
        cfg: AgentConfig,
    ) -> tuple[str, dict[str, Any], str]:
        """Return ``(action, args, rationale)`` per the fixed-priority rules.

        The order matters and is documented at module level: conflict first
        (safety), then poor quality, then an open high-uncertainty crop, then
        another video frame for an ambiguous finding, then finalize.
        """
        regions = finding.candidate_regions
        notable = self._notable_regions(regions)

        # 1. Conflicting evidence -> human review (Req 23.4). Highest priority:
        #    a self-contradicting input is never resolved by guessing.
        if self._has_conflict(regions, cfg):
            return (
                "request_human_review",
                {},
                (
                    "Conflicting evidence: at least one region reads as clearly "
                    "empty while another reads as clearly filled (disagreement "
                    f">= conflict_margin {cfg.conflict_margin}); escalating to "
                    "human review rather than guessing."
                ),
            )

        # 2. Poor image quality -> request a better image. A quality problem
        #    makes both cropping and finalizing unreliable.
        if notable and self._is_poor_quality(finding, notable):
            return (
                "request_better_image",
                {},
                (
                    "Poor image quality: notable regions are unreadable (an "
                    "explicit quality flag, or confidence at/below "
                    f"{_POOR_QUALITY_CONFIDENCE} across notable regions); "
                    "requesting a better image before spending the evidence "
                    "budget."
                ),
            )

        # 3. High uncertainty on a specific, not-yet-cropped region ->
        #    inspect_crop that region (Req 7.4, 23.2). "High uncertainty" ==
        #    low measurement confidence (below cfg.sufficient_uncertainty).
        cropped = self._cropped_region_ids(trace)
        target = self._uncertain_uncropped_region(notable, cfg, cropped)
        if target is not None:
            box = target.bounding_box
            return (
                "inspect_crop",
                {
                    "region_id": target.region_id,
                    "bounding_box": {
                        "x": box.x,
                        "y": box.y,
                        "w": box.w,
                        "h": box.h,
                    },
                },
                (
                    f"High uncertainty on region {target.region_id!r} "
                    f"(confidence {target.uncertainty_value:.3f} < "
                    f"sufficient_uncertainty {cfg.sufficient_uncertainty}) and "
                    "not yet cropped; inspecting a crop to gather sharper "
                    "evidence for that region."
                ),
            )

        # 4. Ambiguous finding + unsampled frames remain (video) ->
        #    analyze_another_frame (Req 23.3). Reached only when the sole open
        #    probe is a region already cropped (so step 3 did not fire) yet the
        #    finding is still ambiguous: gather a fresh frame instead.
        if self._is_ambiguous(notable, cfg) and self._frames_remain(
            finding, trace, cfg
        ):
            return (
                "analyze_another_frame",
                {},
                (
                    "Ambiguous video finding with unsampled frames remaining "
                    "within the iteration budget; analyzing another frame to "
                    "gather more evidence rather than finalizing."
                ),
            )

        # 5. Terminal default: sufficient, nothing notable, or budget spent.
        return (
            "finalize_report",
            {},
            self._finalize_rationale(finding, notable, trace, cfg),
        )

    # -- predicates & helpers (all pure) ----------------------------------

    def _notable_regions(
        self, regions: list[CandidateRegion]
    ) -> list[CandidateRegion]:
        """Regions that look empty enough to be worth probing.

        A region is notable when its measured ``empty_space_ratio`` is at or
        above the configured empty-space threshold. Regions using an
        ``alt_metric`` (``empty_space_ratio is None``) are not counted as
        notable empties here. Order follows the finding's region order (stable),
        so the result is deterministic.
        """
        return [
            r
            for r in regions
            if r.empty_space_ratio is not None
            and r.empty_space_ratio >= self._empty_ratio_threshold
        ]

    def _has_conflict(
        self, regions: list[CandidateRegion], cfg: AgentConfig
    ) -> bool:
        """True when a clearly-empty and a clearly-filled region coexist.

        Conflict (Req 23.4) is measured on ``empty_space_ratio``: a region is
        clearly empty at/above ``empty_ratio_threshold`` and clearly filled
        at/below ``1 - empty_ratio_threshold``. We additionally require the gap
        between the most-empty and most-filled region to be at least
        ``cfg.conflict_margin`` so a borderline spread does not trip human
        review. Regions on an ``alt_metric`` are ignored (no comparable ratio).
        """
        ratios = [
            r.empty_space_ratio
            for r in regions
            if r.empty_space_ratio is not None
        ]
        if len(ratios) < 2:
            return False

        filled_threshold = 1.0 - self._empty_ratio_threshold
        has_empty = any(v >= self._empty_ratio_threshold for v in ratios)
        has_filled = any(v <= filled_threshold for v in ratios)
        if not (has_empty and has_filled):
            return False

        spread = max(ratios) - min(ratios)
        return spread >= cfg.conflict_margin

    def _is_poor_quality(
        self, finding: VisionFinding, notable: list[CandidateRegion]
    ) -> bool:
        """True when a quality problem (low contrast / blur) is signalled.

        Two deterministic signals, either of which is sufficient:

        * an explicit truthy quality flag on any region's ``alt_metric`` (keys
          in :data:`_QUALITY_FLAG_KEYS`), or
        * *every* notable region measured at or below
          :data:`_POOR_QUALITY_CONFIDENCE` (uniformly unreadable confidence).
        """
        for region in finding.candidate_regions:
            if self._alt_metric_flags_quality(region.alt_metric):
                return True
        if notable and all(
            r.uncertainty_value <= _POOR_QUALITY_CONFIDENCE for r in notable
        ):
            return True
        return False

    @staticmethod
    def _alt_metric_flags_quality(alt_metric: dict[str, Any] | None) -> bool:
        """Whether an ``alt_metric`` mapping carries a truthy quality flag.

        Keys are checked in a fixed, sorted order so the scan is independent of
        dict insertion order (determinism, Req 7.4).
        """
        if not alt_metric:
            return False
        for key in sorted(_QUALITY_FLAG_KEYS):
            if key in alt_metric and bool(alt_metric[key]):
                return True
        return False

    def _uncertain_uncropped_region(
        self,
        notable: list[CandidateRegion],
        cfg: AgentConfig,
        cropped_region_ids: frozenset[str],
    ) -> CandidateRegion | None:
        """Pick the most-uncertain notable region not already cropped.

        "Most uncertain" == lowest ``uncertainty_value`` (lowest measurement
        confidence) among notable regions whose confidence is below
        ``cfg.sufficient_uncertainty`` and whose ``region_id`` has not already
        appeared as an ``inspect_crop`` in the trace (the loop guard that
        prevents re-cropping the same region forever).

        Ties break on ``region_id`` so the choice is deterministic regardless of
        input ordering. Returns ``None`` when no such open region exists.
        """
        candidates = [
            r
            for r in notable
            if r.uncertainty_value < cfg.sufficient_uncertainty
            and r.region_id not in cropped_region_ids
        ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda r: (r.uncertainty_value, r.region_id),
        )

    def _is_ambiguous(
        self, notable: list[CandidateRegion], cfg: AgentConfig
    ) -> bool:
        """True when a notable region remains below the sufficiency bar.

        Ambiguity is the complement of sufficiency on the notable set: at least
        one notable region whose confidence is below
        ``cfg.sufficient_uncertainty``. (Whether that region was already cropped
        is handled by the crop step's loop guard, not here.)
        """
        return any(
            r.uncertainty_value < cfg.sufficient_uncertainty for r in notable
        )

    def _frames_remain(
        self,
        finding: VisionFinding,
        trace: list[TraceEntry],
        cfg: AgentConfig,
    ) -> bool:
        """True when the source is video and more frames may still be sampled.

        The policy only sees ``finding``/``trace``/``cfg``, so "unsampled frames
        remain" is read from the trace: the source must be video-derived
        (``source == "video_frame"`` or the finding carries a ``frame_id``), and
        the number of ``analyze_another_frame`` decisions already recorded must
        be below the iteration budget ``cfg.max_iterations``. This keeps the
        another-frame action bounded and deterministic without needing the raw
        frame count here.
        """
        is_video = finding.source == "video_frame" or finding.frame_id is not None
        if not is_video:
            return False
        frames_sampled = sum(
            1
            for entry in trace
            if isinstance(entry, AgentDecision)
            and entry.action == "analyze_another_frame"
        )
        return frames_sampled < cfg.max_iterations

    @staticmethod
    def _cropped_region_ids(trace: list[TraceEntry]) -> frozenset[str]:
        """Region ids already inspected via an ``inspect_crop`` in the trace.

        Used as the loop guard so the policy never re-crops a region it has
        already cropped. Reads the ``region_id`` from each ``inspect_crop``
        decision's ``args``; entries without a usable id are ignored.
        """
        ids: set[str] = set()
        for entry in trace:
            if (
                isinstance(entry, AgentDecision)
                and entry.action == "inspect_crop"
            ):
                region_id = entry.args.get("region_id")
                if isinstance(region_id, str) and region_id:
                    ids.add(region_id)
        return frozenset(ids)

    def _is_sufficient(
        self,
        notable: list[CandidateRegion],
        trace: list[TraceEntry],
        cfg: AgentConfig,
    ) -> bool:
        """Concrete sufficiency predicate (the previously underspecified point).

        Evidence is sufficient to finalize when there is **no open probe left**:
        every notable region is either already confident
        (``uncertainty_value >= cfg.sufficient_uncertainty``) or has already
        been cropped earlier in the trace. With no notable region at all, the
        finding is trivially sufficient. This is exactly the condition under
        which none of the evidence-gathering branches apply.
        """
        if not notable:
            return True
        cropped = self._cropped_region_ids(trace)
        return all(
            r.uncertainty_value >= cfg.sufficient_uncertainty
            or r.region_id in cropped
            for r in notable
        )

    def _finalize_rationale(
        self,
        finding: VisionFinding,
        notable: list[CandidateRegion],
        trace: list[TraceEntry],
        cfg: AgentConfig,
    ) -> str:
        """Explain which terminal condition produced ``finalize_report``."""
        if not notable:
            return (
                "Nothing notable: no region reaches the empty-space threshold, "
                "so there is no open probe; finalizing the report."
            )
        if self._is_sufficient(notable, trace, cfg):
            return (
                "Evidence sufficient: every notable region is either confident "
                f"(>= sufficient_uncertainty {cfg.sufficient_uncertainty}) or "
                "already inspected via a crop in the trace; finalizing."
            )
        # An open probe exists but no evidence-gathering action is available
        # (e.g. a still image whose only open region was already cropped, or the
        # frame budget is spent). The bounded loop finalizes rather than loop.
        return (
            "Iteration budget spent or no further evidence action available for "
            "the remaining ambiguous region(s); finalizing within the bounded "
            f"loop (max_iterations {cfg.max_iterations})."
        )

    @staticmethod
    def _next_iteration(trace: list[TraceEntry]) -> int:
        """Iteration index for the decision about to be recorded.

        Equal to the count of :class:`AgentDecision` entries already in the
        trace, matching the loop's zero-based iteration counter. Pure function
        of the trace (no clock), so it is deterministic.
        """
        return sum(1 for entry in trace if isinstance(entry, AgentDecision))
