"""Property-based tests for the deterministic :class:`FallbackPolicy` (task 6.3).

These Hypothesis properties complement (and do not duplicate) the example-based
tests in ``test_fallback_policy.py``. They exercise the policy as the pure
function it is, building :class:`VisionFinding` inputs directly from the typed
contracts (no ``cv2``, no AWS) over a wide space of generated inputs.

Properties implemented here:

* **Property 17: Fallback policy is deterministic** (Validates: Requirements 7.4)
  - same ``(finding, trace, cfg)`` -> identical action/args/rationale and
    identical ``canonical_json`` of the produced :class:`AgentDecision`, both on
    one instance called twice and across two separate instances;
  - the result is independent of ``candidate_regions`` ordering: shuffling the
    regions yields the SAME action and (for ``inspect_crop``) the SAME chosen
    ``region_id``.

* **Property 19: Ambiguous findings gather more evidence**
  (Validates: Requirements 23.3)
  - a finding constructed to isolate the ambiguity branch (a notable region
    with low measurement confidence, not conflicting and not poor-quality)
    selects an evidence-gathering action (``inspect_crop``, or
    ``analyze_another_frame`` when the uncertain region is already cropped and
    the source is video) rather than ``finalize_report``.

Every property also asserts the produced action is within the allowed five
action set.
"""

from __future__ import annotations

import random
from typing import Any

from hypothesis import assume, given
from hypothesis import strategies as st

from shelfsentry.agent import AGENT_ACTIONS, FallbackPolicy
from shelfsentry.config.settings import AgentConfig
from shelfsentry.data.contracts import (
    AgentDecision,
    BoundingBox,
    CandidateRegion,
    TraceEntry,
    VisionFinding,
)

# Documented thresholds the branch logic uses (see fallback.py / settings.py):
#   empty_space_ratio_threshold ("notable") default ...... 0.6
#   sufficient_uncertainty (ambiguity bar) default ........ 0.7
#   conflict_margin default ............................... 0.25
#   poor-quality confidence floor (fixed) ................. 0.15
_EMPTY_THRESHOLD = 0.6
_POOR_QUALITY_FLOOR = 0.15
_SUFFICIENT_UNCERTAINTY_DEFAULT = 0.7
_QUALITY_FLAG_KEYS = (
    "blur",
    "blurry",
    "low_contrast",
    "poor_quality",
    "quality_poor",
)


# ---------------------------------------------------------------------------
# builders (typed contracts only; no cv2 / AWS)
# ---------------------------------------------------------------------------


def _region(
    region_id: str,
    *,
    empty_space_ratio: float | None,
    uncertainty_value: float,
    alt_metric: dict[str, Any] | None = None,
    x: int = 0,
    y: int = 0,
    w: int = 10,
    h: int = 10,
) -> CandidateRegion:
    return CandidateRegion(
        region_id=region_id,
        bounding_box=BoundingBox(x=x, y=y, w=w, h=h),
        empty_space_ratio=empty_space_ratio,
        alt_metric=alt_metric,
        heuristic_score=0.5,
        uncertainty_value=uncertainty_value,
        visual_caveat="estimate only; may be shadow/occlusion/dividers/quality",
    )


def _finding(
    regions: list[CandidateRegion],
    *,
    source: str = "image",
    frame_id: str | None = None,
) -> VisionFinding:
    return VisionFinding(
        finding_id="f1",
        inspection_id="insp-1",
        source=source,  # type: ignore[arg-type]
        frame_id=frame_id,
        opencv_version="5.0.0",
        candidate_regions=regions,
        uncertainty_interpretations={},
        annotated_output_ref="ref://annotated",
        suggested_follow_up="finalize_report",
    )


def _crop_entry(region_id: str, *, iteration: int = 0) -> AgentDecision:
    """A trace entry recording that ``region_id`` was already cropped."""
    return AgentDecision(
        decision_id=f"prev-{iteration}",
        iteration=iteration,
        action="inspect_crop",
        rationale="prior crop",
        args={"region_id": region_id},
        decided_by="fallback",
    )


def _another_frame_entry(iteration: int) -> AgentDecision:
    return AgentDecision(
        decision_id=f"prev-{iteration}",
        iteration=iteration,
        action="analyze_another_frame",
        rationale="prior frame",
        decided_by="fallback",
    )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# A region id drawn from a small, distinct alphabet so a region set has unique
# ids (the policy identifies regions by id; collisions would be ambiguous).
_region_ids = st.sampled_from([f"r{i}" for i in range(8)])

# alt_metric that may or may not carry a quality flag. Quality-flag keys are
# only ever emitted here with a boolean/truthy value so the generator can both
# trip and not-trip the poor-quality branch.
_alt_metric = st.one_of(
    st.none(),
    st.dictionaries(
        keys=st.sampled_from(list(_QUALITY_FLAG_KEYS) + ["mean_intensity", "note"]),
        values=st.one_of(st.booleans(), st.floats(0.0, 1.0), st.text(max_size=4)),
        max_size=3,
    ),
)


@st.composite
def _regions(draw: st.DrawFn) -> list[CandidateRegion]:
    """A set of 0-5 candidate regions with unique ids spanning the input space.

    Each region independently varies empty_space_ratio (including ``None`` for
    alt-metric-only regions), uncertainty_value, alt_metric quality flags, and
    bounding box, so the generated space covers notable/non-notable,
    confident/uncertain, conflicting, and poor-quality combinations.
    """
    count = draw(st.integers(min_value=0, max_value=5))
    ids = draw(
        st.lists(_region_ids, min_size=count, max_size=count, unique=True)
    )
    regions: list[CandidateRegion] = []
    for rid in ids:
        ratio = draw(
            st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0))
        )
        unc = draw(st.floats(min_value=0.0, max_value=1.0))
        alt = draw(_alt_metric)
        x = draw(st.integers(min_value=0, max_value=50))
        y = draw(st.integers(min_value=0, max_value=50))
        w = draw(st.integers(min_value=1, max_value=40))
        h = draw(st.integers(min_value=1, max_value=40))
        regions.append(
            _region(
                rid,
                empty_space_ratio=ratio,
                uncertainty_value=unc,
                alt_metric=alt,
                x=x,
                y=y,
                w=w,
                h=h,
            )
        )
    return regions


_sources = st.sampled_from(["image", "video_frame", "crop", "frame_compare"])


@st.composite
def _findings(draw: st.DrawFn) -> VisionFinding:
    regions = draw(_regions())
    source = draw(_sources)
    frame_id = draw(st.one_of(st.none(), st.just("frame_000001")))
    return _finding(regions, source=source, frame_id=frame_id)


@st.composite
def _traces(draw: st.DrawFn, finding: VisionFinding) -> list[TraceEntry]:
    """A plausible prior trace: some regions already cropped + prior frames.

    Built from the finding's own region ids so the cropped-region loop guard is
    genuinely exercised (crops of ids that exist in the finding), plus zero or
    more prior ``analyze_another_frame`` entries. Iterations are assigned in
    order so the trace is internally consistent.
    """
    existing_ids = [r.region_id for r in finding.candidate_regions]
    cropped = (
        draw(st.lists(st.sampled_from(existing_ids), max_size=3, unique=True))
        if existing_ids
        else []
    )
    n_frames = draw(st.integers(min_value=0, max_value=3))
    trace: list[TraceEntry] = []
    it = 0
    for rid in cropped:
        trace.append(_crop_entry(rid, iteration=it))
        it += 1
    for _ in range(n_frames):
        trace.append(_another_frame_entry(it))
        it += 1
    return trace


_configs = st.builds(
    AgentConfig,
    max_iterations=st.integers(min_value=1, max_value=8),
    sufficient_uncertainty=st.floats(min_value=0.1, max_value=1.0),
    conflict_margin=st.floats(min_value=0.0, max_value=1.0),
)


# ---------------------------------------------------------------------------
# Property 17: Fallback policy is deterministic
# Validates: Requirements 7.4
# ---------------------------------------------------------------------------


@given(finding=_findings(), cfg=_configs, data=st.data())
def test_property_17_fallback_policy_is_deterministic(
    finding: VisionFinding, cfg: AgentConfig, data: st.DataObject
) -> None:
    """Property 17: Fallback policy is deterministic.

    Validates: Requirements 7.4

    For any generated ``(finding, trace, cfg)``:
      * calling ``decide`` twice on one instance, and once on a fresh instance,
        yields identical action, args, rationale, and canonical JSON;
      * the decision is independent of ``candidate_regions`` ordering: shuffling
        the regions keeps the same action and (for ``inspect_crop``) the same
        chosen ``region_id``.
    """
    trace = data.draw(_traces(finding))

    policy = FallbackPolicy()
    d1 = policy.decide(finding, trace, cfg)
    d2 = policy.decide(finding, trace, cfg)
    d3 = FallbackPolicy().decide(finding, trace, cfg)

    # Allowed-action invariant (Req 6.1): the action is one of the fixed five.
    assert d1.action in AGENT_ACTIONS

    # Deterministic: identical selection on repeated / separate instances.
    assert d1.action == d2.action == d3.action
    assert d1.args == d2.args == d3.args
    assert d1.rationale == d2.rationale == d3.rationale
    assert d1.canonical_json() == d2.canonical_json() == d3.canonical_json()

    # Order-independence: shuffle the regions and re-decide. The trace is built
    # from region ids (not positions), so it remains valid under a shuffle.
    shuffled = list(finding.candidate_regions)
    rng = random.Random(1234)
    rng.shuffle(shuffled)
    shuffled_finding = _finding(
        shuffled, source=finding.source, frame_id=finding.frame_id
    )
    d_shuffled = policy.decide(shuffled_finding, trace, cfg)

    assert d_shuffled.action == d1.action
    if d1.action == "inspect_crop":
        # The most-uncertain uncropped region is picked regardless of ordering.
        assert d_shuffled.args.get("region_id") == d1.args.get("region_id")
        assert d_shuffled.args == d1.args


# ---------------------------------------------------------------------------
# Property 19: Ambiguous findings gather more evidence
# Validates: Requirements 23.3
# ---------------------------------------------------------------------------

_EVIDENCE_ACTIONS = {"inspect_crop", "analyze_another_frame"}


@st.composite
def _ambiguous_case(
    draw: st.DrawFn,
) -> tuple[VisionFinding, list[TraceEntry], AgentConfig, bool]:
    """Build a finding that isolates the ambiguity branch.

    Guarantees, using the documented default thresholds
    (sufficient_uncertainty 0.7, empty threshold 0.6, poor-quality floor 0.15,
    conflict_margin 0.25):

    * at least one *notable* region (empty_space_ratio >= 0.6) whose measurement
      confidence is strictly below ``sufficient_uncertainty`` and strictly above
      the poor-quality floor -> genuinely ambiguous, not poor-quality;
    * NO clearly-filled region (every ratio-bearing region is >= 0.6) and no
      alt_metric quality flags -> neither the conflict nor the poor-quality
      branch can fire;
    * optionally, the single uncertain region is already cropped and the source
      is video with frames remaining -> the open probe must be gathered via
      ``analyze_another_frame`` instead of ``inspect_crop``.

    Returns ``(finding, trace, cfg, expect_another_frame)``.
    """
    cfg = AgentConfig()  # documented defaults
    suff = cfg.sufficient_uncertainty  # 0.7

    # The ambiguous region: notable and uncertain, but above the poor-quality
    # floor so the poor-quality branch does not fire.
    unc = draw(
        st.floats(
            min_value=_POOR_QUALITY_FLOOR + 0.01,
            max_value=suff - 0.01,
            allow_nan=False,
        )
    )
    ratio = draw(st.floats(min_value=_EMPTY_THRESHOLD, max_value=1.0))
    ambiguous = _region(
        "r0", empty_space_ratio=ratio, uncertainty_value=unc, alt_metric=None
    )

    regions: list[CandidateRegion] = [ambiguous]

    # Optional extra regions. To keep the conflict branch from firing, every
    # ratio-bearing extra region is also "empty" (>= 0.6); alt-metric-only
    # regions (ratio=None) are allowed but never carry a quality flag.
    n_extra = draw(st.integers(min_value=0, max_value=2))
    for i in range(n_extra):
        use_alt_only = draw(st.booleans())
        if use_alt_only:
            extra = _region(
                f"r{i + 1}",
                empty_space_ratio=None,
                uncertainty_value=draw(st.floats(0.2, 1.0)),
                alt_metric={"mean_intensity": draw(st.floats(0.0, 1.0))},
            )
        else:
            # Empty + confident, so it adds no new open probe and cannot be a
            # "clearly filled" conflict partner.
            extra = _region(
                f"r{i + 1}",
                empty_space_ratio=draw(
                    st.floats(_EMPTY_THRESHOLD, 1.0)
                ),
                uncertainty_value=draw(st.floats(suff, 1.0)),
                alt_metric=None,
            )
        regions.append(extra)

    # Decide whether to steer into the analyze_another_frame branch: crop the
    # only uncertain region (r0) and make the source a video with frames left.
    expect_another_frame = draw(st.booleans())
    if expect_another_frame:
        source = "video_frame"
        frame_id = "frame_000001"
        trace: list[TraceEntry] = [_crop_entry("r0", iteration=0)]
    else:
        source = draw(st.sampled_from(["image", "video_frame"]))
        frame_id = None
        trace = []

    finding = _finding(regions, source=source, frame_id=frame_id)
    return finding, trace, cfg, expect_another_frame


@given(case=_ambiguous_case())
def test_property_19_ambiguous_findings_gather_more_evidence(
    case: tuple[VisionFinding, list[TraceEntry], AgentConfig, bool],
) -> None:
    """Property 19: Ambiguous findings gather more evidence.

    Validates: Requirements 23.3

    An ambiguous finding (a notable region with low measurement confidence, and
    deliberately neither conflicting nor poor-quality) must select an
    evidence-gathering action rather than ``finalize_report``: ``inspect_crop``
    of the uncertain region, or ``analyze_another_frame`` when that region is
    already cropped and the source is video with frames remaining.
    """
    finding, trace, cfg, expect_another_frame = case
    decision = FallbackPolicy().decide(finding, trace, cfg)

    # Allowed-action invariant (Req 6.1).
    assert decision.action in AGENT_ACTIONS

    # An ambiguous finding never finalizes; it gathers more evidence.
    assert decision.action != "finalize_report"
    assert decision.action in _EVIDENCE_ACTIONS

    if expect_another_frame:
        # The only open probe (r0) is already cropped and the source is video
        # with frames remaining -> sample another frame.
        assert decision.action == "analyze_another_frame"
    else:
        # The uncertain region has not been cropped -> crop it directly.
        assert decision.action == "inspect_crop"
        assert decision.args.get("region_id") == "r0"
