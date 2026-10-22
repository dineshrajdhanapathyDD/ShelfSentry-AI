"""The evaluation runner (task 14.1; Req 25.2, 25.3, 25.4).

:func:`run_evaluation` runs the **real** local active-perception pipeline over
the reproducible labeled set (:mod:`shelfsentry.eval.labeled_set`) and reports
**only measured** results in a single
:class:`~shelfsentry.data.contracts.EvaluationResult` record (Req 25.2, 25.4).

What is measured
----------------
For every labeled sample the runner:

1. renders the sample's synthetic PNG (deterministic);
2. runs the full local pipeline
   (:func:`shelfsentry.pipeline.run_local_inspection`) against a fresh
   :class:`~shelfsentry.data.storage.LocalStorageAdapter` in a temporary
   directory - real OpenCV 5 vision through the MCP tools plus the bounded
   deterministic agent loop, no AWS and no network;
3. also derives the vision-only detection via ``analyze_image_bytes`` +
   ``build_finding`` to decide the measured ``has_gap`` boolean (see below);
4. times the pipeline run with :func:`time.perf_counter`.

Metrics (each measured, none fabricated; Req 25.4):

* **Detection (false positives / false negatives).** The measured per-image
  signal is ``has_gap`` = *does the vision finding flag at least one candidate
  region whose measured ``empty_space_ratio`` is at or above the configured
  ``empty_space_ratio_threshold``*. Compared against the sample's
  ``expected_has_gap`` ground truth:

  - a **false positive** is a sample labeled ``expected_has_gap=False`` for
    which the system flagged a gap;
  - a **false negative** is a sample labeled ``expected_has_gap=True`` for which
    the system flagged no gap.

  Both are collected as :class:`~shelfsentry.data.contracts.EvalExample` records
  carrying the expected and measured values.

* **Agent decision correctness.** EXACTLY what this number measures (Req 25.4,
  honesty): over the subset of samples whose ``expected_action`` is not
  ``None``, it is the fraction whose pipeline-produced report has a *terminal
  decision action* equal to the labeled ``expected_action``. The terminal action
  is the ``action`` of the last :class:`~shelfsentry.data.contracts.AgentDecision`
  in the report's ``decision_trace``. It therefore measures *whether the
  deterministic active-perception loop routes each labeled scene to the intended
  terminal action* - it is NOT a measure of a learned/LLM policy and does not
  cover unlabeled samples. When no sample carries an expected action the metric
  is reported as ``0.0`` and that fact is added to the documented limitations
  (so the ``0.0`` is never mistaken for a measured-poor result).

* **Latency (p50 / p95).** Measured wall-clock milliseconds per pipeline run via
  :func:`time.perf_counter`, reduced to the 50th and 95th percentiles with
  linear interpolation. Environment-dependent (documented).

Honesty guardrails (Req 25.4, 27.2). The result is a measured-only
:class:`EvaluationResult` (``measured_only`` is structurally ``True``); it is
marked ``is_demo=True`` by default because the labels are synthetic; and it
always carries non-empty ``documented_limitations`` drawn from
:data:`shelfsentry.eval.labeled_set.SYNTHETIC_LABEL_LIMITATIONS` plus any
run-specific caveat. Nothing here invents a metric or a confidence score.

Determinism. The vision detection and the agent routing are pure functions of
the (deterministic) sample bytes and fixed config, so the FP/FN classification
and the decision-correctness value are identical across runs. Only the latency
percentiles vary run to run (wall-clock), which is why callers must never assert
latency equality.
"""

from __future__ import annotations

import tempfile
import time
import uuid

from shelfsentry.config.settings import AgentConfig, VisionConfig
from shelfsentry.data.contracts import (
    AgentDecision,
    EvalExample,
    EvaluationResult,
    InspectionReport,
    VisionFinding,
)
from shelfsentry.data.storage import LocalStorageAdapter
from shelfsentry.eval.labeled_set import (
    SYNTHETIC_LABEL_LIMITATIONS,
    EvalSample,
    labeled_set,
    render_sample_png,
)
from shelfsentry.pipeline import run_local_inspection
from shelfsentry.vision.analyze import analyze_image_bytes
from shelfsentry.vision.scoring import build_finding

__all__ = [
    "SampleOutcome",
    "measure_has_gap",
    "terminal_action",
    "percentile",
    "run_evaluation",
]


# ---------------------------------------------------------------------------
# Per-sample measured outcome (internal intermediate, measured only)
# ---------------------------------------------------------------------------


class SampleOutcome:
    """The measured outcome of running one labeled sample through the pipeline.

    Every field is a measurement or a copy of the sample's label - nothing is
    fabricated. Kept as a plain object (not a contract) because it is an
    internal intermediate the runner aggregates into the
    :class:`EvaluationResult`.
    """

    __slots__ = (
        "sample",
        "inspection_id",
        "measured_has_gap",
        "measured_action",
        "latency_ms",
    )

    def __init__(
        self,
        *,
        sample: EvalSample,
        inspection_id: str,
        measured_has_gap: bool,
        measured_action: str | None,
        latency_ms: float,
    ) -> None:
        self.sample = sample
        self.inspection_id = inspection_id
        self.measured_has_gap = measured_has_gap
        self.measured_action = measured_action
        self.latency_ms = latency_ms


# ---------------------------------------------------------------------------
# Measured signals
# ---------------------------------------------------------------------------


def measure_has_gap(finding: VisionFinding, *, empty_ratio_threshold: float) -> bool:
    """Measured per-image ``has_gap`` signal (see module docstring).

    ``True`` iff the finding carries at least one candidate region whose
    measured ``empty_space_ratio`` is at or above ``empty_ratio_threshold`` -
    i.e. the vision heuristic flagged an empty region worth attention. This is a
    pure function of the finding's measured ratios and the configured threshold;
    nothing is estimated beyond the already-measured ratio.
    """
    for region in finding.candidate_regions:
        ratio = region.empty_space_ratio
        if ratio is not None and ratio >= empty_ratio_threshold:
            return True
    return False


def terminal_action(report: InspectionReport) -> str | None:
    """Return the terminal agent action recorded in a report's trace.

    The terminal action is the ``action`` of the last
    :class:`AgentDecision` in ``report.decision_trace``. Returns ``None`` when
    the trace holds no decision (which the bounded loop does not produce, but is
    handled defensively so the metric never raises).
    """
    last_action: str | None = None
    for entry in report.decision_trace:
        if isinstance(entry, AgentDecision):
            last_action = entry.action
    return last_action


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of ``values`` (``pct`` in ``[0, 100]``).

    Returns ``0.0`` for an empty input. Uses the standard
    ``rank = (n - 1) * pct/100`` interpolation so ``percentile(xs, 50)`` is the
    median and the result is monotonic in ``pct`` (hence ``p50 <= p95`` always
    holds for the same sample set). Operates on a sorted copy, leaving the input
    untouched.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


# ---------------------------------------------------------------------------
# Running one sample (the real pipeline + vision-only detection)
# ---------------------------------------------------------------------------


def _run_one_sample(
    sample: EvalSample,
    *,
    cfg_vision: VisionConfig,
    cfg_agent: AgentConfig,
    adapter: LocalStorageAdapter,
) -> SampleOutcome:
    """Run one labeled sample through the real pipeline and measure it.

    Measures the detection ``has_gap`` from a vision-only finding (so the signal
    is read directly from the OpenCV pass, independent of how the agent loop
    later routes), runs the full local pipeline for the agent terminal action
    and the timed latency, and returns a :class:`SampleOutcome`. All values are
    measured; none are fabricated.
    """
    data = render_sample_png(sample)
    inspection_id = f"{sample.sample_id}-{uuid.uuid4().hex[:8]}"

    # Vision-only detection signal (deterministic, measured). Reuse the real
    # analysis + finding assembly; this is the same OpenCV 5 path the pipeline
    # perceives through, read here directly for the has_gap boolean.
    analysis = analyze_image_bytes(data, cfg_vision)
    vision_finding = build_finding(
        analysis,
        inspection_id=inspection_id,
        finding_id=f"{inspection_id}-vision",
        annotated_output_ref=f"evidence://{inspection_id}/vision",
        source="image",
        cfg=cfg_vision,
    )
    measured_has_gap = measure_has_gap(
        vision_finding, empty_ratio_threshold=cfg_vision.empty_space_ratio_threshold
    )

    # Full local pipeline for the agent terminal action + measured latency.
    start = time.perf_counter()
    report = run_local_inspection(
        data,
        sample.content_type,
        inspection_id=inspection_id,
        adapter=adapter,
        cfg_vision=cfg_vision,
        cfg_agent=cfg_agent,
        mode="local",
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    return SampleOutcome(
        sample=sample,
        inspection_id=inspection_id,
        measured_has_gap=measured_has_gap,
        measured_action=terminal_action(report),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# The public entry point
# ---------------------------------------------------------------------------


def run_evaluation(
    *,
    is_demo: bool = True,
    cfg_vision: VisionConfig | None = None,
    cfg_agent: AgentConfig | None = None,
    eval_run_id: str | None = None,
) -> EvaluationResult:
    """Run the labeled set and return a measured-only ``EvaluationResult``.

    Runs the real local pipeline over every sample from
    :func:`shelfsentry.eval.labeled_set.labeled_set`, measures detection
    false positives / false negatives, agent decision correctness, and latency
    percentiles, and assembles one
    :class:`~shelfsentry.data.contracts.EvaluationResult`. Only measured values
    are reported (Req 25.4, 27.2); the record is ``measured_only`` structurally,
    is marked ``is_demo`` (default ``True`` because the labels are synthetic),
    and always carries non-empty documented limitations (Req 25.3).

    :param is_demo: stamp the record as a demo/synthetic record (Req 14.5).
        Defaults to ``True`` because this set's labels are synthetic own-work,
        not real retail ground truth. Pass ``False`` only when running against a
        real, non-demo labeled set (none ships today).
    :param cfg_vision: vision config; defaults to :class:`VisionConfig`.
    :param cfg_agent: agent config; defaults to :class:`AgentConfig`.
    :param eval_run_id: id for this run; a fresh one is generated when omitted.
    :returns: a measured-only :class:`EvaluationResult`.
    """
    cfg_vision = cfg_vision or VisionConfig()
    cfg_agent = cfg_agent or AgentConfig()
    run_id = eval_run_id or f"eval-{uuid.uuid4().hex[:12]}"

    samples = labeled_set()

    outcomes: list[SampleOutcome] = []
    # One temporary storage root for the whole run; each sample uses a distinct
    # inspection id so their media/reports never collide. Cleaned up on exit.
    with tempfile.TemporaryDirectory(prefix="shelfsentry-eval-") as tmp:
        adapter = LocalStorageAdapter(tmp)
        for sample in samples:
            outcomes.append(
                _run_one_sample(
                    sample,
                    cfg_vision=cfg_vision,
                    cfg_agent=cfg_agent,
                    adapter=adapter,
                )
            )

    # --- detection: false positives / false negatives (measured) ---
    false_positives: list[EvalExample] = []
    false_negatives: list[EvalExample] = []
    for outcome in outcomes:
        sample = outcome.sample
        expected = sample.expected_has_gap
        measured = outcome.measured_has_gap
        if measured and not expected:
            false_positives.append(
                EvalExample(
                    sample_id=sample.sample_id,
                    inspection_id=outcome.inspection_id,
                    expected="has_gap=False",
                    actual="has_gap=True",
                    note=(
                        f"{sample.description} The vision heuristic flagged an "
                        "empty region where none was drawn (false positive)."
                    ),
                )
            )
        elif expected and not measured:
            false_negatives.append(
                EvalExample(
                    sample_id=sample.sample_id,
                    inspection_id=outcome.inspection_id,
                    expected=f"has_gap=True (drew {sample.expected_gap_count} gap(s))",
                    actual="has_gap=False",
                    note=(
                        f"{sample.description} The vision heuristic flagged no "
                        "empty region although a gap was drawn (false negative)."
                    ),
                )
            )

    # --- agent decision correctness (measured over labeled subset) ---
    labeled = [o for o in outcomes if o.sample.expected_action is not None]
    if labeled:
        correct = sum(
            1 for o in labeled if o.measured_action == o.sample.expected_action
        )
        agent_decision_correctness = correct / len(labeled)
    else:  # no labeled expected actions: report 0.0 and document why
        agent_decision_correctness = 0.0

    # --- latency percentiles (measured wall-clock) ---
    latencies = [o.latency_ms for o in outcomes]
    p50 = percentile(latencies, 50.0)
    p95 = percentile(latencies, 95.0)

    # --- documented limitations (always non-empty; Req 25.3) ---
    limitations: list[str] = list(SYNTHETIC_LABEL_LIMITATIONS)
    if not labeled:
        limitations.append(
            "No sample in this set carried an expected agent action, so "
            "agent_decision_correctness is reported as 0.0 by convention and "
            "reflects an empty labeled subset, not a measured-poor policy."
        )
    else:
        limitations.append(
            "Agent decision correctness was measured over "
            f"{len(labeled)} of {len(outcomes)} samples that carry an expected "
            "terminal-action label."
        )

    return EvaluationResult(
        eval_run_id=run_id,
        is_demo=is_demo,
        sample_count=len(samples),
        false_positives=false_positives,
        false_negatives=false_negatives,
        agent_decision_correctness=agent_decision_correctness,
        latency_ms_p50=p50,
        latency_ms_p95=p95,
        documented_limitations=limitations,
        measured_only=True,
    )
