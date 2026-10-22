"""Property test for honest evaluation reporting (task 14.2).

**Property 23: Only measured metrics are reported**

**Validates: Requirements 14.4, 25.4, 27.2**

This property asserts that any ``EvaluationResult`` produced by
:func:`shelfsentry.eval.run_evaluation` is MEASURED-ONLY and internally
consistent with the labeled set - i.e. no number in the result is fabricated.
Every reported metric is shown to be *reproducible from measurement*:

* ``measured_only`` is ``True`` and ``is_demo`` is ``True`` (synthetic labels).
* ``sample_count`` equals ``len(labeled_set())``.
* Every false-positive / false-negative example references a ``sample_id`` that
  exists in the labeled set, and the FP and FN id sets are disjoint.
* ``agent_decision_correctness`` is in ``[0, 1]`` AND equals, to float
  tolerance, the value recomputed *independently from measurement*: we re-render
  each labeled sample with :func:`render_sample_png`, re-run the real local
  pipeline with :func:`shelfsentry.pipeline.run_local_inspection`, and read the
  terminal action with :func:`shelfsentry.eval.terminal_action`, then divide the
  count of correct terminal actions by the number of labeled samples. The
  reported correctness is therefore proven reproducible from measurement, not
  invented. The same independent re-measurement reproduces the FP/FN id sets.
* ``latency_ms_p50 <= latency_ms_p95`` and both are ``>= 0``.
* ``documented_limitations`` is non-empty and includes the synthetic-label
  limitation.
* The measured classification is deterministic across runs (identical FP/FN ids
  and identical correctness; latency may differ and is never asserted equal).

Because every full-pipeline run is real OpenCV plus the real agent loop, the
independent reference is measured ONCE at module import and reused, and the
Hypothesis-driven variation over the honest knobs is capped low
(``max_examples`` small) so the suite stays fast. The variation deliberately
keeps ``is_demo`` semantics honest (synthetic labels => ``is_demo=True``) and
only varies the run id, which must not change any measured metric.

These tests use the REAL pipeline (no mocks) and intentionally do NOT duplicate
the example-based assertions in ``test_labeled_set_and_runner.py``; they add the
independent-recomputation (anti-fabrication) property on top.
"""

from __future__ import annotations

import tempfile

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shelfsentry.config.settings import AgentConfig, VisionConfig
from shelfsentry.data.contracts import EvalExample
from shelfsentry.data.storage import LocalStorageAdapter
from shelfsentry.eval import (
    SYNTHETIC_LABEL_LIMITATIONS,
    labeled_set,
    measure_has_gap,
    render_sample_png,
    run_evaluation,
    terminal_action,
)
from shelfsentry.pipeline import run_local_inspection
from shelfsentry.vision.analyze import analyze_image_bytes
from shelfsentry.vision.scoring import build_finding


# ---------------------------------------------------------------------------
# Independent measurement reference (computed once; the whole point of the
# property is that run_evaluation's numbers are reproducible from measurement).
# ---------------------------------------------------------------------------


def _independent_reference(
    *, cfg_vision: VisionConfig, cfg_agent: AgentConfig
) -> dict[str, object]:
    """Re-measure the labeled set from scratch via the public helpers.

    This re-derives the detection FP/FN id sets and the agent-decision
    correctness WITHOUT calling :func:`run_evaluation`, using exactly the
    documented public surface (``render_sample_png`` + the vision analysis path
    for ``has_gap``, and ``run_local_inspection`` + ``terminal_action`` for the
    terminal action). If ``run_evaluation`` fabricated a metric, its reported
    value would not match what this function measures.
    """
    samples = labeled_set()
    fp_ids: set[str] = set()
    fn_ids: set[str] = set()
    labeled_total = 0
    correct = 0

    with tempfile.TemporaryDirectory(prefix="shelfsentry-eval-ref-") as tmp:
        adapter = LocalStorageAdapter(tmp)
        for sample in samples:
            data = render_sample_png(sample)
            inspection_id = f"ref-{sample.sample_id}"

            # Detection has_gap via the same vision-only path the runner uses.
            analysis = analyze_image_bytes(data, cfg_vision)
            finding = build_finding(
                analysis,
                inspection_id=inspection_id,
                finding_id=f"{inspection_id}-vision",
                annotated_output_ref=f"evidence://{inspection_id}/vision",
                source="image",
                cfg=cfg_vision,
            )
            measured_has_gap = measure_has_gap(
                finding,
                empty_ratio_threshold=cfg_vision.empty_space_ratio_threshold,
            )
            if measured_has_gap and not sample.expected_has_gap:
                fp_ids.add(sample.sample_id)
            elif sample.expected_has_gap and not measured_has_gap:
                fn_ids.add(sample.sample_id)

            # Terminal action via the full real pipeline, over the labeled
            # subset only (the denominator of the correctness metric).
            if sample.expected_action is not None:
                labeled_total += 1
                report = run_local_inspection(
                    data,
                    sample.content_type,
                    inspection_id=inspection_id,
                    adapter=adapter,
                    cfg_vision=cfg_vision,
                    cfg_agent=cfg_agent,
                    mode="local",
                )
                if terminal_action(report) == sample.expected_action:
                    correct += 1

    correctness = (correct / labeled_total) if labeled_total else 0.0
    return {
        "fp_ids": fp_ids,
        "fn_ids": fn_ids,
        "labeled_total": labeled_total,
        "correct": correct,
        "correctness": correctness,
    }


# Measured exactly once with the default config; reused across property cases so
# the suite does not re-run the real pipeline for every Hypothesis example.
_DEFAULT_CFG_VISION = VisionConfig()
_DEFAULT_CFG_AGENT = AgentConfig()
_REFERENCE = _independent_reference(
    cfg_vision=_DEFAULT_CFG_VISION, cfg_agent=_DEFAULT_CFG_AGENT
)


def _assert_measured_only_and_reproducible(result, reference) -> None:
    """Assert ``result`` carries only measured, reproducible-from-measurement
    metrics, consistent with the independent ``reference``."""
    samples = labeled_set()
    valid_ids = {s.sample_id for s in samples}

    # Structural honesty flags.
    assert result.measured_only is True
    assert result.is_demo is True
    assert result.sample_count == len(samples)

    # FP/FN are EvalExample records, reference real samples, and are disjoint.
    assert all(isinstance(e, EvalExample) for e in result.false_positives)
    assert all(isinstance(e, EvalExample) for e in result.false_negatives)
    fp_ids = {e.sample_id for e in result.false_positives}
    fn_ids = {e.sample_id for e in result.false_negatives}
    assert fp_ids <= valid_ids
    assert fn_ids <= valid_ids
    assert fp_ids.isdisjoint(fn_ids)

    # The reported FP/FN classification matches an independent re-measurement.
    assert fp_ids == reference["fp_ids"]
    assert fn_ids == reference["fn_ids"]

    # Correctness is a ratio in [0, 1] whose denominator is the number of
    # labeled samples and whose numerator is an integer count, and it equals the
    # independently recomputed value => reproducible from measurement, not
    # invented.
    assert 0.0 <= result.agent_decision_correctness <= 1.0
    labeled_total = reference["labeled_total"]
    assert labeled_total == sum(1 for s in samples if s.expected_action is not None)
    assert labeled_total > 0, "expected at least one labeled sample for this property"
    expected_correct = round(result.agent_decision_correctness * labeled_total)
    # numerator is an integer count exactly reproduced by measurement
    assert abs(result.agent_decision_correctness * labeled_total - expected_correct) < 1e-9
    assert expected_correct == reference["correct"]
    assert result.agent_decision_correctness == reference["correctness"]

    # Latency percentiles measured and ordered.
    assert result.latency_ms_p50 >= 0.0
    assert result.latency_ms_p95 >= 0.0
    assert result.latency_ms_p50 <= result.latency_ms_p95

    # Documented limitations non-empty and include the synthetic-label caveat.
    assert len(result.documented_limitations) > 0
    assert SYNTHETIC_LABEL_LIMITATIONS[0] in result.documented_limitations


# ---------------------------------------------------------------------------
# Property 23 (Validates: Requirements 14.4, 25.4, 27.2)
# ---------------------------------------------------------------------------


@settings(
    max_examples=12,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(
    # Vary only honest knobs: the run id must never change a measured metric,
    # and is_demo stays True because the labels are synthetic (keeping demo
    # semantics honest). Both are driven by Hypothesis to show the invariants
    # hold across configuration variation.
    eval_run_id=st.text(
        alphabet=st.characters(min_codepoint=33, max_codepoint=126),
        min_size=1,
        max_size=24,
    ),
    is_demo=st.just(True),
)
def test_property_only_measured_metrics_are_reported(eval_run_id, is_demo):
    """Property 23: every reported metric is measured and reproducible.

    **Validates: Requirements 14.4, 25.4, 27.2**
    """
    result = run_evaluation(
        is_demo=is_demo,
        eval_run_id=eval_run_id,
        cfg_vision=_DEFAULT_CFG_VISION,
        cfg_agent=_DEFAULT_CFG_AGENT,
    )
    # The run id flows through verbatim (no metric depends on it).
    assert result.eval_run_id == eval_run_id
    _assert_measured_only_and_reproducible(result, _REFERENCE)


@settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(st.integers(min_value=0, max_value=3))
def test_property_measured_classification_is_run_invariant(_seed):
    """The measured FP/FN ids and correctness are identical across runs; only
    latency (wall clock) may differ and is never asserted equal.

    **Validates: Requirements 14.4, 25.4, 27.2**
    """
    first = run_evaluation(
        is_demo=True,
        cfg_vision=_DEFAULT_CFG_VISION,
        cfg_agent=_DEFAULT_CFG_AGENT,
    )
    second = run_evaluation(
        is_demo=True,
        cfg_vision=_DEFAULT_CFG_VISION,
        cfg_agent=_DEFAULT_CFG_AGENT,
    )
    assert {e.sample_id for e in first.false_positives} == {
        e.sample_id for e in second.false_positives
    }
    assert {e.sample_id for e in first.false_negatives} == {
        e.sample_id for e in second.false_negatives
    }
    assert first.agent_decision_correctness == second.agent_decision_correctness
    # Both runs also match the independent measurement reference.
    _assert_measured_only_and_reproducible(first, _REFERENCE)
    _assert_measured_only_and_reproducible(second, _REFERENCE)
