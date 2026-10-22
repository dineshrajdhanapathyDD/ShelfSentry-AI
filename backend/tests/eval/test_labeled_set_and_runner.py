"""Tests for the labeled evaluation set and runner (task 14.1; Req 25.1-25.4).

These assert the properties the task requires of the evaluation set and the
``run_evaluation`` runner, using the REAL local pipeline (no mocks):

* the labeled set is deterministic and non-empty, with reproducible image bytes;
* ``run_evaluation`` returns a valid ``EvaluationResult`` that is measured-only,
  demo-flagged, sized to the set, with FP/FN as ``EvalExample`` instances,
  correctness in ``[0, 1]``, ``p50 <= p95``, and non-empty documented
  limitations;
* running twice yields the SAME false-positive / false-negative classification
  (the vision detection is deterministic); latency may differ and is NOT
  asserted equal.
"""

from __future__ import annotations

from shelfsentry.data.contracts import EvalExample, EvaluationResult
from shelfsentry.eval import (
    SYNTHETIC_LABEL_LIMITATIONS,
    labeled_set,
    render_sample_png,
    run_evaluation,
)


# ---------------------------------------------------------------------------
# Labeled set: deterministic and non-empty (Req 25.1)
# ---------------------------------------------------------------------------


def test_labeled_set_is_non_empty():
    samples = labeled_set()
    assert len(samples) > 0


def test_labeled_set_is_deterministic_across_calls():
    first = labeled_set()
    second = labeled_set()
    # Same samples, in the same order, with the same labels.
    assert [s.sample_id for s in first] == [s.sample_id for s in second]
    assert [s.expected_has_gap for s in first] == [s.expected_has_gap for s in second]
    assert [s.expected_action for s in first] == [s.expected_action for s in second]


def test_labeled_set_has_unique_sample_ids():
    ids = [s.sample_id for s in labeled_set()]
    assert len(ids) == len(set(ids))


def test_labeled_set_contains_both_positive_and_negative_cases():
    # Both FP and FN must be observable, so the set needs gap and no-gap scenes.
    has_gap = [s for s in labeled_set() if s.expected_has_gap]
    no_gap = [s for s in labeled_set() if not s.expected_has_gap]
    assert has_gap, "expected at least one gap (positive) sample"
    assert no_gap, "expected at least one no-gap (negative) sample"


def test_render_sample_png_is_deterministic():
    for sample in labeled_set():
        a = render_sample_png(sample)
        b = render_sample_png(sample)
        assert a == b, f"sample {sample.sample_id} rendered non-deterministically"
        assert a[:8] == b"\x89PNG\r\n\x1a\n", "expected PNG-encoded bytes"


# ---------------------------------------------------------------------------
# run_evaluation returns a valid, measured-only EvaluationResult (Req 25.2-25.4)
# ---------------------------------------------------------------------------


def test_run_evaluation_returns_valid_measured_only_result():
    result = run_evaluation(is_demo=True)

    assert isinstance(result, EvaluationResult)
    # Measured-only and demo-flagged (Req 25.4, 27.2, 14.5).
    assert result.measured_only is True
    assert result.is_demo is True
    # Sized to the labeled set.
    assert result.sample_count == len(labeled_set())

    # FP / FN are EvalExample instances (Req 25.2).
    assert all(isinstance(e, EvalExample) for e in result.false_positives)
    assert all(isinstance(e, EvalExample) for e in result.false_negatives)

    # Decision correctness is a measured fraction in [0, 1] (Req 25.4).
    assert 0.0 <= result.agent_decision_correctness <= 1.0

    # Latency percentiles are measured and ordered (p50 <= p95).
    assert result.latency_ms_p50 >= 0.0
    assert result.latency_ms_p95 >= 0.0
    assert result.latency_ms_p50 <= result.latency_ms_p95

    # Documented limitations are always reported (Req 25.3).
    assert len(result.documented_limitations) > 0
    # The synthetic-label limitations are included.
    for limitation in SYNTHETIC_LABEL_LIMITATIONS:
        assert limitation in result.documented_limitations


def test_run_evaluation_fp_fn_do_not_overlap():
    result = run_evaluation(is_demo=True)
    fp_ids = {e.sample_id for e in result.false_positives}
    fn_ids = {e.sample_id for e in result.false_negatives}
    # A single sample cannot be both a false positive and a false negative.
    assert fp_ids.isdisjoint(fn_ids)
    # Every FP/FN references a real sample from the set.
    valid_ids = {s.sample_id for s in labeled_set()}
    assert fp_ids <= valid_ids
    assert fn_ids <= valid_ids


def test_run_evaluation_result_round_trips_canonically():
    # A valid EvaluationResult serializes and re-validates (contract boundary).
    result = run_evaluation(is_demo=True)
    restored = EvaluationResult.model_validate_json(result.canonical_json())
    assert restored.canonical_json() == result.canonical_json()


# ---------------------------------------------------------------------------
# Determinism of the vision classification across runs (latency may differ)
# ---------------------------------------------------------------------------


def test_run_evaluation_fp_fn_classification_is_deterministic():
    first = run_evaluation(is_demo=True)
    second = run_evaluation(is_demo=True)

    # The FP/FN classification is a pure function of the deterministic vision
    # pass, so the classified sample ids match exactly across runs.
    assert {e.sample_id for e in first.false_positives} == {
        e.sample_id for e in second.false_positives
    }
    assert {e.sample_id for e in first.false_negatives} == {
        e.sample_id for e in second.false_negatives
    }
    # Agent decision correctness is likewise deterministic.
    assert first.agent_decision_correctness == second.agent_decision_correctness
    # Latency is wall-clock and NOT asserted equal here (see module docstring).
