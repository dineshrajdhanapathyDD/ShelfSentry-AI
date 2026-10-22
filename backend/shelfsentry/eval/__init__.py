"""Evaluation: the reproducible labeled evaluation set and a runner that
produces measured-only ``EvaluationResult`` records with documented limitations.

Public interface:

* :func:`labeled_set` - the deterministic, non-empty labeled evaluation set
  (synthetic shelf scenes authored as code, each with a ground-truth label).
* :func:`render_sample_png` - render one labeled sample to deterministic PNG
  bytes.
* :class:`EvalSample` - one labeled sample (scene + ground truth).
* :data:`SYNTHETIC_LABEL_LIMITATIONS` - the documented limitations carried with
  every result (synthetic labels, heuristic metrics, environment-dependent
  latency).
* :func:`run_evaluation` - run the labeled set through the real local pipeline
  and return a measured-only
  :class:`~shelfsentry.data.contracts.EvaluationResult` (Req 25.2-25.4).

See :mod:`shelfsentry.eval.labeled_set` for the labeling methodology and
:mod:`shelfsentry.eval.runner` for exactly what each measured metric means.
"""

from shelfsentry.eval.labeled_set import (
    SYNTHETIC_LABEL_LIMITATIONS,
    EvalSample,
    labeled_set,
    render_sample_png,
)
from shelfsentry.eval.runner import (
    SampleOutcome,
    measure_has_gap,
    percentile,
    run_evaluation,
    terminal_action,
)

__all__ = [
    "EvalSample",
    "SYNTHETIC_LABEL_LIMITATIONS",
    "labeled_set",
    "render_sample_png",
    "SampleOutcome",
    "measure_has_gap",
    "terminal_action",
    "percentile",
    "run_evaluation",
]
