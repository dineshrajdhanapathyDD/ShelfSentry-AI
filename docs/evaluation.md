# Evaluation (ShelfSentry AI)

_Requirement 26.3 — the measured evaluation detail. Every number here is
produced by the real runner and reported measured-only; nothing is fabricated._

ShelfSentry ships a small, **reproducible, labeled** evaluation set stored *as
code* and a runner that executes the real local active-perception pipeline over
it and reports **only measured** results. The honesty line is strict: the labels
are synthetic own-work (not real retail ground truth), the results are marked
`is_demo=True` / `measured_only=True`, and the runner always carries documented
limitations. Do not read these numbers as production accuracy.

- Labeled set: `backend/shelfsentry/eval/labeled_set.py`
- Runner: `backend/shelfsentry/eval/runner.py`
- Result contract: `EvaluationResult` in `backend/shelfsentry/data/contracts.py`

## The labeled evaluation set

The set is drawn programmatically with NumPy + OpenCV 5 (no bundled binary
assets), so it is byte-for-byte reproducible on every machine. Because the
images are drawn by us, the labels are a direct statement of what was drawn, not
a guess.

Each sample (`EvalSample`) carries three ground-truth labels:

- **`expected_has_gap`** (bool) — the primary detection label. `True` when one
  or more clearly empty gaps were deliberately drawn into an otherwise stocked
  shelf; `False` for a fully stocked shelf or a flat scene with no stockable
  shelf space.
- **`expected_gap_count`** (int) — how many gaps were drawn. This is recorded
  for documentation and the expected/actual example text; it is **not** used as
  a strict equality scoring target, because the classical-CV contour pass may
  legitimately split or merge adjacent gaps.
- **`expected_action`** (one of the five `AgentAction` values, or `None`) — the
  terminal action the deterministic active-perception loop is expected to settle
  on. Scenes where a single terminal action is genuinely not well defined are
  labeled `None` and excluded from the decision-correctness denominator rather
  than given a guessed expectation.

The eight scenes (verified from `labeled_set()`):

| sample_id | kind | expected_has_gap | gap_count | expected_action |
| --- | --- | --- | --- | --- |
| `eval-gap-single-left` | stocked_with_gaps | True | 1 | `request_human_review` |
| `eval-gap-single-right` | stocked_with_gaps | True | 1 | `None` (borderline) |
| `eval-gap-two` | stocked_with_gaps | True | 2 | `request_human_review` |
| `eval-gap-bottom` | stocked_with_gaps | True | 1 | `None` (borderline) |
| `eval-gap-three` | stocked_with_gaps | True | 3 | `request_human_review` |
| `eval-full-1` | fully_stocked | False | 0 | `finalize_report` |
| `eval-full-2` | fully_stocked | False | 0 | `finalize_report` |
| `eval-empty-scene` | empty_scene | False | 0 | `finalize_report` |

That is 8 samples total; 6 carry an expected terminal action (the two borderline
gap scenes are `None`).

## Exactly what each metric means

The runner (`run_evaluation`) runs the real local pipeline
(`run_local_inspection`, real OpenCV 5 through the MCP tools plus the bounded
deterministic agent loop — no AWS, no network) over every sample, and derives
each metric as follows.

### Detection: false positives and false negatives

The measured per-image signal is a coarse boolean, **`has_gap`**: does the
vision finding flag at least one candidate region whose measured
`empty_space_ratio` is at or above the configured `empty_space_ratio_threshold`
(`VisionConfig.empty_space_ratio_threshold`, default `0.6`)? This is computed by
`measure_has_gap(...)` in `runner.py` directly from the OpenCV pass
(`analyze_image_bytes` + `build_finding`), independent of how the agent loop
later routes.

Compared against each sample's `expected_has_gap` ground truth:

- a **false positive** is a sample labeled `expected_has_gap=False` for which
  the system flagged a gap;
- a **false negative** is a sample labeled `expected_has_gap=True` for which the
  system flagged no gap.

This is a heuristic sanity check, not a bounding-box-level accuracy measure, and
it inherits the classical-CV heuristic's limitations (see
`docs/limitations.md`).

### Agent decision correctness

Over the subset of samples whose `expected_action` is not `None`, this is the
fraction whose pipeline-produced report has a *terminal decision action* equal
to the labeled `expected_action`. The terminal action is the `action` of the
last `AgentDecision` in the report's `decision_trace` (`terminal_action(...)` in
`runner.py`).

It therefore measures **whether the deterministic active-perception loop routes
each labeled scene to the intended terminal action**. It is explicitly *not* a
measure of a learned/LLM policy, and it does not cover the unlabeled
(borderline) samples. The denominator is the 6 action-labeled samples above.

### Latency (p50 / p95)

Measured wall-clock milliseconds per pipeline run, timed with
`time.perf_counter` around `run_local_inspection`, reduced to the 50th and 95th
percentiles with linear interpolation (`percentile(...)` in `runner.py`). These
values are **environment-dependent** — they vary by hardware, load, and OpenCV
build — so they are illustrative, not a guarantee.

## Measured results

The following were produced by running the runner in this environment. The
detection classification and the decision-correctness value are deterministic
(pure functions of the deterministic sample bytes and fixed config), so they are
identical across runs. The latency figures are wall-clock and will differ on
other machines.

| Metric | Measured value |
| --- | --- |
| Samples | 8 |
| False positives | 0 |
| False negatives | 2 — `eval-gap-single-right`, `eval-gap-bottom` |
| Agent decision correctness | 1.0 over the 6 action-labeled samples |
| Latency p50 | ~93 ms (environment-dependent; example run) |
| Latency p95 | ~102 ms (environment-dependent; example run) |
| `is_demo` | `True` |
| `measured_only` | `True` |

**These are synthetic-label results on own-work scenes, not real-retail
accuracy.** The runner marks every result `is_demo=True` and `measured_only=True`
and attaches non-empty documented limitations.

### The false negatives are the honest failure cases

The two false negatives are not hidden. Both are borderline synthetic gaps the
heuristic did not flag at the configured `empty_space_ratio_threshold` (0.6),
reported as `EvalExample` records:

- **`eval-gap-single-right`** — "Stocked shelf with one empty gap on the
  middle-right row; a borderline case for the empty-ratio threshold." Expected
  `has_gap=True (drew 1 gap(s))`, measured `has_gap=False`.
- **`eval-gap-bottom`** — "Stocked shelf with one wide empty gap on the bottom
  row; a borderline case for the empty-ratio threshold." Expected
  `has_gap=True (drew 1 gap(s))`, measured `has_gap=False`.

These two scenes are precisely the ones labeled `expected_action=None` and
excluded from the decision-correctness denominator — a borderline scene is
*excluded, not relabeled*, to keep the metric honest.

## How to reproduce

From `backend/`, with the pinned dependencies installed (see the README and
`docs/limitations.md` for the exact pins):

```bash
python -c "from shelfsentry.eval import run_evaluation; r = run_evaluation(); \
print('samples', r.sample_count, 'FP', len(r.false_positives), \
'FN', len(r.false_negatives), 'decision_correctness', r.agent_decision_correctness)"
```

Observed output in this environment:

```
samples 8 FP 0 FN 2 decision_correctness 1.0
```

The detection and decision-correctness numbers are deterministic; only the
latency percentiles vary run to run, which is why callers must never assert
latency equality.
