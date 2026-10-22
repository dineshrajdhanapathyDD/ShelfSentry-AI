# ShelfSentry AI - technical report

A concise, honest account of what ShelfSentry AI is, how it is built, what has
been verified, and what has not. Numbers labeled "as last run" are what the
suites and runners reported on the most recent local run documented here; they
are reproducible, not a standing guarantee.

## 1. Problem

Retail shelf inspection is usually framed as one-shot detection: a model looks
at an image, returns a box and a number, and stops. That framing hides two
hard problems. First, a single pass cannot tell an actually empty shelf from a
shadow, an occlusion, a divider, or a dark/blurry photo - yet a one-shot score
presents all of them as the same confident number. Second, there is no record
of *why* a call was made, so a wrong "out of stock" is indistinguishable from a
right one after the fact.

ShelfSentry targets both: make perception a **feedback loop** that gathers more
evidence when the first look is uncertain, and keep an auditable **decision
trace** so every recommendation cites the evidence that produced it - without
ever fabricating a confidence score.

## 2. Approach

Five components compose into one active-perception pipeline:

1. **Substantive OpenCV 5 vision.** `backend/shelfsentry/vision/analyze.py`
   runs a fixed, config-driven classical pipeline: `cv2.Sobel` vertical
   gradient + horizontal projection (optionally corroborated by
   `cv2.HoughLinesP`) for shelf bands; Otsu + adaptive thresholding into a
   "filled" mask; `cv2.morphologyEx` open-then-close; `cv2.Canny` edge density;
   and `cv2.findContours` / `boundingRect` / `contourArea` to delineate gap
   blobs, sorted by a stable key for reproducible boxes. Scoring produces a
   *measured* `empty_space_ratio`, an ordinal `heuristic_score` kept distinct
   from any probability, and an `uncertainty_value` - each finding labeled an
   estimate, each region carrying a visual caveat. The OpenCV version is
   verified at startup (`vision/version.py`) and recorded on every finding.

2. **A bounded Strands active-perception loop.** `backend/shelfsentry/agent/`
   runs the finding through a bounded loop (`max_iterations` default 4) that
   selects one of exactly five actions - `inspect_crop`,
   `analyze_another_frame`, `request_better_image`, `request_human_review`,
   `finalize_report` - records each `AgentDecision` and `ToolExecution`, merges
   returned evidence into the next iteration, and finalizes a report that
   references its supporting trace entries. A deterministic `FallbackPolicy` is
   a pure function of `(finding, trace, cfg)`; in AWS mode Amazon Bedrock is the
   primary decision path with the policy as a recorded fallback.

3. **A restricted MCP tool surface.** `backend/shelfsentry/mcp/` exposes exactly
   four vision tools over a Python MCP SDK stdio transport -
   `analyze_shelf_image`, `inspect_region_crop`, `compare_shelf_frames`,
   `get_annotated_evidence` - each delegating to the existing vision functions.
   Media is resolved by inspection-scoped id; no tool argument is a filesystem
   path, arbitrary URL, shell command, or credential. The agent's only route to
   perception is these tools.

4. **Two-mode storage behind one protocol.** `backend/shelfsentry/data/storage.py`
   defines a single `StorageAdapter` seam. `LocalStorageAdapter` persists JSON
   records and media as files on disk, addressing media only by opaque,
   server-minted `media://<token>` references validated against directory
   traversal. `AwsStorageAdapter` satisfies the same protocol over S3 +
   DynamoDB, so the identical pipeline runs in either mode.

5. **CloudFormation infrastructure.** `infrastructure/template.yaml` is the
   single infrastructure definition (no CDK, no Terraform): a private,
   encrypted, public-access-blocked S3 bucket; DynamoDB tables; API + worker
   Lambdas with explicit timeout/memory; an API Gateway HTTP API; CloudWatch log
   groups with retention and a basic alarm; least-privilege IAM per Lambda; no
   embedded secrets.

## 3. What was built and verified

| Area | Status (as last run) |
| --- | --- |
| Local/mock pipeline end to end (real OpenCV 5, agent loop, MCP tools, local persistence) | Built and verified. |
| Backend test suite (`pytest` + Hypothesis, property tests >= 100 iterations each) | **455 passed.** |
| OpenCV 5 runtime | **Verified** - `opencv-python==5.0.0.93`, `cv2.__version__ == 5.0.0`. |
| Evidence generator (`backend/scripts/generate_demo_evidence.py`) | Built and verified to run (exit 0); produced the `docs/evidence/` artifacts below. |
| Reproducible evaluation runner | Built and verified to run; results over *synthetic* labels. |
| CloudFormation template | Authored. Not deployed in this environment. |

The backend count here (`455 passed`) is the number observed on the run
documented in this report; the README cites a separate earlier run (`428
passed`). Both are "as last run" snapshots - re-run `python -m pytest` from
`backend/` to confirm the current number.

## 4. Evidence of the visual-to-action loop

The script `backend/scripts/generate_demo_evidence.py` runs the real pipeline on
two inputs and writes the artifacts it actually produced into `docs/evidence/`.
It was run; the files below exist and are real pipeline output:

- `report-bundled-sample.json`, `trace-bundled-sample.txt`,
  `evidence-bundled-sample-0.png` - the bundled synthetic shelf sample.
- `report-conflicting-sample.json`, `trace-conflicting-sample.txt`,
  `evidence-conflicting-sample-0.png` - a crafted ambiguous/conflicting
  multi-gap scene (reused from the labeled evaluation set's `eval-gap-two`).

On both inputs the vision pass measures regions that read as clearly empty
alongside regions that read as clearly filled. That is the policy's definition
of **conflicting evidence**, and the agent's recorded decision is exactly that -
the finding causally driving the action. The decision trace transcript
(verbatim from `docs/evidence/trace-conflicting-sample.txt`) records it in one
line:

```
[0] AGENT DECISION  iter=0  action=request_human_review  decided_by=fallback
      rationale: Conflicting evidence: at least one region reads as clearly
      empty while another reads as clearly filled (disagreement >=
      conflict_margin 0.25); escalating to human review rather than guessing.
```

The corresponding `InspectionReport` finalizes with
`status = needs_review`, `requires_human_review = true`, and a recommendation
that explicitly declines to assert an out-of-stock condition. The annotated PNG
is the real rendering of the vision pass's candidate regions at their recorded
bounding boxes, fetched back through the same opaque `media://` reference seam
the MCP tools use. To regenerate: `python scripts/generate_demo_evidence.py`
from `backend/`.

Note on trace length: on these two synthetic scenes the conflict is detected on
the first finding, so the loop escalates on iteration 0 (a one-entry trace).
That is the honest, measured behavior for these inputs - not a truncation. The
bounded multi-step branches (`inspect_crop`, `analyze_another_frame`) are
exercised by the agent property/unit test suite rather than by these particular
samples.

## 5. Measured evaluation behavior

The runner (`backend/shelfsentry/eval/`) scores the deterministic pipeline
against a small, reproducible, **synthetic** labeled set authored as code.
Measured output, as last run:

- samples: **8**
- false positives: **0**
- false negatives: **2** (on borderline synthetic gaps)
- agent decision-correctness: **1.0** over the labeled subset

These are synthetic-label results on own-work scenes, not real-retail accuracy.
The `EvaluationResult` is pinned `is_demo=True` and `measured_only=True` and
always carries its documented limitations (the labels are the authors' own
ground truth for drawn content; the detection metric is a coarse per-image
`has_gap` boolean; decision-correctness covers only the subset with a
well-defined expected terminal action; latency percentiles are
environment-dependent). The two false negatives are a real, reported limitation
of the heuristic on borderline gaps, not hidden.

## 6. Limitations

- **Classical-CV heuristic, not a learned detector.** Metrics are measured
  visual cues, not calibrated out-of-stock probabilities; borderline gaps can be
  missed (see the two evaluation false negatives).
- **Synthetic evaluation.** The labeled set is drawn programmatically; it is a
  reproducible sanity check, not a statistically powered real-retail benchmark.
- **UI is not wired to the backend.** The React app runs against mocked API
  responses; there is no HTTP server connecting it to the Python pipeline in
  this repo. The runnable end-to-end demo is the Python path.
- **Small bounded loop.** `max_iterations` defaults to 4; the loop deliberately
  finalizes or escalates rather than pursuing unbounded evidence gathering.

## 7. What is NOT verified

- **Live AWS deployment.** The CloudFormation stack (S3 / Lambda / API Gateway /
  DynamoDB / CloudWatch) is authored but **not deployed** from this repo. The
  deploy and teardown commands (README, `docs/teardown.md`) are the documented
  path for you to run against your own account; they are unverified against live
  AWS from here.
- **The Bedrock decision path.** In AWS mode Bedrock is the intended primary
  decision path, with the deterministic policy as a recorded fallback. The
  Bedrock path has **not** been exercised against real AWS from this repo.
- **Full WCAG 2.1 AA conformance.** Automated `jest-axe` checks run against the
  four frontend pages, but full conformance additionally requires manual testing
  with assistive technologies and expert review, which was not performed.

## 8. Reproduce the claims in this report

From `backend/`:

```bash
python -c "import cv2; print(cv2.__version__)"          # -> 5.0.0
python -m pytest                                         # -> 455 passed (as last run)
python scripts/generate_demo_evidence.py                 # -> writes docs/evidence/ (exit 0)
python -c "from shelfsentry.eval import run_evaluation; r = run_evaluation(); \
print(r.sample_count, len(r.false_positives), len(r.false_negatives), \
r.agent_decision_correctness)"                           # -> 8 0 2 1.0
```
