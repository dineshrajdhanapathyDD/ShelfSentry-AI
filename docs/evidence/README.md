# ShelfSentry AI - visual-to-action evidence

These artifacts are **real output** from the local active-perception pipeline,
emitted by `backend/scripts/generate_demo_evidence.py`. None of it is
hand-written. They are the submission's evidence that a vision finding causally
drives the agent's action (task 17.3, Req 26.5).

## Files

| File | What it is |
| --- | --- |
| `report-bundled-sample.json` | Canonical `InspectionReport` for the bundled synthetic shelf sample. |
| `trace-bundled-sample.txt` | Human-readable decision-trace transcript for the bundled sample. |
| `evidence-bundled-sample-0.png` | Annotated-evidence PNG the pipeline rendered for the bundled sample, fetched back by its opaque `media://` reference. |
| `report-conflicting-sample.json` | Canonical `InspectionReport` for a crafted ambiguous/conflicting multi-gap scene. |
| `trace-conflicting-sample.txt` | Decision-trace transcript for the conflicting sample. |
| `evidence-conflicting-sample-0.png` | Annotated-evidence PNG for the conflicting sample. |

## What they show

On both inputs the OpenCV 5 vision pass measures regions that read as clearly
empty alongside regions that read as clearly filled. That is the deterministic
policy's definition of **conflicting evidence**, so the agent's recorded
decision is `request_human_review` on iteration 0 - the finding driving the
action - and the report finalizes with `status = needs_review`,
`requires_human_review = true`, declining to assert an out-of-stock condition.

The trace length here is one entry because the conflict is detected on the first
finding; that is the honest measured behavior for these two synthetic scenes,
not a truncation. The multi-step branches (`inspect_crop`,
`analyze_another_frame`) are covered by the agent test suite.

## Regenerate

From `backend/`:

```bash
python scripts/generate_demo_evidence.py
```

The script is deterministic apart from the per-run random inspection-id suffix
and the creation timestamp, so re-running overwrites these files with
equivalent content. The scratch store it persists through
(`backend/.shelfsentry-evidence/`) is gitignored; only these small artifacts are
committed.
