# ShelfSentry AI - five-minute demo script

A presenter-ready walkthrough of ShelfSentry AI's **active-perception** loop,
timed to about five minutes. Every command below is run from the real local
path and has been verified to work; where the UI is not wired to the backend,
that is called out honestly rather than papered over.

## Honest framing (read this before presenting)

- The runnable end-to-end demo is driven from **Python**: real OpenCV 5 vision
  through the restricted MCP tools, the bounded agent loop, and local on-disk
  persistence - no AWS credentials and no network.
- There is **no HTTP server wiring the React UI to the Python backend** in this
  repo (see the README "Running the local demo" note). The frontend therefore
  runs as a UI against its own mocked API responses. Show the UI for the
  experience; show the Python path for the *real* pipeline and evidence. Do not
  claim the UI is displaying live backend output.
- No metric shown is fabricated. The evaluation numbers come from the runner;
  the decision trace comes from the real pipeline.

## Prerequisites (do this before the clock starts)

```bash
# Backend deps (Python >= 3.11). From backend/:
cd backend
python -m pip install -e ".[test]"

# Frontend deps (Node.js >= 20). From frontend/:
cd ../frontend
npm install
```

Optionally pre-generate the evidence artifacts (also done live in Segment 3):

```bash
cd ../backend
python scripts/generate_demo_evidence.py
```

---

## Segment 0 - Framing (0:00 - 0:30, ~30s)

Say, in your own words:

> "A one-shot shelf detector returns a box and a number and stops. ShelfSentry
> treats perception as a feedback loop: the first OpenCV 5 pass produces a
> finding *plus a quantified uncertainty*, and that uncertainty causally drives
> the next action - zoom into a region, analyze another frame, request a better
> image, escalate to a human, or finalize. Every step is recorded in a decision
> trace the final report cites. It never fabricates a confidence score, and
> every empty-looking region carries a visual caveat."

Point out the honest framing above: the real pipeline is Python; the UI uses
mocked data.

---

## Segment 1 - Verify the real OpenCV 5 engine (0:30 - 1:15, ~45s)

Show that the substantive analysis really runs on OpenCV 5, enforced at
startup.

```bash
cd backend
python -c "import cv2; print('cv2', cv2.__version__)"
```

Expected (verified): `cv2 5.0.0`.

Say: "The version is checked at startup in `vision/version.py` and recorded in
every finding; the system fails loudly rather than silently accepting OpenCV 4.
The real work - shelf-band detection, thresholding, morphology, edge and
contour analysis - lives in `vision/analyze.py`."

---

## Segment 2 - Run the full active-perception pipeline (1:15 - 2:45, ~90s)

Run the end-to-end local inspection on the bundled synthetic shelf sample.

```bash
python -c "from shelfsentry.samples import generate_sample_shelf_png; \
from shelfsentry.data.storage import LocalStorageAdapter; \
from shelfsentry.pipeline import run_local_inspection; \
report = run_local_inspection(generate_sample_shelf_png(), 'image/png', \
adapter=LocalStorageAdapter('.shelfsentry-local')); \
print('report', report.inspection_id, 'recommendation', report.final_recommendation); \
print('trace entries', len(report.decision_trace))"
```

Verified behavior on this sample: the pipeline validates the input, perceives
the first finding through the MCP `analyze_shelf_image` tool, runs the bounded
loop, and persists an `InspectionReport`. On the bundled sample the vision pass
finds a region that reads as clearly empty alongside regions that read as
clearly filled - **conflicting evidence** - so the agent's first and only
decision is `request_human_review` rather than guessing an out-of-stock
condition. That is the finding *causally* driving the action.

Talking point: "Notice it did not assert 'out of stock'. The measured evidence
conflicted, so the bounded policy escalated to a human. That restraint is the
honesty line in action."

---

## Segment 3 - Show the visual-to-action evidence (2:45 - 3:45, ~60s)

Generate the committed evidence artifacts straight from the real pipeline.

```bash
python scripts/generate_demo_evidence.py
```

This runs the pipeline on the bundled sample **and** on a crafted
ambiguous/conflicting multi-gap scene, then writes real artifacts into
`docs/evidence/`:

- `report-<name>.json` - the canonical `InspectionReport`.
- `trace-<name>.txt` - a human-readable transcript of the decision trace.
- `evidence-<name>-0.png` - the annotated-evidence PNG the pipeline rendered and
  stored, fetched back by its opaque `media://` reference.

Open `docs/evidence/trace-conflicting-sample.txt` and read the single decision
aloud - it ties the finding to the action in one line:

```
[0] AGENT DECISION  iter=0  action=request_human_review  decided_by=fallback
      rationale: Conflicting evidence: at least one region reads as clearly
      empty while another reads as clearly filled (disagreement >=
      conflict_margin 0.25); escalating to human review rather than guessing.
```

Then open `docs/evidence/evidence-conflicting-sample-0.png` to show the
annotated regions the vision pass drew at the recorded bounding boxes.

Say: "Nothing here is hand-written. The trace, the report, and the annotated
image are all emitted by the same run."

---

## Segment 4 - The restricted MCP tool surface (3:45 - 4:15, ~30s)

Show that the agent only ever touches perception through four restricted tools -
never the filesystem, a URL, a shell, or credentials.

```python
from shelfsentry.data.storage import LocalStorageAdapter, StorageMediaResolver
from shelfsentry.mcp import run_stdio_server
run_stdio_server(StorageMediaResolver(LocalStorageAdapter(".shelfsentry-local")))
```

Say: "This stdio MCP server exposes exactly `analyze_shelf_image`,
`inspect_region_crop`, `compare_shelf_frames`, and `get_annotated_evidence`.
Media is resolved by inspection id, not a path. 'Give the agent vision tools'
never becomes 'give the agent the machine.'" (You can mention this rather than
leaving the server blocking in the foreground during the demo.)

---

## Segment 5 - Measured evaluation + the UI (4:15 - 5:00, ~45s)

Run the honest evaluation runner:

```bash
python -c "from shelfsentry.eval import run_evaluation; r = run_evaluation(); \
print('samples', r.sample_count, 'FP', len(r.false_positives), \
'FN', len(r.false_negatives), 'decision_correctness', r.agent_decision_correctness)"
```

Verified measured output (synthetic labels): `samples 8 FP 0 FN 2
decision_correctness 1.0`. Say: "Eight synthetic labeled scenes, zero false
positives, two false negatives on borderline gaps that we report rather than
hide, and decision-correctness 1.0 over the labeled subset. These are
synthetic-label results, not real-retail accuracy - the runner marks itself
`is_demo` and `measured_only` and always carries its limitations."

Finally, start the UI to show the four pages (Inspection, Results, History,
Evaluation):

```bash
cd ../frontend
npm run dev     # Vite dev server, default http://localhost:5173
```

Say: "This is the UI experience. As noted up front, it runs against mocked API
responses - the real pipeline is the Python path we just ran. The Results page
mirrors what the real report carries: annotated regions, per-region measured
metric and uncertainty, the agent's chosen action and rationale, and the
human-review status."

---

## Timing summary

| Segment | Content | Budget |
| --- | --- | --- |
| 0 | Framing + honest caveat | ~0:30 |
| 1 | OpenCV 5 version check | ~0:45 |
| 2 | Full pipeline run (bundled sample) | ~1:30 |
| 3 | Visual-to-action evidence artifacts | ~1:00 |
| 4 | Restricted MCP tool surface | ~0:30 |
| 5 | Measured evaluation + UI | ~0:45 |
| | **Total** | **~5:00** |

## Commands cited (all verified to run from this repo)

- `python -c "import cv2; print('cv2', cv2.__version__)"` -> `cv2 5.0.0`
- the `run_local_inspection` one-liner -> prints a report id, recommendation,
  and trace-entry count
- `python scripts/generate_demo_evidence.py` -> writes the `docs/evidence/`
  artifacts (exit 0)
- the `run_evaluation` one-liner -> `samples 8 FP 0 FN 2 decision_correctness 1.0`
- `npm run dev` (frontend) -> Vite dev server on `http://localhost:5173`

Re-run any command to confirm; counts are "as last run," not a standing
guarantee.
