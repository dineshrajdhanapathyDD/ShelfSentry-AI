# Limitations (ShelfSentry AI)

_Requirement 26.3 — the documented limitations. These are stated plainly so a
reader never mistakes ShelfSentry's measured behavior for validated real-retail
performance or a product-grade detector._

ShelfSentry uses **classical computer vision** (OpenCV 5: edges, contours,
thresholding, morphology) to *estimate* empty shelf regions, and a **bounded,
deterministic** active-perception loop to decide what to look at next. The honesty
line in the design shows up here as concrete, acknowledged limits.

## Heuristic classical-CV limitations

The substantive analysis lives in `backend/shelfsentry/vision/analyze.py`
(`analyze_image_arrays` / `analyze_image_bytes`) and the scoring/caveat logic in
`backend/shelfsentry/vision/scoring.py`.

- **It is an estimate, not a product detector.** The pipeline locates candidate
  empty regions with edge / contour / threshold heuristics (Sobel + horizontal
  projection for shelf bands, Otsu + adaptive threshold for a filled mask,
  `cv2.morphologyEx`, `cv2.Canny`, `cv2.findContours`). It does not identify
  products, SKUs, or planogram compliance. Every finding is labeled
  `is_estimate=True`.

- **"Empty-looking" is never "proven out of stock."** An empty-looking region
  may instead be shadow, occlusion, shelf dividers, or poor image quality. Every
  candidate region therefore carries a non-empty `visual_caveat` naming those
  four confounders (`build_visual_caveat(...)` in `scoring.py`, enforced
  structurally by the contract's required `visual_caveat` field). The system
  never asserts an out-of-stock condition as fact (Req 27.4).

- **`empty_space_ratio` is a measured estimate, not a probability.** It is
  defined exactly as `empty_pixels / box_area`, clamped to `[0, 1]`
  (`compute_empty_space_ratio`). It estimates empty shelf space; it is not a
  calibrated out-of-stock probability.

- **`heuristic_score` is ordinal, not a probability.** It is a fixed weighted
  blend of the empty-space ratio, relative region size, and edge sparsity
  (weights 0.6 / 0.2 / 0.2 in `scoring.py`), used only to *rank* regions "most
  worth a closer look first." It is explicitly not calibrated and must never be
  presented as the likelihood that a shelf is out of stock
  (`compute_heuristic_score`, `UNCERTAINTY_INTERPRETATIONS`).

- **`uncertainty_value` is a bounded measurement confidence.** It is reduced for
  very small regions, borderline ratios near the empty/filled boundary, and high
  internal edge density (clutter/occlusion), and is clamped to
  `[0.05, 0.95]` — the system never reports absolute certainty
  (`compute_uncertainty_value`).

- **False negatives on borderline gaps.** The detection heuristic can miss a
  genuinely-drawn but borderline gap at the configured
  `empty_space_ratio_threshold` (default 0.6). This is observed and reported in
  the evaluation: 2 of 8 synthetic samples (`eval-gap-single-right`,
  `eval-gap-bottom`) are false negatives. See `docs/evaluation.md`.

## Synthetic-label evaluation limitations

These travel with every evaluation result (verbatim-in-spirit from
`SYNTHETIC_LABEL_LIMITATIONS` in `backend/shelfsentry/eval/labeled_set.py`, plus
a run-specific caveat the runner appends). Content was rephrased for brevity;
the source strings are authoritative.

1. **The evaluation images are synthetic.** They are drawn programmatically with
   NumPy and OpenCV, not photographed from real shelves. Their labels are the
   authors' own ground truth for drawn content and are **not** real retail
   ground truth; measured performance here does not predict real-world accuracy.

2. **The detection metric is a coarse per-image `has_gap` boolean.** It scores
   whether the pipeline flags any region at/above the configured threshold
   against whether a gap was deliberately drawn — a heuristic sanity check, not a
   bounding-box-level accuracy measure, inheriting the classical-CV heuristic's
   limitations.

3. **Agent decision correctness measures the deterministic fallback routing, not
   a learned policy.** It compares the loop's terminal action against a
   hand-labeled expected terminal action, only for the subset of samples where a
   single expected terminal action is well defined. It does not measure a
   learned/LLM policy and does not apply to the unlabeled (borderline) samples.

4. **The set is deliberately small and reproducible** (a handful of scenes) to
   stay fast and asset-free; it is not a statistically powered benchmark.

5. **Latency percentiles are environment-dependent** wall-clock measurements
   (`time.perf_counter` around the real pipeline); absolute values vary by
   hardware, load, and OpenCV build.

6. **Run-specific caveat.** The runner records that agent decision correctness
   was measured over the 6 of 8 samples that carry an expected terminal-action
   label.

## Video limitations

Video analysis (`backend/shelfsentry/vision/video.py`) reduces a clip to a
**bounded set of sampled frames** — at most `cfg.max_frame_count` (default 16)
evenly spaced frame indices — and runs each through the same image path. It does
not analyze every frame, so transient gaps between sampled frames can be missed.
Some backends/containers do not honor exact frame seeks; an unreadable sampled
frame is skipped rather than failing the whole video.

## Agent / decision-path limitations

- **The deterministic fallback policy is the exercised local policy.** The local
  demo and the evaluation run the deterministic `FallbackPolicy`
  (`backend/shelfsentry/agent/fallback.py`) — a pure function of
  `(finding, trace, cfg)`. Its routing (conflict → `request_human_review`, poor
  quality → `request_better_image`, high uncertainty → `inspect_crop`, ambiguous
  video with frames left → `analyze_another_frame`, otherwise
  `finalize_report`) is what the decision-correctness metric measures.

- **Bedrock is primary-but-unexercised-against-real-AWS.** In AWS mode,
  `BedrockDecisionModel` (`backend/shelfsentry/agent/bedrock.py`) is the primary
  decision path, with the deterministic policy as a thin, recorded fallback. That
  Bedrock path has **not** been exercised against real AWS from this repo; its
  behavior here is only verified via injected test clients.

- **Bounded loop.** The loop is capped at `max_iterations` (default 4). It
  gathers more evidence before committing, but only within that budget; a
  genuinely ambiguous scene may finalize once the budget is spent.

## Deployment / wiring limitations

- **The AWS stack is authored but not deployed.** `infrastructure/template.yaml`
  is the single infrastructure definition; it has **not** been deployed in this
  environment, and the Lambda code is a deployable inline placeholder to be
  replaced with the packaged backend at deploy time.

- **No HTTP server wiring between frontend and backend.** The backend API layer
  (`shelfsentry.api`) returns framework-agnostic response envelopes; no HTTP
  server is wired, and the React frontend runs as a UI with its tests against
  mocked API responses. The runnable local demo today is driven from Python.

- **WCAG 2.1 AA.** The frontend runs automated accessibility checks
  (`jest-axe`); full conformance additionally requires manual testing with
  assistive technologies and expert review, which the automated checks do not
  claim to cover.
