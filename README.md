# ShelfSentry AI

**Agentic retail shelf inspection that pairs substantive OpenCV 5 computer
vision with a bounded active-perception agent loop.**

## Live demo

- **Web app (AWS Amplify):** https://main.d3521ybmmu2jz6.amplifyapp.com/
- **API health check:** https://q5a59gmnw9.execute-api.us-east-1.amazonaws.com/health

## Architecture

<img width="1601" height="1041" alt="shelfsenty v1 drawio" src="https://github.com/user-attachments/assets/d5d890f6-16b0-4d8d-9b30-faed9d1d4f32" />


Open the web app and click **Run example (bundled sample)** to run a real
inspection on AWS (OpenCV 5 vision + an Amazon Bedrock-driven active-perception
agent loop), then open **Results** to see the decision trace, **History** for
stored runs, and **Evaluation** for measured results. No sign-in required. The
API endpoint is a public HTTPS URL, not a secret.

> Both URLs are deployed from this repo: the frontend via `amplify.yml`, the
> backend via the single CloudFormation template in `infrastructure/template.yaml`.
> See [`docs/aws-deployment.md`](docs/aws-deployment.md) for the full deploy flow.

### Example results (real output from a live run)

A sample inspection on the deployed stack produced the annotated output image
plus three measured candidate regions. The values below are real measured
output, not illustrative placeholders:

| Region | Empty-space ratio (estimate) | Uncertainty value | Heuristic score |
| --- | --- | --- | --- |
| region_0000 | 77.8% | 0.428 | 0.654 |
| region_0001 | 18.9% | 0.539 | 0.340 |
| region_0002 | 22.3% | 0.446 | 0.354 |

Each region carries a visual caveat, e.g. *"This region is an estimate of empty
shelf space, not a confirmed out-of-stock condition. An empty-looking region
may instead be shadow, occlusion, shelf dividers, or poor image quality."* The
empty-space ratio is a measured value, the heuristic score is an ordinal (not a
probability), and the uncertainty value has a documented interpretation - the
system never asserts out-of-stock as proven.

ShelfSentry AI looks at a retail shelf image (or short video), runs real
classical computer vision to find candidate empty regions with *measured*
metrics and *quantified uncertainty*, and then hands those findings to a
bounded agent that **actively decides what to look at next** - zoom into a
region, analyze another frame, ask for a better image, escalate to a human, or
finalize - until it reaches an evidence-backed report. A restricted Model
Context Protocol (MCP) server is the only surface through which the agent
touches perception, so "give the agent vision tools" never becomes "give the
agent the machine."

## Why active perception

A one-shot detector returns a box and a number and stops. ShelfSentry instead
treats perception as a feedback loop: the first OpenCV pass produces a finding
plus an **uncertainty value**, and that uncertainty *causally* drives the next
action. An ambiguous gap triggers a closer crop; a frame that is too dark
triggers a request for a better image; evidence that conflicts across frames
triggers human review. The agent gathers more evidence before it commits, and
every step it took is recorded in a decision trace that the final report cites.

The system is held to a hard honesty line: it never fabricates metrics or
confidence scores, it keeps the rule-derived heuristic score distinct from any
probability, and it never claims an empty-looking region *proves* an
out-of-stock condition - every candidate region carries a visual caveat
(shadow, occlusion, dividers, image quality).

## What is verified vs. not

This matters, so it is stated up front rather than implied.

| Area | Status |
| --- | --- |
| Local/mock pipeline end to end (real OpenCV 5, agent loop, MCP tools, local persistence) | **Built and verified.** |
| Backend test suite | **Verified** - 497 passed (as last run; see [Testing](#testing)). |
| Frontend build + test suite | **Verified** - production build succeeds; 61 tests passed (as last run). |
| OpenCV 5 runtime | **Verified** - `opencv-python==5.0.0.93`, `cv2.__version__ == 5.0.0`. |
| Reproducible evaluation runner | **Built and verified** to run; results are over *synthetic* labels (see [Evaluation](#evaluation)). |
| CloudFormation template (`infrastructure/template.yaml`) | **Authored.** Not deployed in this environment. |
| Container-image Lambda packaging (OpenCV 5 on arm64) | **Authored.** Dockerfile + build/push scripts added; the image is built/pushed **on your machine** (not from this repo). |
| Live AWS deployment (S3 / Lambda / API Gateway / DynamoDB / Bedrock) | **Not performed here.** The deploy path is documented in `docs/aws-deployment.md`; you must run it, and it is unverified against live AWS from this repo. |
| HTTP wiring between the React frontend and the Python backend | **AWS path built; local HTTP server not included.** The API Lambda (`backend/shelfsentry/awslambda/api.py`) serves the exact routes the frontend client calls; set `VITE_API_BASE_URL` to the stack's `ApiUrl`. No standalone local HTTP server ships in this repo. |

Counts labeled "as last run" are what the suites reported on the most recent
local run; they are not a standing guarantee - re-run the suites to confirm.

## Repository layout

```
backend/           Python: vision, agent, MCP, data, api, config, eval, pipeline (+ tests/)
frontend/          React + TypeScript + Vite + Tailwind web app (src/pages, src/api)
infrastructure/    template.yaml - the single CloudFormation stack (AWS mode)
docs/              architecture.md, aws-deployment.md, aws-architecture.drawio (+ evaluation/limitations/security)
.kiro/             the spec (requirements, design, tasks)
```

Backend package (`backend/shelfsentry/`):

- `vision/` - OpenCV 5 `validate` / `preprocess` / `analyze` / `scoring` /
  `annotate` / `video` / `version`. No LLM dependency.
- `agent/` - the bounded active-perception `loop`, the `DecisionModel`
  abstraction, the deterministic `fallback` policy, `bedrock` (AWS primary
  path), and tool-arg validation.
- `mcp/` - the restricted MCP vision-tool server (four tools).
- `data/` - the seven typed Pydantic v2 `contracts` and the `StorageAdapter`
  (`LocalStorageAdapter`, `AwsStorageAdapter`).
- `api/` - mode-aware, versioned request handlers (framework-agnostic
  envelopes).
- `config/` - typed `settings` and the Local/AWS `mode` switch.
- `eval/` - the reproducible labeled set and the honest reporting runner.
- `pipeline.py` - `run_local_inspection(...)`, the end-to-end local flow.

## Setup

### Backend

Python >= 3.11 (verified on 3.13). From `backend/`:

```bash
python -m pip install -e ".[test]"
# or, equivalently, from the pinned list:
python -m pip install -r requirements.txt
```

All runtime and test dependencies are pinned to exact versions (no `^`/`~`
ranges). The pins are the single source of truth in `backend/pyproject.toml`
and mirrored in `backend/requirements.txt`:

| Dependency | Pin | Role |
| --- | --- | --- |
| `opencv-python` | `5.0.0.93` | OpenCV 5 - `cv2.__version__ == 5.0.0` |
| `numpy` | `2.2.6` | array backing for OpenCV |
| `pydantic` | `2.12.5` | typed data contracts (v2) |
| `strands-agents` | `1.48.0` | agent SDK (AWS-mode decision path) |
| `mcp` | `1.28.1` | Python MCP SDK |
| `pytest` | `9.1.1` | test runner |
| `hypothesis` | `6.170.2` | property-based testing |

### Frontend

Node.js >= 20. From `frontend/`:

```bash
npm install
```

Frontend dependency versions are also pinned exactly in
`frontend/package.json` (React `19.2.0`, Vite `7.3.7`, Tailwind `4.1.16`,
TypeScript `5.9.3`, Vitest `3.2.4`, among others).

## OpenCV 5: the exact version and where it does real work

- **Pinned package:** `opencv-python==5.0.0.93` (the only 5.x wheel currently
  published on PyPI).
- **Runtime version:** `cv2.__version__ == 5.0.0` - the exact OpenCV 5 used.
- **Enforced at startup:** `backend/shelfsentry/vision/version.py`
  (`verify_opencv_version()`) reads `cv2.__version__` and **fails loudly** if
  the major version is not 5. The system never silently accepts OpenCV 4. The
  verified version string is recorded in every `VisionFinding`.

The substantive analysis lives in **`backend/shelfsentry/vision/analyze.py`**
(`analyze_image_arrays` / `analyze_image_bytes`). It is a fixed, config-driven
classical pipeline - deterministic for a given input and `VisionConfig`:

1. **Shelf-region analysis** - `cv2.Sobel` vertical gradient + a horizontal
   projection (`cv2.reduce`) locate shelf bands; an optional `cv2.HoughLinesP`
   pass corroborates the shelf lips.
2. **Thresholding** - `cv2.threshold` with Otsu plus `cv2.adaptiveThreshold`
   combine into a deterministic "filled" (product) mask.
3. **Morphology** - `cv2.morphologyEx` open-then-close (`cv2.getStructuringElement`)
   removes speckle and consolidates regions.
4. **Edge analysis** - `cv2.Canny` edge density inside a band is an empty-space
   cue.
5. **Contour analysis** - `cv2.findContours` + `cv2.boundingRect` +
   `cv2.contourArea` delineate gap blobs, sorted by a stable key so boxes are
   reproducible across runs.

Scoring/uncertainty/caveats and `VisionFinding` assembly (the *measured*
`empty_space_ratio`, the ordinal `heuristic_score` kept distinct from any
probability, and the measurement-confidence `uncertainty_value`) live in
**`backend/shelfsentry/vision/scoring.py`**; annotation drawing is in
`vision/annotate.py`; video frame sampling in `vision/video.py`.

## Running the local demo

The local/mock path runs the full active-perception loop with **no AWS
credentials and no network** - real OpenCV 5 vision through the MCP tools, the
bounded agent loop driven by the deterministic fallback policy, and local
on-disk persistence.

> **Honest note on wiring:** the backend API layer
> (`shelfsentry.api.build_local_api`) returns framework-agnostic response
> envelopes; an HTTP server and a live frontend-to-backend connection are
> **not** wired in this repo. The runnable local demo today is driven from
> Python (and the frontend runs as a UI with its tests against mocked API
> responses). Where you see `http://localhost:8000` in `frontend/.env.example`,
> treat it as the slot a future HTTP adapter would fill, not a server this repo
> starts.

**End to end, from Python** (bundled synthetic sample, example mode):

```bash
cd backend
python -c "from shelfsentry.samples import generate_sample_shelf_png; \
from shelfsentry.data.storage import LocalStorageAdapter; \
from shelfsentry.pipeline import run_local_inspection; \
report = run_local_inspection(generate_sample_shelf_png(), 'image/png', \
adapter=LocalStorageAdapter('.shelfsentry-local')); \
print('report', report.inspection_id, 'recommendation', report.final_recommendation); \
print('trace entries', len(report.decision_trace))"
```

This validates the input, perceives the first finding through the MCP
`analyze_shelf_image` tool, runs the bounded loop (`max_iterations` default 4),
and persists an `InspectionReport` with a populated decision trace under
`.shelfsentry-local/`.

**The restricted MCP vision server** (stdio transport), for an MCP-capable
client:

```python
from shelfsentry.data.storage import LocalStorageAdapter, StorageMediaResolver
from shelfsentry.mcp import run_stdio_server
run_stdio_server(StorageMediaResolver(LocalStorageAdapter(".shelfsentry-local")))
```

It exposes exactly four tools - `analyze_shelf_image`, `inspect_region_crop`,
`compare_shelf_frames`, `get_annotated_evidence` - each delegating to the
vision pipeline. No tool exposes filesystem paths, arbitrary URLs, shell
execution, or credentials.

**The frontend UI** (Inspection, Results, History, Evaluation pages):

```bash
cd frontend
npm run dev     # Vite dev server, default http://localhost:5173
```

## Testing

### Backend (pytest + Hypothesis)

Single run, no watch mode; property tests generate >= 100 examples each:

```bash
cd backend
python -m pytest
```

**As last run:** `497 passed`. Property-based tests follow the design's
numbered Correctness Properties (sorted alongside the code they validate); the
rest are example/unit tests across vision, agent, MCP, data, API, and the
end-to-end pipeline.

### Frontend (Vitest + jest-axe)

```bash
cd frontend
npm test            # single run (vitest run)
npm run build       # tsc --noEmit then a production build
```

**As last run:** `61 passed` across 8 test files, and the production build
completes. Accessibility checks (`jest-axe`) run against the four pages; full
WCAG 2.1 AA conformance additionally requires manual testing with assistive
technologies and expert review, which is not a substitute the automated checks
claim to cover.

## Evaluation

The reproducible labeled set and the honest reporting runner live in
`backend/shelfsentry/eval/`. Run it:

```bash
cd backend
python -c "from shelfsentry.eval import run_evaluation; r = run_evaluation(); \
print('samples', r.sample_count, 'FP', len(r.false_positives), \
'FN', len(r.false_negatives), 'decision_correctness', r.agent_decision_correctness)"
```

**Measured behavior observed from this runner (synthetic labels):** over the
8-sample labeled set, `0` false positives, `2` false negatives on borderline
synthetic gaps, and agent decision-correctness `1.0` over the labeled subset.

These are **synthetic-label results on own-work scenes**, not real-retail
accuracy. The runner marks its output `is_demo=True` and `measured_only=True`
and always carries documented limitations. The false negatives are a real,
reported limitation of the heuristic on borderline gaps, not hidden. Do not
read these numbers as production accuracy.

## Deployment (AWS mode) - container-image Lambda via CloudFormation

AWS mode runs on a **container-image Lambda** (the real OpenCV 5 native
libraries exceed Lambda's zip size limit), provisioned by the **single**
CloudFormation template `infrastructure/template.yaml` (no CDK, no Terraform).
One ARM64 image serves both the API and worker functions; each selects its
handler via `ImageConfig.Command`.

> **You must build and push the ARM64 image to Amazon ECR *before* you deploy
> the CloudFormation stack.** The functions reference an `EcrImageUri`
> parameter; the stack will not create them if the image is absent. Image build
> and push happen **on your machine** (Docker is required) - the Kiro agent
> sandbox cannot build or push container images.

Two steps (full detail, prerequisites, validation, cost, and teardown are in
**[`docs/aws-deployment.md`](docs/aws-deployment.md)**; the editable
architecture diagram is **[`docs/aws-architecture.drawio`](docs/aws-architecture.drawio)**):

```bash
# 1. Build + push the arm64 image to ECR (on your machine; Docker required).
#    Windows PowerShell:
./infrastructure/scripts/build_and_push.ps1 -Region us-east-1 -Repo shelfsentry
#    macOS/Linux:
./infrastructure/scripts/build_and_push.sh us-east-1 shelfsentry
# -> prints the image URI by @sha256 digest (prefer the digest for EcrImageUri)

# 2. Deploy the stack with that image.
aws cloudformation deploy \
  --template-file infrastructure/template.yaml \
  --stack-name shelfsentry-dev \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      EcrImageUri=<acct>.dkr.ecr.us-east-1.amazonaws.com/shelfsentry@sha256:... \
      Environment=dev BedrockModelId=amazon.nova-lite-v1:0 \
  --region us-east-1
```

Read the stack's `ApiUrl` output and point the frontend at it via
`VITE_API_BASE_URL` (see `frontend/.env.example` and `frontend/README.md`). The
agent's decision path uses Amazon Bedrock through the model-agnostic Converse
API; the default model is `amazon.nova-lite-v1:0` (verified available in the
target account/region). The deployed API Lambda serves the exact routes the
frontend client calls (`POST /inspections` multipart, `GET /inspections/{id}`,
`GET /inspections`, `GET /evaluations`, evidence), so the UI is drop-in once
`VITE_API_BASE_URL` is set.

> **What is verified vs. not:** the template passes
> `aws cloudformation validate-template`; Nova Lite was confirmed to return the
> agent's JSON action shape via Converse; the Lambda transport handlers pass a
> dedicated test suite. The **live stack deploy and end-to-end `curl`
> validation have not been performed from this repo** - you run them (see
> `docs/aws-deployment.md` section 5) and verify against your account.
## Cleanup / teardown

Because all stateful resources use `Retain` deletion and update-replace
policies (data-safety; the stack never silently destroys your data), teardown
is a deliberate two-step:

1. **Empty the media S3 bucket first.** A `Retain` bucket cannot be deleted by
   CloudFormation while it still holds objects, so the stack delete will fail
   on a non-empty bucket. Remove the objects explicitly (this deletes data -
   make sure that is what you want):

   ```bash
   aws s3 rm s3://<MediaBucketName output> --recursive
   ```

2. **Delete the stack:**

   ```bash
   aws cloudformation delete-stack --stack-name shelfsentry-dev
   ```

Retained resources (the S3 bucket once emptied, the DynamoDB tables, and the
log groups) may persist by policy after the stack is gone; delete them
individually if you want them fully removed. This teardown has not been
executed from this repo.

## AWS workload and why each service

In AWS mode (provisioned only by `template.yaml`):

| Service | Why it is used |
| --- | --- |
| **S3** | Stores inspection media (images/frames) and annotated evidence. Private, encrypted at rest, public access blocked; large uploads use presigned PUT URLs so bytes never transit the API Lambda. |
| **DynamoDB** | One single table stores inspection status, results, the decision-trace records, async job status, and the media manifest, distinguished by a `record_type` attribute. Key-value access by id fits the access pattern (see `backend/shelfsentry/data/aws_storage.py`). |
| **Lambda** | Two container-image functions (one shared arm64 image): a synchronous inspection API handler and an async worker for longer videos, each with explicit timeout and memory. A container image (not a zip) is required because OpenCV 5's native libraries exceed the zip limit. |
| **API Gateway (HTTP API)** | Fronts the API Lambda as the inspection HTTP endpoint. |
| **CloudWatch Logs + Alarm** | Log groups with explicit retention and a basic errors alarm for observability. |
| **Amazon Bedrock** | The primary agent decision path in AWS mode (the deterministic fallback policy backs it). Scoped `bedrock:InvokeModel` for the configured model only. |
| **IAM** | Least-privilege role per Lambda; no wildcard admin; no embedded secrets. |

## Further documentation

- `docs/architecture.md` - component architecture and the OpenCV -> agent ->
  MCP tool -> new evidence -> final decision sequence diagram.
- `docs/aws-deployment.md` - the full AWS container-image deployment guide
  (build/push to ECR, deploy the stack, validate end to end, cost, teardown).
- `docs/aws-architecture.drawio` - editable AWS architecture diagram
  (implemented vs. planned components and data flow).
- `docs/evaluation.md`, `docs/limitations.md`, `docs/security.md` - measured
  evaluation detail, documented limitations, and the security boundary.
- `.kiro/specs/shelfsentry-ai-active-perception/` - the full requirements,
  design, and task plan.
