# ShelfSentry AI - Architecture

This document describes the component architecture of ShelfSentry AI and
contains the end-to-end sequence diagram for the active-perception loop:
OpenCV analysis -> agent -> MCP tool -> new evidence -> final decision.

Every component named here maps to a real module in the repository; paths are
given so the description can be checked against the code.

## Design principles

- **Perception is independent of the LLM.** The OpenCV 5 vision pipeline has no
  LLM or agent dependency. It produces measured findings and quantified
  uncertainty; the agent reasons *over* those findings but never performs
  pixel work itself.
- **Typed contracts at every seam.** All cross-component data is a validated
  Pydantic v2 model, so boundaries reject malformed data with a
  field-identifying error and serialize canonically.
- **The agent loop is bounded and auditable.** It iterates a configurable,
  capped number of times, records every decision and tool call in an ordered
  trace, and the final report cites that trace.
- **The agent reaches perception only through restricted MCP tools.** The loop
  never imports the MCP SDK directly and tools expose no filesystem, URL,
  shell, or credential surface.
- **One storage seam, two implementations.** A `StorageAdapter` protocol backs
  both local on-disk storage and AWS (S3 + DynamoDB), so the same orchestration
  runs in either mode unchanged.
- **Honest by construction.** Heuristic scores are kept distinct from
  probabilities, empty regions always carry visual caveats, and the evaluation
  runner reports measured values only.

## Components

### 1. Vision pipeline (OpenCV 5) - `backend/shelfsentry/vision/`

Independent of the agent/LLM. Stages:

- `validate.py` - decodability, content type, size, empty-file checks; raises a
  structured validation error naming the specific condition.
- `preprocess.py` - deterministic decode, grayscale/HSV, blur, CLAHE.
- `analyze.py` - **the substantive classical analysis**:
  `cv2.Sobel`/`cv2.reduce` shelf-band detection (optional `cv2.HoughLinesP`),
  Otsu + adaptive thresholding, `cv2.morphologyEx`, `cv2.Canny`, and
  `cv2.findContours`/`boundingRect`/`contourArea`, emitting
  deterministically-ordered candidate regions.
- `scoring.py` - the measured `empty_space_ratio`, the ordinal
  `heuristic_score` (never a probability), the measurement-confidence
  `uncertainty_value` with a documented interpretation, the per-region visual
  caveat, and `VisionFinding` assembly (records the verified OpenCV version,
  `is_estimate=True`).
- `annotate.py` - draws regions at exactly the recorded bounding boxes.
- `video.py` - uniform frame sampling bounded by `max_frame_count`, stable
  frame ids, per-frame analysis via the same image path.
- `version.py` - startup OpenCV 5 verification (`cv2.__version__`, fails loudly
  on non-5).

Determinism is a hard requirement: every stage is a pure, config-driven
function of the input bytes plus `VisionConfig`, so repeated runs produce
byte-identical boxes.

### 2. Typed data contracts - `backend/shelfsentry/data/contracts.py`

The seven typed contracts - `InspectionRequest`, `VisionFinding`,
`CandidateRegion`, `AgentDecision`, `ToolExecution`, `InspectionReport`
(with a `TraceEntry` union), and `EvaluationResult` - plus shared
`BoundingBox`/`AgentAction` types. Each carries a `schema_version`, validates
at boundaries, and serializes to canonical JSON.

### 3. Agent loop (Strands, bounded) - `backend/shelfsentry/agent/`

- `decision.py` - the `DecisionModel` protocol, the fixed five-action set
  `{inspect_crop, analyze_another_frame, request_better_image,
  request_human_review, finalize_report}`, and `InjectedDecisionModel` for
  offline/causality tests.
- `fallback.py` - the deterministic `FallbackPolicy`: a pure function of
  finding + trace (high uncertainty -> `inspect_crop`; ambiguous + frames
  remain -> `analyze_another_frame`; poor quality -> `request_better_image`;
  conflicting evidence -> `request_human_review`; sufficient/at cap ->
  `finalize_report`).
- `bedrock.py` - `BedrockDecisionModel`, the **primary** decision path in AWS
  mode; raises on unavailability/error so the loop can record a fallback.
- `tools.py` - `validate_tool_args`, which rejects invalid tool invocations
  before execution and records a `ToolExecution(status="rejected")`.
- `loop.py` - `run_agent_loop(...)`: records each `AgentDecision`, terminates
  on `finalize_report` (sufficiency) or `request_human_review` (escalation),
  otherwise validates args and executes the backing tool via a small
  `ToolClient` protocol, merges returned evidence into the finding for the next
  iteration, caps at `max_iterations` (default 4), and finalizes an
  `InspectionReport` that references the ordered decision trace.

The loop depends only on the `DecisionModel` and `ToolClient` protocols - never
on the MCP SDK - which is what lets the same loop run with a fake client in
tests, the local dispatcher in local mode, and a real MCP client in AWS mode.

### 4. MCP server (restricted tools) - `backend/shelfsentry/mcp/server.py`

Exposes exactly four tools over a Strands-compatible transport -
`analyze_shelf_image`, `inspect_region_crop`, `compare_shelf_frames`,
`get_annotated_evidence` - each delegating to the vision functions (no
duplicate image-processing logic) and resolving media **by inspection id**
through a `MediaResolver`. Every invocation is validated against the tool's
schema before execution. The capability boundary is strict: no tool argument is
a filesystem path, an arbitrary URL, a shell command, or a credential.

`VisionToolDispatcher` holds the tool logic and is importable without starting
a transport, so the local pipeline and tests exercise the exact same real tool
surface the agent uses.

### 5. Storage adapter - `backend/shelfsentry/data/storage.py`, `data/aws_storage.py`

A `StorageAdapter` protocol (media put/get, report save/get, list, job
save/get) with two implementations: `LocalStorageAdapter` (JSON records + files
on disk) and `AwsStorageAdapter` (S3 for media/evidence, DynamoDB for
status/results/decision-trace/jobs, presigned-upload minting). A
`StorageMediaResolver` shim lets the MCP tools resolve media by id over either.

### 6. Mode switch - `backend/shelfsentry/config/mode.py`

`SHELFSENTRY_MODE=local|aws` resolves once at startup into a frozen
`RuntimeConfig`. Local mode requires no AWS config. AWS mode requires the
non-secret resource identifiers (S3 bucket, DynamoDB table, region) and fails
loudly naming every missing key; it never silently falls back to local.

### 7. Pipeline orchestration - `backend/shelfsentry/pipeline.py`

`run_local_inspection(...)` composes validate -> store -> perceive (via the MCP
dispatcher) -> decide (bounded loop with the deterministic model) -> persist,
all locally with no network. It depends only on the `StorageAdapter` and
`DecisionModel` protocols, so swapping in the AWS adapter and the Bedrock model
runs the identical flow in AWS mode.

### 8. API layer - `backend/shelfsentry/api/`

Mode-aware handlers (`build_local_api` / `LocalApi`) that validate requests,
route to the pipeline, persist via the adapter, and return versioned,
mode-stamped envelopes (`apiVersion`, `mode`, `data | error`). The envelopes
are framework-agnostic Pydantic models - a FastAPI/Lambda adapter would
serialize them; this repo does not wire an HTTP server.

### 9. Frontend - `frontend/src/`

React + TypeScript + Vite + Tailwind. Four pages (`pages/`): Inspection,
Results, History, Evaluation. A thin typed API client (`api/`) mirrors the
versioned contracts, surfaces structured validation errors, and reads the
`mode` field for demo-vs-real labeling.

### 10. Infrastructure - `infrastructure/template.yaml`

The single CloudFormation stack for AWS mode (S3, DynamoDB, Lambda, API Gateway
HTTP API, CloudWatch, least-privilege IAM, Bedrock invoke scope). Authored and
template-validatable; **not deployed from this repo**.

## The active-perception feedback loop

The core flow turns an initial OpenCV finding into an evidence-backed decision
by looping: the agent picks an action based on the finding's uncertainty, the
MCP tool runs real perception, the returned finding merges back in, and the
loop repeats until it finalizes or escalates - bounded by `max_iterations`.

The deterministic `FallbackPolicy` is the local decision model and the recorded
safe fallback in AWS mode. `BedrockDecisionModel` is the primary path in AWS
mode; it has not been exercised against real AWS from this repo, so the diagram
shows the decision as "DecisionModel" with the local/deterministic model as the
exercised implementation.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant API as API / pipeline<br/>(run_local_inspection)
    participant Vision as Vision pipeline<br/>(vision/analyze + scoring)
    participant Loop as Agent loop<br/>(agent/loop.run_agent_loop)
    participant Model as DecisionModel<br/>(FallbackPolicy / Bedrock)
    participant MCP as MCP tools<br/>(VisionToolDispatcher)
    participant Store as StorageAdapter<br/>(Local / AWS)

    User->>API: submit InspectionRequest (image/video bytes)
    API->>Vision: validate + preprocess + analyze (OpenCV 5)
    Vision-->>API: VisionFinding (regions, empty_space_ratio,<br/>heuristic_score, uncertainty_value, caveats)
    Note over Vision,MCP: the first finding is produced through<br/>the MCP analyze_shelf_image tool
    API->>Store: store media + annotated evidence
    API->>Loop: run_agent_loop(initial finding, model, tools, cfg)

    loop until finalize / human-review / max_iterations (default 4)
        Loop->>Model: decide(finding, trace)
        Model-->>Loop: AgentDecision (action + rationale)
        Note right of Loop: record AgentDecision in the trace
        alt action needs a tool (inspect_crop / analyze_another_frame)
            Loop->>Loop: validate_tool_args(decision)
            Loop->>MCP: inspect_region_crop / next frame
            MCP->>Vision: re-run OpenCV 5 on crop / next frame
            Vision-->>MCP: new VisionFinding
            MCP->>Store: store new annotated evidence
            MCP-->>Loop: ToolResult (new finding + evidence ref)
            Note right of Loop: record ToolExecution;<br/>merge new evidence into finding
        else finalize_report (sufficient evidence)
            Loop->>Loop: stop iterating
        else request_human_review (conflicting evidence)
            Loop->>Loop: stop; mark needs_review
        end
    end

    Loop-->>API: InspectionReport (recommendation + decision trace)
    API->>Store: save_report(report)
    API-->>User: versioned response (apiVersion, mode, report)
```

Key points the diagram reflects, grounded in the code:

- The **first** finding and every follow-up probe go through the same real
  OpenCV 5 perception seam (the MCP dispatcher), so there is no second,
  divergent analysis path.
- Each iteration appends an `AgentDecision` and, when a tool runs, a
  `ToolExecution` immediately after, sharing one iteration index - the trace is
  causal and ordered (`agent/loop.py`).
- Invalid tool arguments are rejected **before** execution; the loop records
  the rejection and finalizes rather than spinning, and the `max_iterations`
  cap independently guarantees termination.
- The final `InspectionReport` references the decision trace that supports its
  recommendation and carries the human-review status.

## Honesty notes

- The **deterministic fallback policy** is the exercised local decision model
  and the recorded safe fallback in AWS mode. Amazon **Bedrock** is the
  intended primary path in AWS mode but has not been run against real AWS from
  this repo.
- The **AWS stack** (`template.yaml`) is authored and template-validatable but
  not deployed here.
- Evaluation numbers come from the runner over **synthetic** labels; see
  `docs/evaluation.md` and `docs/limitations.md`.
