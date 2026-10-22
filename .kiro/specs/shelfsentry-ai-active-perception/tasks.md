# Implementation Plan: ShelfSentry AI (Active Perception)

## Overview

This plan builds ShelfSentry AI greenfield, **local/mock first**, with AWS as a verified follow-on. Early waves stand up typed contracts, the OpenCV 5 vision pipeline, the bounded Strands agent loop, the MCP server, local storage, the local API, and the frontend so the full active-perception demo is reachable **without AWS credentials**. Later waves add the CloudFormation stack and the AWS integration. Documentation and submission artifacts, and every task that depends on a live AWS deployment, are marked optional (`*`) so local/mock demo value does not depend on them.

Backend is Python (Pydantic v2, OpenCV 5, Strands Agents, Python MCP SDK). Frontend is React + TypeScript + Vite + Tailwind. Property-based tests follow the design's Correctness Properties (Properties 1-23) and run with at least 100 generated iterations each.

## Tasks

- [x] 1. Project scaffolding, typed data contracts, and OpenCV 5 version verification
  - [x] 1.1 Create backend project structure and pinned dependencies
    - Create `backend/` package layout (`shelfsentry/` with `vision/`, `agent/`, `mcp/`, `data/`, `api/`, `config/`, `eval/` subpackages) and a `tests/` tree
    - Add `pyproject.toml`/`requirements.txt` pinning exact versions for OpenCV 5 (`opencv-python`), NumPy, Pydantic v2, Strands Agents SDK, the Python MCP SDK, and the test stack (pytest + a property-based testing library such as Hypothesis)
    - Configure the test runner for single-run execution (no watch mode) and property tests at >=100 iterations
    - _Requirements: 26.4_

  - [x] 1.2 Define the seven typed Pydantic v2 contracts
    - Implement `InspectionRequest`, `VisionFinding`, `CandidateRegion`, `AgentDecision`, `ToolExecution`, `InspectionReport` (with `TraceEntry` union), and `EvaluationResult` with `schema_version` on models and shared `BoundingBox`/`AgentAction` types
    - Implement boundary validation that rejects malformed data with a field-identifying structured error, and canonical JSON serialization (sorted keys, stable formatting)
    - _Requirements: 10.1, 10.2, 10.3, 10.5_

  - [x]* 1.3 Write property tests for contract serialization and boundary validation
    - **Property 21: Equivalent typed objects serialize identically**
    - **Validates: Requirements 10.3**
    - Add example-based tests that malformed boundary data is rejected with a field-identifying error (Req 10.2, 10.5)

  - [x] 1.4 Implement typed configuration and OpenCV 5 startup version verification
    - Implement `VisionConfig`/`AgentConfig` and config loading (accepted content types, max size, max frame count, kernel sizes/thresholds, `max_iterations` default 4)
    - Implement `vision/version.py` `opencv_version()` that reads `cv2.__version__`, verifies it is OpenCV 5 at startup, and fails loudly otherwise
    - _Requirements: 2.1, 6.6, 26.4_

  - [x]* 1.5 Write unit test for OpenCV 5 version verification
    - Assert `opencv_version()` returns the exact runtime version and that a non-5 version fails startup
    - _Requirements: 2.1, 26.4_

- [x] 2. Vision pipeline: validation and preprocessing
  - [x] 2.1 Implement input validation
    - Implement `vision/validate.py` `validate_input(...)` checking empty file, decodability as image/video (`cv2.imdecode`, `cv2.VideoCapture`), accepted content type, and max size, each raising a structured validation error naming the specific condition
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 2.2 Implement preprocessing
    - Implement `vision/preprocess.py` (decode, BGR->gray/HSV via `cv2.cvtColor`, `cv2.GaussianBlur`, histogram equalization/CLAHE) with fixed, config-driven parameters for determinism
    - _Requirements: 2.2_

  - [x]* 2.3 Write property tests for input validation
    - **Property 1: Bad inputs are always rejected with the matching condition**
    - **Validates: Requirements 1.3, 1.4, 1.5, 1.6**
    - **Property 2: Valid inputs are accepted**
    - **Validates: Requirements 1.7**

- [x] 3. Vision pipeline: OpenCV 5 analysis, scoring, annotation, and video
  - [x] 3.1 Implement shelf-region, edge, contour, and morphology analysis
    - Implement `vision/analyze.py` using Sobel/gradient + horizontal projection (optional `cv2.HoughLinesP`) for shelf bands, `cv2.Canny` edges, `cv2.findContours`/`boundingRect`/`contourArea`, `cv2.morphologyEx`, and Otsu/adaptive thresholding
    - Sort contours by a stable key (top-left `(y, x)` then area) and emit `(x, y, w, h)` integer-pixel boxes in one documented coordinate system for deterministic output
    - _Requirements: 2.2, 5.1, 5.2_

  - [x] 3.2 Implement scoring, uncertainty, caveats, and finding assembly
    - Implement `vision/scoring.py` computing measured `empty_space_ratio` (or a documented alternative metric), a `heuristic_score` kept distinct from any probability, and an `uncertainty_value`
    - Assemble `VisionFinding` with per-finding uncertainty interpretations, `is_estimate=True`, a visual caveat on each candidate region, a `suggested_follow_up`, and the recorded exact OpenCV version
    - _Requirements: 2.3, 2.5, 2.6, 4.1, 4.2, 4.3, 4.6, 4.7, 27.4_

  - [x] 3.3 Implement annotation rendering
    - Implement `vision/annotate.py` `render_annotation(...)` drawing each candidate region at exactly the recorded bounding-box coordinates and returning the `Annotated_Output` bytes; wire `annotated_output_ref` into the finding
    - _Requirements: 2.4, 4.5, 5.3_

  - [x] 3.4 Implement video frame sampling and per-frame analysis
    - Implement `vision/video.py` uniform sampling of at most `M` evenly spaced frame indices, assign stable unique `frame_id`s (`frame_{index:06d}`), analyze each frame with the same image path, and set `frame_id` on video-derived regions/findings
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 4.4_

  - [x] 3.5 Implement region inspection and frame comparison entry points
    - Implement `inspect_region(data, box, cfg)` (crop re-analysis) and `compare_frames(a, b, cfg)` returning `VisionFinding`s, reusing the shared analysis path; expose the full vision public interface
    - _Requirements: 2.2, 4.1_

- [x] 4. Vision test suite
  - [x]* 4.1 Write candidate-region field and caveat property tests
    - **Property 3: Every finding records the exact OpenCV version**
    - **Validates: Requirements 2.1**
    - **Property 4: All vision results are labeled as estimates**
    - **Validates: Requirements 2.5**
    - **Property 5: Every candidate region carries required, well-typed fields**
    - **Validates: Requirements 2.6, 4.2**
    - **Property 6: Every uncertainty value has a documented interpretation**
    - **Validates: Requirements 4.3**
    - **Property 7: Empty-looking regions always carry a visual caveat**
    - **Validates: Requirements 4.7, 27.4**

  - [x]* 4.2 Write reproducibility and annotation-match property tests
    - **Property 8: Annotation coordinates are reproducible**
    - **Validates: Requirements 5.1**
    - **Property 9: Annotations match recorded bounding boxes**
    - **Validates: Requirements 5.3**
    - Covers the repeated-run coordinate consistency test (Req 22.4)

  - [x]* 4.3 Write video sampling property tests
    - **Property 10: Video sampling stays within the configured limit**
    - **Validates: Requirements 3.1, 3.3, 3.4**
    - **Property 11: Sampled frames have unique identifiers**
    - **Validates: Requirements 3.2**
    - Covers video frame extraction test (Req 22.3)

  - [x]* 4.4 Write unit tests for normal and adverse conditions
    - Cover valid/invalid/empty/corrupt inputs and varied lighting, shadows, dividers, occlusions, and multiple regions; assert region fields and visual caveats
    - _Requirements: 22.1, 22.2_

- [x] 5. Checkpoint - vision pipeline verified
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Agent loop: decision model abstraction and fallback policy
  - [x] 6.1 Define the DecisionModel interface and the five-action set
    - Implement the `DecisionModel` protocol, the fixed `AgentAction` set `{inspect_crop, analyze_another_frame, request_better_image, request_human_review, finalize_report}`, and `InjectedDecisionModel` for offline/causality tests
    - _Requirements: 6.1, 7.3_

  - [x] 6.2 Implement the deterministic FallbackPolicy
    - Implement `FallbackPolicy.decide(...)` as a pure function of finding + trace per the reference logic (high uncertainty -> inspect_crop; ambiguous + frames remain -> analyze_another_frame; poor quality -> request_better_image; conflicting evidence -> request_human_review; sufficient/cap -> finalize_report)
    - _Requirements: 7.4_

  - [x]* 6.3 Write property tests for the fallback policy
    - **Property 17: Fallback policy is deterministic**
    - **Validates: Requirements 7.4**
    - **Property 19: Ambiguous findings gather more evidence**
    - **Validates: Requirements 23.3**
    - _Requirements: 23.3_

- [x] 7. Agent loop: tool-arg validation, bounded loop, and trace
  - [x] 7.1 Implement tool-argument validation
    - Implement `validate_tool_args(decision)` that validates arguments before execution and rejects invalid invocations with a structured error, recording `ToolExecution.status="rejected"`
    - _Requirements: 6.4_

  - [x] 7.2 Implement the bounded agent loop with decision trace and finalization
    - Implement `run_agent_loop(...)`: record each `AgentDecision`, validate args and call tools via the `MCPToolClient`, record each `ToolExecution`, merge returned evidence into the next iteration, cap at `max_iterations` (default 4), stop on sufficiency, stop and mark `needs_review` on `request_human_review`, and finalize an `InspectionReport` referencing supporting trace entries
    - _Requirements: 6.3, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10_

  - [x] 7.3 Wire the Bedrock decision path with recorded fallback
    - Implement `BedrockDecisionModel` as the primary path in AWS Mode that parses a validated action selection, and switch to `FallbackPolicy` recording `used_fallback=true` with a reason when Bedrock is unavailable or errors
    - _Requirements: 7.1, 7.2_

  - [x]* 7.4 Write property tests for loop bounds, causality, trace, and fallback
    - **Property 12: Findings causally influence the selected action**
    - **Validates: Requirements 6.2**
    - **Property 13: Invalid tool arguments are rejected before execution**
    - **Validates: Requirements 6.4, 8.3**
    - **Property 14: Iteration count is bounded**
    - **Validates: Requirements 6.6, 6.7**
    - **Property 15: Reports reference existing trace evidence**
    - **Validates: Requirements 6.10**
    - **Property 16: Model errors cause a recorded fallback**
    - **Validates: Requirements 7.2**
    - **Property 18: inspect_crop consumes the preceding finding's region**
    - **Validates: Requirements 23.2**
    - **Property 20: Conflicting evidence triggers human review**
    - **Validates: Requirements 23.4**

  - [x]* 7.5 Write unit tests for agent causality, safety, and trace integrity
    - Different findings -> different actions (Req 23.1); max iterations respected (Req 23.5); invalid args rejected (Req 23.6); model error -> fallback used and recorded (Req 23.7); recorded Decision_Trace matches the tool calls made (Req 23.8)
    - _Requirements: 23.1, 23.5, 23.6, 23.7, 23.8_

- [x] 8. Checkpoint - agent loop verified
  - Ensure all tests pass, ask the user if questions arise.

- [x] 9. MCP server with restricted vision tools
  - [x] 9.1 Implement the MCP server and four tools reusing vision functions
    - Implement the server with a maintained Python MCP SDK over a Strands-compatible transport exposing `analyze_shelf_image`, `inspect_region_crop`, `compare_shelf_frames`, and `get_annotated_evidence`, each delegating to the existing vision functions (no duplicate image-processing logic) and resolving media by inspection id through the `StorageAdapter`
    - _Requirements: 8.1, 8.2, 8.4_

  - [x] 9.2 Implement MCP tool-schema argument validation
    - Validate every invocation against the tool's JSON schema and reject invalid arguments before execution with a structured error
    - _Requirements: 8.3_

  - [x]* 9.3 Write MCP argument-validation and security-boundary tests
    - Assert tool schemas expose no filesystem path, arbitrary URL, shell-execution, or unrestricted-credential argument, and that invalid arguments are rejected (Property 13)
    - **Property 13: Invalid tool arguments are rejected before execution**
    - **Validates: Requirements 6.4, 8.3**
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 10. Local storage adapter and end-to-end local pipeline
  - [x] 10.1 Implement the StorageAdapter protocol and LocalStorageAdapter
    - Define the `StorageAdapter` protocol (media put/get, report save/get, list_reports, job save/get) and implement `LocalStorageAdapter` using JSON records and files on disk; add bundled licensed sample inputs for example mode
    - _Requirements: 15.4_

  - [x] 10.2 Wire the mode-agnostic end-to-end local inspection flow
    - Compose validate -> analyze -> agent loop (injected/deterministic model) -> MCP tools -> persist report through `LocalStorageAdapter`, running real OpenCV 5 with no AWS credentials
    - _Requirements: 15.1, 15.2, 15.3_

  - [x]* 10.3 Write an integration test for the local end-to-end pipeline
    - Run a bundled sample through the full local flow and assert a persisted `InspectionReport` with a populated decision trace
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

- [x] 11. Local API layer and mode configuration switch
  - [x] 11.1 Implement local API handlers with versioned responses
    - Implement API handlers that validate requests, route to the pipeline, persist via the adapter, and return responses carrying `apiVersion` and `mode`; include history listing with search/filter and evaluation record retrieval
    - _Requirements: 10.4, 16.1_

  - [x] 11.2 Implement the fail-loud mode configuration switch
    - Implement the `SHELFSENTRY_MODE=local|aws` `Config_Switch` resolved once at startup; in AWS Mode validate required configuration and fail startup with a message naming the missing configuration; never substitute local/mock results while AWS Mode is selected
    - _Requirements: 16.4, 17.1, 17.2, 17.3_

  - [x]* 11.3 Write property and example tests for API versioning and mode switch
    - **Property 22: Every API response includes an API version**
    - **Validates: Requirements 10.4**
    - Add example tests: API request validation and rejection of invalid requests (Req 24.1); AWS Mode with missing config fails startup with a naming message and no local substitution (Req 17.2, 17.3)
    - _Requirements: 24.1_

- [x] 12. Checkpoint - local/mock demo reachable end to end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Frontend application
  - [x] 13.1 Scaffold the frontend and typed API client
    - Set up React + TypeScript + Vite + Tailwind; implement a thin typed API client mirroring the versioned contracts, surfacing structured validation errors, and reading the `mode` field for demo-vs-real labeling
    - _Requirements: 10.4_

  - [x] 13.2 Implement the Inspection page
    - Upload image/video or pick a bundled sample; client-side type/size validation mirroring server limits; inspection progress; error and empty states; WCAG 2.1 AA conformance
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6_

  - [x] 13.3 Implement the Results page
    - Original + annotated side-by-side with highlighted regions; per-region measured metric and uncertainty interpretation; agent initial finding, chosen action + rationale, tool results; final recommendation + human-review status; WCAG 2.1 AA conformance
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x] 13.4 Implement the History page
    - List with status + timestamp; search/filter controls; detail drill-in; empty state; WCAG 2.1 AA conformance
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

  - [x] 13.5 Implement the Evaluation page
    - Display measured test-set performance from `EvaluationResult` records (FP/FN examples, decision correctness, latency), documented limitations, demo-vs-real distinction, measured-only with no fabricated metrics; WCAG 2.1 AA conformance
    - _Requirements: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 27.2_

  - [x]* 13.6 Write frontend component and accessibility tests
    - Test empty/error/progress states, demo-vs-real labeling, and automated accessibility checks across the four pages
    - _Requirements: 11.6, 12.5, 13.5, 14.6_

- [x] 14. Reproducible evaluation set and honest reporting
  - [x] 14.1 Implement the labeled evaluation set and runner
    - Add a small reproducible labeled evaluation set and a runner that produces `EvaluationResult` records with measured false positives, false negatives, agent decision correctness, and latency, plus documented limitations; report measured results only
    - _Requirements: 25.1, 25.2, 25.3, 25.4_

  - [x]* 14.2 Write property test for honest reporting
    - **Property 23: Only measured metrics are reported**
    - **Validates: Requirements 14.4, 25.4, 27.2**

- [x] 15. CloudFormation infrastructure template
  - [x]* 15.1 Author the CloudFormation template.yaml
    - Define all AWS resources in a single `template.yaml`: private/encrypted/public-access-blocked S3 bucket with lifecycle/retention; DynamoDB table(s) for status/results/decision-trace/jobs with configurable names and deletion/retention policies; Lambda functions with explicit timeout+memory (vision, API, worker); API Gateway HTTP API; CloudWatch log groups with explicit retention and basic alarms; least-privilege IAM per component; presigned-upload Lambda; thin async job path (job record + worker entry point); no embedded secrets
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5, 19.1, 19.2, 19.3, 19.4, 19.5, 20.1, 20.2, 20.3_

  - [x]* 15.2 Write CloudFormation template validation and security-constraint tests
    - Validate `template.yaml` (cfn-lint / validate-template) and assert S3 private+encrypted+public-access-blocked, least-privilege IAM, and retention/deletion policies present
    - _Requirements: 24.3, 24.4_

- [x] 16. AWS integration
  - [x]* 16.1 Implement the AwsStorageAdapter
    - Implement `AwsStorageAdapter` using S3 for media and DynamoDB for status/results/decision-trace/job records, including presigned-upload minting, behind the same `StorageAdapter` protocol
    - _Requirements: 16.1, 18.3, 18.4_

  - [x]* 16.2 Wire AWS Mode decision path and async worker with retry/failure handling
    - Enable the real Strands + Bedrock primary path in AWS Mode, implement the async worker entry point that records job status in DynamoDB for polling, and apply configured retry/backoff surfacing a failed status when retries are exhausted (never substituting mock results)
    - _Requirements: 16.2, 16.3, 16.4, 19.4, 19.5_

  - [x]* 16.3 Write AWS integration, API, and infrastructure-security tests
    - Cover missing S3 objects, failed jobs, and result retrieval; API request validation; IAM and storage security constraints
    - _Requirements: 24.1, 24.2, 24.4_

- [x] 17. Documentation and submission artifacts
  - [x]* 17.1 Write README and architecture documentation
    - Write a README covering setup, architecture, testing, deployment, and cleanup, and `docs/architecture.md` containing the OpenCV -> agent -> MCP -> new-evidence -> final-decision sequence diagram
    - _Requirements: 26.1, 26.2_

  - [x]* 17.2 Write evaluation, limitations, and security docs plus pinned-dependency notes
    - Write `docs/evaluation.md`, `docs/limitations.md`, and `docs/security.md`; document pinned dependency versions and the exact OpenCV 5 version and where it performs substantive analysis
    - _Requirements: 26.3, 26.4_

  - [x]* 17.3 Assemble submission artifacts and teardown guide
    - Include licensed sample inputs, a 5-minute demo script, a technical report, and visual-to-action evidence (screenshots or recordings); write the teardown guide describing stack removal honoring deletion/retention policies
    - _Requirements: 26.5, 21.1, 21.2_

- [x] 18. Final checkpoint - full system verified
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional. Optional sub-tasks here are either tests or AWS-deployment-dependent and documentation work; skipping them still leaves the local/mock demo fully reachable (end of Task 12).
- The local/mock pipeline (Tasks 1-12) runs real OpenCV 5 and the full agent + MCP loop with no AWS credentials, and is the primary path for tests and the demo.
- AWS work (Tasks 15-16) is a verified follow-on; the `template.yaml` is the only infrastructure definition.
- Each task references specific requirement sub-clauses for traceability.
- Property tests reference the design's Correctness Properties and are placed close to the implementation they validate.
- Infrastructure checks (CloudFormation validation, IAM/storage constraints) use integration/snapshot assertions rather than property-based testing.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.4"] },
    { "id": 2, "tasks": ["1.3", "1.5", "2.1", "2.2"] },
    { "id": 3, "tasks": ["2.3", "3.1"] },
    { "id": 4, "tasks": ["3.2"] },
    { "id": 5, "tasks": ["3.3", "3.4", "3.5"] },
    { "id": 6, "tasks": ["4.1", "4.2", "4.3", "4.4"] },
    { "id": 7, "tasks": ["6.1"] },
    { "id": 8, "tasks": ["6.2"] },
    { "id": 9, "tasks": ["6.3", "7.1"] },
    { "id": 10, "tasks": ["7.2"] },
    { "id": 11, "tasks": ["7.3"] },
    { "id": 12, "tasks": ["7.4", "7.5"] },
    { "id": 13, "tasks": ["9.1"] },
    { "id": 14, "tasks": ["9.2"] },
    { "id": 15, "tasks": ["9.3", "10.1"] },
    { "id": 16, "tasks": ["10.2"] },
    { "id": 17, "tasks": ["10.3", "11.1", "11.2"] },
    { "id": 18, "tasks": ["11.3", "13.1"] },
    { "id": 19, "tasks": ["13.2", "13.3", "13.4", "13.5"] },
    { "id": 20, "tasks": ["13.6", "14.1", "15.1", "16.1"] },
    { "id": 21, "tasks": ["14.2", "15.2", "16.2", "17.1", "17.2"] },
    { "id": 22, "tasks": ["16.3", "17.3"] }
  ]
}
```
