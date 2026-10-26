/**
 * TypeScript mirrors of the backend's versioned data contracts and response
 * envelopes (task 13.1; Req 10.4).
 *
 * These types are a faithful, UI-facing projection of the Pydantic v2 contracts
 * in `backend/shelfsentry/data/contracts.py` and the response envelopes in
 * `backend/shelfsentry/api/envelope.py`. They are intentionally thin: the UI
 * consumes exactly the fields the backend emits, including the per-model
 * `schema_version`, the envelope `apiVersion`, and the `mode` label used for
 * demo-vs-real presentation (Req 14.5).
 *
 * Keep this file in sync with the backend contracts when `SCHEMA_VERSION` or the
 * envelope `API_VERSION` changes shape.
 */

// ---------------------------------------------------------------------------
// Shared literal sets (mirror contracts.AGENT_ACTIONS / MCP_TOOL_NAMES)
// ---------------------------------------------------------------------------

/** One of the fixed five agent actions (Req 6.1). */
export type AgentAction =
  | 'inspect_crop'
  | 'analyze_another_frame'
  | 'request_better_image'
  | 'request_human_review'
  | 'finalize_report';

/** One of the four restricted MCP vision tool names (Req 8.2). */
export type McpToolName =
  | 'analyze_shelf_image'
  | 'inspect_region_crop'
  | 'compare_shelf_frames'
  | 'get_annotated_evidence';

/** Operating mode carried on every response and report (Req 17). */
export type ApiMode = 'local' | 'aws';

/** Report lifecycle status. */
export type ReportStatus = 'complete' | 'failed' | 'needs_review';

/** Source of a vision finding. */
export type FindingSource = 'image' | 'video_frame' | 'crop' | 'frame_compare';

/** MCP tool execution outcome. */
export type ToolExecutionStatus = 'ok' | 'rejected' | 'error';

/** Who decided an agent action. */
export type DecidedBy = 'bedrock' | 'fallback' | 'injected';

// ---------------------------------------------------------------------------
// Shared value type
// ---------------------------------------------------------------------------

/**
 * An integer-pixel bounding box, origin top-left, `(x, y, w, h)` relative to the
 * analyzed input's dimensions (Req 5.2).
 */
export interface BoundingBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

// ---------------------------------------------------------------------------
// VisionFinding + CandidateRegion (Req 2, 4, 5, 27)
// ---------------------------------------------------------------------------

/** A single region of interest within an analyzed input (Req 4.2). */
export interface CandidateRegion {
  region_id: string;
  bounding_box: BoundingBox;
  /** Measured empty-shelf fraction in [0, 1], or null when an alt metric is used. */
  empty_space_ratio: number | null;
  /** Documented alternative metric when `empty_space_ratio` is N/A. */
  alt_metric: Record<string, unknown> | null;
  /** Rule-derived ordinal score; explicitly NOT a probability (Req 2.6). */
  heuristic_score: number;
  /** Confidence over the measurement, interpreted via the finding map (Req 4.3). */
  uncertainty_value: number;
  /** Source frame id for video-derived regions (Req 4.4). */
  frame_id: string | null;
  /** Human-readable caveat attached to every region (Req 4.7, 27.4). */
  visual_caveat: string;
}

/** A structured, uncertainty-quantified result from the vision pipeline. */
export interface VisionFinding {
  schema_version: string;
  finding_id: string;
  inspection_id: string;
  source: FindingSource;
  frame_id: string | null;
  /** Exact runtime OpenCV version recorded for the analysis (Req 2.1). */
  opencv_version: string;
  candidate_regions: CandidateRegion[];
  /** Per-uncertainty-value documented meaning (Req 4.3). */
  uncertainty_interpretations: Record<string, string>;
  annotated_output_ref: string;
  suggested_follow_up: AgentAction;
  /** Classical CV results are always labeled estimates (Req 2.5). */
  is_estimate: true;
}

// ---------------------------------------------------------------------------
// Decision trace: AgentDecision | ToolExecution (Req 6, 7)
// ---------------------------------------------------------------------------

/** A record of one action the agent selected, including its rationale. */
export interface AgentDecision {
  schema_version: string;
  /** Discriminator for the decision_trace union. */
  entry_type: 'agent_decision';
  decision_id: string;
  iteration: number;
  action: AgentAction;
  rationale: string;
  args: Record<string, unknown>;
  decided_by: DecidedBy;
  used_fallback: boolean;
  fallback_reason: string | null;
}

/** A record of one MCP tool call the agent made and its result. */
export interface ToolExecution {
  schema_version: string;
  /** Discriminator for the decision_trace union. */
  entry_type: 'tool_execution';
  execution_id: string;
  iteration: number;
  tool: McpToolName;
  args: Record<string, unknown>;
  result_ref: string;
  status: ToolExecutionStatus;
  error: string | null;
}

/**
 * An ordered Decision_Trace entry, discriminated on `entry_type` (Req 6.10).
 * Narrow with `entry.entry_type === 'agent_decision'` / `'tool_execution'`.
 */
export type TraceEntry = AgentDecision | ToolExecution;

// ---------------------------------------------------------------------------
// InspectionReport + summary (Req 6, 12, 13, 16)
// ---------------------------------------------------------------------------

/** The evidence-backed final output of an inspection. */
export interface InspectionReport {
  schema_version: string;
  inspection_id: string;
  mode: ApiMode;
  final_recommendation: string;
  requires_human_review: boolean;
  initial_finding_id: string;
  decision_trace: TraceEntry[];
  supporting_evidence_refs: string[];
  created_at: string;
  status: ReportStatus;
}

/**
 * A compact History-list row projected from an `InspectionReport` (Req 13.1).
 * Mirrors `storage.InspectionSummary`.
 */
export interface InspectionSummary {
  inspection_id: string;
  mode: ApiMode;
  status: ReportStatus;
  final_recommendation: string;
  requires_human_review: boolean;
  created_at: string;
}

// ---------------------------------------------------------------------------
// EvaluationResult (Req 14, 25, 27)
// ---------------------------------------------------------------------------

/** A single labeled example from the evaluation set (measured only; Req 25.2). */
export interface EvalExample {
  sample_id: string;
  inspection_id: string | null;
  expected: string;
  actual: string;
  note: string | null;
}

/** A record of measured performance against a labeled evaluation set. */
export interface EvaluationResult {
  schema_version: string;
  eval_run_id: string;
  is_demo: boolean;
  sample_count: number;
  false_positives: EvalExample[];
  false_negatives: EvalExample[];
  /** Measured-only decision correctness in [0, 1] (Req 14.4, 27.2). */
  agent_decision_correctness: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
  documented_limitations: string[];
  /** Results are measured only and never fabricated (Req 25.4, 27.2). */
  measured_only: true;
}

/**
 * A summary of an inspection's primary vision finding, surfaced for the UI.
 * Not a distinct backend contract; some endpoints may inline a `VisionFinding`.
 */
export type InspectionSummaryFinding = VisionFinding;

// ---------------------------------------------------------------------------
// Response envelopes (mirror api/envelope.py) (Req 10.4, 10.5)
// ---------------------------------------------------------------------------

/** Stable machine-readable error codes the API layer may return. */
export type ApiErrorCode = 'validation' | 'not_found' | 'bad_request';

/** The structured `error` object inside an error envelope (Req 10.5). */
export interface ApiErrorDetail {
  code: ApiErrorCode;
  message: string;
  /** Offending field/condition when applicable (Req 10.5). */
  field?: string | null;
  /** Optional structured context (never raw media bytes). */
  details?: Record<string, unknown> | null;
}

/** Success envelope: `{ apiVersion, mode, data }` (Req 10.4). */
export interface ApiSuccess<T> {
  apiVersion: string;
  mode: ApiMode;
  data: T;
}

/** Structured error envelope: `{ apiVersion, mode, error }` (Req 10.4, 10.5). */
export interface ApiErrorEnvelope {
  apiVersion: string;
  mode: ApiMode;
  error: ApiErrorDetail;
}

/**
 * The full response union the server returns for any endpoint: either a typed
 * success payload or a structured error, both carrying `apiVersion` and `mode`.
 */
export type ApiResponse<T> = ApiSuccess<T> | ApiErrorEnvelope;

// ---------------------------------------------------------------------------
// Annotated-evidence descriptor (mirror handlers._AnnotatedEvidence)
// ---------------------------------------------------------------------------

/**
 * The transport descriptor returned by the annotated-evidence endpoint. The
 * bytes arrive base64-encoded over JSON; a transport adapter may instead serve
 * raw bytes, in which case the client exposes a blob URL (see client.ts).
 */
export interface AnnotatedEvidence {
  inspection_id: string;
  finding_id: string;
  ref: string;
  byte_length: number;
  /** Base64-encoded annotated image bytes, when delivered inline. */
  data_base64?: string;
  /**
   * Optional projection of the initial VisionFinding, derived from a real
   * OpenCV 5 analysis of the stored image. Present when the backend can derive
   * it; absent (never fabricated) otherwise. The Results page renders region
   * overlays and per-region detail from this when present.
   */
  primary_finding?: {
    candidate_regions: CandidateRegion[];
    uncertainty_interpretations: Record<string, string>;
    opencv_version: string;
  };
}
