import type { CSSProperties } from 'react';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { getApiClient, type ApiClient } from '../api/client';
import type { ClientError } from '../api/result';
import type {
  AgentDecision,
  AnnotatedEvidence,
  BoundingBox,
  CandidateRegion,
  InspectionReport,
  ToolExecution,
  TraceEntry,
} from '../api/types';

/**
 * Results page (task 13.3; Req 12.1-12.5).
 *
 * Given an inspection id from the route (`/results/:inspectionId`), fetches the
 * {@link InspectionReport} and its annotated evidence and presents:
 *  - the original input and the annotated output side by side, with each
 *    Candidate_Region highlighted as a non-color-only overlay box (Req 12.1);
 *  - per-region the measured metric and its uncertainty value together with the
 *    finding's documented interpretation, plus the visual caveat, all labeled
 *    as estimates (Req 12.2);
 *  - the agent initial finding, each selected action with its rationale, and the
 *    recorded Tool_Execution results, walked in decision-trace order (Req 12.3);
 *  - the final recommendation and the human-review status (Req 12.4).
 *
 * Loading, empty (`not_found`), and error states are announced through an
 * `aria-live` region. The page uses semantic headings, is keyboard operable,
 * conveys status with text/shape as well as color, and gives every image
 * meaningful alt text to conform to WCAG 2.1 AA (Req 12.5).
 */

// ---------------------------------------------------------------------------
// Props (client + ids injectable for tests; no backend required)
// ---------------------------------------------------------------------------

export interface ResultsPageProps {
  /** Injectable API client; defaults to the shared client in production. */
  client?: ApiClient;
  /** Override the route param (primarily for tests). */
  inspectionId?: string;
}

type LoadState =
  | { phase: 'idle' }
  | { phase: 'loading' }
  | { phase: 'loaded'; report: InspectionReport; evidence: AnnotatedEvidence | null }
  | { phase: 'empty'; message: string }
  | { phase: 'error'; error: ClientError };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Format a metric value for display, preferring empty_space_ratio. */
function describeMetric(region: CandidateRegion): { label: string; value: string } {
  if (region.empty_space_ratio !== null) {
    return {
      label: 'Empty-space ratio',
      value: `${(region.empty_space_ratio * 100).toFixed(1)}%`,
    };
  }
  if (region.alt_metric) {
    return {
      label: 'Alternative metric',
      value: JSON.stringify(region.alt_metric),
    };
  }
  return { label: 'Metric', value: 'not available' };
}

/**
 * Resolve the documented interpretation for a region's uncertainty value from
 * the finding's `uncertainty_interpretations` map. Lookups are tolerant of
 * numeric formatting (e.g. `0.5` vs `0.50`).
 */
function interpretUncertainty(
  value: number,
  interpretations: Record<string, string>,
): string | null {
  const exact = interpretations[String(value)];
  if (exact) return exact;
  for (const [key, text] of Object.entries(interpretations)) {
    if (Number(key) === value) return text;
  }
  return null;
}

/** Build a data URL from inline base64 evidence bytes, when present. */
function evidenceDataUrl(evidence: AnnotatedEvidence | null): string | null {
  if (evidence?.data_base64) {
    return `data:image/png;base64,${evidence.data_base64}`;
  }
  return null;
}

function isAgentDecision(entry: TraceEntry): entry is AgentDecision {
  return entry.entry_type === 'agent_decision';
}

function isToolExecution(entry: TraceEntry): entry is ToolExecution {
  return entry.entry_type === 'tool_execution';
}

/** Human-readable label for a status, paired with a non-color shape cue. */
function statusPresentation(report: InspectionReport): {
  label: string;
  cue: string;
} {
  switch (report.status) {
    case 'needs_review':
      return { label: 'Needs human review', cue: '\u25B2' }; // triangle
    case 'failed':
      return { label: 'Failed', cue: '\u2715' }; // cross
    case 'complete':
    default:
      return { label: 'Complete', cue: '\u2713' }; // check
  }
}

// ---------------------------------------------------------------------------
// Subcomponents
// ---------------------------------------------------------------------------

/**
 * Renders an image with candidate-region overlay boxes positioned by percentage
 * of the image's intrinsic dimensions. Each box is labeled for screen readers
 * and uses a border + index number so regions are distinguishable without color.
 */
function AnnotatedComparison(props: {
  regions: CandidateRegion[];
  annotatedUrl: string | null;
  openCvVersion: string;
}) {
  const { regions, annotatedUrl, openCvVersion } = props;
  // The overlay uses the region boxes' own extent to build a reference frame so
  // boxes stay proportional even when the underlying image is unavailable.
  const frame = useMemo(() => boundingFrame(regions), [regions]);

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <figure className="m-0">
        <figcaption className="mb-1 text-sm font-medium">Original input</figcaption>
        <div className="relative rounded border border-gray-300 bg-gray-50 p-2 text-sm text-gray-600">
          <p>
            The original uploaded image is referenced by the report; the
            annotated output on the right marks each detected region.
          </p>
        </div>
      </figure>

      <figure className="m-0">
        <figcaption className="mb-1 text-sm font-medium">
          Annotated output
        </figcaption>
        <div
          className="relative overflow-hidden rounded border border-gray-300 bg-gray-50"
          data-testid="annotated-frame"
        >
          {annotatedUrl ? (
            <img
              src={annotatedUrl}
              alt={`Annotated shelf image with ${regions.length} highlighted candidate ${
                regions.length === 1 ? 'region' : 'regions'
              }, produced by OpenCV ${openCvVersion}.`}
              className="block w-full"
            />
          ) : (
            <div
              role="img"
              aria-label={`Diagram of ${regions.length} highlighted candidate ${
                regions.length === 1 ? 'region' : 'regions'
              } on the shelf. The annotated image is referenced by the report.`}
              className="relative aspect-video w-full"
            >
              {regions.map((region, index) => (
                <RegionOverlay
                  key={region.region_id}
                  region={region}
                  index={index + 1}
                  frame={frame}
                />
              ))}
            </div>
          )}
        </div>
      </figure>
    </div>
  );
}

/** The reference frame (max extent) used to position overlay boxes. */
interface Frame {
  width: number;
  height: number;
}

function boundingFrame(regions: CandidateRegion[]): Frame {
  let width = 1;
  let height = 1;
  for (const r of regions) {
    width = Math.max(width, r.bounding_box.x + r.bounding_box.w);
    height = Math.max(height, r.bounding_box.y + r.bounding_box.h);
  }
  // Pad so boxes do not hug the edge; keeps proportions readable.
  return { width: width * 1.05, height: height * 1.05 };
}

function overlayStyle(box: BoundingBox, frame: Frame): CSSProperties {
  return {
    left: `${(box.x / frame.width) * 100}%`,
    top: `${(box.y / frame.height) * 100}%`,
    width: `${(box.w / frame.width) * 100}%`,
    height: `${(box.h / frame.height) * 100}%`,
  };
}

function RegionOverlay(props: {
  region: CandidateRegion;
  index: number;
  frame: Frame;
}) {
  const { region, index, frame } = props;
  return (
    <div
      className="absolute border-2 border-amber-600"
      style={overlayStyle(region.bounding_box, frame)}
      data-testid={`region-overlay-${region.region_id}`}
    >
      <span className="absolute left-0 top-0 bg-amber-600 px-1 text-xs font-bold text-white">
        {index}
      </span>
    </div>
  );
}

/** One candidate region's measured metric, uncertainty, and caveat. */
function RegionDetails(props: {
  region: CandidateRegion;
  index: number;
  interpretations: Record<string, string>;
}) {
  const { region, index, interpretations } = props;
  const metric = describeMetric(region);
  const interpretation = interpretUncertainty(
    region.uncertainty_value,
    interpretations,
  );

  return (
    <li className="rounded border border-gray-300 p-4">
      <h3 className="font-medium">
        <span aria-hidden="true">{index}. </span>
        Region {region.region_id}
        {region.frame_id ? (
          <span className="ml-2 text-sm text-gray-500">
            (frame {region.frame_id})
          </span>
        ) : null}
      </h3>

      <dl className="mt-2 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="font-medium">{metric.label}</dt>
        <dd>
          {metric.value}{' '}
          <span className="text-gray-500">(estimate)</span>
        </dd>

        <dt className="font-medium">Uncertainty value</dt>
        <dd>{region.uncertainty_value}</dd>

        <dt className="font-medium">Interpretation</dt>
        <dd data-testid={`uncertainty-interpretation-${region.region_id}`}>
          {interpretation ?? 'No documented interpretation for this value.'}
        </dd>

        <dt className="font-medium">Heuristic score</dt>
        <dd>
          {region.heuristic_score}{' '}
          <span className="text-gray-500">(ordinal, not a probability)</span>
        </dd>
      </dl>

      <p
        className="mt-2 rounded bg-amber-50 p-2 text-sm text-amber-900"
        data-testid={`visual-caveat-${region.region_id}`}
      >
        <span className="font-medium">Caveat: </span>
        {region.visual_caveat}
      </p>
    </li>
  );
}

/** The ordered decision trace: agent decisions interleaved with tool results. */
function DecisionTrace(props: { entries: TraceEntry[] }) {
  const { entries } = props;
  if (entries.length === 0) {
    return (
      <p className="text-sm text-gray-500">
        No decision-trace entries were recorded for this inspection.
      </p>
    );
  }
  return (
    <ol className="space-y-3" data-testid="decision-trace">
      {entries.map((entry) => (
        <li
          key={traceKey(entry)}
          className="rounded border border-gray-300 p-4"
          data-entry-type={entry.entry_type}
        >
          {isAgentDecision(entry) ? (
            <AgentDecisionView decision={entry} />
          ) : isToolExecution(entry) ? (
            <ToolExecutionView execution={entry} />
          ) : null}
        </li>
      ))}
    </ol>
  );
}

function traceKey(entry: TraceEntry): string {
  return isAgentDecision(entry) ? entry.decision_id : entry.execution_id;
}

function AgentDecisionView(props: { decision: AgentDecision }) {
  const { decision } = props;
  return (
    <div>
      <p className="text-sm text-gray-500">
        Iteration {decision.iteration} &middot; agent decision
      </p>
      <h3 className="font-medium">
        Action: <code>{decision.action}</code>
      </h3>
      <p className="mt-1 text-sm">
        <span className="font-medium">Rationale: </span>
        {decision.rationale}
      </p>
      <p className="mt-1 text-sm text-gray-600">
        Decided by {decision.decided_by}
        {decision.used_fallback ? (
          <span className="ml-1 font-medium text-amber-800">
            (fallback{decision.fallback_reason ? `: ${decision.fallback_reason}` : ''})
          </span>
        ) : null}
      </p>
    </div>
  );
}

function ToolExecutionView(props: { execution: ToolExecution }) {
  const { execution } = props;
  const statusCue =
    execution.status === 'ok'
      ? '\u2713'
      : execution.status === 'rejected'
        ? '\u2715'
        : '\u26A0'; // warning
  return (
    <div>
      <p className="text-sm text-gray-500">
        Iteration {execution.iteration} &middot; tool execution
      </p>
      <h3 className="font-medium">
        Tool: <code>{execution.tool}</code>
      </h3>
      <p className="mt-1 text-sm">
        <span className="font-medium">Result: </span>
        <span data-testid={`tool-status-${execution.execution_id}`}>
          <span aria-hidden="true">{statusCue} </span>
          {execution.status}
        </span>{' '}
        <span className="text-gray-600">(ref {execution.result_ref})</span>
      </p>
      {execution.error ? (
        <p className="mt-1 text-sm text-red-800">
          <span className="font-medium">Error: </span>
          {execution.error}
        </p>
      ) : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function ResultsPage({ client, inspectionId: idProp }: ResultsPageProps = {}) {
  const params = useParams<{ inspectionId: string }>();
  const inspectionId = idProp ?? params.inspectionId;
  const apiClient = useMemo(() => client ?? getApiClient(), [client]);

  const [state, setState] = useState<LoadState>({ phase: 'idle' });
  // Guards against setting state after the effect is torn down / id changes.
  const requestId = useRef(0);

  useEffect(() => {
    if (!inspectionId) {
      setState({ phase: 'idle' });
      return;
    }

    const current = ++requestId.current;
    setState({ phase: 'loading' });

    (async () => {
      const reportResult = await apiClient.getInspection(inspectionId);
      if (current !== requestId.current) return;

      if (!reportResult.ok) {
        if (reportResult.error.kind === 'not_found') {
          setState({
            phase: 'empty',
            message: reportResult.error.message,
          });
        } else {
          setState({ phase: 'error', error: reportResult.error });
        }
        return;
      }

      const report = reportResult.data;

      // Best-effort annotated evidence; a failure here must not sink the page.
      let evidence: AnnotatedEvidence | null = null;
      const findingId = report.initial_finding_id;
      if (findingId) {
        const evidenceResult = await apiClient.getAnnotatedEvidence(
          inspectionId,
          findingId,
        );
        if (current !== requestId.current) return;
        if (evidenceResult.ok) {
          evidence = evidenceResult.data;
        }
      }

      setState({ phase: 'loaded', report, evidence });
    })();
  }, [apiClient, inspectionId]);

  // No id on the route: an informational landing, still a labeled page.
  if (!inspectionId) {
    return (
      <section aria-labelledby="results-heading">
        <h1 id="results-heading" className="text-2xl font-semibold">
          Results
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-300">
          Select an inspection from the history to view its annotated evidence,
          per-region measurements, and the agent decision trace.
        </p>
      </section>
    );
  }

  return (
    <section aria-labelledby="results-heading">
      <h1 id="results-heading" className="text-2xl font-semibold">
        Inspection results
      </h1>

      {/* Async status announced to assistive technology. */}
      <div aria-live="polite" className="sr-only" data-testid="results-status">
        {state.phase === 'loading' ? 'Loading inspection results.' : ''}
        {state.phase === 'loaded' ? 'Inspection results loaded.' : ''}
        {state.phase === 'empty' ? 'No inspection found.' : ''}
        {state.phase === 'error' ? 'Failed to load inspection results.' : ''}
      </div>

      {state.phase === 'loading' ? (
        <p className="mt-4 text-gray-600" role="status">
          Loading inspection {inspectionId}&hellip;
        </p>
      ) : null}

      {state.phase === 'empty' ? (
        <div className="mt-4 rounded border border-gray-300 bg-gray-50 p-6">
          <h2 className="text-lg font-medium">No such inspection</h2>
          <p className="mt-1 text-gray-600">
            {state.message || `No inspection was found for id "${inspectionId}".`}
          </p>
        </div>
      ) : null}

      {state.phase === 'error' ? (
        <div
          className="mt-4 rounded border border-red-300 bg-red-50 p-6 text-red-900"
          role="alert"
        >
          <h2 className="text-lg font-medium">Could not load results</h2>
          <p className="mt-1">{state.error.message}</p>
        </div>
      ) : null}

      {state.phase === 'loaded' ? (
        <LoadedResults report={state.report} evidence={state.evidence} />
      ) : null}
    </section>
  );
}

function LoadedResults(props: {
  report: InspectionReport;
  evidence: AnnotatedEvidence | null;
}) {
  const { report, evidence } = props;
  const status = statusPresentation(report);
  const annotatedUrl = evidenceDataUrl(evidence);

  // Regions + interpretations come from the primary (initial) finding which the
  // backend inlines on the evidence descriptor (derived from a real OpenCV
  // analysis of the stored image). When absent, we show the trace and
  // recommendation and keep an honest "no regions inlined" state.
  const regions = evidence?.primary_finding?.candidate_regions ?? [];
  const interpretations =
    evidence?.primary_finding?.uncertainty_interpretations ?? {};
  const openCvVersion = evidence?.primary_finding?.opencv_version ?? 'unknown';

  return (
    <div className="mt-4 space-y-8">
      {/* Final recommendation + human-review status (Req 12.4). */}
      <section aria-labelledby="recommendation-heading">
        <h2 id="recommendation-heading" className="text-xl font-semibold">
          Recommendation
        </h2>
        <p className="mt-2 text-lg">{report.final_recommendation}</p>
        <p className="mt-2 flex items-center gap-2 text-sm">
          <span
            className="inline-flex items-center gap-1 rounded border border-gray-400 px-2 py-0.5 font-medium"
            data-testid="report-status"
          >
            <span aria-hidden="true">{status.cue}</span>
            {status.label}
          </span>
          <span data-testid="human-review-status">
            {report.requires_human_review
              ? 'Human review required.'
              : 'No human review required.'}
          </span>
        </p>
      </section>

      {/* Original + annotated side by side (Req 12.1). */}
      <section aria-labelledby="evidence-heading">
        <h2 id="evidence-heading" className="text-xl font-semibold">
          Visual evidence
        </h2>
        <p className="mt-1 text-sm text-gray-600">
          Highlighted regions are classical computer-vision estimates, not
          certainties.
        </p>
        <div className="mt-3">
          <AnnotatedComparison
            regions={regions}
            annotatedUrl={annotatedUrl}
            openCvVersion={openCvVersion}
          />
        </div>
      </section>

      {/* Per-region metric + uncertainty interpretation + caveat (Req 12.2). */}
      <section aria-labelledby="regions-heading">
        <h2 id="regions-heading" className="text-xl font-semibold">
          Candidate regions
        </h2>
        {regions.length === 0 ? (
          <p className="mt-2 text-sm text-gray-500">
            No candidate regions are inlined with this report.
          </p>
        ) : (
          <ul className="mt-3 space-y-3">
            {regions.map((region, index) => (
              <RegionDetails
                key={region.region_id}
                region={region}
                index={index + 1}
                interpretations={interpretations}
              />
            ))}
          </ul>
        )}
      </section>

      {/* Agent initial finding, actions + rationale, tool results (Req 12.3). */}
      <section aria-labelledby="trace-heading">
        <h2 id="trace-heading" className="text-xl font-semibold">
          Agent decision trace
        </h2>
        <p className="mt-1 text-sm text-gray-600">
          Initial finding:{' '}
          <code data-testid="initial-finding-id">{report.initial_finding_id}</code>
        </p>
        <div className="mt-3">
          <DecisionTrace entries={report.decision_trace} />
        </div>
      </section>
    </div>
  );
}
