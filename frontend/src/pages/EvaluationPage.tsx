import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getApiClient } from '../api';
import type { ApiClient } from '../api';
import type { ClientError } from '../api/result';
import type { EvalExample, EvaluationResult } from '../api/types';

/**
 * Evaluation page (task 13.5; Req 14.1-14.6, 27.2).
 *
 * Shows measured test-set performance read from `EvaluationResult` records via
 * the typed API client (`listEvaluations` / `getEvaluation`). Every number on
 * this page comes from a recorded `EvaluationResult`; nothing is derived,
 * rounded into existence, or fabricated. When the server returns no records the
 * page renders an honest empty state with no numbers at all (Req 14.4, 27.2).
 *
 * Each record is labeled demo-vs-real using `EvaluationResult.is_demo`, with a
 * visible text label (not a color-only cue) so the distinction survives for
 * colorblind and screen-reader users (Req 14.5, WCAG 1.4.1).
 *
 * Async lifecycle (loading / error / empty / data) is announced through an
 * `aria-live` region, metrics are presented in captioned tables with scoped
 * headers, and all controls are native and keyboard operable (Req 14.6,
 * WCAG 2.1 AA).
 */

type LoadState =
  | { status: 'loading' }
  | { status: 'error'; error: ClientError }
  | { status: 'ready'; records: EvaluationResult[]; mode: string };

export interface EvaluationPageProps {
  /** Inject an API client for tests; defaults to the shared client. */
  client?: Pick<ApiClient, 'listEvaluations'>;
}

/** Format a ratio in [0, 1] as a percentage string for display. */
function formatPercent(ratio: number): string {
  return `${(ratio * 100).toFixed(1)}%`;
}

/** Format a millisecond latency value for display. */
function formatMs(ms: number): string {
  return `${Math.round(ms)} ms`;
}

export function EvaluationPage({ client }: EvaluationPageProps = {}) {
  const api = useMemo(() => client ?? getApiClient(), [client]);
  const [state, setState] = useState<LoadState>({ status: 'loading' });
  const requestIdRef = useRef(0);

  const load = useCallback(async () => {
    const requestId = ++requestIdRef.current;
    setState({ status: 'loading' });
    const result = await api.listEvaluations();
    // Ignore a stale response if a newer load has started (e.g. retry).
    if (requestId !== requestIdRef.current) return;
    if (result.ok) {
      setState({
        status: 'ready',
        records: result.data,
        mode: result.mode,
      });
    } else {
      setState({ status: 'error', error: result.error });
    }
  }, [api]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <section aria-labelledby="evaluation-heading">
      <h1 id="evaluation-heading" className="text-2xl font-semibold">
        Evaluation
      </h1>
      <p className="mt-2 text-gray-600 dark:text-gray-300">
        Measured test-set performance from recorded evaluation runs. Every value
        shown is a measured result; no metrics are estimated or fabricated.
      </p>

      {/* Announce async lifecycle changes to assistive technology. */}
      <div aria-live="polite" className="mt-6">
        {state.status === 'loading' && <LoadingState />}
        {state.status === 'error' && (
          <ErrorState error={state.error} onRetry={() => void load()} />
        )}
        {state.status === 'ready' &&
          (state.records.length === 0 ? (
            <EmptyState />
          ) : (
            <ResultsList records={state.records} />
          ))}
      </div>
    </section>
  );
}

function LoadingState() {
  return (
    <p role="status" className="text-gray-600 dark:text-gray-300">
      Loading evaluation runs&hellip;
    </p>
  );
}

function EmptyState() {
  return (
    <div
      role="status"
      className="rounded border border-gray-200 bg-gray-50 p-4 text-gray-700 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-200"
    >
      <p className="font-medium">No evaluation runs recorded yet</p>
      <p className="mt-1 text-sm">
        Performance metrics appear here once a labeled evaluation set has been
        run. Until then there is nothing measured to report.
      </p>
    </div>
  );
}

function ErrorState({
  error,
  onRetry,
}: {
  error: ClientError;
  onRetry: () => void;
}) {
  return (
    <div
      role="alert"
      className="rounded border border-red-300 bg-red-50 p-4 text-red-900 dark:border-red-700 dark:bg-red-950 dark:text-red-100"
    >
      <p className="font-medium">Could not load evaluation runs</p>
      <p className="mt-1 text-sm">{error.message}</p>
      {error.field ? (
        <p className="mt-1 text-sm">
          Field: <span className="font-mono">{error.field}</span>
        </p>
      ) : null}
      <button
        type="button"
        onClick={onRetry}
        className="mt-3 rounded bg-red-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-red-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-red-500"
      >
        Retry
      </button>
    </div>
  );
}

function ResultsList({ records }: { records: EvaluationResult[] }) {
  return (
    <ul className="space-y-8">
      {records.map((record) => (
        <li key={record.eval_run_id}>
          <EvaluationCard record={record} />
        </li>
      ))}
    </ul>
  );
}

/** A visible, non-color-only badge distinguishing demo from real records. */
function DataKindBadge({ isDemo }: { isDemo: boolean }) {
  const label = isDemo ? 'Demo data' : 'Real data';
  // A leading glyph plus the explicit text label gives a non-color cue that
  // survives for colorblind users and is read verbatim by screen readers.
  const glyph = isDemo ? '\u25CB' : '\u25CF'; // open circle vs filled circle
  const classes = isDemo
    ? 'border-amber-500 bg-amber-100 text-amber-900 dark:bg-amber-900 dark:text-amber-100'
    : 'border-emerald-600 bg-emerald-100 text-emerald-900 dark:bg-emerald-900 dark:text-emerald-100';
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ${classes}`}
    >
      <span aria-hidden="true">{glyph}</span>
      {label}
    </span>
  );
}

function EvaluationCard({ record }: { record: EvaluationResult }) {
  const headingId = `eval-${record.eval_run_id}-heading`;
  return (
    <article
      aria-labelledby={headingId}
      className="rounded border border-gray-200 p-4 dark:border-gray-700"
    >
      <header className="flex flex-wrap items-center justify-between gap-2">
        <h2 id={headingId} className="text-lg font-semibold">
          Run <span className="font-mono">{record.eval_run_id}</span>
        </h2>
        <DataKindBadge isDemo={record.is_demo} />
      </header>
      <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
        {record.sample_count} labeled sample
        {record.sample_count === 1 ? '' : 's'} evaluated.
      </p>

      <SummaryMetrics record={record} />

      <ExampleTable
        title="False-positive examples"
        caption={`False positives for run ${record.eval_run_id}: cases flagged as empty that were not.`}
        emptyMessage="No false-positive examples were recorded for this run."
        examples={record.false_positives}
      />

      <ExampleTable
        title="False-negative examples"
        caption={`False negatives for run ${record.eval_run_id}: empty cases that were missed.`}
        emptyMessage="No false-negative examples were recorded for this run."
        examples={record.false_negatives}
      />

      <Limitations limitations={record.documented_limitations} />
    </article>
  );
}

function SummaryMetrics({ record }: { record: EvaluationResult }) {
  return (
    <figure className="mt-4">
      <figcaption className="text-sm font-medium text-gray-700 dark:text-gray-200">
        Measured performance summary
      </figcaption>
      <table className="mt-2 w-full border-collapse text-left text-sm">
        <caption className="sr-only">
          Measured performance metrics for evaluation run {record.eval_run_id}
        </caption>
        <thead>
          <tr>
            <th scope="col" className="border-b border-gray-200 py-1 pr-4 dark:border-gray-700">
              Metric
            </th>
            <th scope="col" className="border-b border-gray-200 py-1 dark:border-gray-700">
              Measured value
            </th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row" className="py-1 pr-4 font-normal">
              Agent decision correctness
            </th>
            <td className="py-1 font-mono">
              {formatPercent(record.agent_decision_correctness)}
            </td>
          </tr>
          <tr>
            <th scope="row" className="py-1 pr-4 font-normal">
              False positives
            </th>
            <td className="py-1 font-mono">{record.false_positives.length}</td>
          </tr>
          <tr>
            <th scope="row" className="py-1 pr-4 font-normal">
              False negatives
            </th>
            <td className="py-1 font-mono">{record.false_negatives.length}</td>
          </tr>
          <tr>
            <th scope="row" className="py-1 pr-4 font-normal">
              Processing latency (p50)
            </th>
            <td className="py-1 font-mono">{formatMs(record.latency_ms_p50)}</td>
          </tr>
          <tr>
            <th scope="row" className="py-1 pr-4 font-normal">
              Processing latency (p95)
            </th>
            <td className="py-1 font-mono">{formatMs(record.latency_ms_p95)}</td>
          </tr>
        </tbody>
      </table>
    </figure>
  );
}

function ExampleTable({
  title,
  caption,
  emptyMessage,
  examples,
}: {
  title: string;
  caption: string;
  emptyMessage: string;
  examples: EvalExample[];
}) {
  return (
    <section className="mt-4">
      <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-100">
        {title}
      </h3>
      {examples.length === 0 ? (
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
          {emptyMessage}
        </p>
      ) : (
        <table className="mt-2 w-full border-collapse text-left text-sm">
          <caption className="sr-only">{caption}</caption>
          <thead>
            <tr>
              <th scope="col" className="border-b border-gray-200 py-1 pr-4 dark:border-gray-700">
                Sample
              </th>
              <th scope="col" className="border-b border-gray-200 py-1 pr-4 dark:border-gray-700">
                Expected
              </th>
              <th scope="col" className="border-b border-gray-200 py-1 pr-4 dark:border-gray-700">
                Actual
              </th>
              <th scope="col" className="border-b border-gray-200 py-1 dark:border-gray-700">
                Note
              </th>
            </tr>
          </thead>
          <tbody>
            {examples.map((example) => (
              <tr key={example.sample_id}>
                <th scope="row" className="py-1 pr-4 font-mono font-normal">
                  {example.sample_id}
                </th>
                <td className="py-1 pr-4">{example.expected}</td>
                <td className="py-1 pr-4">{example.actual}</td>
                <td className="py-1">{example.note ?? '\u2014'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function Limitations({ limitations }: { limitations: string[] }) {
  return (
    <section className="mt-4">
      <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-100">
        Documented failure cases and limitations
      </h3>
      {limitations.length === 0 ? (
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
          No limitations were documented for this run.
        </p>
      ) : (
        <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-gray-700 dark:text-gray-200">
          {limitations.map((limitation, index) => (
            <li key={index}>{limitation}</li>
          ))}
        </ul>
      )}
    </section>
  );
}
