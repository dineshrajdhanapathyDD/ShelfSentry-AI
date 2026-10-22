import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  getApiClient,
  type ApiClient,
  type ClientError,
  type InspectionSummary,
  type ListInspectionsQuery,
  type ReportStatus,
} from '../api';

/**
 * History page (task 13.4; Req 13.1-13.5).
 *
 * Lists previous inspections (status + timestamp) from the typed API client,
 * with accessible search / status / limit filter controls, a link from each row
 * to the Results detail route (`/results/:inspectionId`, learned read-only from
 * App.tsx), and honest loading / error / empty states.
 *
 * Accessibility (WCAG 2.1 AA):
 *  - The listing is a semantic `<table>` with a `<caption>` and column headers.
 *  - Every filter control has an associated `<label>`; the whole group sits in a
 *    labeled `<form>` landmark and is fully keyboard operable with visible focus
 *    (focus styling lives in index.css / Tailwind ring utilities).
 *  - Status is conveyed with text plus a non-color icon glyph, never color
 *    alone (WCAG 1.4.1).
 *  - Async state changes are announced through an `aria-live="polite"` region.
 */

/** Statuses the filter can select, plus an "all" sentinel for no filter. */
const STATUS_OPTIONS: ReadonlyArray<{ value: ReportStatus | 'all'; label: string }> = [
  { value: 'all', label: 'All statuses' },
  { value: 'complete', label: 'Complete' },
  { value: 'needs_review', label: 'Needs review' },
  { value: 'failed', label: 'Failed' },
];

/** Non-color cue (glyph + text) for each status (WCAG 1.4.1). */
const STATUS_PRESENTATION: Record<
  ReportStatus,
  { icon: string; label: string; className: string }
> = {
  complete: {
    icon: '\u2714', // heavy check mark
    label: 'Complete',
    className: 'text-green-800 dark:text-green-300',
  },
  needs_review: {
    icon: '\u26A0', // warning sign
    label: 'Needs review',
    className: 'text-amber-800 dark:text-amber-300',
  },
  failed: {
    icon: '\u2716', // heavy multiplication x
    label: 'Failed',
    className: 'text-red-800 dark:text-red-300',
  },
};

const SEARCH_DEBOUNCE_MS = 300;

/** Props for {@link HistoryPage}. `client` is injectable for tests. */
export interface HistoryPageProps {
  /** API client used to list inspections (defaults to the shared client). */
  client?: ApiClient;
  /** Override the search debounce (ms). Tests set this to 0 for determinism. */
  searchDebounceMs?: number;
}

type LoadState =
  | { phase: 'loading' }
  | { phase: 'error'; error: ClientError }
  | { phase: 'ready'; rows: InspectionSummary[] };

/** Format an ISO timestamp for display, falling back to the raw value. */
function formatTimestamp(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) {
    return iso;
  }
  return parsed.toLocaleString();
}

/** Render a status with a non-color glyph cue and visible text. */
function StatusCell({ status }: { status: ReportStatus }) {
  const presentation = STATUS_PRESENTATION[status];
  return (
    <span className={`inline-flex items-center gap-1.5 ${presentation.className}`}>
      <span aria-hidden="true">{presentation.icon}</span>
      <span>{presentation.label}</span>
    </span>
  );
}

export function HistoryPage({
  client,
  searchDebounceMs = SEARCH_DEBOUNCE_MS,
}: HistoryPageProps = {}) {
  const apiClient = useMemo(() => client ?? getApiClient(), [client]);

  // Controlled filter inputs.
  const [searchInput, setSearchInput] = useState('');
  const [statusFilter, setStatusFilter] = useState<ReportStatus | 'all'>('all');
  const [limit, setLimit] = useState<number | 'all'>('all');

  // The debounced text that actually drives the query.
  const [debouncedText, setDebouncedText] = useState('');

  const [state, setState] = useState<LoadState>({ phase: 'loading' });

  // Unique ids so labels bind to controls even if the page mounts twice.
  const searchId = useId();
  const statusId = useId();
  const limitId = useId();

  // Debounce the free-text search into `debouncedText`.
  useEffect(() => {
    if (searchDebounceMs <= 0) {
      setDebouncedText(searchInput);
      return;
    }
    const handle = setTimeout(() => {
      setDebouncedText(searchInput);
    }, searchDebounceMs);
    return () => clearTimeout(handle);
  }, [searchInput, searchDebounceMs]);

  // Guard against out-of-order responses: only the latest request may commit.
  const requestSeq = useRef(0);

  const load = useCallback(
    async (query: ListInspectionsQuery) => {
      const seq = ++requestSeq.current;
      setState({ phase: 'loading' });
      const result = await apiClient.listInspections(query);
      if (seq !== requestSeq.current) {
        return; // A newer request superseded this one.
      }
      if (result.ok) {
        setState({ phase: 'ready', rows: result.data });
      } else {
        setState({ phase: 'error', error: result.error });
      }
    },
    [apiClient],
  );

  // Re-run the listing whenever an effective filter changes.
  useEffect(() => {
    const query: ListInspectionsQuery = {};
    const trimmed = debouncedText.trim();
    if (trimmed) query.text = trimmed;
    if (statusFilter !== 'all') query.status = statusFilter;
    if (limit !== 'all') query.limit = limit;
    void load(query);
  }, [debouncedText, statusFilter, limit, load]);

  const rowCount = state.phase === 'ready' ? state.rows.length : 0;
  const liveMessage =
    state.phase === 'loading'
      ? 'Loading inspections...'
      : state.phase === 'error'
        ? `Could not load inspections: ${state.error.message}`
        : rowCount === 0
          ? 'No inspections match the current filters.'
          : `Showing ${rowCount} inspection${rowCount === 1 ? '' : 's'}.`;

  return (
    <section aria-labelledby="history-heading">
      <h1 id="history-heading" className="text-2xl font-semibold">
        History
      </h1>
      <p className="mt-2 text-gray-600 dark:text-gray-300">
        Previous inspections with their status and time. Search and filter, then
        open any row to view its full results.
      </p>

      <form
        className="mt-6 flex flex-wrap items-end gap-4"
        aria-label="Filter inspections"
        onSubmit={(event) => event.preventDefault()}
      >
        <div className="flex flex-col gap-1">
          <label htmlFor={searchId} className="text-sm font-medium">
            Search
          </label>
          <input
            id={searchId}
            type="search"
            value={searchInput}
            onChange={(event) => setSearchInput(event.target.value)}
            placeholder="Search recommendation or id"
            className="rounded-md border border-gray-300 px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-900"
          />
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor={statusId} className="text-sm font-medium">
            Status
          </label>
          <select
            id={statusId}
            value={statusFilter}
            onChange={(event) =>
              setStatusFilter(event.target.value as ReportStatus | 'all')
            }
            className="rounded-md border border-gray-300 px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-900"
          >
            {STATUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label htmlFor={limitId} className="text-sm font-medium">
            Show
          </label>
          <select
            id={limitId}
            value={String(limit)}
            onChange={(event) => {
              const next = event.target.value;
              setLimit(next === 'all' ? 'all' : Number(next));
            }}
            className="rounded-md border border-gray-300 px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-900"
          >
            <option value="all">All</option>
            <option value="10">Latest 10</option>
            <option value="25">Latest 25</option>
            <option value="50">Latest 50</option>
          </select>
        </div>
      </form>

      {/* Polite announcer for async state changes (WCAG 4.1.3). */}
      <p className="sr-only" role="status" aria-live="polite">
        {liveMessage}
      </p>

      <div className="mt-6">
        {state.phase === 'loading' && (
          <p className="text-gray-600 dark:text-gray-300">Loading inspections...</p>
        )}

        {state.phase === 'error' && (
          <div
            role="alert"
            className="rounded-md border border-red-300 bg-red-50 p-4 text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-200"
          >
            <p className="font-medium">We could not load your inspection history.</p>
            <p className="mt-1 text-sm">{state.error.message}</p>
            {state.error.field && (
              <p className="mt-1 text-sm">Field: {state.error.field}</p>
            )}
            <button
              type="button"
              onClick={() => {
                const query: ListInspectionsQuery = {};
                const trimmed = debouncedText.trim();
                if (trimmed) query.text = trimmed;
                if (statusFilter !== 'all') query.status = statusFilter;
                if (limit !== 'all') query.limit = limit;
                void load(query);
              }}
              className="mt-3 rounded-md border border-red-400 px-3 py-2 text-sm font-medium hover:bg-red-100 dark:hover:bg-red-900"
            >
              Try again
            </button>
          </div>
        )}

        {state.phase === 'ready' && rowCount === 0 && (
          <div className="rounded-md border border-gray-200 bg-gray-50 p-6 text-center dark:border-gray-800 dark:bg-gray-900">
            <p className="font-medium">No inspections yet</p>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
              Once you run an inspection it will appear here. Try adjusting your
              search or filters, or start a new inspection.
            </p>
          </div>
        )}

        {state.phase === 'ready' && rowCount > 0 && (
          <table className="w-full border-collapse text-left text-sm">
            <caption className="sr-only">
              Previous inspections, newest first, with status and time.
            </caption>
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-800">
                <th scope="col" className="py-2 pr-4 font-semibold">
                  Inspection
                </th>
                <th scope="col" className="py-2 pr-4 font-semibold">
                  Status
                </th>
                <th scope="col" className="py-2 pr-4 font-semibold">
                  Recommendation
                </th>
                <th scope="col" className="py-2 pr-4 font-semibold">
                  Created
                </th>
              </tr>
            </thead>
            <tbody>
              {state.rows.map((row) => (
                <tr
                  key={row.inspection_id}
                  className="border-b border-gray-100 dark:border-gray-900"
                >
                  <td className="py-2 pr-4">
                    <Link
                      to={`/results/${encodeURIComponent(row.inspection_id)}`}
                      className="font-medium text-blue-700 underline hover:text-blue-900 dark:text-blue-300"
                    >
                      {row.inspection_id}
                    </Link>
                  </td>
                  <td className="py-2 pr-4">
                    <StatusCell status={row.status} />
                  </td>
                  <td className="py-2 pr-4 text-gray-700 dark:text-gray-300">
                    {row.final_recommendation}
                  </td>
                  <td className="py-2 pr-4 text-gray-700 dark:text-gray-300">
                    <time dateTime={row.created_at}>
                      {formatTimestamp(row.created_at)}
                    </time>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
