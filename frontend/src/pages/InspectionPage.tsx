import { useId, useMemo, useRef, useState } from 'react';
import {
  ACCEPTED_CONTENT_TYPES,
  MAX_FILE_SIZE_BYTES,
  getApiClient,
  validateUploadFile,
  type ApiClient,
  type ClientError,
  type InspectionReport,
} from '../api';

/**
 * The Inspection page (task 13.2; Req 11.1-11.6).
 *
 * Lets a User either upload an image/video file or run the bundled sample in
 * "example mode", validates a selected file client-side against the mirrored
 * server limits before submission (Req 11.2), runs the inspection through the
 * typed API client, and renders progress / success / error / empty states
 * (Req 11.3-11.5).
 *
 * The {@link ApiClient} is injectable so tests drive the page without a live
 * backend. Every state transition is announced through an `aria-live` region,
 * status is conveyed with text (not color alone), the file input is labeled,
 * and all controls are native/keyboard-operable (Req 11.6, WCAG 2.1 AA).
 */

/** One of the mutually-exclusive view states of the page. */
type Phase =
  | { kind: 'empty' }
  | { kind: 'running' }
  | { kind: 'success'; report: InspectionReport }
  | { kind: 'error'; error: ClientError };

export interface InspectionPageProps {
  /** Injectable API client (defaults to the shared default client). */
  client?: ApiClient;
}

/** Human-readable byte size for the size-limit hint (e.g. "25 MB"). */
function formatBytes(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return `${Number.isInteger(mb) ? mb : mb.toFixed(1)} MB`;
}

export function InspectionPage({ client }: InspectionPageProps = {}) {
  const apiClient = useMemo(() => client ?? getApiClient(), [client]);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<ClientError | null>(null);
  const [phase, setPhase] = useState<Phase>({ kind: 'empty' });

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const fileInputId = useId();
  const fileHintId = useId();
  const fileErrorId = useId();

  const isRunning = phase.kind === 'running';
  const acceptAttr = ACCEPTED_CONTENT_TYPES.join(',');

  function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    // Validate immediately so the User sees a rejection before submitting.
    setFileError(file ? validateUploadFile(file) : null);
  }

  async function runInspection(
    create: () => ReturnType<ApiClient['createInspection']>,
  ) {
    setPhase({ kind: 'running' });
    const result = await create();
    if (result.ok) {
      setPhase({ kind: 'success', report: result.data });
    } else {
      setPhase({ kind: 'error', error: result.error });
    }
  }

  function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile) {
      setFileError({
        kind: 'client_validation',
        message: 'Choose an image or video file before starting the inspection.',
        field: 'media',
      });
      return;
    }
    // Re-validate at submit time; never submit a rejected file (Req 11.2).
    const validationError = validateUploadFile(selectedFile);
    if (validationError) {
      setFileError(validationError);
      return;
    }
    setFileError(null);
    void runInspection(() => apiClient.createInspection({ file: selectedFile }));
  }

  function handleExampleMode() {
    setFileError(null);
    void runInspection(() => apiClient.createInspection({ exampleMode: true }));
  }

  function handleReset() {
    setSelectedFile(null);
    setFileError(null);
    setPhase({ kind: 'empty' });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }

  return (
    <section aria-labelledby="inspection-heading">
      <h1 id="inspection-heading" className="text-2xl font-semibold">
        Inspection
      </h1>
      <p className="mt-2 text-gray-600 dark:text-gray-300">
        Upload a shelf image or video, or run the bundled sample, to start an
        active-perception inspection.
      </p>

      <form
        className="mt-6 space-y-4"
        onSubmit={handleSubmit}
        aria-describedby={fileHintId}
        noValidate
      >
        <fieldset
          className="space-y-2 border border-gray-200 dark:border-gray-800 rounded-md p-4"
          disabled={isRunning}
        >
          <legend className="text-sm font-medium px-1">Shelf media</legend>

          <label htmlFor={fileInputId} className="block text-sm font-medium">
            Image or video file
          </label>
          <input
            ref={fileInputRef}
            id={fileInputId}
            name="media"
            type="file"
            accept={acceptAttr}
            onChange={handleFileChange}
            aria-describedby={
              fileError ? `${fileHintId} ${fileErrorId}` : fileHintId
            }
            aria-invalid={fileError ? true : undefined}
            className="block w-full text-sm"
          />
          <p id={fileHintId} className="text-sm text-gray-500">
            Accepted types: {ACCEPTED_CONTENT_TYPES.join(', ')}. Maximum size{' '}
            {formatBytes(MAX_FILE_SIZE_BYTES)}.
          </p>

          {fileError ? (
            <p
              id={fileErrorId}
              role="alert"
              className="text-sm font-medium text-red-700 dark:text-red-400"
            >
              <span aria-hidden="true">⚠ </span>
              {fileError.message}
            </p>
          ) : null}

          <div className="flex flex-wrap gap-3 pt-2">
            <button
              type="submit"
              className="rounded-md bg-blue-700 px-4 py-2 text-sm font-medium text-white hover:bg-blue-800 disabled:opacity-60"
            >
              Start inspection
            </button>
            <button
              type="button"
              onClick={handleExampleMode}
              className="rounded-md border border-blue-700 px-4 py-2 text-sm font-medium text-blue-800 hover:bg-blue-100 dark:text-blue-200 dark:hover:bg-blue-900 disabled:opacity-60"
            >
              Run example (bundled sample)
            </button>
          </div>
        </fieldset>
      </form>

      {/* Status region: every phase change is announced to assistive tech. */}
      <div aria-live="polite" className="mt-6">
        {phase.kind === 'empty' ? (
          <EmptyState />
        ) : phase.kind === 'running' ? (
          <RunningState />
        ) : phase.kind === 'success' ? (
          <SuccessState report={phase.report} onReset={handleReset} />
        ) : (
          <ErrorState error={phase.error} onReset={handleReset} />
        )}
      </div>
    </section>
  );
}

/** Empty state shown before any inspection has been started (Req 11.5). */
function EmptyState() {
  return (
    <section
      aria-labelledby="inspection-empty-heading"
      className="rounded-md border border-dashed border-gray-300 dark:border-gray-700 p-6 text-center"
    >
      <h2
        id="inspection-empty-heading"
        className="text-base font-medium text-gray-700 dark:text-gray-200"
      >
        No inspection yet
      </h2>
      <p className="mt-1 text-sm text-gray-500">
        Choose a file or run the bundled sample to see results here.
      </p>
    </section>
  );
}

/** Progress state shown while an inspection is in flight (Req 11.3). */
function RunningState() {
  return (
    <section
      aria-labelledby="inspection-running-heading"
      className="rounded-md border border-blue-200 dark:border-blue-900 bg-blue-50 dark:bg-blue-950 p-6"
    >
      <h2
        id="inspection-running-heading"
        className="flex items-center gap-2 text-base font-medium text-blue-900 dark:text-blue-100"
      >
        <span
          className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-blue-700 border-t-transparent"
          aria-hidden="true"
        />
        Inspection in progress
      </h2>
      <p className="mt-1 text-sm text-blue-800 dark:text-blue-200" role="status">
        Running the vision pipeline and agent loop. This may take a moment.
      </p>
    </section>
  );
}

/** Success state announcing the completed report (Req 11.3). */
function SuccessState({
  report,
  onReset,
}: {
  report: InspectionReport;
  onReset: () => void;
}) {
  return (
    <section
      aria-labelledby="inspection-success-heading"
      className="rounded-md border border-green-200 dark:border-green-900 bg-green-50 dark:bg-green-950 p-6"
    >
      <h2
        id="inspection-success-heading"
        className="text-base font-medium text-green-900 dark:text-green-100"
      >
        <span aria-hidden="true">✓ </span>
        Inspection complete
      </h2>
      <dl className="mt-2 text-sm text-green-900 dark:text-green-100">
        <div className="flex gap-2">
          <dt className="font-medium">Report id:</dt>
          <dd>
            <code>{report.inspection_id}</code>
          </dd>
        </div>
        <div className="flex gap-2">
          <dt className="font-medium">Status:</dt>
          <dd>{report.status}</dd>
        </div>
        <div className="flex gap-2">
          <dt className="font-medium">Human review:</dt>
          <dd>{report.requires_human_review ? 'Required' : 'Not required'}</dd>
        </div>
      </dl>
      <p className="mt-2 text-sm text-green-900 dark:text-green-100">
        {report.final_recommendation}
      </p>
      <div className="mt-4">
        <a
          href={`/results/${encodeURIComponent(report.inspection_id)}`}
          className="text-sm font-medium text-blue-800 underline dark:text-blue-200"
        >
          View full report
        </a>
      </div>
      <div className="mt-3">
        <button
          type="button"
          onClick={onReset}
          className="rounded-md border border-gray-300 dark:border-gray-700 px-3 py-1.5 text-sm font-medium"
        >
          Start another inspection
        </button>
      </div>
    </section>
  );
}

/** Error state describing a failed inspection (Req 11.4). */
function ErrorState({
  error,
  onReset,
}: {
  error: ClientError;
  onReset: () => void;
}) {
  return (
    <section
      aria-labelledby="inspection-error-heading"
      className="rounded-md border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950 p-6"
    >
      <h2
        id="inspection-error-heading"
        className="text-base font-medium text-red-800 dark:text-red-300"
      >
        <span aria-hidden="true">⚠ </span>
        Inspection failed
      </h2>
      <p className="mt-2 text-sm text-red-800 dark:text-red-300">
        {error.message}
      </p>
      <dl className="mt-2 text-sm text-red-800 dark:text-red-300">
        <div className="flex gap-2">
          <dt className="font-medium">Reason:</dt>
          <dd>{error.kind}</dd>
        </div>
        {error.field ? (
          <div className="flex gap-2">
            <dt className="font-medium">Field:</dt>
            <dd>{error.field}</dd>
          </div>
        ) : null}
      </dl>
      <div className="mt-4">
        <button
          type="button"
          onClick={onReset}
          className="rounded-md border border-gray-300 dark:border-gray-700 px-3 py-1.5 text-sm font-medium"
        >
          Try again
        </button>
      </div>
    </section>
  );
}
