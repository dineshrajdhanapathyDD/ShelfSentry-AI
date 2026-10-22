/**
 * The thin, typed ShelfSentry API client (task 13.1; Req 10.4, 10.5, 14.5).
 *
 * Responsibilities, kept deliberately small:
 *  - Call each backend endpoint (create/get/list inspections, list/get
 *    evaluations, get annotated evidence) and return typed results.
 *  - Read the envelope `apiVersion` and `mode` off every response so the UI can
 *    label demo-vs-real data (Req 14.5) and reason about the API version
 *    (Req 10.4).
 *  - Surface the server's structured validation errors to the UI as a typed
 *    `ApiResult` instead of throwing raw (Req 10.5).
 *  - Mirror the server's client-side upload limits before sending (Req 11.2).
 *
 * The client is transport-agnostic about the error path: whether the server
 * signals an error via HTTP status or an in-body error envelope, the response
 * body carries `{ apiVersion, mode, error }` and is mapped onto `ApiResult`.
 *
 * `fetch` is injectable so tests can drive the client without a live server.
 */

import { EXPECTED_API_VERSION, resolveApiBaseUrl } from './config';
import {
  type ApiErr,
  type ApiResult,
  clientErrorFromDetail,
  err,
  ok,
} from './result';
import { validateUploadFile } from './validation';
import type {
  AnnotatedEvidence,
  ApiErrorEnvelope,
  ApiMode,
  ApiResponse,
  EvaluationResult,
  InspectionReport,
  InspectionSummary,
  ReportStatus,
} from './types';

/** Minimal `fetch` signature the client depends on (injectable for tests). */
export type FetchLike = (
  input: string,
  init?: RequestInit,
) => Promise<Response>;

/** Options for constructing an {@link ApiClient}. */
export interface ApiClientOptions {
  /** Override the base URL (defaults to {@link resolveApiBaseUrl}). */
  baseUrl?: string;
  /** Inject a `fetch` implementation (defaults to the global `fetch`). */
  fetchImpl?: FetchLike;
}

/** Filters for the history listing (Req 13.1, 13.2). */
export interface ListInspectionsQuery {
  status?: ReportStatus;
  text?: string;
  limit?: number;
}

/** Input for creating an inspection. Exactly one source must be provided. */
export type CreateInspectionInput =
  | { exampleMode: true; inspectionId?: string }
  | { file: File; exampleMode?: false; inspectionId?: string };

function isErrorEnvelope(body: unknown): body is ApiErrorEnvelope {
  return (
    typeof body === 'object' &&
    body !== null &&
    'error' in body &&
    typeof (body as { error: unknown }).error === 'object'
  );
}

/**
 * A typed client for the ShelfSentry API. All methods resolve to an
 * `ApiResult<T>` and never reject for API-level errors.
 */
export class ApiClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: FetchLike;

  constructor(options: ApiClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? resolveApiBaseUrl()).replace(/\/+$/, '');
    const injected = options.fetchImpl;
    if (injected) {
      this.fetchImpl = injected;
    } else if (typeof fetch === 'function') {
      this.fetchImpl = (input, init) => fetch(input, init);
    } else {
      // No fetch available and none injected: fail loudly at call time.
      this.fetchImpl = () =>
        Promise.reject(new Error('No fetch implementation available.'));
    }
  }

  /** The resolved base URL this client targets. */
  getBaseUrl(): string {
    return this.baseUrl;
  }

  /** The envelope API version this client expects (Req 10.4). */
  getExpectedApiVersion(): string {
    return EXPECTED_API_VERSION;
  }

  // -- endpoints ----------------------------------------------------------

  /**
   * Create (run) an inspection. In `exampleMode` the bundled server sample is
   * used; otherwise the provided `file` is validated client-side (Req 11.2),
   * then uploaded as multipart form data.
   */
  async createInspection(
    input: CreateInspectionInput,
  ): Promise<ApiResult<InspectionReport>> {
    const body = new FormData();

    if ('file' in input && input.file) {
      const clientError = validateUploadFile(input.file);
      if (clientError) {
        return err(clientError);
      }
      body.append('media', input.file, input.file.name);
      body.append('content_type', input.file.type);
    } else if ('exampleMode' in input && input.exampleMode) {
      body.append('example_mode', 'true');
    } else {
      return err({
        kind: 'client_validation',
        message: 'Provide a file or set exampleMode to create an inspection.',
        field: 'media',
      });
    }

    if (input.inspectionId) {
      body.append('inspection_id', input.inspectionId);
    }

    return this.request<InspectionReport>('/inspections', {
      method: 'POST',
      body,
    });
  }

  /** Fetch a single inspection report by id. */
  async getInspection(
    inspectionId: string,
  ): Promise<ApiResult<InspectionReport>> {
    const path = `/inspections/${encodeURIComponent(inspectionId)}`;
    return this.request<InspectionReport>(path, { method: 'GET' });
  }

  /** List inspection history summaries with optional status/text/limit filters. */
  async listInspections(
    query: ListInspectionsQuery = {},
  ): Promise<ApiResult<InspectionSummary[]>> {
    const params = new URLSearchParams();
    if (query.status) params.set('status', query.status);
    if (query.text) params.set('text', query.text);
    if (typeof query.limit === 'number') params.set('limit', String(query.limit));
    const qs = params.toString();
    const path = qs ? `/inspections?${qs}` : '/inspections';
    return this.request<InspectionSummary[]>(path, { method: 'GET' });
  }

  /** List measured evaluation records (empty when none exist; never fabricated). */
  async listEvaluations(): Promise<ApiResult<EvaluationResult[]>> {
    return this.request<EvaluationResult[]>('/evaluations', { method: 'GET' });
  }

  /** Fetch a single evaluation record by run id. */
  async getEvaluation(
    evalRunId: string,
  ): Promise<ApiResult<EvaluationResult>> {
    const path = `/evaluations/${encodeURIComponent(evalRunId)}`;
    return this.request<EvaluationResult>(path, { method: 'GET' });
  }

  /** Fetch the annotated-evidence descriptor for a finding (Req 12). */
  async getAnnotatedEvidence(
    inspectionId: string,
    findingId: string,
  ): Promise<ApiResult<AnnotatedEvidence>> {
    const path =
      `/inspections/${encodeURIComponent(inspectionId)}` +
      `/evidence/${encodeURIComponent(findingId)}`;
    return this.request<AnnotatedEvidence>(path, { method: 'GET' });
  }

  // -- core request/parse -------------------------------------------------

  /**
   * Perform a request and map the response onto an `ApiResult<T>`.
   *
   * Reads `apiVersion` and `mode` off the envelope (Req 10.4, 14.5). A
   * well-formed error envelope becomes a typed failure (Req 10.5); a network
   * failure or a non-JSON/malformed body becomes a `network`/`transport`
   * failure. No raw exception escapes to the caller.
   */
  private async request<T>(
    path: string,
    init: RequestInit,
  ): Promise<ApiResult<T>> {
    const url = `${this.baseUrl}${path}`;
    let response: Response;
    try {
      response = await this.fetchImpl(url, {
        headers: { Accept: 'application/json' },
        ...init,
      });
    } catch (cause) {
      return err({
        kind: 'network',
        message:
          cause instanceof Error
            ? `Request to ${path} failed: ${cause.message}`
            : `Request to ${path} failed.`,
      });
    }

    let body: unknown;
    try {
      const text = await response.text();
      body = text ? (JSON.parse(text) as unknown) : undefined;
    } catch {
      return err({
        kind: 'transport',
        message: `Response from ${path} was not valid JSON.`,
        status: response.status,
      });
    }

    return this.mapEnvelope<T>(body, response.status, path);
  }

  /** Map a parsed response body (an envelope) onto an `ApiResult<T>`. */
  private mapEnvelope<T>(
    body: unknown,
    status: number,
    path: string,
  ): ApiResult<T> {
    if (typeof body !== 'object' || body === null) {
      return this.transportFailure(status, path);
    }

    const envelope = body as Partial<ApiResponse<T>>;
    const apiVersion =
      typeof (envelope as { apiVersion?: unknown }).apiVersion === 'string'
        ? (envelope as { apiVersion: string }).apiVersion
        : undefined;
    const mode = (envelope as { mode?: ApiMode }).mode;

    if (isErrorEnvelope(body)) {
      const detail = body.error;
      return {
        ...err(clientErrorFromDetail(detail, status), { apiVersion, mode }),
      } satisfies ApiErr;
    }

    if ('data' in envelope && apiVersion !== undefined && mode !== undefined) {
      return ok<T>((envelope as { data: T }).data, apiVersion, mode);
    }

    return this.transportFailure(status, path);
  }

  private transportFailure(status: number, path: string): ApiErr {
    return err({
      kind: 'transport',
      message:
        `Response from ${path} did not match the expected ` +
        `{ apiVersion, mode, data | error } envelope.`,
      status,
    });
  }
}

/** A lazily-constructed default client pointed at the resolved base URL. */
let defaultClient: ApiClient | null = null;

/** Return a shared default {@link ApiClient} (constructed on first use). */
export function getApiClient(): ApiClient {
  if (defaultClient === null) {
    defaultClient = new ApiClient();
  }
  return defaultClient;
}
