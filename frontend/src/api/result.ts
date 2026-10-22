/**
 * A small typed `Result` for the API client (task 13.1; Req 10.5).
 *
 * The client never throws raw errors at the UI. Every call resolves to a
 * discriminated `ApiResult<T>`: either `{ ok: true, data, apiVersion, mode }`
 * on success, or `{ ok: false, error }` carrying a structured `ClientError`.
 * This lets pages branch exhaustively (`if (res.ok) ... else ...`) and render
 * honest error/empty states instead of catching exceptions.
 */

import type { ApiErrorDetail, ApiMode } from './types';

/**
 * The structured error surfaced to the UI. It unifies three sources into one
 * shape the pages can render:
 *  - `validation` / `not_found` / `bad_request`: the server's structured error
 *    envelope, passed through verbatim (code/message/field/details).
 *  - `client_validation`: a client-side pre-flight failure (type/size mirror of
 *    the server limits) raised before any request is sent (Req 11.2).
 *  - `network`: the request could not be completed (offline, DNS, CORS, abort).
 *  - `transport`: a malformed or non-JSON response, or an unexpected HTTP shape.
 */
export interface ClientError {
  kind:
    | 'validation'
    | 'not_found'
    | 'bad_request'
    | 'client_validation'
    | 'network'
    | 'transport';
  message: string;
  /** Offending field/condition when applicable. */
  field?: string | null;
  /** Optional structured context carried through from the server. */
  details?: Record<string, unknown> | null;
  /** HTTP status code when the failure came from an HTTP response. */
  status?: number;
}

/** Success branch of {@link ApiResult}. */
export interface ApiOk<T> {
  ok: true;
  data: T;
  /** Envelope API version (Req 10.4). */
  apiVersion: string;
  /** Operating mode for demo-vs-real labeling (Req 14.5). */
  mode: ApiMode;
}

/** Failure branch of {@link ApiResult}. */
export interface ApiErr {
  ok: false;
  error: ClientError;
  /** Present when the failure arrived inside a well-formed envelope. */
  apiVersion?: string;
  mode?: ApiMode;
}

/** The value every client method resolves to. Never rejects for API errors. */
export type ApiResult<T> = ApiOk<T> | ApiErr;

/** Build a success result. */
export function ok<T>(data: T, apiVersion: string, mode: ApiMode): ApiOk<T> {
  return { ok: true, data, apiVersion, mode };
}

/** Build a failure result from a structured client error. */
export function err(
  error: ClientError,
  envelope?: { apiVersion?: string; mode?: ApiMode },
): ApiErr {
  return { ok: false, error, apiVersion: envelope?.apiVersion, mode: envelope?.mode };
}

/** Map a server error-envelope detail onto a {@link ClientError}. */
export function clientErrorFromDetail(
  detail: ApiErrorDetail,
  status?: number,
): ClientError {
  return {
    kind: detail.code,
    message: detail.message,
    field: detail.field ?? null,
    details: detail.details ?? null,
    status,
  };
}
