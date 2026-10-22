/**
 * API client configuration and client-side upload limits (task 13.1).
 *
 * The base URL is configurable through `VITE_API_BASE_URL` so the client can
 * point at the local dev server or the deployed API (Req: configurable base
 * URL). It defaults to a local dev endpoint when the env var is unset.
 *
 * The accepted content types and max upload size mirror the backend's
 * `VisionConfig` defaults (`DEFAULT_ACCEPTED_CONTENT_TYPES`,
 * `DEFAULT_MAX_FILE_SIZE_BYTES`) so the Inspection page can perform the same
 * client-side type/size checks the server enforces (Req 11.2). The server
 * remains the source of truth; these are a fast-fail mirror, not a replacement.
 */

/** Default local dev API endpoint used when `VITE_API_BASE_URL` is unset. */
export const DEFAULT_API_BASE_URL = 'http://localhost:8000';

/**
 * Resolve the API base URL from the Vite environment, falling back to the local
 * dev endpoint. A trailing slash is trimmed so path joining is predictable.
 */
export function resolveApiBaseUrl(): string {
  const fromEnv =
    typeof import.meta !== 'undefined' &&
    import.meta.env &&
    typeof import.meta.env.VITE_API_BASE_URL === 'string'
      ? import.meta.env.VITE_API_BASE_URL
      : undefined;
  const raw = (fromEnv && fromEnv.trim()) || DEFAULT_API_BASE_URL;
  return raw.replace(/\/+$/, '');
}

/**
 * Content types accepted for inspection input, mirroring the backend
 * `DEFAULT_ACCEPTED_CONTENT_TYPES` (Req 1.5, 11.2).
 */
export const ACCEPTED_CONTENT_TYPES: readonly string[] = [
  'image/jpeg',
  'image/png',
  'image/webp',
  'image/bmp',
  'video/mp4',
  'video/quicktime',
  'video/x-msvideo', // .avi
];

/**
 * Maximum accepted upload size in bytes (25 MiB), mirroring the backend
 * `DEFAULT_MAX_FILE_SIZE_BYTES` (Req 1.6, 11.2).
 */
export const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024;

/** The envelope API version this client is written against (mirror `API_VERSION`). */
export const EXPECTED_API_VERSION = 'v1';
