/**
 * Client-side upload validation that mirrors the server's limits (task 13.1;
 * Req 11.2). This is a fast-fail pre-flight only: the backend's `validate_input`
 * remains the source of truth (Req 1.3-1.6). Returning a structured
 * `client_validation` error lets the Inspection page surface the exact failing
 * condition before a request is sent.
 */

import { ACCEPTED_CONTENT_TYPES, MAX_FILE_SIZE_BYTES } from './config';
import type { ClientError } from './result';

/** The specific client-side validation condition that failed. */
export type ClientValidationCondition =
  | 'empty_file'
  | 'unsupported_content_type'
  | 'size_exceeded';

/**
 * Validate a selected file against the mirrored content-type and size limits.
 * Returns a {@link ClientError} (kind `client_validation`) when it fails, or
 * `null` when the file passes the client-side checks.
 */
export function validateUploadFile(file: File): ClientError | null {
  if (file.size === 0) {
    return {
      kind: 'client_validation',
      message: 'The selected file is empty.',
      field: 'empty_file',
    };
  }
  if (!ACCEPTED_CONTENT_TYPES.includes(file.type)) {
    return {
      kind: 'client_validation',
      message:
        `Unsupported content type "${file.type || 'unknown'}". ` +
        `Accepted types: ${ACCEPTED_CONTENT_TYPES.join(', ')}.`,
      field: 'unsupported_content_type',
    };
  }
  if (file.size > MAX_FILE_SIZE_BYTES) {
    return {
      kind: 'client_validation',
      message:
        `File is ${file.size} bytes, which exceeds the ` +
        `${MAX_FILE_SIZE_BYTES}-byte limit.`,
      field: 'size_exceeded',
    };
  }
  return null;
}
