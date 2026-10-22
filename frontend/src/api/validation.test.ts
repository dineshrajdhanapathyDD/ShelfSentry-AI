import { describe, expect, it } from 'vitest';
import { validateUploadFile } from './validation';
import { MAX_FILE_SIZE_BYTES } from './config';

/** Build a `File` with a controllable size and type for validation tests. */
function fileOfSize(bytes: number, type: string, name = 'f'): File {
  const blob = new Blob([new Uint8Array(bytes)], { type });
  return new File([blob], name, { type });
}

describe('validateUploadFile (client-side mirror of server limits)', () => {
  it('accepts a valid, within-limit image', () => {
    const file = fileOfSize(1024, 'image/png', 'shelf.png');
    expect(validateUploadFile(file)).toBeNull();
  });

  it('rejects an empty file', () => {
    const file = fileOfSize(0, 'image/png');
    const error = validateUploadFile(file);
    expect(error?.field).toBe('empty_file');
  });

  it('rejects an unsupported content type', () => {
    const file = fileOfSize(16, 'application/pdf', 'doc.pdf');
    const error = validateUploadFile(file);
    expect(error?.field).toBe('unsupported_content_type');
  });

  it('rejects a file over the size limit', () => {
    const file = fileOfSize(MAX_FILE_SIZE_BYTES + 1, 'image/jpeg');
    const error = validateUploadFile(file);
    expect(error?.field).toBe('size_exceeded');
  });
});
