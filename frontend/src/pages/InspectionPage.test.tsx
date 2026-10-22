import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'jest-axe';
import { InspectionPage } from './InspectionPage';
import type { ApiClient } from '../api';
import { MAX_FILE_SIZE_BYTES, ok, err } from '../api';
import type { InspectionReport } from '../api';

/**
 * Tests for the Inspection page (task 13.2; Req 11.1-11.6).
 *
 * The API client is mocked via the injectable `client` prop so no backend is
 * needed. We exercise the four view states (empty / progress / success / error)
 * plus client-side file validation and an automated accessibility check.
 */

function makeReport(overrides: Partial<InspectionReport> = {}): InspectionReport {
  return {
    schema_version: '1.0',
    inspection_id: 'insp-42',
    mode: 'local',
    final_recommendation: 'Shelf appears adequately stocked.',
    requires_human_review: false,
    initial_finding_id: 'find-1',
    decision_trace: [],
    supporting_evidence_refs: [],
    created_at: '2024-01-01T00:00:00Z',
    status: 'complete',
    ...overrides,
  };
}

/** Build a stub ApiClient whose `createInspection` is a controllable mock. */
function makeClient(
  createInspection: ApiClient['createInspection'],
): ApiClient {
  return { createInspection } as unknown as ApiClient;
}

/** Make a File of a given byte size and MIME type without allocating bytes. */
function makeFile(name: string, type: string, size: number): File {
  const file = new File(['x'], name, { type });
  Object.defineProperty(file, 'size', { value: size });
  return file;
}

describe('InspectionPage', () => {
  it('renders the empty state before any inspection has started', () => {
    const client = makeClient(vi.fn());
    render(<InspectionPage client={client} />);

    expect(
      screen.getByRole('heading', { level: 1, name: /inspection/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('heading', { name: /no inspection yet/i }),
    ).toBeInTheDocument();
  });

  it('exposes a labeled file input and both actions', () => {
    const client = makeClient(vi.fn());
    render(<InspectionPage client={client} />);

    expect(screen.getByLabelText(/image or video file/i)).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /start inspection/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /run example/i }),
    ).toBeInTheDocument();
  });

  it('shows a validation message and does not submit an oversize file', async () => {
    const user = userEvent.setup();
    const createInspection = vi.fn();
    render(<InspectionPage client={makeClient(createInspection)} />);

    const oversize = makeFile('big.jpg', 'image/jpeg', MAX_FILE_SIZE_BYTES + 1);
    await user.upload(screen.getByLabelText(/image or video file/i), oversize);

    // A validation alert is shown right away...
    expect(await screen.findByRole('alert')).toHaveTextContent(/exceeds/i);

    // ...and starting the inspection does not call the client.
    await user.click(screen.getByRole('button', { name: /start inspection/i }));
    expect(createInspection).not.toHaveBeenCalled();
  });

  it('shows a validation message for an unsupported content type', async () => {
    const user = userEvent.setup();
    const createInspection = vi.fn();
    render(<InspectionPage client={makeClient(createInspection)} />);

    // `userEvent.upload` honors the input's `accept` filter and would drop a
    // text file before it reaches the component. Fire the change directly so the
    // component's own type validation runs (as it would for a drag/drop or a
    // file whose extension slips past the OS picker).
    const input = screen.getByLabelText(/image or video file/i);
    const bad = makeFile('notes.txt', 'text/plain', 10);
    fireEvent.change(input, { target: { files: [bad] } });

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /unsupported content type/i,
    );
    await user.click(screen.getByRole('button', { name: /start inspection/i }));
    expect(createInspection).not.toHaveBeenCalled();
  });

  it('does not submit when no file is selected', async () => {
    const user = userEvent.setup();
    const createInspection = vi.fn();
    render(<InspectionPage client={makeClient(createInspection)} />);

    await user.click(screen.getByRole('button', { name: /start inspection/i }));

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /choose an image or video/i,
    );
    expect(createInspection).not.toHaveBeenCalled();
  });

  it('shows progress then a success indication with the report id', async () => {
    const user = userEvent.setup();
    let resolve!: (value: ReturnType<typeof ok<InspectionReport>>) => void;
    const pending = new Promise<ReturnType<typeof ok<InspectionReport>>>((r) => {
      resolve = r;
    });
    const createInspection = vi.fn(() => pending);
    render(<InspectionPage client={makeClient(createInspection)} />);

    const valid = makeFile('shelf.jpg', 'image/jpeg', 1024);
    await user.upload(screen.getByLabelText(/image or video file/i), valid);
    await user.click(screen.getByRole('button', { name: /start inspection/i }));

    // Progress state is visible while the promise is pending.
    expect(
      await screen.findByRole('heading', { name: /inspection in progress/i }),
    ).toBeInTheDocument();
    expect(createInspection).toHaveBeenCalledWith({ file: valid });

    // Resolve the inspection; the success state renders with the report id.
    resolve(ok(makeReport(), 'v1', 'local'));
    expect(
      await screen.findByRole('heading', { name: /inspection complete/i }),
    ).toBeInTheDocument();
    expect(screen.getByText('insp-42')).toBeInTheDocument();
  });

  it('runs example mode without a selected file', async () => {
    const user = userEvent.setup();
    const createInspection = vi.fn(async () =>
      ok(makeReport({ inspection_id: 'example-1' }), 'v1', 'local'),
    );
    render(<InspectionPage client={makeClient(createInspection)} />);

    await user.click(screen.getByRole('button', { name: /run example/i }));

    expect(
      await screen.findByRole('heading', { name: /inspection complete/i }),
    ).toBeInTheDocument();
    expect(createInspection).toHaveBeenCalledWith({ exampleMode: true });
    expect(screen.getByText('example-1')).toBeInTheDocument();
  });

  it('renders the error state when the inspection returns a failure', async () => {
    const user = userEvent.setup();
    const createInspection = vi.fn(async () =>
      err({
        kind: 'validation',
        message: 'The uploaded file could not be decoded as an image.',
        field: 'media',
      }),
    );
    render(<InspectionPage client={makeClient(createInspection)} />);

    const valid = makeFile('shelf.jpg', 'image/jpeg', 1024);
    await user.upload(screen.getByLabelText(/image or video file/i), valid);
    await user.click(screen.getByRole('button', { name: /start inspection/i }));

    expect(
      await screen.findByRole('heading', { name: /inspection failed/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/could not be decoded as an image/i),
    ).toBeInTheDocument();
  });

  it('has no automatically-detectable accessibility violations', async () => {
    const client = makeClient(vi.fn());
    const { container } = render(<InspectionPage client={client} />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });

  it('has no accessibility violations in the error state', async () => {
    const user = userEvent.setup();
    const createInspection = vi.fn(async () =>
      err({ kind: 'network', message: 'Request failed: offline.' }),
    );
    const { container } = render(
      <InspectionPage client={makeClient(createInspection)} />,
    );

    await user.click(screen.getByRole('button', { name: /run example/i }));
    await waitFor(() =>
      expect(
        screen.getByRole('heading', { name: /inspection failed/i }),
      ).toBeInTheDocument(),
    );

    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
