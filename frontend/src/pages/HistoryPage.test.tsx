import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { axe } from 'jest-axe';
import { HistoryPage } from './HistoryPage';
import type { ApiClient, ListInspectionsQuery } from '../api';
import { ok, err } from '../api';
import type { InspectionSummary } from '../api';

/**
 * Tests for the History page (task 13.4; Req 13.1-13.5).
 *
 * The typed API client is mocked at the `listInspections` boundary, so no
 * backend or `fetch` is involved. Routing is provided by a `MemoryRouter` so we
 * can assert the detail-link hrefs resolve to the Results route.
 */

function summary(overrides: Partial<InspectionSummary> = {}): InspectionSummary {
  return {
    inspection_id: 'insp-001',
    mode: 'local',
    status: 'complete',
    final_recommendation: 'Shelf adequately stocked.',
    requires_human_review: false,
    created_at: '2024-05-01T10:00:00.000Z',
    ...overrides,
  };
}

/**
 * Build a mock ApiClient whose `listInspections` returns a successful envelope
 * carrying the provided rows. Returns the spy so tests can assert call args.
 */
function mockClient(rows: InspectionSummary[]) {
  const listInspections = vi.fn(
    async (_query?: ListInspectionsQuery) => ok(rows, 'v1', 'local' as const),
  );
  const client = { listInspections } as unknown as ApiClient;
  return { client, listInspections };
}

function renderPage(client: ApiClient) {
  return render(
    <MemoryRouter>
      <HistoryPage client={client} searchDebounceMs={0} />
    </MemoryRouter>,
  );
}

describe('HistoryPage', () => {
  it('renders a list of summaries with status and timestamp', async () => {
    const rows = [
      summary({ inspection_id: 'insp-aaa', status: 'complete' }),
      summary({
        inspection_id: 'insp-bbb',
        status: 'needs_review',
        final_recommendation: 'Possible gap near top-left.',
        created_at: '2024-05-02T14:30:00.000Z',
      }),
    ];
    const { client } = mockClient(rows);
    renderPage(client);

    // Semantic table with column headers (WCAG).
    const table = await screen.findByRole('table');
    expect(
      within(table).getByRole('columnheader', { name: /status/i }),
    ).toBeInTheDocument();
    expect(
      within(table).getByRole('columnheader', { name: /created/i }),
    ).toBeInTheDocument();

    // Both rows render with an id link and a status text cue. Scope the status
    // text to the table so it doesn't collide with the filter's <option>s.
    expect(screen.getByRole('link', { name: 'insp-aaa' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'insp-bbb' })).toBeInTheDocument();
    expect(within(table).getByText('Complete')).toBeInTheDocument();
    expect(within(table).getByText('Needs review')).toBeInTheDocument();

    // Timestamp is rendered as a machine-readable <time>.
    const timeEl = table.querySelector('time[datetime="2024-05-02T14:30:00.000Z"]');
    expect(timeEl).not.toBeNull();
  });

  it('calls the client once on mount with no filters', async () => {
    const { client, listInspections } = mockClient([summary()]);
    renderPage(client);
    await screen.findByRole('table');
    expect(listInspections).toHaveBeenCalledTimes(1);
    expect(listInspections).toHaveBeenLastCalledWith({});
  });

  it('typing in search triggers a filtered client call', async () => {
    const user = userEvent.setup();
    const { client, listInspections } = mockClient([summary()]);
    renderPage(client);
    await screen.findByRole('table');

    const search = screen.getByLabelText(/search/i);
    await user.type(search, 'gap');

    await waitFor(() => {
      expect(listInspections).toHaveBeenLastCalledWith({ text: 'gap' });
    });
  });

  it('changing the status filter triggers a filtered client call and updates the list', async () => {
    const user = userEvent.setup();
    const listInspections = vi.fn(async (query?: ListInspectionsQuery) => {
      if (query?.status === 'failed') {
        return ok([summary({ inspection_id: 'insp-failed', status: 'failed' })], 'v1', 'local' as const);
      }
      return ok([summary({ inspection_id: 'insp-ok', status: 'complete' })], 'v1', 'local' as const);
    });
    const client = { listInspections } as unknown as ApiClient;
    renderPage(client);

    await screen.findByRole('link', { name: 'insp-ok' });

    const statusSelect = screen.getByLabelText(/status/i);
    await user.selectOptions(statusSelect, 'failed');

    await waitFor(() => {
      expect(listInspections).toHaveBeenLastCalledWith({ status: 'failed' });
    });
    expect(
      await screen.findByRole('link', { name: 'insp-failed' }),
    ).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: 'insp-ok' })).not.toBeInTheDocument();
  });

  it('links each row to the Results detail route for its inspection id', async () => {
    const { client } = mockClient([summary({ inspection_id: 'insp-xyz' })]);
    renderPage(client);

    const link = await screen.findByRole('link', { name: 'insp-xyz' });
    expect(link).toHaveAttribute('href', '/results/insp-xyz');
  });

  it('renders an empty state when there are no inspections', async () => {
    const { client } = mockClient([]);
    renderPage(client);

    expect(await screen.findByText(/no inspections yet/i)).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('renders a structured error state when the client fails', async () => {
    const listInspections = vi.fn(async () =>
      err({ kind: 'network', message: 'Request to /inspections failed.' }),
    );
    const client = { listInspections } as unknown as ApiClient;
    renderPage(client);

    const alert = await screen.findByRole('alert');
    expect(within(alert).getByText(/request to \/inspections failed/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
  });

  it('has no automatically-detectable accessibility violations', async () => {
    const { client } = mockClient([
      summary({ inspection_id: 'insp-a11y', status: 'needs_review' }),
    ]);
    const { container } = renderPage(client);
    await screen.findByRole('table');

    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
