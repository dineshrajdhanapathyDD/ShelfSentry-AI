import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { axe } from 'jest-axe';
import { ResultsPage } from './ResultsPage';
import type { ApiClient } from '../api/client';
import { err, ok } from '../api/result';
import type {
  AnnotatedEvidence,
  CandidateRegion,
  InspectionReport,
  VisionFinding,
} from '../api/types';

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const REGION_A: CandidateRegion = {
  region_id: 'region-a',
  bounding_box: { x: 10, y: 20, w: 120, h: 80 },
  empty_space_ratio: 0.62,
  alt_metric: null,
  heuristic_score: 3,
  uncertainty_value: 0.5,
  frame_id: null,
  visual_caveat:
    'A dark gap may be shadow rather than empty shelf; verify before restock.',
};

const REGION_B: CandidateRegion = {
  region_id: 'region-b',
  bounding_box: { x: 200, y: 40, w: 90, h: 140 },
  empty_space_ratio: null,
  alt_metric: { texture_variance: 0.12 },
  heuristic_score: 1,
  uncertainty_value: 0.9,
  frame_id: 'frame_000003',
  visual_caveat: 'Low-texture region; measurement may reflect packaging, not stock.',
};

const PRIMARY_FINDING: VisionFinding = {
  schema_version: '1.0.0',
  finding_id: 'finding-1',
  inspection_id: 'insp-123',
  source: 'image',
  frame_id: null,
  opencv_version: '5.0.0',
  candidate_regions: [REGION_A, REGION_B],
  uncertainty_interpretations: {
    '0.5': 'Moderate confidence: the region is likely empty but shadows are possible.',
    '0.9': 'High confidence in the measurement for this region.',
  },
  annotated_output_ref: 'insp-123/annotated.png',
  suggested_follow_up: 'inspect_crop',
  is_estimate: true,
};

/**
 * A realistic report whose decision trace alternates an AgentDecision and the
 * ToolExecution it triggered, ending in a finalize decision. The primary
 * finding is inlined under `primary_finding` so the page can render regions.
 */
function makeReport(overrides: Partial<InspectionReport> = {}): InspectionReport {
  const base: InspectionReport = {
    schema_version: '1.0.0',
    inspection_id: 'insp-123',
    mode: 'local',
    final_recommendation: 'Restock the two flagged regions and verify shadowed gap.',
    requires_human_review: true,
    initial_finding_id: 'finding-1',
    decision_trace: [
      {
        schema_version: '1.0.0',
        entry_type: 'agent_decision',
        decision_id: 'dec-1',
        iteration: 1,
        action: 'inspect_crop',
        rationale: 'High uncertainty on region-a warrants a closer crop inspection.',
        args: { region_id: 'region-a' },
        decided_by: 'fallback',
        used_fallback: true,
        fallback_reason: 'Bedrock unavailable in local mode.',
      },
      {
        schema_version: '1.0.0',
        entry_type: 'tool_execution',
        execution_id: 'exec-1',
        iteration: 1,
        tool: 'inspect_region_crop',
        args: { region_id: 'region-a' },
        result_ref: 'insp-123/crop-1.json',
        status: 'ok',
        error: null,
      },
      {
        schema_version: '1.0.0',
        entry_type: 'agent_decision',
        decision_id: 'dec-2',
        iteration: 2,
        action: 'finalize_report',
        rationale: 'Crop confirmed the empty region; sufficient evidence to finalize.',
        args: {},
        decided_by: 'fallback',
        used_fallback: false,
        fallback_reason: null,
      },
    ],
    supporting_evidence_refs: ['insp-123/crop-1.json'],
    created_at: '2024-05-01T12:00:00Z',
    status: 'needs_review',
  };
  // Inline the primary finding for region rendering (the page reads it when present).
  return { ...base, ...overrides, primary_finding: PRIMARY_FINDING } as InspectionReport & {
    primary_finding: VisionFinding;
  };
}

const EVIDENCE: AnnotatedEvidence = {
  inspection_id: 'insp-123',
  finding_id: 'finding-1',
  ref: 'insp-123/annotated.png',
  byte_length: 4,
  data_base64: 'iVBORw0KGgo=',
};

// ---------------------------------------------------------------------------
// Client stubs
// ---------------------------------------------------------------------------

function stubClient(partial: Partial<ApiClient>): ApiClient {
  return partial as unknown as ApiClient;
}

function renderPage(client: ApiClient, id = 'insp-123') {
  return render(
    <MemoryRouter initialEntries={[`/results/${id}`]}>
      <Routes>
        <Route path="/results/:inspectionId" element={<ResultsPage client={client} />} />
      </Routes>
    </MemoryRouter>,
  );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ResultsPage', () => {
  it('renders regions with their measured metric, uncertainty interpretation, and caveat', async () => {
    const client = stubClient({
      getInspection: vi.fn().mockResolvedValue(ok(makeReport(), 'v1', 'local')),
      getAnnotatedEvidence: vi.fn().mockResolvedValue(ok(EVIDENCE, 'v1', 'local')),
    });

    renderPage(client);

    // Metric for region-a (empty_space_ratio -> percent, labeled estimate).
    expect(await screen.findByText(/62\.0%/)).toBeInTheDocument();
    expect(screen.getAllByText(/\(estimate\)/i).length).toBeGreaterThan(0);

    // Documented interpretation for the 0.5 uncertainty value.
    expect(
      screen.getByTestId('uncertainty-interpretation-region-a'),
    ).toHaveTextContent(/moderate confidence/i);
    // And the 0.9 value interpretation for region-b.
    expect(
      screen.getByTestId('uncertainty-interpretation-region-b'),
    ).toHaveTextContent(/high confidence/i);

    // Visual caveat on each region.
    expect(screen.getByTestId('visual-caveat-region-a')).toHaveTextContent(
      /shadow rather than empty shelf/i,
    );
    expect(screen.getByTestId('visual-caveat-region-b')).toHaveTextContent(
      /low-texture region/i,
    );
  });

  it('renders the decision trace (action + rationale + tool result) in order', async () => {
    const client = stubClient({
      getInspection: vi.fn().mockResolvedValue(ok(makeReport(), 'v1', 'local')),
      getAnnotatedEvidence: vi.fn().mockResolvedValue(ok(EVIDENCE, 'v1', 'local')),
    });

    renderPage(client);

    const trace = await screen.findByTestId('decision-trace');
    const items = within(trace).getAllByRole('listitem');
    expect(items).toHaveLength(3);

    // Order: agent_decision -> tool_execution -> agent_decision.
    expect(items[0]).toHaveAttribute('data-entry-type', 'agent_decision');
    expect(items[0]).toHaveTextContent(/inspect_crop/);
    expect(items[0]).toHaveTextContent(/closer crop inspection/i);

    expect(items[1]).toHaveAttribute('data-entry-type', 'tool_execution');
    expect(items[1]).toHaveTextContent(/inspect_region_crop/);
    expect(items[1]).toHaveTextContent(/\bok\b/);

    expect(items[2]).toHaveAttribute('data-entry-type', 'agent_decision');
    expect(items[2]).toHaveTextContent(/finalize_report/);

    // Initial finding id surfaced.
    expect(screen.getByTestId('initial-finding-id')).toHaveTextContent('finding-1');
  });

  it('renders the final recommendation and human-review status', async () => {
    const client = stubClient({
      getInspection: vi.fn().mockResolvedValue(ok(makeReport(), 'v1', 'local')),
      getAnnotatedEvidence: vi.fn().mockResolvedValue(ok(EVIDENCE, 'v1', 'local')),
    });

    renderPage(client);

    expect(
      await screen.findByText(/restock the two flagged regions/i),
    ).toBeInTheDocument();
    expect(screen.getByTestId('report-status')).toHaveTextContent(/needs human review/i);
    expect(screen.getByTestId('human-review-status')).toHaveTextContent(
      /human review required/i,
    );
  });

  it('shows the empty state when the inspection id is not found', async () => {
    const client = stubClient({
      getInspection: vi
        .fn()
        .mockResolvedValue(
          err({ kind: 'not_found', message: 'No inspection with id "missing".' }),
        ),
      getAnnotatedEvidence: vi.fn(),
    });

    renderPage(client, 'missing');

    expect(await screen.findByText(/no such inspection/i)).toBeInTheDocument();
    expect(screen.getByText(/no inspection with id "missing"/i)).toBeInTheDocument();
    // Evidence is never fetched when the report is missing.
    expect(client.getAnnotatedEvidence).not.toHaveBeenCalled();
  });

  it('shows an error state for non-not_found failures', async () => {
    const client = stubClient({
      getInspection: vi
        .fn()
        .mockResolvedValue(err({ kind: 'network', message: 'Request failed.' })),
      getAnnotatedEvidence: vi.fn(),
    });

    renderPage(client);

    expect(await screen.findByRole('alert')).toHaveTextContent(/request failed/i);
  });

  it('renders regions even when annotated evidence is unavailable', async () => {
    const client = stubClient({
      getInspection: vi.fn().mockResolvedValue(ok(makeReport(), 'v1', 'local')),
      getAnnotatedEvidence: vi
        .fn()
        .mockResolvedValue(err({ kind: 'not_found', message: 'no evidence' })),
    });

    renderPage(client);

    // Falls back to an overlay diagram with an accessible label.
    expect(
      await screen.findByRole('img', { name: /highlighted candidate/i }),
    ).toBeInTheDocument();
    expect(screen.getByTestId('region-overlay-region-a')).toBeInTheDocument();
  });

  it('has no automatically-detectable accessibility violations', async () => {
    const client = stubClient({
      getInspection: vi.fn().mockResolvedValue(ok(makeReport(), 'v1', 'local')),
      getAnnotatedEvidence: vi.fn().mockResolvedValue(ok(EVIDENCE, 'v1', 'local')),
    });

    const { container } = renderPage(client);

    // Wait for content to load before auditing.
    await screen.findByText(/restock the two flagged regions/i);

    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});
