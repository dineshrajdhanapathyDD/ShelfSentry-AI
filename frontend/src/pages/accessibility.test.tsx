import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { axe } from 'jest-axe';
import { InspectionPage } from './InspectionPage';
import { ResultsPage } from './ResultsPage';
import { HistoryPage } from './HistoryPage';
import { EvaluationPage } from './EvaluationPage';
import type { ApiClient } from '../api';
import { err, ok } from '../api/result';
import type {
  AnnotatedEvidence,
  CandidateRegion,
  EvaluationResult,
  InspectionReport,
  InspectionSummary,
  VisionFinding,
} from '../api/types';

/**
 * Centralized cross-cutting accessibility and state coverage for the four
 * pages (task 13.6; Req 11.6, 12.5, 13.5, 14.6 — WCAG 2.1 AA).
 *
 * The per-page co-located tests already assert each page's primary state is
 * accessible (and, for Inspection, the error state). This file makes the
 * *cross-cutting* coverage explicit and centralized: every page is mounted with
 * a mocked client injected via its props (no backend, no `fetch`) in each of its
 * key view states — primary, empty, and error (plus the loading/progress state
 * where a page has a distinct async one) — and audited with jest-axe.
 *
 * It intentionally does NOT duplicate the behavioral assertions in the
 * co-located page tests; it only fills the automated-a11y gaps those tests leave
 * (empty/error/loading audits that each page test did not already run), plus the
 * demo-vs-real labeling distinctness on the Evaluation page.
 *
 * Every client is a prop-injected stub, so these tests exercise real component
 * markup with no network involved.
 */

// ---------------------------------------------------------------------------
// Shared fixtures (minimal, reused across states)
// ---------------------------------------------------------------------------

const REGION: CandidateRegion = {
  region_id: 'region-a',
  bounding_box: { x: 10, y: 20, w: 120, h: 80 },
  empty_space_ratio: 0.62,
  alt_metric: null,
  heuristic_score: 3,
  uncertainty_value: 0.5,
  frame_id: null,
  visual_caveat: 'A dark gap may be shadow rather than empty shelf.',
};

const FINDING: VisionFinding = {
  schema_version: '1.0.0',
  finding_id: 'finding-1',
  inspection_id: 'insp-123',
  source: 'image',
  frame_id: null,
  opencv_version: '5.0.0',
  candidate_regions: [REGION],
  uncertainty_interpretations: {
    '0.5': 'Moderate confidence: likely empty but shadows are possible.',
  },
  annotated_output_ref: 'insp-123/annotated.png',
  suggested_follow_up: 'inspect_crop',
  is_estimate: true,
};

function makeReport(): InspectionReport & { primary_finding: VisionFinding } {
  const base: InspectionReport = {
    schema_version: '1.0.0',
    inspection_id: 'insp-123',
    mode: 'local',
    final_recommendation: 'Restock the flagged region and verify shadowed gap.',
    requires_human_review: true,
    initial_finding_id: 'finding-1',
    decision_trace: [
      {
        schema_version: '1.0.0',
        entry_type: 'agent_decision',
        decision_id: 'dec-1',
        iteration: 1,
        action: 'finalize_report',
        rationale: 'Sufficient evidence to finalize.',
        args: {},
        decided_by: 'fallback',
        used_fallback: false,
        fallback_reason: null,
      },
    ],
    supporting_evidence_refs: [],
    created_at: '2024-05-01T12:00:00Z',
    status: 'needs_review',
  };
  return { ...base, primary_finding: FINDING };
}

const EVIDENCE: AnnotatedEvidence = {
  inspection_id: 'insp-123',
  finding_id: 'finding-1',
  ref: 'insp-123/annotated.png',
  byte_length: 4,
  data_base64: 'iVBORw0KGgo=',
};

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

const EVAL_REAL: EvaluationResult = {
  schema_version: '1.0.0',
  eval_run_id: 'run-real-001',
  is_demo: false,
  sample_count: 20,
  false_positives: [],
  false_negatives: [],
  agent_decision_correctness: 0.85,
  latency_ms_p50: 120,
  latency_ms_p95: 340,
  documented_limitations: ['Struggles under strong backlighting.'],
  measured_only: true,
};

const EVAL_DEMO: EvaluationResult = {
  ...EVAL_REAL,
  eval_run_id: 'run-demo-001',
  is_demo: true,
  sample_count: 5,
  agent_decision_correctness: 1.0,
  latency_ms_p50: 90,
  latency_ms_p95: 150,
  documented_limitations: ['Demo set is tiny and not representative.'],
};

function stub(partial: Partial<ApiClient>): ApiClient {
  return partial as unknown as ApiClient;
}

/** Audit a rendered container and assert there are no detectable violations. */
async function expectNoViolations(container: HTMLElement) {
  const results = await axe(container);
  expect(results).toHaveNoViolations();
}

// ---------------------------------------------------------------------------
// Inspection page: primary (empty), progress, and error states
// ---------------------------------------------------------------------------

describe('Accessibility: InspectionPage states (Req 11.6)', () => {
  it('is accessible in its empty (primary) state', async () => {
    const client = stub({ createInspection: vi.fn() });
    const { container } = render(<InspectionPage client={client} />);
    // Empty state is the initial render.
    await screen.findByRole('heading', { name: /no inspection yet/i });
    await expectNoViolations(container);
  });

  it('is accessible in its in-progress state', async () => {
    // A never-resolving promise keeps the page in the running/progress phase.
    const pending = new Promise<never>(() => {});
    const client = stub({ createInspection: vi.fn(() => pending) });
    const user = userEvent.setup();
    const { container } = render(<InspectionPage client={client} />);

    // Example mode needs no file selection to enter the running phase.
    await user.click(screen.getByRole('button', { name: /run example/i }));
    await screen.findByRole('heading', { name: /inspection in progress/i });

    await expectNoViolations(container);
  });

  it('is accessible in its error state', async () => {
    const client = stub({
      createInspection: vi.fn(async () =>
        err({ kind: 'network', message: 'Request failed: offline.' }),
      ),
    });
    const user = userEvent.setup();
    const { container } = render(<InspectionPage client={client} />);

    await user.click(screen.getByRole('button', { name: /run example/i }));
    await screen.findByRole('heading', { name: /inspection failed/i });

    await expectNoViolations(container);
  });
});

// ---------------------------------------------------------------------------
// Results page: loading, loaded (primary), empty (not_found), error states
// ---------------------------------------------------------------------------

function renderResults(client: ApiClient, id = 'insp-123') {
  return render(
    <MemoryRouter initialEntries={[`/results/${id}`]}>
      <Routes>
        <Route
          path="/results/:inspectionId"
          element={<ResultsPage client={client} />}
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe('Accessibility: ResultsPage states (Req 12.5)', () => {
  it('is accessible in its loading state', async () => {
    // getInspection never resolves, so the page stays in the loading phase.
    const client = stub({
      getInspection: vi.fn(() => new Promise<never>(() => {})),
      getAnnotatedEvidence: vi.fn(),
    });
    const { container } = renderResults(client);

    // The visible loading paragraph (role=status) names the inspection id;
    // a separate sr-only announcer also says "Loading inspection results.",
    // so scope to the status role to disambiguate.
    await screen.findByRole('status');
    await expectNoViolations(container);
  });

  it('is accessible in its loaded (primary) state', async () => {
    const client = stub({
      getInspection: vi.fn().mockResolvedValue(ok(makeReport(), 'v1', 'local')),
      getAnnotatedEvidence: vi.fn().mockResolvedValue(ok(EVIDENCE, 'v1', 'local')),
    });
    const { container } = renderResults(client);

    await screen.findByText(/restock the flagged region/i);
    await expectNoViolations(container);
  });

  it('is accessible in its empty (not-found) state', async () => {
    const client = stub({
      getInspection: vi
        .fn()
        .mockResolvedValue(
          err({ kind: 'not_found', message: 'No inspection with id "missing".' }),
        ),
      getAnnotatedEvidence: vi.fn(),
    });
    const { container } = renderResults(client, 'missing');

    await screen.findByText(/no such inspection/i);
    await expectNoViolations(container);
  });

  it('is accessible in its error state', async () => {
    const client = stub({
      getInspection: vi
        .fn()
        .mockResolvedValue(err({ kind: 'network', message: 'Request failed.' })),
      getAnnotatedEvidence: vi.fn(),
    });
    const { container } = renderResults(client);

    await screen.findByRole('alert');
    await expectNoViolations(container);
  });
});

// ---------------------------------------------------------------------------
// History page: list (primary), empty, and error states
// ---------------------------------------------------------------------------

function renderHistory(client: ApiClient) {
  return render(
    <MemoryRouter>
      <HistoryPage client={client} searchDebounceMs={0} />
    </MemoryRouter>,
  );
}

describe('Accessibility: HistoryPage states (Req 13.5)', () => {
  it('is accessible in its populated (primary) state', async () => {
    const client = stub({
      listInspections: vi.fn(async () =>
        ok(
          [
            summary({ inspection_id: 'insp-a', status: 'complete' }),
            summary({ inspection_id: 'insp-b', status: 'needs_review' }),
            summary({ inspection_id: 'insp-c', status: 'failed' }),
          ],
          'v1',
          'local' as const,
        ),
      ),
    });
    const { container } = renderHistory(client);

    await screen.findByRole('table');
    await expectNoViolations(container);
  });

  it('is accessible in its empty state', async () => {
    const client = stub({
      listInspections: vi.fn(async () => ok([], 'v1', 'local' as const)),
    });
    const { container } = renderHistory(client);

    await screen.findByText(/no inspections yet/i);
    await expectNoViolations(container);
  });

  it('is accessible in its error state', async () => {
    const client = stub({
      listInspections: vi.fn(async () =>
        err({ kind: 'network', message: 'Request to /inspections failed.' }),
      ),
    });
    const { container } = renderHistory(client);

    await screen.findByRole('alert');
    await expectNoViolations(container);
  });
});

// ---------------------------------------------------------------------------
// Evaluation page: data (primary), empty, and error states + demo-vs-real label
// ---------------------------------------------------------------------------

describe('Accessibility: EvaluationPage states (Req 14.6)', () => {
  it('is accessible in its data (primary) state', async () => {
    const listEvaluations = vi
      .fn()
      .mockResolvedValue({ ok: true, data: [EVAL_REAL, EVAL_DEMO], apiVersion: 'v1', mode: 'local' });
    const { container } = render(
      <EvaluationPage client={{ listEvaluations } as unknown as ApiClient} />,
    );

    // Wait for both cards to render (each demo/real badge is unique text).
    await screen.findByText('Real data');
    await screen.findByText('Demo data');
    await expectNoViolations(container);
  });

  it('is accessible in its empty state', async () => {
    const listEvaluations = vi
      .fn()
      .mockResolvedValue({ ok: true, data: [], apiVersion: 'v1', mode: 'local' });
    const { container } = render(
      <EvaluationPage client={{ listEvaluations } as unknown as ApiClient} />,
    );

    await screen.findByText(/no evaluation runs recorded yet/i);
    await expectNoViolations(container);
  });

  it('is accessible in its error state', async () => {
    const listEvaluations = vi.fn().mockResolvedValue({
      ok: false,
      error: { kind: 'network', message: 'Request to /evaluations failed.' },
    });
    const { container } = render(
      <EvaluationPage client={{ listEvaluations } as unknown as ApiClient} />,
    );

    await screen.findByRole('alert');
    await expectNoViolations(container);
  });
});

// ---------------------------------------------------------------------------
// Evaluation page: demo-vs-real labeling is a visible, non-color-only cue
// that stays distinct whether a record is demo or real (Req 14.5).
// ---------------------------------------------------------------------------

describe('EvaluationPage demo-vs-real labeling (Req 14.5)', () => {
  it('renders a visible "Real data" label (not color-only) for a real record', async () => {
    const listEvaluations = vi
      .fn()
      .mockResolvedValue({ ok: true, data: [EVAL_REAL], apiVersion: 'v1', mode: 'local' });
    render(<EvaluationPage client={{ listEvaluations } as unknown as ApiClient} />);

    // The distinction survives as literal text a screen reader reads verbatim.
    expect(await screen.findByText('Real data')).toBeInTheDocument();
    expect(screen.queryByText('Demo data')).not.toBeInTheDocument();
  });

  it('renders a visible "Demo data" label (not color-only) for a demo record', async () => {
    const listEvaluations = vi
      .fn()
      .mockResolvedValue({ ok: true, data: [EVAL_DEMO], apiVersion: 'v1', mode: 'local' });
    render(<EvaluationPage client={{ listEvaluations } as unknown as ApiClient} />);

    expect(await screen.findByText('Demo data')).toBeInTheDocument();
    expect(screen.queryByText('Real data')).not.toBeInTheDocument();
  });

  it('renders both labels as distinct text when demo and real records coexist', async () => {
    const listEvaluations = vi
      .fn()
      .mockResolvedValue({ ok: true, data: [EVAL_REAL, EVAL_DEMO], apiVersion: 'v1', mode: 'local' });
    render(<EvaluationPage client={{ listEvaluations } as unknown as ApiClient} />);

    expect(await screen.findByText('Real data')).toBeInTheDocument();
    expect(screen.getByText('Demo data')).toBeInTheDocument();
  });
});
