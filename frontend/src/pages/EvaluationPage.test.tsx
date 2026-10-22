import { describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { axe } from 'jest-axe';
import { EvaluationPage } from './EvaluationPage';
import type { ApiClient } from '../api';
import type { ApiResult } from '../api/result';
import type { EvaluationResult } from '../api/types';

/**
 * Tests for the Evaluation page (task 13.5; Req 14.1-14.6, 27.2).
 *
 * The typed API client is mocked so no backend is needed. Each test drives the
 * page with a `listEvaluations` stub returning a typed `ApiResult`.
 */

type EvalClient = Pick<ApiClient, 'listEvaluations'>;

function makeClient(
  result: ApiResult<EvaluationResult[]>,
): { client: EvalClient; listEvaluations: ReturnType<typeof vi.fn> } {
  const listEvaluations = vi.fn().mockResolvedValue(result);
  return { client: { listEvaluations } as unknown as EvalClient, listEvaluations };
}

function okResult(records: EvaluationResult[]): ApiResult<EvaluationResult[]> {
  return { ok: true, data: records, apiVersion: 'v1', mode: 'local' };
}

const realRecord: EvaluationResult = {
  schema_version: '1.0.0',
  eval_run_id: 'run-real-001',
  is_demo: false,
  sample_count: 20,
  false_positives: [
    {
      sample_id: 'fp-sample-7',
      inspection_id: 'insp-7',
      expected: 'stocked',
      actual: 'empty',
      note: 'Shadow misread as a gap',
    },
  ],
  false_negatives: [
    {
      sample_id: 'fn-sample-3',
      inspection_id: 'insp-3',
      expected: 'empty',
      actual: 'stocked',
      note: null,
    },
  ],
  agent_decision_correctness: 0.85,
  latency_ms_p50: 120,
  latency_ms_p95: 340,
  documented_limitations: [
    'Struggles under strong backlighting.',
    'Not validated on reflective packaging.',
  ],
  measured_only: true,
};

const demoRecord: EvaluationResult = {
  schema_version: '1.0.0',
  eval_run_id: 'run-demo-001',
  is_demo: true,
  sample_count: 5,
  false_positives: [],
  false_negatives: [],
  agent_decision_correctness: 1.0,
  latency_ms_p50: 90,
  latency_ms_p95: 150,
  documented_limitations: ['Demo set is tiny and not representative.'],
  measured_only: true,
};

describe('EvaluationPage', () => {
  it('renders measured metrics from a provided EvaluationResult fixture', async () => {
    const { client } = makeClient(okResult([realRecord]));
    render(<EvaluationPage client={client} />);

    // Metrics summary: correctness, FP/FN counts, latency p50/p95.
    expect(await screen.findByText('85.0%')).toBeInTheDocument();
    expect(screen.getByText('120 ms')).toBeInTheDocument();
    expect(screen.getByText('340 ms')).toBeInTheDocument();

    // False-positive example surfaced with its expected/actual/note.
    expect(screen.getByText('fp-sample-7')).toBeInTheDocument();
    expect(screen.getByText('Shadow misread as a gap')).toBeInTheDocument();

    // False-negative example surfaced.
    expect(screen.getByText('fn-sample-3')).toBeInTheDocument();

    // Documented limitations surfaced.
    expect(
      screen.getByText('Struggles under strong backlighting.'),
    ).toBeInTheDocument();
    expect(
      screen.getByText('Not validated on reflective packaging.'),
    ).toBeInTheDocument();

    // Counts appear in the summary table.
    const summary = screen.getByRole('table', {
      name: /measured performance metrics for evaluation run run-real-001/i,
    });
    const fpRow = within(summary).getByRole('rowheader', {
      name: /false positives/i,
    });
    expect(fpRow.parentElement).toHaveTextContent('1');
  });

  it('distinguishes demo records from real records with a non-color-only text label', async () => {
    const { client } = makeClient(okResult([realRecord, demoRecord]));
    render(<EvaluationPage client={client} />);

    // Both labels are rendered as visible text (not color-only cues).
    expect(await screen.findByText('Demo data')).toBeInTheDocument();
    expect(screen.getByText('Real data')).toBeInTheDocument();
  });

  it('shows an honest empty state with no fabricated numbers when the list is empty', async () => {
    const { client } = makeClient(okResult([]));
    const { container } = render(<EvaluationPage client={client} />);

    expect(
      await screen.findByText(/no evaluation runs recorded yet/i),
    ).toBeInTheDocument();

    // No numeric metrics are fabricated in the empty state.
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\bms\b/)).not.toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
    // No bare digits anywhere in the rendered output.
    expect(container.textContent ?? '').not.toMatch(/\d/);
  });

  it('renders a structured error state and can retry', async () => {
    const listEvaluations = vi
      .fn()
      .mockResolvedValueOnce({
        ok: false,
        error: {
          kind: 'network',
          message: 'Request to /evaluations failed: offline',
        },
      } satisfies ApiResult<EvaluationResult[]>)
      .mockResolvedValueOnce(okResult([realRecord]));
    const client = { listEvaluations } as unknown as EvalClient;

    render(<EvaluationPage client={client} />);

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/could not load evaluation runs/i);
    expect(alert).toHaveTextContent(/offline/i);

    await userEvent.click(
      screen.getByRole('button', { name: /retry/i }),
    );

    expect(await screen.findByText('85.0%')).toBeInTheDocument();
    expect(listEvaluations).toHaveBeenCalledTimes(2);
  });

  it('has no automatically-detectable accessibility violations', async () => {
    const { client } = makeClient(okResult([realRecord, demoRecord]));
    const { container } = render(<EvaluationPage client={client} />);

    await screen.findByText('85.0%');
    await waitFor(async () => {
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });
});
