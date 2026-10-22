import { describe, expect, it, vi } from 'vitest';
import { ApiClient, type FetchLike } from './client';
import type {
  ApiErrorEnvelope,
  ApiSuccess,
  InspectionReport,
  InspectionSummary,
} from './types';

const BASE = 'http://api.test';

/** Build a `fetch` stub that returns one JSON body with a given status. */
function jsonFetch(body: unknown, status = 200): FetchLike {
  return vi.fn(async () =>
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  );
}

function makeReport(overrides: Partial<InspectionReport> = {}): InspectionReport {
  return {
    schema_version: '1.0',
    inspection_id: 'insp-1',
    mode: 'local',
    final_recommendation: 'Shelf appears stocked.',
    requires_human_review: false,
    initial_finding_id: 'find-1',
    decision_trace: [],
    supporting_evidence_refs: [],
    created_at: '2024-01-01T00:00:00Z',
    status: 'complete',
    ...overrides,
  };
}

describe('ApiClient envelope handling', () => {
  it('returns typed data and reads apiVersion + mode on success', async () => {
    const report = makeReport();
    const envelope: ApiSuccess<InspectionReport> = {
      apiVersion: 'v1',
      mode: 'local',
      data: report,
    };
    const client = new ApiClient({ baseUrl: BASE, fetchImpl: jsonFetch(envelope) });

    const res = await client.getInspection('insp-1');

    expect(res.ok).toBe(true);
    if (res.ok) {
      expect(res.apiVersion).toBe('v1');
      expect(res.mode).toBe('local'); // demo-vs-real labeling source (Req 14.5)
      expect(res.data.inspection_id).toBe('insp-1');
    }
  });

  it('surfaces the mode for real/aws responses', async () => {
    const envelope: ApiSuccess<InspectionReport> = {
      apiVersion: 'v1',
      mode: 'aws',
      data: makeReport({ mode: 'aws' }),
    };
    const client = new ApiClient({ baseUrl: BASE, fetchImpl: jsonFetch(envelope) });

    const res = await client.getInspection('insp-1');

    expect(res.ok && res.mode).toBe('aws');
  });

  it('maps a structured error envelope to a typed failure (not a throw)', async () => {
    const envelope: ApiErrorEnvelope = {
      apiVersion: 'v1',
      mode: 'local',
      error: {
        code: 'not_found',
        message: "No inspection found for id 'missing'.",
        field: 'inspection_id',
      },
    };
    const client = new ApiClient({
      baseUrl: BASE,
      fetchImpl: jsonFetch(envelope, 404),
    });

    const res = await client.getInspection('missing');

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.error.kind).toBe('not_found');
      expect(res.error.field).toBe('inspection_id');
      expect(res.error.status).toBe(404);
      expect(res.apiVersion).toBe('v1');
      expect(res.mode).toBe('local');
    }
  });

  it('maps a validation error envelope with its field', async () => {
    const envelope: ApiErrorEnvelope = {
      apiVersion: 'v1',
      mode: 'local',
      error: { code: 'validation', message: 'content_type is required.', field: 'content_type' },
    };
    const client = new ApiClient({
      baseUrl: BASE,
      fetchImpl: jsonFetch(envelope, 400),
    });

    const res = await client.createInspection({ exampleMode: true });

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.error.kind).toBe('validation');
      expect(res.error.field).toBe('content_type');
    }
  });

  it('returns a transport error for a malformed envelope', async () => {
    const client = new ApiClient({
      baseUrl: BASE,
      fetchImpl: jsonFetch({ unexpected: true }),
    });

    const res = await client.listEvaluations();

    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.error.kind).toBe('transport');
  });

  it('returns a network error when fetch rejects', async () => {
    const failing: FetchLike = vi.fn(async () => {
      throw new Error('offline');
    });
    const client = new ApiClient({ baseUrl: BASE, fetchImpl: failing });

    const res = await client.listInspections();

    expect(res.ok).toBe(false);
    if (!res.ok) {
      expect(res.error.kind).toBe('network');
      expect(res.error.message).toContain('offline');
    }
  });

  it('builds the history query string from status/text/limit', async () => {
    const summaries: InspectionSummary[] = [];
    const envelope: ApiSuccess<InspectionSummary[]> = {
      apiVersion: 'v1',
      mode: 'local',
      data: summaries,
    };
    const spy = vi.fn(async () =>
      new Response(JSON.stringify(envelope), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    const client = new ApiClient({ baseUrl: BASE, fetchImpl: spy });

    await client.listInspections({ status: 'needs_review', text: 'milk', limit: 5 });

    const calledUrl = spy.mock.calls[0]?.[0] as string;
    expect(calledUrl).toContain('/inspections?');
    expect(calledUrl).toContain('status=needs_review');
    expect(calledUrl).toContain('text=milk');
    expect(calledUrl).toContain('limit=5');
  });
});
