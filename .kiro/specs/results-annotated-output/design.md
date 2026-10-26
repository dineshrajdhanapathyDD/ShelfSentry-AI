# Design - Show annotated output and candidate regions on the Results page (AWS mode)

## Overview

On the deployed app, the Results page renders a placeholder instead of the processed image: the "Annotated output" figure shows "Diagram of 0 highlighted candidate regions ... annotated image is referenced by the report", and the "Candidate regions" section shows "No candidate regions are inlined with this report."

This is not a styling issue; the data the page needs never arrives in AWS mode. This design fixes two independent, code-confirmed root causes so the real OpenCV 5 annotated image and the measured candidate regions display on the Results page in AWS mode, while preserving honest "not available" fallbacks and the Req 9 security boundary.

Root cause A - annotated image bytes are never served in AWS mode. In `backend/shelfsentry/api/handlers.py` the evidence resolver is wired ONLY for a local adapter (`StorageMediaResolver(adapter)` guarded by `isinstance(adapter, LocalStorageAdapter)`). In AWS mode the adapter is an `AwsStorageAdapter`, so the resolver is `None` and `get_annotated_evidence` returns `not_found`. The frontend then has `evidence = null`, `annotatedUrl = null`, and shows the placeholder. `StorageMediaResolver` only uses `get_media_ref_for` / `get_media` / `put_media`, all of which `AwsStorageAdapter` already implements; the async worker already uses `StorageMediaResolver(adapter)` over the AWS adapter. The guard is simply too strict.

Root cause B - candidate regions are not carried on the report. `InspectionReport` has `initial_finding_id` + `decision_trace` but no inlined finding. The Results page reads `finding.primary_finding` candidate regions, which do not exist on the report, so regions are always empty - even in local mode.

Goals: (1) annotated image displays in AWS mode; (2) candidate regions + metric / uncertainty / caveat display; (3) honest "not available" when evidence or a finding genuinely does not exist (never fabricated); (4) no local-mode regression; (5) security boundary unchanged.

Non-goals: no in-browser image processing (image is produced server-side by OpenCV 5 and served as bytes); no change to the agent loop, MCP tools, or the vision pipeline.

## Architecture

End-to-end data flow for the Results page (AWS mode):

```
ResultsPage
  -> GET /inspections/{id}                 (report: trace + initial_finding_id)
  -> GET /inspections/{id}/evidence/{fid}  (annotated bytes + finding projection)
        API Lambda handler (LocalApi.get_annotated_evidence)
          -> StorageMediaResolver(adapter)         [Fix A: build for AWS adapter]
               -> get_evidence_ref(id, fid)        -> media ref
               -> get_media(ref)                   -> annotated PNG bytes (S3)
          -> finding projection (regions, interpretations, opencv_version)
                                                     [Fix B: from real analysis]
  -> frontend renders base64 image + overlay boxes + region details
```

Fix A unblocks the image bytes (the handler already base64-encodes them in `api.py` `_evidence_response`; the frontend already turns `data_base64` into a data URL). Fix B supplies the regions the overlay and per-region sections need. No CloudFormation/template change is required; backend ships as a new container image (OpenCV 5 build assertion unchanged) via `EcrImageUri`, frontend via Amplify.

## Components and Interfaces

### backend/shelfsentry/api/handlers.py (Fix A + Fix B)

Replace the `isinstance` guard with a capability check so the resolver is built for any adapter that structurally supports media ops:

```python
def _supports_media(adapter) -> bool:
    return all(
        callable(getattr(adapter, name, None))
        for name in ("get_media_ref_for", "get_media", "put_media")
    )

self._resolver = StorageMediaResolver(adapter) if _supports_media(adapter) else None
```

Both Local and AWS adapters pass; a partial/fake adapter still yields `None` and the handler returns the honest "not available" error. `get_annotated_evidence` is unchanged in contract but now resolves in AWS mode. Extend the returned descriptor to also carry the finding projection (Fix B).

### backend/shelfsentry/awslambda/api.py

`_evidence_response` already base64-encodes `ev.data` into `data_base64`. Extend it to pass through the optional finding projection (regions, interpretations, opencv_version) when present.

### backend/shelfsentry/data/storage.py

Widen the `StorageMediaResolver.__init__` type annotation from `LocalStorageAdapter` to the `StorageAdapter` Protocol (annotation only; no behavior change). This matches its actual structural contract, already exercised by the worker.

### frontend/src/pages/ResultsPage.tsx

Read `candidate_regions`, `uncertainty_interpretations`, and `opencv_version` from the evidence result's finding projection (`evidence.primary_finding`) instead of the nonexistent `report.primary_finding`. No change to the overlay, region-detail, caveat, or honesty labeling logic.

### frontend/src/api/types.ts

Add an optional `primary_finding` projection to the `AnnotatedEvidence` type (candidate regions, interpretations, opencv version), all optional so the "not available" path still type-checks.

## Data Models

No stored-contract field is renamed or removed. Additive only. The AnnotatedEvidence descriptor (API to client) gains an optional `primary_finding` projection:

```
AnnotatedEvidence {
  inspection_id: string
  finding_id: string
  ref: string
  byte_length: number
  data_base64?: string                 # already present
  primary_finding?: {                  # NEW (optional)
    candidate_regions: CandidateRegion[]   # real, from OpenCV analysis
    uncertainty_interpretations: Record<string,string>
    opencv_version: string
  }
}
```

The `primary_finding` is a projection of the initial `VisionFinding`, derived from a real OpenCV 5 analysis of the stored image (the MCP dispatcher already re-derives a finding deterministically from stored bytes when none is cached). Regions are therefore measured, never fabricated, and always match the annotated image they accompany.

## Correctness Properties

### Property 1: Evidence present implies image rendered

If stored annotated bytes exist for `initial_finding_id`, the evidence endpoint returns them (base64) and the Results page renders an image element, not the placeholder.

**Validates: Requirements 12.1**

### Property 2: Absence is honest, never fabricated

If no image or evidence exists for the requested finding, the endpoint returns `not_found` and the page shows the "not available" state; no invented image or region is produced.

**Validates: Requirements 9, 27.2, 27.4**

### Property 3: Regions are measured, not invented

The regions shown are exactly those produced by a real OpenCV 5 analysis of the stored image for that inspection, and are labeled estimates with visual caveats.

**Validates: Requirements 12.2, 2.5, 4.7**

### Property 4: No local-mode regression

Local-mode Results behavior is unchanged (same resolver construction result, same rendering).

**Validates: Requirements 12.5**

### Property 5: Security boundary preserved

Media is addressed only by server-minted opaque media refs and inspection-scoped ids; no path or URL supplied by the client is dereferenced.

**Validates: Requirements 9**

## Error Handling

- Missing image or evidence: resolver returns `None`, handler returns a structured `not_found` envelope; the page keeps its placeholder text.
- Malformed/absent finding id: structured `validation`/`not_found`, never a raw exception across the boundary (existing behavior).
- Evidence byte read failure (stale ref): `not_found` with an honest message; no partial/garbled image is emitted.
- The finding projection is best-effort: if it cannot be derived, the annotated image still renders and the regions section shows the existing honest empty state rather than failing the whole response.

## Testing Strategy

Backend unit tests (in-memory fake S3/DDB, no AWS, no Bedrock):
- resolver IS built for the AWS adapter (capability check);
- `get_annotated_evidence` returns base64 bytes for a stored finding;
- returns `not_found` when no evidence exists (no fabrication);
- evidence response carries real candidate regions when an image is stored, and omits/does-not-fabricate them when absent.

Frontend tests: ResultsPage renders an image element when the evidence result carries `data_base64`; renders overlay boxes + region details when `primary_finding` is present; keeps the honest placeholder otherwise.

Regression: existing ResultsPage and handler tests stay green; full backend suite and frontend tests pass; production build succeeds.

Live verification after redeploy: open the deployed Results page for a real inspection and confirm the annotated image renders and regions populate; confirm an honest "not available" state for an inspection with no stored evidence.

## Rollout and Risks

Rollout: backend to a new container image in ECR, update `EcrImageUri` via the single CloudFormation template; frontend via Amplify build. No template change.

Risks:
- If the finding id stored for evidence differs from `initial_finding_id`, Fix A resolves nothing; mitigation is to verify the seeded id equals the advertised `initial_finding_id` and look up by that id.
- Returning region data on the evidence endpoint grows the payload; bounded by the small number of candidate regions per inspection (acceptable at demo scale).