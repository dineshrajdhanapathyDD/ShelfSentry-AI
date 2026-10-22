# ShelfSentry AI - frontend

React + TypeScript + Vite + Tailwind CSS frontend for ShelfSentry AI.

This package currently contains the **scaffold and the typed API client**
(task 13.1). The four pages (Inspection, Results, History, Evaluation) are
placeholders filled in by tasks 13.2-13.5.

## Setup

From this `frontend/` directory, with Node.js >= 20:

```bash
npm install
```

All dependency versions are pinned exactly (no `^`/`~` ranges) in
`package.json` for reproducibility.

## Scripts

| Script | What it does |
| --- | --- |
| `npm run dev` | Start the Vite dev server (default http://localhost:5173). |
| `npm run build` | Type-check (`tsc --noEmit`) then produce a production build. |
| `npm run typecheck` | Type-check only. |
| `npm test` | Run the test suite once (no watch). |
| `npm run test:watch` | Run tests in watch mode (opt-in). |

## Configuration

The API client base URL is configurable via `VITE_API_BASE_URL`. Copy
`.env.example` to `.env.local` and adjust as needed:

```bash
cp .env.example .env.local
```

It defaults to `http://localhost:8000` when unset. For **AWS mode**, set it to
the deployed stack's CloudFormation **`ApiUrl`** output (the API Gateway HTTP
API endpoint), for example:

```bash
# read it from the deployed stack
aws cloudformation describe-stacks --stack-name shelfsentry-dev \
  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" \
  --output text --region us-east-1
# then in .env.local
VITE_API_BASE_URL=https://<apiId>.execute-api.us-east-1.amazonaws.com
```

The endpoint is a public HTTPS URL, not a secret. The browser never holds AWS
credentials. See [`../docs/aws-deployment.md`](../docs/aws-deployment.md) for
the full deploy flow (build + push the ARM64 image to ECR, then deploy the
CloudFormation stack).

The client calls these backend routes (served by the API Lambda; see
`backend/shelfsentry/awslambda/api.py`):

| Client method | Route |
| --- | --- |
| `createInspection` | `POST /inspections` (multipart form-data) |
| `getInspection` | `GET /inspections/{id}` |
| `listInspections` | `GET /inspections?status&text&limit` |
| `listEvaluations` / `getEvaluation` | `GET /evaluations` / `GET /evaluations/{id}` |
| `getAnnotatedEvidence` | `GET /inspections/{id}/evidence/{findingId}` |

## Typed API client

`src/api/` mirrors the backend's versioned contracts and response envelopes:

- `types.ts` - TypeScript mirrors of the backend Pydantic contracts
  (`InspectionReport`, `CandidateRegion`, `BoundingBox`, `AgentDecision`,
  `ToolExecution`, `VisionFinding`, `InspectionSummary`, `EvaluationResult`) and
  the `ApiResponse<T>` / error envelope (`{ apiVersion, mode, data | error }`).
- `client.ts` - a thin typed client (`ApiClient`) that calls each endpoint,
  reads the `mode` field for demo-vs-real labeling (Req 14.5), and returns a
  typed `ApiResult` that surfaces structured validation errors to the UI
  instead of throwing (Req 10.5).
- `config.ts` - base URL resolution and the client-side type/size limits that
  mirror the server (Req 11.2).
- `validation.ts` - client-side upload pre-flight validation.
- `result.ts` - the `ApiResult` / `ClientError` discriminated unions.

## Accessibility

The shell provides semantic landmarks and a skip link. Tests use `jest-axe`
to assert no automatically-detectable accessibility violations; the full
pages (tasks 13.2-13.6) target WCAG 2.1 Level AA.
