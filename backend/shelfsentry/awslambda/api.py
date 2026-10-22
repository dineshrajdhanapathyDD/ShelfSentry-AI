"""Synchronous API Lambda handler (API Gateway HTTP API v2 proxy).

Adapts API Gateway HTTP API v2 (payload format 2.0) events to the existing,
framework-agnostic :class:`~shelfsentry.api.handlers.LocalApi` and returns the
versioned, mode-stamped envelope as JSON. No core perception/decision logic
lives here - this is pure transport.

Routes (method + path) - these mirror EXACTLY what the frontend typed client
(frontend/src/api/client.ts) calls, so the deployed API is drop-in for the UI:

* ``GET  /health``                                   -> liveness + mode/model
* ``POST /inspections``                              -> run an inspection, return report
* ``GET  /inspections/{id}``                         -> fetch a stored report
* ``GET  /inspections``                              -> history (status/text/limit)
* ``GET  /evaluations``                              -> measured eval records (may be empty)
* ``GET  /evaluations/{id}``                         -> one eval record
* ``GET  /inspections/{id}/evidence/{findingId}``    -> annotated-evidence descriptor
* ``GET  /jobs/{id}``                                -> async job status (programmatic)

Create request body: the frontend sends ``multipart/form-data`` with fields
``media`` (the file), ``content_type``, optional ``example_mode=true``, and
optional ``inspection_id``. For programmatic callers (curl) a JSON body with
``media_base64`` / ``content_type`` / ``example_mode`` is also accepted. Both
run the REAL OpenCV 5 pipeline and the bounded agent loop with Bedrock as the
primary decision path (Req 16.2, 16.4) and return an ``InspectionReport``
envelope synchronously, which is what the UI expects.

A programmatic JSON caller MAY instead POST a ``video/*`` with ``media_base64``
AND ``async=true`` to hand the job to the worker Lambda (``WORKER_FUNCTION_NAME``)
with ``InvocationType=Event`` and poll ``GET /jobs/{id}`` (Req 19.4, 19.5). The
browser UI does not use the async path (it has no job-polling screen); it always
gets a synchronous report.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from shelfsentry.api.handlers import LocalApi
from shelfsentry.aws_worker import run_aws_inspection
from shelfsentry.config.mode import resolve_runtime_config
from shelfsentry.data.aws_storage import AwsStorageAdapter
from shelfsentry.data.storage import JobRecord

# Non-secret config read from the environment (set by CloudFormation).
_ENV_WORKER_FN = "WORKER_FUNCTION_NAME"
_ENV_MODEL_ID = "SHELFSENTRY_BEDROCK_MODEL_ID"
_ENV_KEY_PREFIX = "SHELFSENTRY_KEY_PREFIX"

_VIDEO_PREFIX = "video/"

# CORS headers echoed on every response. API Gateway HTTP API also applies its
# own CORS config; these make the Lambda-proxy responses explicit and keep the
# browser happy whether or not the gateway rewrites them.
_CORS = {
    "access-control-allow-origin": "*",
    "access-control-allow-methods": "GET,POST,OPTIONS",
    "access-control-allow-headers": "content-type,accept",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_adapter() -> AwsStorageAdapter:
    """Build the AWS storage adapter from the resolved runtime config.

    ``resolve_runtime_config`` enforces that AWS mode has every required
    non-secret key (bucket, table, region) and never silently falls back to
    local mode (Req 17.2, 17.3). boto3 clients are created lazily on first use.
    """
    cfg = resolve_runtime_config()
    if not cfg.is_aws or cfg.aws is None:
        raise RuntimeError(
            "API Lambda requires SHELFSENTRY_MODE=aws with S3/DDB/region set."
        )
    return AwsStorageAdapter(
        bucket=cfg.aws.s3_bucket,
        table=cfg.aws.ddb_table,
        region=cfg.aws.region,
        key_prefix=os.environ.get(_ENV_KEY_PREFIX, ""),
    )


def _aws_runner(
    data: bytes,
    content_type: str,
    *,
    adapter: Any,
    inspection_id: str | None = None,
    mode: Any = "aws",
) -> Any:
    """Adapt the runner signature LocalApi expects to ``run_aws_inspection``.

    ``LocalApi`` passes ``mode=`` to its injected runner, but the AWS runner
    derives the mode itself (it always stamps ``mode="aws"`` on the report), so
    this shim absorbs and drops ``mode`` and forwards the rest unchanged. It
    adds no behavior; it only reconciles the two signatures without modifying
    either core module.
    """
    return run_aws_inspection(
        data, content_type, adapter=adapter, inspection_id=inspection_id
    )


class _AwsEvalStore:
    """Adapt the AWS adapter's eval methods to the ``EvalStore`` protocol.

    ``LocalApi`` expects an eval store exposing ``list_results()`` /
    ``get_result(id)``; the AWS adapter exposes ``list_evals()`` / ``get_eval``.
    This thin shim bridges the two names and returns measured records only (it
    invents nothing - an absent record yields an empty list / None).
    """

    def __init__(self, adapter: AwsStorageAdapter) -> None:
        self._adapter = adapter

    def list_results(self) -> Any:
        return self._adapter.list_evals()

    def get_result(self, eval_run_id: str) -> Any:
        return self._adapter.get_eval(eval_run_id)


def _build_api(adapter: AwsStorageAdapter) -> LocalApi:
    """Wire the handler set for AWS mode: AWS adapter + Bedrock-backed runner.

    The eval store reads measured EvaluationResult records persisted in the same
    single table, so GET /evaluations serves real stored runs (and honestly
    returns an empty list until one is recorded) - never a fabricated record.
    """
    return LocalApi(
        adapter, runner=_aws_runner, mode="aws", eval_store=_AwsEvalStore(adapter)
    )


def _response(status: int, body: Any) -> dict[str, Any]:
    """Build an HTTP API v2 proxy response from an envelope/plain mapping."""
    payload = body.model_dump() if hasattr(body, "model_dump") else body
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json", **_CORS},
        "body": json.dumps(payload, default=str),
    }


def _error(status: int, code: str, message: str, field: str | None = None) -> dict[str, Any]:
    detail: dict[str, Any] = {"code": code, "message": message}
    if field is not None:
        detail["field"] = field
    return _response(status, {"apiVersion": "v1", "mode": "aws", "error": detail})


def _status_for(envelope: Any) -> int:
    """Map an envelope to an HTTP status (success 200; errors by code)."""
    if getattr(envelope, "ok", False):
        return 200
    code = getattr(getattr(envelope, "error", None), "code", None)
    return {"validation": 400, "bad_request": 400, "not_found": 404}.get(code, 400)


def _route(event: dict[str, Any]) -> tuple[str, str]:
    """Extract (method, path) from an HTTP API v2 event."""
    ctx = event.get("requestContext", {}) or {}
    http = ctx.get("http", {}) or {}
    method = str(http.get("method", "GET")).upper()
    path = str(http.get("path", event.get("rawPath", "/")))
    return method, path


def _raw_body_bytes(event: dict[str, Any]) -> bytes:
    """Return the raw request body as bytes, decoding base64 when flagged."""
    raw = event.get("body")
    if raw is None:
        return b""
    if event.get("isBase64Encoded"):
        return base64.b64decode(raw)
    return raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)


def _content_type_header(event: dict[str, Any]) -> str:
    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    return str(headers.get("content-type", ""))


def _extract_boundary(content_type_header: str) -> bytes | None:
    """Pull the boundary token out of a multipart Content-Type header."""
    for part in content_type_header.split(";"):
        part = part.strip()
        if part.lower().startswith("boundary="):
            b = part[len("boundary="):].strip().strip('"')
            return b.encode("utf-8") if b else None
    return None


def _parse_disposition(header_block: bytes) -> tuple[str | None, str | None, str | None]:
    """Return (field_name, filename, content_type) from a part's header block."""
    name: str | None = None
    filename: str | None = None
    ctype: str | None = None
    for line in header_block.split(b"\r\n"):
        try:
            text = line.decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - defensive
            continue
        low = text.lower()
        if low.startswith("content-disposition:"):
            for seg in text.split(";"):
                seg = seg.strip()
                if seg.lower().startswith("name="):
                    name = seg[len("name="):].strip().strip('"')
                elif seg.lower().startswith("filename="):
                    filename = seg[len("filename="):].strip().strip('"')
        elif low.startswith("content-type:"):
            ctype = text.split(":", 1)[1].strip()
    return name, filename, ctype


def _parse_multipart(body: bytes, content_type_header: str) -> dict[str, Any]:
    """Parse a multipart/form-data body by splitting on the declared boundary.

    A direct byte-level splitter (no email/cgi dependency). This preserves
    binary file bodies (e.g. PNG bytes) exactly, which the email-policy parser
    did not reliably do for real browser FormData routed through API Gateway.
    Only the fields the frontend sends are needed: ``media`` (file),
    ``content_type``, ``example_mode``, ``inspection_id``.
    """
    boundary = _extract_boundary(content_type_header)
    if not boundary:
        return {}
    delim = b"--" + boundary

    out: dict[str, Any] = {}
    # Split on the boundary; drop the preamble and the closing "--" epilogue.
    for chunk in body.split(delim):
        if chunk in (b"", b"--", b"--\r\n", b"\r\n"):
            continue
        # A part is "\r\n<headers>\r\n\r\n<content>\r\n"; trim the leading CRLF.
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        sep = chunk.find(b"\r\n\r\n")
        if sep == -1:
            continue
        header_block = chunk[:sep]
        content = chunk[sep + 4:]
        name, filename, ctype = _parse_disposition(header_block)
        if not name:
            continue
        if filename is not None:
            out[name] = {"filename": filename, "bytes": content, "type": ctype or ""}
        else:
            out[name] = content.decode("utf-8", "replace").strip()
    return out


def _decode_json(event: dict[str, Any]) -> dict[str, Any]:
    raw = _raw_body_bytes(event)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _new_ids() -> tuple[str, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return (
        f"insp-{stamp}-{uuid.uuid4().hex[:12]}",
        f"job-{stamp}-{uuid.uuid4().hex[:12]}",
    )


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """API Gateway HTTP API v2 entry point (Req 19.1)."""
    method, path = _route(event)
    norm = path.rstrip("/") or "/"

    # CORS preflight.
    if method == "OPTIONS":
        return {"statusCode": 204, "headers": _CORS, "body": ""}

    # Liveness does not need AWS; answer before building the adapter.
    if method == "GET" and norm in ("/", "/health"):
        return _response(
            200,
            {
                "apiVersion": "v1",
                "mode": "aws",
                "status": "ok",
                "bedrockModelId": os.environ.get(_ENV_MODEL_ID, "(default)"),
            },
        )

    adapter = _build_adapter()
    api = _build_api(adapter)

    if method == "POST" and norm == "/inspections":
        return _create_inspection(event, api, adapter)

    # Evidence: /inspections/{id}/evidence/{findingId}
    if method == "GET" and path.startswith("/inspections/") and "/evidence/" in path:
        rest = path[len("/inspections/"):]
        inspection_id, _, tail = rest.partition("/evidence/")
        finding_id = tail
        env = api.get_annotated_evidence(inspection_id, finding_id)
        return _evidence_response(env)

    if method == "GET" and path.startswith("/inspections/"):
        inspection_id = path[len("/inspections/"):]
        return _response(_status_for(env := api.get_inspection(inspection_id)), env)

    if method == "GET" and norm == "/inspections":
        qs = event.get("queryStringParameters") or {}
        limit = qs.get("limit")
        env = api.list_inspections(
            status=qs.get("status"),
            text=qs.get("text"),
            limit=int(limit) if limit not in (None, "") else None,
        )
        return _response(_status_for(env), env)

    if method == "GET" and norm == "/evaluations":
        return _response(_status_for(env := api.list_evaluations()), env)

    if method == "GET" and path.startswith("/evaluations/"):
        eval_run_id = path[len("/evaluations/"):]
        return _response(_status_for(env := api.get_evaluation(eval_run_id)), env)

    if method == "GET" and path.startswith("/jobs/"):
        job_id = path[len("/jobs/"):]
        job = adapter.get_job(job_id)
        if job is None:
            return _error(404, "not_found", f"no job {job_id!r}")
        return _response(200, {"apiVersion": "v1", "mode": "aws", "data": job.model_dump()})

    return _error(404, "not_found", f"no route for {method} {path}")


def _evidence_response(env: Any) -> dict[str, Any]:
    """Serialize an annotated-evidence envelope, base64-encoding the bytes.

    The frontend's ``AnnotatedEvidence`` type carries an optional
    ``data_base64``; the descriptor returned by the handler holds raw ``data``
    bytes, so convert them here (never emit raw bytes into JSON).
    """
    if not getattr(env, "ok", False):
        return _response(_status_for(env), env)
    ev = env.data
    payload = {
        "apiVersion": env.apiVersion,
        "mode": env.mode,
        "data": {
            "inspection_id": ev.inspection_id,
            "finding_id": ev.finding_id,
            "ref": ev.ref,
            "byte_length": ev.byte_length,
            "data_base64": base64.b64encode(ev.data).decode("ascii"),
        },
    }
    return _response(200, payload)


def _create_inspection(
    event: dict[str, Any], api: LocalApi, adapter: AwsStorageAdapter
) -> dict[str, Any]:
    """Run an inspection and return its report synchronously.

    Accepts the frontend's ``multipart/form-data`` (fields ``media``,
    ``content_type``, ``example_mode``, ``inspection_id``) or a JSON body with
    ``media_base64`` / ``content_type`` / ``example_mode`` (programmatic). A JSON
    caller may set ``async=true`` with a ``video/*`` type to hand the job to the
    worker Lambda and receive a queued JobRecord instead.
    """
    ct_header = _content_type_header(event).lower()
    example_mode = False
    content_type: str | None = None
    media: bytes | None = None
    inspection_id: str | None = None
    want_async = False

    if ct_header.startswith("multipart/form-data"):
        fields = _parse_multipart(_raw_body_bytes(event), ct_header)
        example_mode = str(fields.get("example_mode", "")).lower() == "true"
        content_type = fields.get("content_type") or None
        inspection_id = fields.get("inspection_id") or None
        media_field = fields.get("media")
        if isinstance(media_field, dict):
            media = media_field.get("bytes")
            # Fall back to the uploaded part's own content type if not given.
            content_type = content_type or media_field.get("type")
    else:
        body = _decode_json(event)
        example_mode = bool(body.get("example_mode"))
        content_type = body.get("content_type")
        inspection_id = body.get("inspection_id") or None
        want_async = bool(body.get("async"))
        if body.get("media_base64"):
            try:
                media = base64.b64decode(body["media_base64"])
            except (ValueError, TypeError):
                return _error(400, "validation", "media_base64 is not valid base64", "media_base64")

    is_video = isinstance(content_type, str) and content_type.startswith(_VIDEO_PREFIX)

    # Programmatic opt-in async path for large videos (not used by the browser).
    if want_async and is_video:
        return _enqueue_video_job(media, content_type, adapter, inspection_id)

    # Synchronous path: images, the bundled example, and bounded videos. Returns
    # an InspectionReport envelope, which is exactly what the frontend consumes.
    env = api.create_inspection(
        media=media,
        content_type=content_type,
        example_mode=example_mode,
        inspection_id=inspection_id,
    )
    return _response(_status_for(env), env)


def _enqueue_video_job(
    media: bytes | None,
    content_type: str,
    adapter: AwsStorageAdapter,
    inspection_id: str | None,
) -> dict[str, Any]:
    """Persist a queued JobRecord and invoke the worker Lambda asynchronously."""
    worker_fn = os.environ.get(_ENV_WORKER_FN)
    if not worker_fn:
        return _error(500, "bad_request", f"{_ENV_WORKER_FN} is not configured")
    if not media:
        return _error(400, "validation", "async video requires media_base64 bytes", "media_base64")

    gen_insp, job_id = _new_ids()
    inspection_id = inspection_id or gen_insp
    now = _utc_now_iso()
    job = JobRecord(
        job_id=job_id,
        inspection_id=inspection_id,
        status="queued",
        created_at=now,
        updated_at=now,
        error=None,
    )
    adapter.save_job(job)

    import boto3  # lazy: only when an async handoff is actually needed

    payload = {
        "job_id": job_id,
        "inspection_id": inspection_id,
        "content_type": content_type,
        "media_base64": base64.b64encode(media).decode("ascii"),
    }
    boto3.client("lambda").invoke(
        FunctionName=worker_fn,
        InvocationType="Event",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    return _response(202, {"apiVersion": "v1", "mode": "aws", "data": job.model_dump()})