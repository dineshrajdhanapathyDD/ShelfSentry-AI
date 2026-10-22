"""Async worker Lambda handler (container image; OpenCV 5 + Bedrock).

Invoked asynchronously by the API Lambda (``InvocationType=Event``) for longer
video jobs. It adapts the invoke event to the existing
:func:`shelfsentry.aws_worker.run_async_job`, which records the job's
``queued -> running -> succeeded/failed`` transitions in DynamoDB for polling
(Req 19.4, 19.5) and runs the REAL OpenCV 5 vision + bounded agent loop with
Bedrock as the primary decision path (Req 16.2, 16.4). No logic lives here
beyond decoding the event and building the AWS adapter.

Event payload (JSON) produced by the API Lambda::

    {"job_id": "...", "inspection_id": "...",
     "content_type": "video/mp4", "media_base64": "<base64 bytes>"}

The worker never returns a mock result: a failure (including exhausted retries)
is recorded as a ``failed`` JobRecord carrying the error, never swallowed.
"""

from __future__ import annotations

import base64
import os
from typing import Any

from shelfsentry.aws_worker import run_async_job
from shelfsentry.config.mode import resolve_runtime_config
from shelfsentry.data.aws_storage import AwsStorageAdapter

_ENV_KEY_PREFIX = "SHELFSENTRY_KEY_PREFIX"


def _build_adapter() -> AwsStorageAdapter:
    cfg = resolve_runtime_config()
    if not cfg.is_aws or cfg.aws is None:
        raise RuntimeError(
            "Worker Lambda requires SHELFSENTRY_MODE=aws with S3/DDB/region set."
        )
    return AwsStorageAdapter(
        bucket=cfg.aws.s3_bucket,
        table=cfg.aws.ddb_table,
        region=cfg.aws.region,
        key_prefix=os.environ.get(_ENV_KEY_PREFIX, ""),
    )


def handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Async worker entry point (Req 19.4, 19.5)."""
    job_id = event.get("job_id")
    inspection_id = event.get("inspection_id")
    content_type = event.get("content_type")
    media_b64 = event.get("media_base64")

    if not (job_id and inspection_id and content_type and media_b64):
        # A malformed invoke cannot be recorded against a job; fail loudly so
        # the async retry/DLQ machinery surfaces it (never a silent success).
        raise ValueError(
            "worker event requires job_id, inspection_id, content_type, media_base64"
        )

    data = base64.b64decode(media_b64)
    adapter = _build_adapter()

    job = run_async_job(
        job_id,
        inspection_id,
        data,
        content_type,
        adapter=adapter,
    )
    return {"job_id": job.job_id, "status": job.status, "error": job.error}
