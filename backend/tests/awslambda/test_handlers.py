"""Transport-layer tests for the AWS Lambda handlers (container-image deploy).

These tests exercise ONLY the new transport/wiring code in
:mod:`shelfsentry.awslambda` - event routing, envelope/status mapping, the
multipart (frontend) and JSON (programmatic) create paths, the opt-in async
video handoff, and the evaluations/evidence routes - against in-memory fakes.
There is NO real AWS call and NO real Bedrock call here. The real OpenCV 5 +
Bedrock path is validated by a live smoke test after deployment (documented in
docs/aws-deployment.md), not mocked into a fake "success" here (Req 16.4).
"""

from __future__ import annotations

import base64
import json

import pytest

from shelfsentry.awslambda import api as api_mod
from shelfsentry.awslambda import worker as worker_mod
from shelfsentry.data.aws_storage import AwsStorageAdapter
from shelfsentry.data.storage import JobRecord


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, *, Bucket, Key):
        return {"Body": _Body(self.objects[(Bucket, Key)])}


class FakeDynamo:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def put_item(self, *, TableName, Item):
        self.items[Item["pk"]["S"]] = Item
        return {}

    def get_item(self, *, TableName, Key):
        item = self.items.get(Key["pk"]["S"])
        return {"Item": item} if item is not None else {}

    def scan(self, **kwargs):
        return {"Items": list(self.items.values())}


class FakeLambdaClient:
    def __init__(self) -> None:
        self.invocations: list[dict] = []

    def invoke(self, *, FunctionName, InvocationType, Payload):
        self.invocations.append(
            {
                "FunctionName": FunctionName,
                "InvocationType": InvocationType,
                "Payload": json.loads(Payload.decode("utf-8")),
            }
        )
        return {"StatusCode": 202}


@pytest.fixture
def aws_env(monkeypatch):
    monkeypatch.setenv("SHELFSENTRY_MODE", "aws")
    monkeypatch.setenv("SHELFSENTRY_S3_BUCKET", "test-bucket")
    monkeypatch.setenv("SHELFSENTRY_DDB_TABLE", "test-table")
    monkeypatch.setenv("SHELFSENTRY_AWS_REGION", "us-east-1")
    monkeypatch.setenv("WORKER_FUNCTION_NAME", "shelfsentry-dev-worker")
    monkeypatch.setenv("SHELFSENTRY_BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")


@pytest.fixture
def fake_adapter():
    return AwsStorageAdapter(
        bucket="test-bucket",
        table="test-table",
        region="us-east-1",
        s3_client=FakeS3(),
        ddb_client=FakeDynamo(),
    )


def _event(method, path, *, body=None, qs=None, headers=None, raw=None, b64=False):
    ev = {
        "requestContext": {"http": {"method": method, "path": path}},
        "rawPath": path,
        "headers": headers or {},
    }
    if qs is not None:
        ev["queryStringParameters"] = qs
    if raw is not None:
        ev["body"] = raw
        ev["isBase64Encoded"] = b64
    elif body is not None:
        ev["body"] = json.dumps(body)
        ev["isBase64Encoded"] = False
    return ev


def _multipart(fields: dict, files: dict | None = None) -> tuple[str, str]:
    """Build a multipart/form-data body like the browser's FormData.

    Returns (content_type_header, body_string). ``fields`` are plain text parts;
    ``files`` maps name -> (filename, bytes, content_type).
    """
    boundary = "----shelfsentrytestboundary"
    parts = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        )
    body = "".join(parts)
    # Files appended as latin-1 so arbitrary bytes survive the str round-trip.
    file_sections = b""
    for name, (filename, data, ctype) in (files or {}).items():
        head = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {ctype}\r\n\r\n"
        ).encode("utf-8")
        file_sections += head + data + b"\r\n"
    tail = f"--{boundary}--\r\n".encode("utf-8")
    full = body.encode("utf-8") + file_sections + tail
    ct = f"multipart/form-data; boundary={boundary}"
    return ct, base64.b64encode(full).decode("ascii")


# -- tests -------------------------------------------------------------------


def test_health_reports_aws_mode_and_model(aws_env):
    resp = api_mod.handler(_event("GET", "/health"))
    assert resp["statusCode"] == 200
    payload = json.loads(resp["body"])
    assert payload["mode"] == "aws"
    assert payload["status"] == "ok"
    assert payload["bedrockModelId"] == "amazon.nova-lite-v1:0"
    # CORS header present for the browser.
    assert resp["headers"]["access-control-allow-origin"] == "*"


def test_options_preflight_is_204(aws_env):
    resp = api_mod.handler(_event("OPTIONS", "/inspections"))
    assert resp["statusCode"] == 204
    assert resp["headers"]["access-control-allow-origin"] == "*"


def test_unknown_route_is_404(aws_env):
    resp = api_mod.handler(_event("GET", "/nope"))
    assert resp["statusCode"] == 404
    assert json.loads(resp["body"])["error"]["code"] == "not_found"


def test_multipart_parser_extracts_fields_and_file():
    ct, b64body = _multipart(
        {"content_type": "image/png", "example_mode": "false"},
        {"media": ("shelf.png", b"\x89PNG\r\n\x1a\nDATA", "image/png")},
    )
    body = base64.b64decode(b64body)
    parsed = api_mod._parse_multipart(body, ct)
    assert parsed["content_type"] == "image/png"
    assert parsed["example_mode"] == "false"
    assert isinstance(parsed["media"], dict)
    assert parsed["media"]["filename"] == "shelf.png"
    assert parsed["media"]["bytes"] == b"\x89PNG\r\n\x1a\nDATA"
    assert parsed["media"]["type"] == "image/png"


def test_multipart_create_passes_bytes_to_runner(aws_env, fake_adapter, monkeypatch):
    """A real browser-style multipart PNG upload must reach the runner as
    NON-EMPTY bytes (regression for the "media is required" server error)."""
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)

    captured = {}

    def fake_runner(data, content_type, *, adapter, inspection_id=None, mode="aws"):
        # Record what the handler forwarded, then return a minimal report-like
        # object the envelope can serialize. We assert on the bytes, not vision.
        captured["len"] = len(data)
        captured["content_type"] = content_type
        from shelfsentry.data.contracts import InspectionReport
        return InspectionReport(
            inspection_id=inspection_id or "insp-x",
            mode="aws",
            final_recommendation="stub",
            requires_human_review=False,
            initial_finding_id="f-1",
            decision_trace=[],
            supporting_evidence_refs=[],
            created_at="2026-01-01T00:00:00+00:00",
            status="complete",
        )

    # Inject the stub runner so no real OpenCV/Bedrock runs in this unit test.
    monkeypatch.setattr(api_mod, "_build_api",
                        lambda adapter: __import__("shelfsentry.api.handlers", fromlist=["LocalApi"]).LocalApi(adapter, runner=fake_runner, mode="aws"))

    png = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4  # binary incl. CR/LF bytes
    ct, b64 = _multipart({"content_type": "image/png"}, {"media": ("shelf.png", png, "image/png")})
    resp = api_mod.handler(_event("POST", "/inspections", raw=b64, b64=True,
                                  headers={"content-type": ct}))
    assert resp["statusCode"] == 200, resp["body"]
    # The exact bytes uploaded must have reached the runner (non-empty).
    assert captured["len"] == len(png)
    assert captured["content_type"] == "image/png"


def test_get_missing_job_is_404(aws_env, fake_adapter, monkeypatch):
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)
    resp = api_mod.handler(_event("GET", "/jobs/job-does-not-exist"))
    assert resp["statusCode"] == 404


def test_get_existing_job_returns_record(aws_env, fake_adapter, monkeypatch):
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)
    fake_adapter.save_job(
        JobRecord(
            job_id="job-1",
            inspection_id="insp-1",
            status="running",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:01+00:00",
            error=None,
        )
    )
    resp = api_mod.handler(_event("GET", "/jobs/job-1"))
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["data"]["status"] == "running"


def test_get_missing_inspection_is_404(aws_env, fake_adapter, monkeypatch):
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)
    resp = api_mod.handler(_event("GET", "/inspections/insp-missing"))
    assert resp["statusCode"] == 404
    assert json.loads(resp["body"])["error"]["code"] == "not_found"


def test_list_inspections_empty_ok(aws_env, fake_adapter, monkeypatch):
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)
    resp = api_mod.handler(_event("GET", "/inspections"))
    assert resp["statusCode"] == 200
    assert json.loads(resp["body"])["data"] == []


def test_list_evaluations_is_empty_not_fabricated(aws_env, fake_adapter, monkeypatch):
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)
    resp = api_mod.handler(_event("GET", "/evaluations"))
    assert resp["statusCode"] == 200
    # No eval store wired -> honest empty list, never a fabricated record.
    assert json.loads(resp["body"])["data"] == []


def test_evidence_route_on_aws_adapter_is_not_found(aws_env, fake_adapter, monkeypatch):
    # The AWS adapter does not back the evidence resolver; the handler returns a
    # structured not_found rather than fabricating bytes (honest limitation).
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)
    resp = api_mod.handler(
        _event("GET", "/inspections/insp-1/evidence/find-1")
    )
    assert resp["statusCode"] == 404
    assert json.loads(resp["body"])["error"]["code"] == "not_found"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_async_video_opt_in_enqueues_job(aws_env, fake_adapter, monkeypatch):
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)
    fake_lambda = FakeLambdaClient()
    import boto3

    monkeypatch.setattr(boto3, "client", lambda name, *a, **k: fake_lambda)

    media_b64 = base64.b64encode(b"fake-mp4-bytes").decode("ascii")
    resp = api_mod.handler(
        _event(
            "POST",
            "/inspections",
            body={"content_type": "video/mp4", "media_base64": media_b64, "async": True},
        )
    )
    assert resp["statusCode"] == 202
    data = json.loads(resp["body"])["data"]
    assert data["status"] == "queued"
    assert len(fake_lambda.invocations) == 1
    inv = fake_lambda.invocations[0]
    assert inv["InvocationType"] == "Event"
    assert inv["Payload"]["job_id"] == data["job_id"]
    assert inv["Payload"]["content_type"] == "video/mp4"


def test_async_video_without_bytes_is_validation_error(aws_env, fake_adapter, monkeypatch):
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)
    resp = api_mod.handler(
        _event("POST", "/inspections", body={"content_type": "video/mp4", "async": True})
    )
    assert resp["statusCode"] == 400
    assert json.loads(resp["body"])["error"]["code"] == "validation"


def test_bad_base64_is_validation_error(aws_env, fake_adapter, monkeypatch):
    monkeypatch.setattr(api_mod, "_build_adapter", lambda: fake_adapter)
    # A '%' is outside the base64 alphabet and makes b64decode raise.
    resp = api_mod.handler(
        _event(
            "POST",
            "/inspections",
            body={"content_type": "image/png", "media_base64": "%%%%"},
        )
    )
    assert resp["statusCode"] == 400


def test_worker_rejects_malformed_event(aws_env, fake_adapter, monkeypatch):
    monkeypatch.setattr(worker_mod, "_build_adapter", lambda: fake_adapter)
    with pytest.raises(ValueError):
        worker_mod.handler({"job_id": "x"})


def test_api_requires_aws_mode(monkeypatch):
    monkeypatch.delenv("SHELFSENTRY_MODE", raising=False)
    with pytest.raises(RuntimeError):
        api_mod.handler(_event("GET", "/inspections"))