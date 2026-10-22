"""Tests for the AWS persistence layer (task 16.1; Req 16.1, 18.3, 18.4, 9).

Everything runs against in-memory FAKE S3 and DynamoDB clients - there is NO
real AWS call anywhere in this module. The fakes implement only the methods the
adapter uses:

* S3: ``put_object`` / ``get_object`` / ``generate_presigned_url``,
* DynamoDB: ``put_item`` / ``get_item`` / ``scan``.

Covers:

* ``put_media`` / ``get_media`` round-trip via fake S3,
* ``save_report`` / ``get_report`` round-trip via fake DynamoDB,
* ``list_reports`` status filter + text search + limit via fake scan,
* ``save_job`` / ``get_job`` round-trip,
* presigned-upload minting calls ``generate_presigned_url`` with the scoped
  bucket/key and returns the URL (Req 18.3),
* a crafted / foreign media ref is rejected and cannot read outside the
  configured bucket/prefix (Req 9),
* :class:`AwsStorageAdapter` satisfies the ``StorageAdapter`` Protocol,
* the existing ``StorageMediaResolver`` shim works against the AWS adapter,
* importing :mod:`shelfsentry.data` does NOT import boto3.
"""

from __future__ import annotations

import sys

import pytest

from shelfsentry.data import (
    AwsStorageAdapter,
    HistoryQuery,
    InspectionReport,
    InspectionSummary,
    InvalidMediaRef,
    JobRecord,
    RecordNotFound,
    StorageAdapter,
    StorageMediaResolver,
)
from shelfsentry.mcp import MediaResolver


# ---------------------------------------------------------------------------
# In-memory fakes (no real AWS, no boto3)
# ---------------------------------------------------------------------------


class _S3NotFound(Exception):
    """Mimics a botocore NoSuchKey error shape for the fake S3 client."""

    def __init__(self) -> None:
        super().__init__("NoSuchKey")
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class _Body:
    """Minimal S3 streaming-body stand-in with a ``read()`` method."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3:
    """In-memory S3: stores objects in a dict keyed by (bucket, key)."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.presign_calls: list[dict] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict:
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        try:
            data = self.objects[(Bucket, Key)]
        except KeyError:
            raise _S3NotFound()
        return {"Body": _Body(data)}

    def generate_presigned_url(self, operation: str, *, Params: dict, ExpiresIn: int) -> str:
        self.presign_calls.append(
            {"operation": operation, "Params": dict(Params), "ExpiresIn": ExpiresIn}
        )
        bucket = Params["Bucket"]
        key = Params["Key"]
        return f"https://{bucket}.s3.amazonaws.com/{key}?X-Amz-Signature=fake&expires={ExpiresIn}"


class FakeDynamo:
    """In-memory DynamoDB: stores items in a dict keyed by the partition key.

    Supports the attribute-value shapes the adapter uses: string (``S``) and a
    string-valued map (``M``). ``scan`` honors the adapter's record_type filter.
    """

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def put_item(self, *, TableName: str, Item: dict) -> dict:
        pk = Item["pk"]["S"]
        self.items[pk] = Item
        return {}

    def get_item(self, *, TableName: str, Key: dict) -> dict:
        pk = Key["pk"]["S"]
        item = self.items.get(pk)
        return {"Item": item} if item is not None else {}

    def scan(
        self,
        *,
        TableName: str,
        FilterExpression: str | None = None,
        ExpressionAttributeNames: dict | None = None,
        ExpressionAttributeValues: dict | None = None,
        ExclusiveStartKey: dict | None = None,
    ) -> dict:
        wanted = None
        if ExpressionAttributeValues and ":rt" in ExpressionAttributeValues:
            wanted = ExpressionAttributeValues[":rt"]["S"]
        items = []
        for item in self.items.values():
            if wanted is None or item.get("record_type", {}).get("S") == wanted:
                items.append(item)
        return {"Items": items}


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_s3() -> FakeS3:
    return FakeS3()


@pytest.fixture()
def fake_ddb() -> FakeDynamo:
    return FakeDynamo()


@pytest.fixture()
def adapter(fake_s3: FakeS3, fake_ddb: FakeDynamo) -> AwsStorageAdapter:
    return AwsStorageAdapter(
        bucket="test-bucket",
        table="test-table",
        region="us-east-1",
        key_prefix="shelfsentry/",
        s3_client=fake_s3,
        ddb_client=fake_ddb,
    )


def _report(
    inspection_id: str,
    *,
    status: str = "complete",
    recommendation: str = "Shelf looks adequately stocked.",
    created_at: str = "2024-01-01T00:00:00Z",
    requires_human_review: bool = False,
    mode: str = "aws",
) -> InspectionReport:
    return InspectionReport(
        inspection_id=inspection_id,
        mode=mode,  # type: ignore[arg-type]
        final_recommendation=recommendation,
        requires_human_review=requires_human_review,
        initial_finding_id=f"{inspection_id}-image",
        decision_trace=[],
        supporting_evidence_refs=[],
        created_at=created_at,
        status=status,  # type: ignore[arg-type]
    )


def _job(job_id: str = "job-1", *, status: str = "queued") -> JobRecord:
    return JobRecord(
        job_id=job_id,
        inspection_id="insp-1",
        status=status,  # type: ignore[arg-type]
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:01Z",
        error=None,
    )


# ---------------------------------------------------------------------------
# No-boto3-on-import guarantee
# ---------------------------------------------------------------------------


def test_importing_data_does_not_import_boto3():
    # Assert the import-time property in a FRESH interpreter so this process's
    # already-loaded modules (and any test that imported boto3) cannot mask it,
    # and so we never mutate this process's sys.modules.
    import subprocess

    code = (
        "import sys; import shelfsentry.data; "
        "assert 'boto3' not in sys.modules, 'shelfsentry.data must not import boto3'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_aws_adapter_satisfies_storage_adapter_protocol(adapter: AwsStorageAdapter):
    assert isinstance(adapter, StorageAdapter)


def test_settings_constructor_path():
    from shelfsentry.config import AwsSettings

    settings = AwsSettings(s3_bucket="b", ddb_table="t", region="eu-west-1")
    a = AwsStorageAdapter(settings, s3_client=FakeS3(), ddb_client=FakeDynamo())
    assert isinstance(a, StorageAdapter)


# ---------------------------------------------------------------------------
# Media round-trip via fake S3
# ---------------------------------------------------------------------------


def test_put_get_media_round_trip(adapter: AwsStorageAdapter, fake_s3: FakeS3):
    data = b"\x89PNG\r\n\x1a\n media bytes \x00\x01\x02"
    ref = adapter.put_media("insp-1", data, "image")
    assert ref.startswith("media://")
    assert adapter.get_media(ref) == data

    # The object actually landed under the scoped media prefix in the bucket.
    (bucket, key), stored = next(iter(fake_s3.objects.items()))
    assert bucket == "test-bucket"
    assert key.startswith("shelfsentry/media/")
    assert key.endswith(".bin")
    assert stored == data


def test_put_media_mints_distinct_refs(adapter: AwsStorageAdapter):
    ref_a = adapter.put_media("insp-1", b"a", "image")
    ref_b = adapter.put_media("insp-1", b"b", "frame:1")
    assert ref_a != ref_b
    assert adapter.get_media(ref_a) == b"a"
    assert adapter.get_media(ref_b) == b"b"


def test_get_missing_media_raises_record_not_found(adapter: AwsStorageAdapter):
    good_token = "0" * 32
    with pytest.raises(RecordNotFound):
        adapter.get_media(f"media://{good_token}")


# ---------------------------------------------------------------------------
# Report round-trip via fake DynamoDB
# ---------------------------------------------------------------------------


def test_save_get_report_round_trip(adapter: AwsStorageAdapter, fake_ddb: FakeDynamo):
    report = _report("insp-42", recommendation="Two gaps flagged for review.")
    adapter.save_report(report)

    # Stored as a DynamoDB item carrying the canonical JSON.
    assert any(i.get("record_type", {}).get("S") == "report" for i in fake_ddb.items.values())

    loaded = adapter.get_report("insp-42")
    assert loaded is not None
    assert loaded == report
    assert loaded.canonical_json() == report.canonical_json()


def test_get_missing_report_returns_none(adapter: AwsStorageAdapter):
    assert adapter.get_report("does-not-exist") is None


def test_save_report_overwrites(adapter: AwsStorageAdapter):
    adapter.save_report(_report("insp-1", status="needs_review"))
    adapter.save_report(_report("insp-1", status="complete"))
    loaded = adapter.get_report("insp-1")
    assert loaded is not None
    assert loaded.status == "complete"


# ---------------------------------------------------------------------------
# list_reports: status filter + text search + limit via fake scan
# ---------------------------------------------------------------------------


def test_list_reports_empty_store(adapter: AwsStorageAdapter):
    assert adapter.list_reports(HistoryQuery()) == []


def test_list_reports_status_filter(adapter: AwsStorageAdapter):
    adapter.save_report(_report("a", status="complete", created_at="2024-01-01T00:00:00Z"))
    adapter.save_report(_report("b", status="needs_review", created_at="2024-01-02T00:00:00Z"))
    adapter.save_report(_report("c", status="failed", created_at="2024-01-03T00:00:00Z"))

    needs = adapter.list_reports(HistoryQuery(status="needs_review"))
    assert [s.inspection_id for s in needs] == ["b"]
    assert all(isinstance(s, InspectionSummary) for s in needs)


def test_list_reports_text_search_matches_id_and_recommendation(adapter: AwsStorageAdapter):
    adapter.save_report(
        _report("alpha-1", recommendation="Restock the cereal aisle.", created_at="2024-01-01T00:00:00Z")
    )
    adapter.save_report(
        _report("beta-2", recommendation="No action needed.", created_at="2024-01-02T00:00:00Z")
    )

    by_text = adapter.list_reports(HistoryQuery(text="CEREAL"))
    assert [s.inspection_id for s in by_text] == ["alpha-1"]

    by_id = adapter.list_reports(HistoryQuery(text="beta"))
    assert [s.inspection_id for s in by_id] == ["beta-2"]


def test_list_reports_limit_and_newest_first(adapter: AwsStorageAdapter):
    adapter.save_report(_report("old", created_at="2024-01-01T00:00:00Z"))
    adapter.save_report(_report("mid", created_at="2024-02-01T00:00:00Z"))
    adapter.save_report(_report("new", created_at="2024-03-01T00:00:00Z"))

    limited = adapter.list_reports(HistoryQuery(limit=2))
    assert [s.inspection_id for s in limited] == ["new", "mid"]


def test_list_reports_combined_filter_search_limit(adapter: AwsStorageAdapter):
    adapter.save_report(
        _report("x1", status="complete", recommendation="gap on shelf", created_at="2024-01-01T00:00:00Z")
    )
    adapter.save_report(
        _report("x2", status="complete", recommendation="gap detected", created_at="2024-02-01T00:00:00Z")
    )
    adapter.save_report(
        _report("x3", status="failed", recommendation="gap error", created_at="2024-03-01T00:00:00Z")
    )

    result = adapter.list_reports(HistoryQuery(status="complete", text="gap", limit=1))
    assert [s.inspection_id for s in result] == ["x2"]


def test_list_reports_ignores_non_report_items(adapter: AwsStorageAdapter):
    # Media manifests and jobs share the table but must not appear in History.
    adapter.put_media("insp-1", b"x", "image")
    adapter.save_job(_job("j1"))
    adapter.save_report(_report("only-report"))

    rows = adapter.list_reports(HistoryQuery())
    assert [s.inspection_id for s in rows] == ["only-report"]


# ---------------------------------------------------------------------------
# Job round-trip via fake DynamoDB
# ---------------------------------------------------------------------------


def test_save_get_job_round_trip(adapter: AwsStorageAdapter):
    job = _job("job-abc", status="running")
    adapter.save_job(job)
    loaded = adapter.get_job("job-abc")
    assert loaded is not None
    assert loaded == job


def test_get_missing_job_returns_none(adapter: AwsStorageAdapter):
    assert adapter.get_job("no-such-job") is None


def test_save_job_overwrites_status(adapter: AwsStorageAdapter):
    adapter.save_job(_job("j", status="queued"))
    adapter.save_job(_job("j", status="succeeded"))
    loaded = adapter.get_job("j")
    assert loaded is not None
    assert loaded.status == "succeeded"


# ---------------------------------------------------------------------------
# Presigned upload minting (Req 18.3)
# ---------------------------------------------------------------------------


def test_mint_presigned_upload_scoped_bucket_key_and_returns_url(
    adapter: AwsStorageAdapter, fake_s3: FakeS3
):
    result = adapter.mint_presigned_upload(
        "insp-99", "image", content_type="image/png", expires_in=600
    )
    assert set(result) == {"ref", "url"}
    assert result["ref"].startswith("media://")
    assert result["url"].startswith("https://")

    # generate_presigned_url was called exactly once, for a put_object into the
    # configured bucket under the scoped media prefix, with the given expiry.
    assert len(fake_s3.presign_calls) == 1
    call = fake_s3.presign_calls[0]
    assert call["operation"] == "put_object"
    assert call["Params"]["Bucket"] == "test-bucket"
    assert call["Params"]["Key"].startswith("shelfsentry/media/")
    assert call["Params"]["Key"].endswith(".bin")
    assert call["Params"]["ContentType"] == "image/png"
    assert call["ExpiresIn"] == 600


def test_presigned_ref_resolves_after_client_uploads(
    adapter: AwsStorageAdapter, fake_s3: FakeS3
):
    result = adapter.mint_presigned_upload("insp-77", "image")
    ref = result["ref"]
    # The slot is recorded immediately, before the upload lands.
    assert adapter.get_media_ref_for("insp-77", "image") == ref

    # Simulate the client PUTting bytes to the presigned key.
    key = fake_s3.presign_calls[0]["Params"]["Key"]
    fake_s3.put_object(Bucket="test-bucket", Key=key, Body=b"uploaded-bytes")

    assert adapter.get_media(ref) == b"uploaded-bytes"


def test_presigned_url_uses_default_expiry(adapter: AwsStorageAdapter, fake_s3: FakeS3):
    adapter.mint_presigned_upload("insp-1", "image")
    assert fake_s3.presign_calls[0]["ExpiresIn"] == 900  # DEFAULT_PRESIGN_EXPIRY_SECONDS


# ---------------------------------------------------------------------------
# Security boundary: crafted / foreign media refs are rejected (Req 9)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crafted",
    [
        "media://../../../../etc/passwd",
        "media://..\\..\\windows\\system32\\config\\sam",
        "media:///etc/passwd",
        "media://C:\\Windows\\System32\\drivers\\etc\\hosts",
        "media://%2e%2e%2f%2e%2e%2fsecret",
        "media://deadbeef\x00.bin",
        "media://DEADBEEFDEADBEEFDEADBEEFDEADBEEF",  # uppercase hex not allowed
        "media://short",
        "media://" + "f" * 31,  # wrong length
        "s3://other-bucket/some/key",  # foreign S3 URL
        "s3://test-bucket/shelfsentry/media/" + "a" * 32 + ".bin",  # even our own bucket as URL
        "file:///etc/passwd",
        "/etc/passwd",
        "../../etc/passwd",
        "shelfsentry/media/" + "a" * 32 + ".bin",  # bare key, no scheme
    ],
)
def test_crafted_or_foreign_media_ref_is_rejected(adapter: AwsStorageAdapter, crafted: str):
    with pytest.raises(InvalidMediaRef):
        adapter.get_media(crafted)


def test_crafted_ref_never_reaches_s3(adapter: AwsStorageAdapter, fake_s3: FakeS3):
    # Plant an object in a FOREIGN bucket and under a sibling prefix in ours.
    fake_s3.objects[("other-bucket", "secret")] = b"TOP SECRET"
    fake_s3.objects[("test-bucket", "secret")] = b"OURS BUT OUTSIDE PREFIX"

    for crafted in (
        "s3://other-bucket/secret",
        "media://../secret",
        "media://../../secret",
        "media:///secret",
    ):
        with pytest.raises(InvalidMediaRef):
            adapter.get_media(crafted)

    # Neither foreign object was surfaced via the adapter.
    assert fake_s3.objects[("other-bucket", "secret")] == b"TOP SECRET"
    assert fake_s3.objects[("test-bucket", "secret")] == b"OURS BUT OUTSIDE PREFIX"


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "..\\escape", "a/b", "a\\b", "with:colon", "", "with\x00nul"],
)
def test_crafted_inspection_id_for_report_is_rejected(adapter: AwsStorageAdapter, bad_id: str):
    report = _report("placeholder")
    object.__setattr__(report, "inspection_id", bad_id)
    with pytest.raises(InvalidMediaRef):
        adapter.save_report(report)


def test_crafted_job_id_is_rejected(adapter: AwsStorageAdapter):
    job = _job("placeholder")
    object.__setattr__(job, "job_id", "../../evil")
    with pytest.raises(InvalidMediaRef):
        adapter.save_job(job)


# ---------------------------------------------------------------------------
# StorageMediaResolver shim works against the AWS adapter
# ---------------------------------------------------------------------------


def test_resolver_shim_satisfies_media_resolver_protocol(adapter: AwsStorageAdapter):
    resolver = StorageMediaResolver(adapter)  # type: ignore[arg-type]
    assert isinstance(resolver, MediaResolver)


def test_resolver_image_round_trip(adapter: AwsStorageAdapter):
    resolver = StorageMediaResolver(adapter)  # type: ignore[arg-type]
    resolver.put_image("insp-1", b"image-bytes")
    assert resolver.get_image("insp-1") == b"image-bytes"


def test_resolver_frame_round_trip(adapter: AwsStorageAdapter):
    resolver = StorageMediaResolver(adapter)  # type: ignore[arg-type]
    resolver.put_frame("insp-1", "f0", b"frame-zero")
    resolver.put_frame("insp-1", "f1", b"frame-one")
    assert resolver.get_frame("insp-1", "f0") == b"frame-zero"
    assert resolver.get_frame("insp-1", "f1") == b"frame-one"


def test_resolver_evidence_round_trip(adapter: AwsStorageAdapter):
    resolver = StorageMediaResolver(adapter)  # type: ignore[arg-type]
    assert resolver.get_evidence_ref("insp-1", "finding-1") is None
    ref = resolver.put_evidence("insp-1", "finding-1", b"annotated-png")
    assert ref.startswith("media://")
    assert resolver.get_evidence_ref("insp-1", "finding-1") == ref
    assert adapter.get_media(ref) == b"annotated-png"


def test_resolver_missing_image_raises(adapter: AwsStorageAdapter):
    resolver = StorageMediaResolver(adapter)  # type: ignore[arg-type]
    with pytest.raises(RecordNotFound):
        resolver.get_image("unknown")


def test_resolver_media_is_inspection_scoped(adapter: AwsStorageAdapter):
    resolver = StorageMediaResolver(adapter)  # type: ignore[arg-type]
    resolver.put_image("insp-a", b"a-img")
    resolver.put_image("insp-b", b"b-img")
    assert resolver.get_image("insp-a") == b"a-img"
    assert resolver.get_image("insp-b") == b"b-img"


# ---------------------------------------------------------------------------
# Lazy boto3: missing clients + no boto3 raises a clear error only on use
# ---------------------------------------------------------------------------


def test_no_injected_clients_and_no_boto3_raises_clear_error_on_use(monkeypatch):
    # Import the adapter + error class from the live module (robust to any
    # earlier test that reloaded shelfsentry.data and changed class identity).
    from shelfsentry.data.aws_storage import AwsClientUnavailable, AwsStorageAdapter

    a = AwsStorageAdapter(bucket="b", table="t", region="us-east-1")

    # Force the lazy boto3 import to fail, simulating an env without boto3.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("No module named 'boto3'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # Construction already happened without error; only USE triggers the import.
    with pytest.raises(AwsClientUnavailable):
        a.get_report("anything")


def test_eval_round_trip_and_list(adapter: AwsStorageAdapter) -> None:
    """save_eval persists a measured EvaluationResult; get_eval/list_evals read
    it back verbatim. Nothing is fabricated: an unsaved id returns None and an
    empty table lists nothing."""
    from shelfsentry.data.contracts import EvaluationResult

    # Empty to start - honest empty state (no fabrication).
    assert adapter.list_evals() == []
    assert adapter.get_eval("eval-missing") is None

    rec = EvaluationResult(
        eval_run_id="eval-abc123",
        is_demo=True,
        sample_count=8,
        false_positives=[],
        false_negatives=[],
        agent_decision_correctness=1.0,
        latency_ms_p50=12.5,
        latency_ms_p95=40.0,
        documented_limitations=["synthetic labels"],
    )
    adapter.save_eval(rec)

    got = adapter.get_eval("eval-abc123")
    assert got is not None
    assert got.eval_run_id == "eval-abc123"
    assert got.sample_count == 8
    assert got.agent_decision_correctness == 1.0
    assert got.measured_only is True

    listed = adapter.list_evals()
    assert len(listed) == 1
    assert listed[0].eval_run_id == "eval-abc123"
