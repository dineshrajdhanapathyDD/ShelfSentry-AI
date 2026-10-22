"""AWS-mode runner + async worker tests (task 16.2; Req 16.2-16.4, 19.4, 19.5).

Everything runs against the REAL
:class:`~shelfsentry.data.aws_storage.AwsStorageAdapter` wired to in-memory FAKE
S3 and DynamoDB clients - there is NO real AWS call and NO real Bedrock call
anywhere in this module. The decision model is injected: an
``InjectedDecisionModel`` simulates "Bedrock available", and a raising model
forces the loop's built-in recorded fallback. The retry tests use an injected
fake sleeper plus a call-counting fake so attempts are asserted without sleeping.

Covered:

* a full AWS-mode inspection persists a report with ``mode="aws"`` and a
  populated decision trace, with real OpenCV 5 evidence stored in (fake) S3
  (Req 16.2, 16.4);
* a raising PRIMARY model yields ``used_fallback=True`` in the trace - a real
  deterministic decision, never a mock result (Req 16.2, 7.2, 16.4);
* the async worker transitions a job ``queued`` -> ``running`` -> ``succeeded``
  on success and ``-> failed`` on error, persisting each transition so
  ``get_job`` can poll it (Req 19.4, 19.5);
* a transient AWS failure that recovers within the budget eventually succeeds,
  and one that never recovers surfaces a failed job after the configured
  attempts (Req 16.3).
"""

from __future__ import annotations

import pytest

from shelfsentry.agent.decision import InjectedDecisionModel
from shelfsentry.agent.retry import RetryExhausted, RetryPolicy
from shelfsentry.aws_worker import run_async_job, run_aws_inspection
from shelfsentry.config.settings import AgentConfig, VisionConfig
from shelfsentry.data.aws_storage import AwsStorageAdapter
from shelfsentry.data.contracts import AgentDecision, InspectionReport
from shelfsentry.samples import ensure_samples


# ---------------------------------------------------------------------------
# In-memory fakes (no real AWS, no boto3) - mirrors tests/data/test_aws_storage
# ---------------------------------------------------------------------------


class _S3NotFound(Exception):
    def __init__(self) -> None:
        super().__init__("NoSuchKey")
        self.response = {
            "Error": {"Code": "NoSuchKey"},
            "ResponseMetadata": {"HTTPStatusCode": 404},
        }


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3:
    """In-memory S3 keyed by (bucket, key)."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict:
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        try:
            data = self.objects[(Bucket, Key)]
        except KeyError:
            raise _S3NotFound()
        return {"Body": _Body(data)}

    def generate_presigned_url(self, operation, *, Params, ExpiresIn) -> str:  # pragma: no cover - unused here
        return "https://example.invalid/presigned"


class FakeDynamo:
    """In-memory DynamoDB keyed by the partition key; honors the record_type scan filter."""

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}
        self.put_item_calls = 0

    def put_item(self, *, TableName: str, Item: dict) -> dict:
        self.put_item_calls += 1
        self.items[Item["pk"]["S"]] = Item
        return {}

    def get_item(self, *, TableName: str, Key: dict) -> dict:
        item = self.items.get(Key["pk"]["S"])
        return {"Item": item} if item is not None else {}

    def scan(self, *, TableName, FilterExpression=None, ExpressionAttributeNames=None,
             ExpressionAttributeValues=None, ExclusiveStartKey=None) -> dict:
        wanted = None
        if ExpressionAttributeValues and ":rt" in ExpressionAttributeValues:
            wanted = ExpressionAttributeValues[":rt"]["S"]
        items = [
            i for i in self.items.values()
            if wanted is None or i.get("record_type", {}).get("S") == wanted
        ]
        return {"Items": items}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def fake_ddb() -> FakeDynamo:
    return FakeDynamo()


@pytest.fixture
def adapter(fake_s3: FakeS3, fake_ddb: FakeDynamo) -> AwsStorageAdapter:
    return AwsStorageAdapter(
        bucket="test-bucket",
        table="test-table",
        region="us-east-1",
        key_prefix="shelfsentry/",
        s3_client=fake_s3,
        ddb_client=fake_ddb,
    )


@pytest.fixture
def sample_png() -> bytes:
    data = ensure_samples().read_bytes()
    assert data
    return data


def _no_sleep() -> RetryPolicy:
    """A retry policy whose injected sleeper never actually sleeps."""
    return RetryPolicy(max_attempts=3, base_delay=0.0, sleep=lambda _d: None)


class _RaisingModel:
    """A PRIMARY decision model that always errors (simulates a Bedrock outage)."""

    def decide(self, finding, trace, cfg):  # noqa: ANN001 - test double
        raise RuntimeError("bedrock unavailable (simulated)")


# ---------------------------------------------------------------------------
# Full AWS-mode inspection: persisted report, mode="aws", real evidence
# ---------------------------------------------------------------------------


def test_aws_inspection_persists_report_mode_aws_with_trace(
    adapter: AwsStorageAdapter, fake_s3: FakeS3, sample_png: bytes
):
    # Simulate "Bedrock available": an injected model finalizes immediately with
    # a real, bedrock-attributed decision (no live Bedrock call).
    model = InjectedDecisionModel(
        decisions=[
            AgentDecision(
                decision_id="bedrock-0",
                iteration=0,
                action="finalize_report",
                rationale="evidence sufficient",
                decided_by="bedrock",
            )
        ]
    )

    report = run_aws_inspection(
        sample_png,
        "image/png",
        inspection_id="aws-insp-001",
        adapter=adapter,
        model=model,
        retry_policy=_no_sleep(),
    )

    # Report persisted, stamped mode="aws" (Req 16.2), never a mock (Req 16.4).
    assert isinstance(report, InspectionReport)
    assert report.mode == "aws"
    assert report.inspection_id == "aws-insp-001"

    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert len(decisions) >= 1
    assert decisions[0].decided_by == "bedrock"
    assert decisions[0].used_fallback is False

    # Retrievable from DynamoDB round-trip (byte-for-byte canonical JSON).
    fetched = adapter.get_report("aws-insp-001")
    assert fetched is not None
    assert fetched.canonical_json() == report.canonical_json()

    # Real OpenCV 5 evidence was rendered and stored under the scoped S3 prefix.
    image_ref = adapter.get_media_ref_for("aws-insp-001", "image")
    assert image_ref is not None
    assert adapter.get_media(image_ref) == sample_png
    evidence_ref = adapter.get_media_ref_for(
        "aws-insp-001", f"evidence:{report.initial_finding_id}"
    )
    assert evidence_ref is not None
    assert adapter.get_media(evidence_ref)  # non-empty annotated bytes
    assert all(k.startswith("shelfsentry/media/") for (_b, k) in fake_s3.objects)


def test_aws_inspection_raising_primary_model_records_fallback_not_mock(
    adapter: AwsStorageAdapter, sample_png: bytes
):
    """A Bedrock error -> the loop's deterministic fallback, recorded in the trace.

    The result is a real finding-derived decision with ``used_fallback=True`` and
    a reason, NOT a substituted mock result (Req 16.2, 7.2, 16.4).
    """
    report = run_aws_inspection(
        sample_png,
        "image/png",
        inspection_id="aws-fallback-001",
        adapter=adapter,
        model=_RaisingModel(),
        retry_policy=_no_sleep(),
    )

    assert report.mode == "aws"
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert decisions, "must record at least one decision"
    first = decisions[0]
    assert first.used_fallback is True
    assert first.decided_by == "fallback"
    assert first.fallback_reason and "RuntimeError" in first.fallback_reason
    # Still persisted as a real AWS-mode report, retrievable for polling.
    assert adapter.get_report("aws-fallback-001") is not None


def test_aws_inspection_validates_input_before_any_aws_write(
    adapter: AwsStorageAdapter, fake_s3: FakeS3, fake_ddb: FakeDynamo
):
    """Invalid input is rejected up front; no media/report is written (Req 1)."""
    from shelfsentry.vision.errors import InputValidationError

    with pytest.raises(InputValidationError):
        run_aws_inspection(
            b"",  # empty file -> structured validation error
            "image/png",
            inspection_id="aws-bad-001",
            adapter=adapter,
            model=InjectedDecisionModel(decisions=[]),
            retry_policy=_no_sleep(),
        )
    assert fake_s3.objects == {}
    assert fake_ddb.items == {}


# ---------------------------------------------------------------------------
# Async worker: job lifecycle transitions persisted for polling (Req 19.4/19.5)
# ---------------------------------------------------------------------------


def _clock():
    """A monotonic fake ISO clock so updated_at advances deterministically."""
    counter = {"n": 0}

    def _now() -> str:
        counter["n"] += 1
        return f"2024-01-01T00:00:{counter['n']:02d}Z"

    return _now


def test_async_job_succeeds_persists_queued_running_succeeded(
    adapter: AwsStorageAdapter, sample_png: bytes
):
    saved_statuses: list[str] = []
    original_save = adapter.save_job

    def _tracking_save(job):  # noqa: ANN001
        saved_statuses.append(job.status)
        original_save(job)

    adapter.save_job = _tracking_save  # type: ignore[method-assign]

    model = InjectedDecisionModel(
        decisions=[
            AgentDecision(
                decision_id="bedrock-0",
                iteration=0,
                action="finalize_report",
                rationale="done",
                decided_by="bedrock",
            )
        ]
    )

    job = run_async_job(
        "job-ok-1",
        "aws-job-insp-1",
        sample_png,
        "image/png",
        adapter=adapter,
        model=model,
        retry_policy=_no_sleep(),
        now=_clock(),
    )

    # Each transition was persisted in order (Req 19.5).
    assert saved_statuses == ["queued", "running", "succeeded"]
    assert job.status == "succeeded"
    assert job.error is None

    # Pollable: the terminal record is readable via get_job.
    polled = adapter.get_job("job-ok-1")
    assert polled is not None
    assert polled.status == "succeeded"
    assert polled.inspection_id == "aws-job-insp-1"
    # The inspection report itself was produced and persisted.
    assert adapter.get_report("aws-job-insp-1") is not None


def test_async_job_failure_transitions_to_failed_with_error(
    adapter: AwsStorageAdapter, sample_png: bytes
):
    """An inspection error records a FAILED job carrying the error (Req 16.3)."""
    saved_statuses: list[str] = []
    original_save = adapter.save_job

    def _tracking_save(job):  # noqa: ANN001
        saved_statuses.append(job.status)
        original_save(job)

    adapter.save_job = _tracking_save  # type: ignore[method-assign]

    class _AlwaysFailsModel:
        # Not used: we fail earlier via a broken save_report. But provide a
        # valid model so the loop can run if reached.
        def decide(self, finding, trace, cfg):  # noqa: ANN001
            return AgentDecision(
                decision_id="bedrock-0",
                iteration=0,
                action="finalize_report",
                rationale="done",
                decided_by="bedrock",
            )

    # Make the FINAL save_report raise so the inspection fails after work.
    def _boom_report(_report):  # noqa: ANN001
        raise RuntimeError("dynamodb put_item failed (simulated, non-recovering)")

    adapter.save_report = _boom_report  # type: ignore[method-assign]

    job = run_async_job(
        "job-fail-1",
        "aws-job-insp-2",
        sample_png,
        "image/png",
        adapter=adapter,
        model=_AlwaysFailsModel(),
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0, sleep=lambda _d: None),
        now=_clock(),
    )

    assert saved_statuses == ["queued", "running", "failed"]
    assert job.status == "failed"
    assert job.error and "retries exhausted" in job.error

    polled = adapter.get_job("job-fail-1")
    assert polled is not None
    assert polled.status == "failed"
    assert polled.error


# ---------------------------------------------------------------------------
# Retry: transient recovery vs exhaustion surfacing a failed job (Req 16.3)
# ---------------------------------------------------------------------------


def test_transient_aws_failure_recovers_within_budget(
    adapter: AwsStorageAdapter, sample_png: bytes
):
    """A save_report that fails once then succeeds completes the inspection."""
    calls = {"n": 0}
    real_save = adapter.save_report

    def _flaky_save(report):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("throttled (simulated transient)")
        real_save(report)

    adapter.save_report = _flaky_save  # type: ignore[method-assign]

    sleeps: list[float] = []
    policy = RetryPolicy(max_attempts=3, base_delay=0.0, sleep=sleeps.append)

    model = InjectedDecisionModel(
        decisions=[
            AgentDecision(
                decision_id="bedrock-0",
                iteration=0,
                action="finalize_report",
                rationale="done",
                decided_by="bedrock",
            )
        ]
    )

    report = run_aws_inspection(
        sample_png,
        "image/png",
        inspection_id="aws-retry-ok",
        adapter=adapter,
        model=model,
        retry_policy=policy,
    )

    assert report.mode == "aws"
    assert calls["n"] == 2  # failed once, succeeded on the retry
    assert len(sleeps) == 1  # exactly one backoff sleep happened
    assert adapter.get_report("aws-retry-ok") is not None


def test_transient_aws_failure_never_recovers_surfaces_failed_job(
    adapter: AwsStorageAdapter, sample_png: bytes
):
    """A save_report that always fails exhausts retries -> RetryExhausted -> failed job."""
    save_calls = {"n": 0}

    def _always_fail(_report):  # noqa: ANN001
        save_calls["n"] += 1
        raise RuntimeError("persistent throttling (simulated)")

    adapter.save_report = _always_fail  # type: ignore[method-assign]

    sleeps: list[float] = []
    policy = RetryPolicy(max_attempts=4, base_delay=0.0, sleep=sleeps.append)

    model = InjectedDecisionModel(
        decisions=[
            AgentDecision(
                decision_id="bedrock-0",
                iteration=0,
                action="finalize_report",
                rationale="done",
                decided_by="bedrock",
            )
        ]
    )

    # Direct call surfaces RetryExhausted after exactly max_attempts attempts.
    with pytest.raises(RetryExhausted) as excinfo:
        run_aws_inspection(
            sample_png,
            "image/png",
            inspection_id="aws-retry-doomed",
            adapter=adapter,
            model=model,
            retry_policy=policy,
        )
    assert save_calls["n"] == 4
    assert excinfo.value.attempts == 4
    assert len(sleeps) == 3  # slept before each of the 3 retries

    # Through the async worker, the same exhaustion is recorded as a FAILED job.
    save_calls["n"] = 0
    sleeps.clear()
    job = run_async_job(
        "job-doomed-1",
        "aws-retry-doomed-2",
        sample_png,
        "image/png",
        adapter=adapter,
        model=model,
        retry_policy=RetryPolicy(max_attempts=4, base_delay=0.0, sleep=sleeps.append),
        now=_clock(),
    )
    assert job.status == "failed"
    assert job.error and "retries exhausted" in job.error
    assert adapter.get_job("job-doomed-1").status == "failed"  # type: ignore[union-attr]
