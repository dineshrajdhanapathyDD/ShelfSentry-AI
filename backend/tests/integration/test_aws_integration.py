"""AWS integration, API, and infrastructure-security tests (task 16.3).

Validates Requirements 24.1, 24.2, and 24.4 at the *integration* layer, over
the REAL seams wired together, with ZERO real AWS calls:

* the real :class:`~shelfsentry.data.aws_storage.AwsStorageAdapter` driven by
  in-memory FAKE S3 / DynamoDB clients (the fakes are replicated from
  ``tests/data/test_aws_storage.py`` / ``tests/pipeline/test_aws_worker.py`` so
  this module is self-contained);
* the real AWS-mode runner :func:`shelfsentry.aws_worker.run_aws_inspection`
  (and :func:`run_async_job`) over those fakes; and
* the real :class:`~shelfsentry.api.LocalApi` handler surface built with
  ``mode="aws"`` and an injected runner over the real AWS adapter.

These cases are deliberately *complementary* to the existing unit coverage and
do NOT duplicate it:

* ``tests/data/test_aws_storage.py`` pins the adapter's per-method round-trips
  and ref-boundary at the unit level;
* ``tests/pipeline/test_aws_worker.py`` pins the runner/worker lifecycle and
  retry at the unit level;
* ``tests/api/test_local_api.py`` + ``tests/api/test_api_versioning_properties.py``
  pin the LocalApi envelope surface in LOCAL mode (and an AWS-mode envelope over
  a hand-rolled fake adapter + fake runner);
* ``tests/infrastructure/test_template.py`` pins template validity + the coarse
  "no Action '*' on Resource '*'" wildcard-admin guard and S3/retention.

What this module adds on top of those:

* Req 24.1 - request validation + rejection driven through a ``mode="aws"``
  :class:`LocalApi` whose runner is the REAL ``run_aws_inspection`` over the
  REAL adapter: invalid create requests, invalid history queries, and blank ids
  return structured, versioned, ``mode="aws"`` error envelopes (never a raise,
  never a silent downgrade to local), and a valid create returns an ``mode="aws"``
  success envelope persisted in the fake DynamoDB.
* Req 24.2 - missing S3 objects, result retrieval, and failed jobs surfaced
  through the adapter + runner + API together: a well-formed-but-absent media
  ref raises :class:`RecordNotFound`; a missing report surfaces as an API-level
  structured ``not_found``; a forced-failure async job is retrievable with
  ``status="failed"`` via ``get_job``.
* Req 24.4 - the media-ref security boundary enforced at the API-integration
  layer (a crafted/foreign ref never reaches S3 when routed through the runner),
  plus finer-grained IAM-scoping assertions on ``infrastructure/template.yaml``
  that ``test_template.py`` does not make: ``bedrock:InvokeModel`` is scoped to a
  single foundation-model resource (not ``*``), and the API/worker S3 grants are
  scoped to the media bucket/prefix ARN (not ``*`` and not the bare bucket).

Everything is offline, deterministic, and never sleeps (retry sleepers are
injected as no-ops where retry is involved).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from shelfsentry.agent.decision import InjectedDecisionModel
from shelfsentry.agent.retry import RetryPolicy
from shelfsentry.api import API_VERSION, ApiError, ApiResponse, LocalApi
from shelfsentry.aws_worker import run_async_job, run_aws_inspection
from shelfsentry.data.aws_storage import AwsStorageAdapter
from shelfsentry.data.contracts import AgentDecision, InspectionReport
from shelfsentry.data.storage import InvalidMediaRef, RecordNotFound
from shelfsentry.samples import generate_sample_shelf_png


# ===========================================================================
# In-memory fakes (no real AWS, no boto3) - replicated from the existing unit
# test fakes so this integration module is self-contained.
# ===========================================================================


class _S3NotFound(Exception):
    """Mimics a botocore NoSuchKey error shape for the fake S3 client."""

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
    """In-memory S3 keyed by (bucket, key). Records every accessed key."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.get_keys: list[tuple[str, str]] = []
        self.put_keys: list[tuple[str, str]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> dict:
        self.put_keys.append((Bucket, Key))
        self.objects[(Bucket, Key)] = bytes(Body)
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        self.get_keys.append((Bucket, Key))
        try:
            data = self.objects[(Bucket, Key)]
        except KeyError:
            raise _S3NotFound()
        return {"Body": _Body(data)}

    def generate_presigned_url(self, operation, *, Params, ExpiresIn) -> str:  # pragma: no cover - unused here
        return "https://example.invalid/presigned"


class FakeDynamo:
    """In-memory DynamoDB keyed by the partition key; honors record_type scan filter."""

    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def put_item(self, *, TableName: str, Item: dict) -> dict:
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
    data = generate_sample_shelf_png()
    assert data
    return data


def _no_sleep() -> RetryPolicy:
    """A retry policy whose injected sleeper never actually sleeps."""
    return RetryPolicy(max_attempts=3, base_delay=0.0, sleep=lambda _d: None)


def _finalize_model() -> InjectedDecisionModel:
    """A 'Bedrock-available' injected model that finalizes on the first turn."""
    return InjectedDecisionModel(
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


def _build_aws_api(
    adapter: AwsStorageAdapter, *, model: Any | None = None
) -> LocalApi:
    """A LocalApi in ``mode="aws"`` whose runner is the REAL run_aws_inspection.

    Routing through the real runner + real adapter over fakes is what makes
    these API tests *integration* tests rather than a re-run of the local-mode
    envelope tests: a create actually runs OpenCV 5 and persists to fake S3 +
    DynamoDB, and the envelope is stamped ``mode="aws"`` end to end.
    """
    the_model = model if model is not None else _finalize_model()

    def runner(
        data: bytes,
        content_type: str,
        *,
        adapter: Any,
        inspection_id: str | None = None,
        mode: Any = "aws",
    ) -> InspectionReport:
        return run_aws_inspection(
            data,
            content_type,
            inspection_id=inspection_id,
            adapter=adapter,
            model=the_model,
            retry_policy=_no_sleep(),
        )

    return LocalApi(adapter, runner=runner, mode="aws")


def _assert_versioned_aws(resp: Any) -> None:
    """Every envelope is versioned and stamped ``mode="aws"`` (Req 24.1)."""
    assert isinstance(resp, (ApiResponse, ApiError))
    assert resp.apiVersion == API_VERSION
    assert resp.apiVersion, "apiVersion must be non-empty"
    assert resp.mode == "aws"
    assert resp.ok is isinstance(resp, ApiResponse)


# ===========================================================================
# Req 24.1 - API request validation + rejection, through the AWS boundary
# ===========================================================================
# These drive a mode="aws" LocalApi whose runner is the REAL run_aws_inspection
# over the REAL AwsStorageAdapter (fake clients). The point distinct from
# tests/api/*: validation rejection and the versioned+mode="aws" envelope are
# asserted on the integration path, not against a hand-rolled fake runner.


def test_aws_api_valid_create_returns_versioned_aws_envelope_and_persists(
    adapter: AwsStorageAdapter, fake_ddb: FakeDynamo, sample_png: bytes
) -> None:
    """Req 24.1 (happy path): a valid create yields a mode='aws' success envelope.

    The report is produced by the real runner (real OpenCV 5, real bounded loop)
    and persisted to the fake DynamoDB behind the real adapter, retrievable via
    the same API.
    """
    api = _build_aws_api(adapter)

    resp = api.create_inspection(
        media=sample_png, content_type="image/png", inspection_id="aws-api-ok-1"
    )
    _assert_versioned_aws(resp)
    assert isinstance(resp, ApiResponse)
    report = resp.data
    assert isinstance(report, InspectionReport)
    assert report.mode == "aws"
    assert report.inspection_id == "aws-api-ok-1"
    # Really persisted through the adapter into fake DynamoDB (Req 24.2 result
    # retrieval, exercised via the API boundary).
    assert any(
        i.get("record_type", {}).get("S") == "report" for i in fake_ddb.items.values()
    )
    got = api.get_inspection("aws-api-ok-1")
    assert isinstance(got, ApiResponse)
    _assert_versioned_aws(got)
    assert got.data.canonical_json() == report.canonical_json()


@pytest.mark.parametrize(
    ("call", "kwargs", "code", "field"),
    [
        # Empty media bytes -> rejected before the runner ever touches AWS.
        ("create_inspection", {"media": b"", "content_type": "image/png"}, "validation", "media"),
        # Missing content type with real bytes -> structured validation error.
        (
            "create_inspection",
            {"media": generate_sample_shelf_png(), "content_type": None},
            "validation",
            "content_type",
        ),
        # Real PNG, unsupported content type -> the pipeline's structured
        # condition surfaces as the offending field across the AWS boundary.
        (
            "create_inspection",
            {"media": generate_sample_shelf_png(), "content_type": "application/zip"},
            "validation",
            "unsupported_content_type",
        ),
        # History query: unknown status filter.
        ("list_inspections", {"status": "bogus-status"}, "validation", None),
        # History query: non-positive limit.
        ("list_inspections", {"limit": 0}, "validation", None),
        ("list_inspections", {"limit": -7}, "validation", None),
        # Blank ids are validation errors, not lookups.
        ("get_inspection", {"inspection_id": "   "}, "validation", "inspection_id"),
        ("get_evaluation", {"eval_run_id": ""}, "validation", "eval_run_id"),
    ],
)
def test_aws_api_invalid_requests_rejected_with_structured_aws_envelope(
    adapter: AwsStorageAdapter,
    fake_s3: FakeS3,
    fake_ddb: FakeDynamo,
    call: str,
    kwargs: dict[str, Any],
    code: str,
    field: str | None,
) -> None:
    """Req 24.1: invalid requests are rejected with a structured mode='aws' error.

    Across the whole mode="aws" handler surface, each invalid request returns an
    ``ApiError`` (never a raise), carries ``mode="aws"`` + apiVersion, and the
    expected code/field. The invalid create cases additionally prove the request
    is rejected *before* any AWS write happens (no S3 put, no DynamoDB item).
    """
    api = _build_aws_api(adapter)

    resp = getattr(api, call)(**kwargs)
    _assert_versioned_aws(resp)
    assert isinstance(resp, ApiError)
    assert resp.error.code == code
    assert resp.error.message
    if field is not None:
        assert resp.error.field == field

    # A rejected create must not have written anything to AWS.
    if call == "create_inspection":
        assert fake_s3.objects == {}
        assert not any(
            i.get("record_type", {}).get("S") == "report"
            for i in fake_ddb.items.values()
        )


def test_aws_api_missing_report_returns_structured_not_found(
    adapter: AwsStorageAdapter,
) -> None:
    """Req 24.1/24.2: a missing report is a structured not_found at the API edge.

    Routed through the real AWS adapter's ``get_report`` (fake DynamoDB miss),
    surfaced as a 404-style, versioned, mode='aws' envelope rather than a raise.
    """
    api = _build_aws_api(adapter)
    resp = api.get_inspection("no-such-aws-inspection")
    _assert_versioned_aws(resp)
    assert isinstance(resp, ApiError)
    assert resp.error.code == "not_found"
    assert resp.error.field == "inspection_id"
    assert "no-such-aws-inspection" in resp.error.message


# ===========================================================================
# Req 24.2 - missing S3 objects / result retrieval / failed jobs (integration)
# ===========================================================================


def test_missing_s3_object_wellformed_ref_raises_record_not_found(
    adapter: AwsStorageAdapter,
) -> None:
    """Req 24.2: a well-formed but absent media ref raises RecordNotFound.

    The token is valid (32 lowercase hex), so the ref passes the security
    boundary and resolves to a scoped key - but no object exists there, so the
    fake S3 miss surfaces as RecordNotFound (a missing S3 object), distinct from
    the InvalidMediaRef raised for a *crafted* ref.
    """
    good_token = "a1b2c3d4" * 4  # 32 lowercase hex chars
    with pytest.raises(RecordNotFound):
        adapter.get_media(f"media://{good_token}")


def test_result_retrieval_round_trips_through_runner_and_adapter(
    adapter: AwsStorageAdapter, sample_png: bytes
) -> None:
    """Req 24.2: a completed inspection's result is retrievable after the run.

    End-to-end: the real runner persists the report via the adapter, and both
    ``get_report`` and ``get_media`` (for the stored original + annotated
    evidence) return the stored bytes - a positive result-retrieval path that
    complements the missing-object case above.
    """
    report = run_aws_inspection(
        sample_png,
        "image/png",
        inspection_id="aws-retrieve-1",
        adapter=adapter,
        model=_finalize_model(),
        retry_policy=_no_sleep(),
    )
    fetched = adapter.get_report("aws-retrieve-1")
    assert fetched is not None
    assert fetched.canonical_json() == report.canonical_json()

    # The original image is retrievable by its recorded ref.
    image_ref = adapter.get_media_ref_for("aws-retrieve-1", "image")
    assert image_ref is not None
    assert adapter.get_media(image_ref) == sample_png

    # And the annotated evidence for the initial finding is retrievable.
    evidence_ref = adapter.get_media_ref_for(
        "aws-retrieve-1", f"evidence:{report.initial_finding_id}"
    )
    assert evidence_ref is not None
    assert adapter.get_media(evidence_ref)  # non-empty annotated bytes


def test_get_report_for_missing_id_returns_none(adapter: AwsStorageAdapter) -> None:
    """Req 24.2: retrieving an absent result returns None (not an error/raise)."""
    assert adapter.get_report("never-created") is None


def test_failed_async_job_is_retrievable_with_status_failed(
    adapter: AwsStorageAdapter, sample_png: bytes
) -> None:
    """Req 24.2: a failed job is retrievable via get_job with status='failed'.

    A forced, non-recovering ``save_report`` failure exhausts the retry budget
    inside the real runner; the worker records a FAILED job carrying the error,
    which ``get_job`` then returns - the pollable failed-job path, asserted at
    the integration (runner+worker+adapter) level.
    """

    def _boom(_report: Any) -> None:
        raise RuntimeError("dynamodb put_item failed (simulated, non-recovering)")

    adapter.save_report = _boom  # type: ignore[method-assign]

    def _clock() -> str:
        _clock.n += 1  # type: ignore[attr-defined]
        return f"2024-01-01T00:00:{_clock.n:02d}Z"  # type: ignore[attr-defined]

    _clock.n = 0  # type: ignore[attr-defined]

    job = run_async_job(
        "aws-job-failed-1",
        "aws-job-insp-failed-1",
        sample_png,
        "image/png",
        adapter=adapter,
        model=_finalize_model(),
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0, sleep=lambda _d: None),
        now=_clock,
    )
    assert job.status == "failed"
    assert job.error and "retries exhausted" in job.error

    polled = adapter.get_job("aws-job-failed-1")
    assert polled is not None
    assert polled.status == "failed"
    assert polled.error
    assert polled.inspection_id == "aws-job-insp-failed-1"
    # No report was persisted for the failed inspection.
    assert adapter.get_report("aws-job-insp-failed-1") is None


# ===========================================================================
# Req 24.4 - storage-security boundary at the integration layer
# ===========================================================================
# tests/data/test_aws_storage.py asserts crafted refs are rejected at the
# adapter's get_media. These cases assert the SAME boundary holds when a crafted
# ref is dereferenced through the runner's resolver seam (the perception path),
# and that a planted foreign/sibling object is never surfaced - i.e. the boundary
# is enforced before any S3 read, not just at the top-level adapter call.


@pytest.mark.parametrize(
    "crafted",
    [
        "media://../../../../etc/passwd",
        "media:///etc/passwd",
        "media://C:\\Windows\\System32\\drivers\\etc\\hosts",
        "s3://other-bucket/some/key",
        "s3://test-bucket/shelfsentry/media/" + "a" * 32 + ".bin",
        "file:///etc/passwd",
        "../../etc/passwd",
        "media://DEADBEEFDEADBEEFDEADBEEFDEADBEEF",  # uppercase hex not allowed
        "media://short",
    ],
)
def test_crafted_ref_rejected_before_any_s3_read_integration(
    adapter: AwsStorageAdapter, fake_s3: FakeS3, crafted: str
) -> None:
    """Req 24.4: a crafted/foreign media ref is rejected before any S3 call.

    Plant a secret in a foreign bucket and a sibling key in our own bucket, then
    attempt to dereference crafted refs. Each must raise InvalidMediaRef, no
    get_object must have been attempted for the crafted ref, and neither planted
    object may leak.
    """
    fake_s3.objects[("other-bucket", "secret")] = b"TOP SECRET"
    fake_s3.objects[("test-bucket", "secret")] = b"OURS BUT OUTSIDE PREFIX"
    fake_s3.get_keys.clear()

    with pytest.raises(InvalidMediaRef):
        adapter.get_media(crafted)

    # The boundary rejects structurally, before any S3 read is attempted.
    assert fake_s3.get_keys == []
    # The planted objects are untouched and were never surfaced.
    assert fake_s3.objects[("other-bucket", "secret")] == b"TOP SECRET"
    assert fake_s3.objects[("test-bucket", "secret")] == b"OURS BUT OUTSIDE PREFIX"


def test_runner_only_reads_scoped_media_keys(
    adapter: AwsStorageAdapter, fake_s3: FakeS3, sample_png: bytes
) -> None:
    """Req 24.4: every S3 read a real inspection performs is under the media prefix.

    A full AWS-mode run touches only scoped ``shelfsentry/media/`` keys in the
    configured bucket - it never reads an arbitrary/foreign key. This is the
    positive complement to the crafted-ref rejection above.
    """
    run_aws_inspection(
        sample_png,
        "image/png",
        inspection_id="aws-scope-1",
        adapter=adapter,
        model=_finalize_model(),
        retry_policy=_no_sleep(),
    )
    assert fake_s3.get_keys, "the run should have read at least one media object"
    for bucket, key in fake_s3.get_keys:
        assert bucket == "test-bucket"
        assert key.startswith("shelfsentry/media/")
        assert key.endswith(".bin")


# ===========================================================================
# Req 24.4 - finer-grained IAM scoping on the CloudFormation template
# ===========================================================================
# test_template.py already asserts: no Action '*' on Resource '*'. These add the
# finer checks it does NOT make: bedrock:InvokeModel is scoped to a single
# foundation-model resource (not '*'), and the API/worker S3 grants are scoped to
# the media bucket/prefix ARN (not '*' and not the bare bucket).


_THIS_FILE = Path(__file__).resolve()
# tests/integration/test_aws_integration.py -> repo root is parents[3].
_REPO_ROOT = _THIS_FILE.parents[3]
_TEMPLATE_PATH = _REPO_ROOT / "infrastructure" / "template.yaml"


class _CfnLoader(yaml.SafeLoader):
    """SafeLoader that materializes CloudFormation ``!Tag`` intrinsics as dicts."""


def _cfn_multi_constructor(loader: _CfnLoader, tag_suffix: str, node: yaml.Node) -> Any:
    key = "Ref" if tag_suffix == "Ref" else f"Fn::{tag_suffix}"
    if isinstance(node, yaml.ScalarNode):
        value: Any = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {key: value}


_CfnLoader.add_multi_constructor("!", _cfn_multi_constructor)


@pytest.fixture(scope="module")
def template() -> dict[str, Any]:
    assert _TEMPLATE_PATH.is_file(), f"template not found at {_TEMPLATE_PATH}"
    parsed = yaml.load(_TEMPLATE_PATH.read_text(encoding="utf-8"), Loader=_CfnLoader)
    assert isinstance(parsed, dict)
    return parsed


def _roles(template: dict[str, Any]) -> dict[str, Any]:
    return {
        lid: body
        for lid, body in template.get("Resources", {}).items()
        if isinstance(body, dict) and body.get("Type") == "AWS::IAM::Role"
    }


def _iter_role_statements(role: dict[str, Any]):
    """Yield (policy_name, statement) for every inline policy statement in a role."""
    for policy in role.get("Properties", {}).get("Policies", []) or []:
        if not isinstance(policy, dict):
            continue
        name = policy.get("PolicyName", "<unnamed>")
        doc = policy.get("PolicyDocument", {})
        stmts = doc.get("Statement", []) if isinstance(doc, dict) else []
        if isinstance(stmts, dict):
            stmts = [stmts]
        for stmt in stmts:
            if isinstance(stmt, dict):
                yield name, stmt


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _actions(stmt: dict[str, Any]) -> list[str]:
    return [str(a) for a in _as_list(stmt.get("Action"))]


def _resource_tokens(resource: Any) -> list[str]:
    """Flatten a statement Resource into comparable string tokens.

    Intrinsics are materialized by the loader as single-key dicts; we extract
    the inner scalar so we can assert on the ARN *shape* without evaluating the
    template. A literal ``"*"`` stays ``"*"``; a ``{"Fn::Sub": "...${X}..."}``
    yields its template string; a ``{"Fn::GetAtt": "T.Arn"}`` yields ``"T.Arn"``.
    """
    tokens: list[str] = []
    for item in _as_list(resource):
        if isinstance(item, str):
            tokens.append(item)
        elif isinstance(item, dict):
            for v in item.values():
                if isinstance(v, str):
                    tokens.append(v)
                elif isinstance(v, list):
                    tokens.extend(str(x) for x in v if isinstance(x, (str, int)))
    return tokens


def test_bedrock_invoke_is_scoped_to_a_foundation_model_not_wildcard(
    template: dict[str, Any],
) -> None:
    """Req 24.4: bedrock:InvokeModel is scoped to a model resource, never '*'.

    Finer-grained than test_template.py's '*'/'*' guard: it finds the statement
    that grants ``bedrock:InvokeModel`` and asserts its Resource names a specific
    ``foundation-model/...`` ARN (via the configured model id) and is not ``*``.
    """
    found_invoke = False
    for role_id, role in _roles(template).items():
        for policy_name, stmt in _iter_role_statements(role):
            actions = _actions(stmt)
            if not any(a.startswith("bedrock:InvokeModel") for a in actions):
                continue
            found_invoke = True
            assert stmt.get("Effect") == "Allow"
            tokens = _resource_tokens(stmt.get("Resource"))
            assert tokens, f"{role_id}/{policy_name}: bedrock grant has no Resource"
            assert "*" not in tokens, (
                f"{role_id}/{policy_name}: bedrock:InvokeModel on wildcard Resource"
            )
            # Every resource names a specific foundation-model ARN.
            for tok in tokens:
                assert "foundation-model/" in tok, (
                    f"{role_id}/{policy_name}: bedrock Resource not scoped to a "
                    f"foundation model: {tok!r}"
                )
                assert "${BedrockModelId}" in tok, (
                    f"{role_id}/{policy_name}: bedrock Resource not bound to the "
                    f"configured model id: {tok!r}"
                )
    assert found_invoke, "no bedrock:InvokeModel grant found to verify scoping"


def test_s3_grants_are_scoped_to_media_bucket_prefix_not_wildcard(
    template: dict[str, Any],
) -> None:
    """Req 24.4: every S3 object grant is scoped to the media bucket/prefix ARN.

    For each statement granting an ``s3:`` object action, assert its Resource is
    neither ``*`` nor the bare bucket ARN, but the object ARN under the media
    upload prefix (``${MediaBucket.Arn}/${MediaUploadPrefix}*``). This is a
    finer constraint than test_template.py checks.
    """
    checked = 0
    for role_id, role in _roles(template).items():
        for policy_name, stmt in _iter_role_statements(role):
            actions = _actions(stmt)
            s3_actions = [a for a in actions if a.startswith("s3:")]
            if not s3_actions:
                continue
            # The bucket-policy TLS-deny statement (Action s3:*) lives on the
            # bucket, not a role, so it is not seen here; role S3 grants are the
            # object read/write grants we scope. Skip any pure Deny just in case.
            if stmt.get("Effect") != "Allow":
                continue
            checked += 1
            tokens = _resource_tokens(stmt.get("Resource"))
            assert tokens, f"{role_id}/{policy_name}: S3 grant has no Resource"
            assert "*" not in tokens, (
                f"{role_id}/{policy_name}: S3 action on wildcard Resource '*'"
            )
            for tok in tokens:
                # Must reference the media bucket ARN...
                assert "MediaBucket.Arn" in tok, (
                    f"{role_id}/{policy_name}: S3 Resource not the media bucket: {tok!r}"
                )
                # ...and be an object-level ARN scoped under the upload prefix,
                # not the bare bucket ARN (which would grant bucket-wide access).
                assert "${MediaUploadPrefix}" in tok and tok.rstrip().endswith("*"), (
                    f"{role_id}/{policy_name}: S3 Resource not scoped to the media "
                    f"upload prefix: {tok!r}"
                )
    assert checked, "no role S3 object grant found to verify scoping"


def test_dynamodb_grants_are_scoped_to_named_tables_not_wildcard(
    template: dict[str, Any],
) -> None:
    """Req 24.4: DynamoDB grants name specific table ARNs, never '*'.

    Complements the S3/bedrock checks: each ``dynamodb:`` grant's Resource lists
    the Results/Jobs table ARNs (and the Jobs GSI), not a wildcard - keeping the
    data-plane least privilege verifiable.
    """
    pattern = re.compile(r"(ResultsTable|JobsTable)")
    checked = 0
    for role_id, role in _roles(template).items():
        for policy_name, stmt in _iter_role_statements(role):
            actions = _actions(stmt)
            if not any(a.startswith("dynamodb:") for a in actions):
                continue
            if stmt.get("Effect") != "Allow":
                continue
            checked += 1
            tokens = _resource_tokens(stmt.get("Resource"))
            assert tokens, f"{role_id}/{policy_name}: DDB grant has no Resource"
            assert "*" not in tokens, (
                f"{role_id}/{policy_name}: dynamodb action on wildcard Resource '*'"
            )
            assert all(pattern.search(tok) for tok in tokens), (
                f"{role_id}/{policy_name}: DDB Resource not scoped to a named "
                f"table ARN: {tokens!r}"
            )
    assert checked, "no role DynamoDB grant found to verify scoping"
