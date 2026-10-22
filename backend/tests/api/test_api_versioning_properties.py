"""Property + example tests for API versioning and the mode switch (task 11.3).

This module complements the extensive example coverage in
``tests/api/test_local_api.py`` and ``tests/config/test_mode.py`` without
duplicating it. It adds three things those files do not:

* **Property 22 (Req 10.4)** - a Hypothesis-driven property that drives the
  *whole* :class:`~shelfsentry.api.LocalApi` handler surface (built by
  :func:`shelfsentry.api.build_local_api` over a real tmp-dir
  :class:`~shelfsentry.data.storage.LocalStorageAdapter`) across a *variety* of
  calls - valid and invalid, success and error - and asserts that **every**
  returned envelope (:class:`ApiResponse` or :class:`ApiError`) is a well-formed
  versioned + mode-stamped envelope. No raw exception ever crosses the handler
  boundary; the response always carries ``apiVersion == API_VERSION`` (non-empty)
  and a ``mode`` in ``{"local", "aws"}``.

* **Req 24.1 examples** - representative invalid requests return the structured
  error envelope with a ``code`` and an offending ``field`` (empty media, bad
  content type, unknown status filter, non-positive limit, blank ids). These are
  distinct invalid-request cases chosen to span the handler surface rather than
  re-asserting the same case as the existing file.

* **Req 17.2 / 17.3 examples** - the fail-loud mode switch: AWS mode with missing
  config raises :class:`ModeConfigurationError` naming the missing keys and never
  returns a local config; and an API built for AWS mode (``mode="aws"`` with an
  injected runner/adapter, so no real AWS is needed) stamps ``mode == "aws"`` on
  its envelopes - i.e. the mode label is never silently downgraded to local.

Everything runs offline, with no AWS credentials and no network: the property
exercises the real local pipeline + storage over a tmp dir, and the AWS-mode
example injects a fake runner/adapter so no cloud call is ever made.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shelfsentry.api import (
    API_VERSION,
    ApiError,
    ApiResponse,
    LocalApi,
    build_local_api,
)
from shelfsentry.api.envelope import ApiMode
from shelfsentry.config import (
    ENV_AWS_REGION,
    ENV_DDB_TABLE,
    ENV_MODE,
    ENV_S3_BUCKET,
    Mode,
    ModeConfigurationError,
    resolve_runtime_config,
)
from shelfsentry.data.contracts import AgentDecision, InspectionReport
from shelfsentry.data.storage import HistoryQuery, InspectionSummary, LocalStorageAdapter
from shelfsentry.samples import generate_sample_shelf_png

# Valid mode labels an envelope may carry (Req 10.4 + mode labeling, Req 17).
_VALID_MODES = frozenset({"local", "aws"})


# ---------------------------------------------------------------------------
# Shared envelope invariant: the heart of Property 22
# ---------------------------------------------------------------------------


def _assert_versioned_envelope(resp: Any, *, expected_mode: ApiMode) -> None:
    """Assert ``resp`` is a well-formed versioned + mode-stamped envelope.

    This is the universal post-condition every handler must satisfy regardless
    of input (Req 10.4): the result is one of the two envelope types, carries a
    non-empty ``apiVersion`` equal to :data:`API_VERSION`, and a ``mode`` in the
    allowed set (and equal to the mode the API was built for). A raw exception
    or a ``None`` here would be a Req 10.4/10.5 defect.
    """
    assert isinstance(resp, (ApiResponse, ApiError)), (
        f"handler returned a non-envelope value: {type(resp)!r}"
    )
    assert resp.apiVersion == API_VERSION
    assert resp.apiVersion, "apiVersion must be non-empty"
    assert resp.mode in _VALID_MODES
    assert resp.mode == expected_mode
    # ApiResponse.ok is True, ApiError.ok is False - both branchable.
    assert resp.ok is isinstance(resp, ApiResponse)
    if isinstance(resp, ApiError):
        # A structured error always carries a code + a human message (Req 10.5).
        assert resp.error.code in {"validation", "not_found", "bad_request"}
        assert resp.error.message


# ---------------------------------------------------------------------------
# Property 22: Every API response includes an API version
# Validates: Requirements 10.4
# ---------------------------------------------------------------------------

# A handler "call plan" is a tuple (handler_name, kwargs). The strategy below
# generates plans spanning the whole handler surface with a mix of valid and
# invalid arguments so the property sees success *and* error envelopes.

# Small pools so invalid calls are common and ids sometimes collide with the
# one inspection we create below (exercising both found and missing paths).
_ID_POOL = st.sampled_from(
    ["", "   ", "prop-exists", "does-not-exist", "api-create-x", "  weird id  "]
)
# status spans the three valid values, an unknown filter, None, and blanks.
_STATUS_POOL = st.sampled_from(
    [None, "complete", "failed", "needs_review", "bogus", "", "COMPLETE"]
)
_TEXT_POOL = st.sampled_from([None, "", "prop", "no-match-xyz", "   "])
# limit spans valid (>=1), invalid (<=0), and None.
_LIMIT_POOL = st.sampled_from([None, 1, 2, 0, -1, -100])
# content types span accepted and unsupported.
_CT_POOL = st.sampled_from(
    [None, "", "image/png", "image/jpeg", "application/zip", "text/plain", "video/mp4"]
)
# media spans empty, a tiny non-decodable blob, and a real PNG.
_MEDIA_POOL = st.sampled_from([None, b"", b"not-an-image", b"\x00\x01\x02"])


@st.composite
def _call_plans(draw: st.DrawFn) -> tuple[str, dict[str, Any]]:
    """Generate a (handler_name, kwargs) plan across the handler surface."""
    handler = draw(
        st.sampled_from(
            [
                "create_inspection",
                "get_inspection",
                "list_inspections",
                "list_evaluations",
                "get_evaluation",
                "get_annotated_evidence",
            ]
        )
    )
    if handler == "create_inspection":
        # Either example mode (always valid input) or an explicit request with
        # possibly-invalid media/content_type so both success and error arise.
        if draw(st.booleans()):
            return handler, {"example_mode": True}
        return handler, {
            "media": draw(_MEDIA_POOL),
            "content_type": draw(_CT_POOL),
        }
    if handler == "get_inspection":
        return handler, {"inspection_id": draw(_ID_POOL)}
    if handler == "list_inspections":
        return handler, {
            "status": draw(_STATUS_POOL),
            "text": draw(_TEXT_POOL),
            "limit": draw(_LIMIT_POOL),
        }
    if handler == "list_evaluations":
        return handler, {}
    if handler == "get_evaluation":
        return handler, {"eval_run_id": draw(_ID_POOL)}
    # get_annotated_evidence
    return handler, {
        "inspection_id": draw(_ID_POOL),
        "finding_id": draw(_ID_POOL),
    }


@settings(max_examples=150)
@given(plan=_call_plans())
def test_property22_every_response_is_versioned_and_mode_stamped(
    tmp_path_factory: pytest.TempPathFactory, plan: tuple[str, dict[str, Any]]
) -> None:
    """Property 22 / Req 10.4: every handler call returns a versioned envelope.

    Drives the real LocalApi over a fresh tmp-dir LocalStorageAdapter across the
    full handler surface with valid and invalid arguments. Whatever the input,
    the handler returns a well-formed ``ApiResponse``/``ApiError`` carrying
    ``apiVersion == API_VERSION`` and ``mode == "local"`` - never a raw
    exception crossing the boundary (that would be a Req 10.4/10.5 defect).
    """
    # A fresh store per example keeps examples independent; seed exactly one
    # known inspection so get/evidence calls hit both found and missing paths.
    base = tmp_path_factory.mktemp("prop22")
    adapter = LocalStorageAdapter(base)
    api = build_local_api(adapter)
    seeded = api.create_inspection(example_mode=True, inspection_id="prop-exists")
    assert isinstance(seeded, ApiResponse)  # sanity: the known seed succeeded

    handler_name, kwargs = plan
    handler = getattr(api, handler_name)

    # The whole point of the property: the call must not raise. If it does, that
    # is a real defect (an exception escaping the envelope boundary), so we let
    # it surface as a test failure rather than swallowing it.
    resp = handler(**kwargs)
    _assert_versioned_envelope(resp, expected_mode="local")


def test_property22_known_evidence_path_also_versioned(tmp_path) -> None:
    """A valid evidence fetch (the success branch) is likewise versioned.

    The generated property leans toward missing-evidence ids; this pins the
    success branch so the property's coverage of that path is anchored.
    """
    adapter = LocalStorageAdapter(tmp_path)
    api = build_local_api(adapter)
    created = api.create_inspection(example_mode=True, inspection_id="prop-ev")
    assert isinstance(created, ApiResponse)
    resp = api.get_annotated_evidence("prop-ev", created.data.initial_finding_id)
    _assert_versioned_envelope(resp, expected_mode="local")
    assert isinstance(resp, ApiResponse)
    assert resp.data.byte_length > 0


# ---------------------------------------------------------------------------
# Req 24.1: API request validation and rejection of invalid requests
# (representative invalid requests -> structured error envelope w/ code + field)
# ---------------------------------------------------------------------------


@pytest.fixture
def local_api(tmp_path) -> LocalApi:
    """A local API over a tmp-dir adapter, for the example tests below."""
    return build_local_api(LocalStorageAdapter(tmp_path))


@pytest.mark.parametrize(
    ("call", "kwargs", "expected_code", "expected_field"),
    [
        # Empty media bytes with an otherwise-fine content type (Req 24.1, 10.5).
        (
            "create_inspection",
            {"media": b"", "content_type": "image/png"},
            "validation",
            "media",
        ),
        # A real PNG but an unsupported content type -> the pipeline's own
        # structured condition surfaces as the offending field (Req 1.5, 24.1).
        (
            "create_inspection",
            {"media": generate_sample_shelf_png(), "content_type": "application/zip"},
            "validation",
            "unsupported_content_type",
        ),
        # Unknown status filter on the history list (Req 24.1).
        ("list_inspections", {"status": "not-a-status"}, "validation", None),
        # Non-positive limit on the history list (Req 24.1).
        ("list_inspections", {"limit": -5}, "validation", None),
        # Blank ids are rejected as validation errors, not treated as lookups.
        ("get_inspection", {"inspection_id": "   "}, "validation", "inspection_id"),
        ("get_evaluation", {"eval_run_id": ""}, "validation", "eval_run_id"),
        (
            "get_annotated_evidence",
            {"inspection_id": " ", "finding_id": "f"},
            "validation",
            "inspection_id",
        ),
    ],
)
def test_invalid_requests_return_structured_error_envelope(
    local_api: LocalApi,
    call: str,
    kwargs: dict[str, Any],
    expected_code: str,
    expected_field: str | None,
) -> None:
    """Representative invalid requests return a structured error (Req 24.1, 10.5).

    Each case targets a different offending input across the handler surface and
    asserts the response is an ``ApiError`` carrying the expected ``code`` and,
    where the handler can name it, the offending ``field`` - never a raised
    exception.
    """
    resp = getattr(local_api, call)(**kwargs)
    _assert_versioned_envelope(resp, expected_mode="local")
    assert isinstance(resp, ApiError)
    assert resp.error.code == expected_code
    assert resp.error.message
    if expected_field is not None:
        assert resp.error.field == expected_field


# ---------------------------------------------------------------------------
# Req 17.2 / 17.3: fail-loud mode switch, no silent local substitution
# ---------------------------------------------------------------------------

_AWS_KEYS = (ENV_S3_BUCKET, ENV_DDB_TABLE, ENV_AWS_REGION)


def test_aws_mode_missing_config_fails_loud_names_keys_no_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AWS mode with missing config raises naming the keys, never returns local.

    Req 17.2: startup fails with a message identifying each missing key.
    Req 17.3 / 16.4: the resolver never silently substitutes a local config -
    it raises instead of returning one.
    """
    monkeypatch.setenv(ENV_MODE, "aws")
    for key in _AWS_KEYS:
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ModeConfigurationError) as excinfo:
        resolve_runtime_config()

    err = excinfo.value
    # Every missing key is both recorded and named in the human message.
    assert set(err.missing_keys) == set(_AWS_KEYS)
    message = str(err)
    for key in _AWS_KEYS:
        assert key in message
    # It must not have quietly produced a local runtime instead.
    assert not isinstance(err, Mode)  # defensive: an error, not a mode value


def test_aws_mode_partial_config_still_raises_and_never_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially-configured AWS selection raises; it never yields a LOCAL config.

    Pins Req 17.3 at the call site: whatever the resolver returns on the AWS
    path, it is an AWS config or an exception - we assert the function does not
    return a RuntimeConfig whose mode is LOCAL for an AWS selection.
    """
    monkeypatch.setenv(ENV_MODE, "aws")
    monkeypatch.setenv(ENV_S3_BUCKET, "only-bucket")
    monkeypatch.delenv(ENV_DDB_TABLE, raising=False)
    monkeypatch.delenv(ENV_AWS_REGION, raising=False)

    returned_local = False
    try:
        cfg = resolve_runtime_config()
        # If it ever returns on the AWS path, it must NOT be a LOCAL downgrade.
        returned_local = cfg.mode is Mode.LOCAL
    except ModeConfigurationError:
        pass  # expected fail-loud path
    assert returned_local is False


def _fake_aws_report(inspection_id: str) -> InspectionReport:
    """Build a minimal valid report as a fake AWS runner would return."""
    decision = AgentDecision(
        decision_id="d0",
        iteration=0,
        action="finalize_report",
        rationale="sufficient evidence (injected aws runner)",
        decided_by="injected",
    )
    return InspectionReport(
        inspection_id=inspection_id,
        mode="aws",
        final_recommendation="no action required (injected aws runner)",
        requires_human_review=False,
        initial_finding_id="finding-0",
        decision_trace=[decision],
        supporting_evidence_refs=[],
        created_at="2024-01-01T00:00:00Z",
        status="complete",
    )


class _FakeAwsAdapter:
    """A tiny in-memory adapter standing in for the AWS storage seam.

    Lets us build a ``mode="aws"`` API without any boto3/network, so the test
    can prove the envelope is stamped ``mode == "aws"`` (Req 17.3: the selected
    mode is carried through, never silently downgraded to local).
    """

    def __init__(self) -> None:
        self._reports: dict[str, InspectionReport] = {}

    def save_report(self, report: InspectionReport) -> None:
        self._reports[report.inspection_id] = report

    def get_report(self, inspection_id: str) -> InspectionReport | None:
        return self._reports.get(inspection_id)

    def list_reports(self, query: HistoryQuery) -> list[InspectionSummary]:
        rows = [InspectionSummary.from_report(r) for r in self._reports.values()]
        if query.status is not None:
            rows = [r for r in rows if r.status == query.status]
        if query.limit is not None:
            rows = rows[: query.limit]
        return rows

    # Unused-by-this-test protocol members, present for completeness.
    def put_media(self, inspection_id: str, data: bytes, kind: str) -> str: ...
    def get_media(self, ref: str) -> bytes: ...
    def save_job(self, job: Any) -> None: ...
    def get_job(self, job_id: str) -> Any: ...


def test_aws_mode_api_stamps_mode_aws_on_envelopes() -> None:
    """A LocalApi built for AWS mode stamps ``mode == "aws"`` (Req 17.3).

    Construct the handler set with ``mode="aws"`` and an injected fake
    runner/adapter (no real AWS). Both a success envelope (create) and a
    not-found error envelope (missing get) must carry ``mode == "aws"`` - the
    selected mode is never silently relabeled local.
    """

    def fake_runner(
        data: bytes,
        content_type: str,
        *,
        adapter: Any,
        inspection_id: str | None = None,
        mode: Any = "aws",
    ) -> InspectionReport:
        report = _fake_aws_report(inspection_id or "aws-1")
        adapter.save_report(report)
        return report

    adapter = _FakeAwsAdapter()
    api = LocalApi(adapter, runner=fake_runner, mode="aws")

    created = api.create_inspection(
        media=b"ignored-by-fake-runner", content_type="image/png", inspection_id="aws-1"
    )
    _assert_versioned_envelope(created, expected_mode="aws")
    assert isinstance(created, ApiResponse)
    assert created.data.mode == "aws"
    assert created.mode == "aws"

    # The error branch is likewise stamped aws (never downgraded to local).
    missing = api.get_inspection("no-such-aws-id")
    _assert_versioned_envelope(missing, expected_mode="aws")
    assert isinstance(missing, ApiError)
    assert missing.error.code == "not_found"

    # A validation-error branch on the AWS-mode API is also aws-stamped.
    bad = api.create_inspection(media=b"", content_type="image/png")
    _assert_versioned_envelope(bad, expected_mode="aws")
    assert isinstance(bad, ApiError)
    assert bad.error.code == "validation"
