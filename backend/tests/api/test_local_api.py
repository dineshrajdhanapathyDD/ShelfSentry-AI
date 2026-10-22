"""Example tests for the local API handler layer (task 11.1; Req 10.4, 10.5, 16.1).

Exercises the framework-agnostic handler set built by
:func:`shelfsentry.api.build_local_api` over a real
:class:`~shelfsentry.data.storage.LocalStorageAdapter` rooted in a tmp dir
(dependency-injected, no AWS, no network). The handlers compose over the real
local pipeline and the real storage adapter.

What is pinned here:

* ``create_inspection`` on the bundled sample returns a **versioned**,
  **mode-stamped** success envelope (``apiVersion`` + ``mode == "local"``;
  Req 10.4) whose ``data`` is an :class:`InspectionReport` with a populated
  decision trace;
* ``get_inspection`` returns the stored report, and on a **missing id** returns
  a **structured error envelope** (not an exception; Req 10.5);
* ``list_inspections`` supports ``status`` / ``text`` / ``limit`` filtering;
* evaluation listing returns an **empty, non-fabricated** list when no records
  exist (Req 25.4, 27.2);
* an invalid create (bad content type / empty bytes) returns the **structured
  error envelope** (Req 1, 10.5).
"""

from __future__ import annotations

import pytest

from shelfsentry.api import (
    API_VERSION,
    ApiError,
    ApiResponse,
    build_local_api,
)
from shelfsentry.data.contracts import AgentDecision, InspectionReport
from shelfsentry.data.storage import InspectionSummary, LocalStorageAdapter
from shelfsentry.samples import generate_sample_shelf_png


@pytest.fixture
def adapter(tmp_path) -> LocalStorageAdapter:
    """A real local storage adapter rooted in a throwaway tmp dir."""
    return LocalStorageAdapter(tmp_path)


@pytest.fixture
def api(adapter: LocalStorageAdapter):
    """The local handler set wired over the tmp-dir adapter (local defaults)."""
    return build_local_api(adapter)


def _assert_versioned_local(envelope) -> None:
    """Every response carries apiVersion + mode == 'local' (Req 10.4)."""
    assert envelope.apiVersion == API_VERSION
    assert envelope.apiVersion, "apiVersion must be non-empty"
    assert envelope.mode == "local"


# ---------------------------------------------------------------------------
# create_inspection
# ---------------------------------------------------------------------------


def test_create_inspection_example_mode_returns_versioned_report(api) -> None:
    """Example mode runs the bundled sample -> versioned envelope + trace."""
    resp = api.create_inspection(example_mode=True)

    assert isinstance(resp, ApiResponse)
    assert resp.ok
    _assert_versioned_local(resp)

    report = resp.data
    assert isinstance(report, InspectionReport)
    assert report.mode == "local"
    # The report carries a populated decision trace with >= 1 AgentDecision.
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert len(decisions) >= 1


def test_create_inspection_with_explicit_bytes(api) -> None:
    """Passing raw sample bytes + content type works the same as example mode."""
    resp = api.create_inspection(
        media=generate_sample_shelf_png(),
        content_type="image/png",
        inspection_id="api-create-001",
    )
    assert isinstance(resp, ApiResponse)
    assert resp.data.inspection_id == "api-create-001"
    _assert_versioned_local(resp)


def test_create_inspection_persists_so_get_can_retrieve(api, adapter) -> None:
    """A created report is persisted via the adapter (Req 15.4 through the API)."""
    created = api.create_inspection(example_mode=True, inspection_id="api-persist-1")
    assert isinstance(created, ApiResponse)
    # The adapter now holds it.
    assert adapter.get_report("api-persist-1") is not None


# ---------------------------------------------------------------------------
# create_inspection - invalid requests -> structured error envelope (Req 10.5)
# ---------------------------------------------------------------------------


def test_create_inspection_empty_bytes_returns_structured_error(api) -> None:
    """Empty media returns a structured validation error, not an exception."""
    resp = api.create_inspection(media=b"", content_type="image/png")

    assert isinstance(resp, ApiError)
    assert not resp.ok
    _assert_versioned_local(resp)
    assert resp.error.code == "validation"
    assert resp.error.message


def test_create_inspection_bad_content_type_returns_structured_error(api) -> None:
    """A non-image/video content type is mapped to the structured error envelope.

    The bytes are a real PNG, so this exercises the pipeline's own
    InputValidationError (unsupported_content_type) being mapped to the envelope
    rather than raised across the boundary (Req 1.5, 10.5).
    """
    resp = api.create_inspection(
        media=generate_sample_shelf_png(), content_type="application/zip"
    )

    assert isinstance(resp, ApiError)
    _assert_versioned_local(resp)
    assert resp.error.code == "validation"
    # The structured validation condition is surfaced as the offending field.
    assert resp.error.field == "unsupported_content_type"


def test_create_inspection_missing_content_type_returns_structured_error(api) -> None:
    """Missing content_type (with bytes) is a structured validation error."""
    resp = api.create_inspection(media=generate_sample_shelf_png(), content_type=None)
    assert isinstance(resp, ApiError)
    assert resp.error.code == "validation"
    assert resp.error.field == "content_type"


# ---------------------------------------------------------------------------
# get_inspection
# ---------------------------------------------------------------------------


def test_get_inspection_returns_stored_report(api) -> None:
    """After create, get_inspection returns the same stored report."""
    created = api.create_inspection(example_mode=True, inspection_id="api-get-001")
    assert isinstance(created, ApiResponse)

    resp = api.get_inspection("api-get-001")
    assert isinstance(resp, ApiResponse)
    _assert_versioned_local(resp)
    assert resp.data.inspection_id == "api-get-001"
    # Same record: byte-for-byte canonical round-trip.
    assert resp.data.canonical_json() == created.data.canonical_json()


def test_get_inspection_missing_id_returns_structured_error(api) -> None:
    """A missing id returns a 404-style structured error envelope, not a raise."""
    resp = api.get_inspection("does-not-exist")

    assert isinstance(resp, ApiError)
    assert not resp.ok
    _assert_versioned_local(resp)
    assert resp.error.code == "not_found"
    assert resp.error.field == "inspection_id"
    assert "does-not-exist" in resp.error.message


def test_get_inspection_blank_id_returns_validation_error(api) -> None:
    """A blank id is a validation error (distinct from not_found)."""
    resp = api.get_inspection("   ")
    assert isinstance(resp, ApiError)
    assert resp.error.code == "validation"


# ---------------------------------------------------------------------------
# list_inspections - status / text / limit
# ---------------------------------------------------------------------------


def test_list_inspections_empty_history_is_empty_envelope(api) -> None:
    """With nothing stored, listing returns an empty, versioned envelope."""
    resp = api.list_inspections()
    assert isinstance(resp, ApiResponse)
    _assert_versioned_local(resp)
    assert resp.data == []


def test_list_inspections_supports_status_text_limit(api) -> None:
    """Create a few reports, then filter by status, text substring, and limit."""
    created = [
        api.create_inspection(example_mode=True, inspection_id=f"api-list-{i}")
        for i in range(3)
    ]
    for c in created:
        assert isinstance(c, ApiResponse)
    status = created[0].data.status

    # Unfiltered: all three present.
    all_resp = api.list_inspections()
    assert isinstance(all_resp, ApiResponse)
    ids = {s.inspection_id for s in all_resp.data}
    assert {"api-list-0", "api-list-1", "api-list-2"}.issubset(ids)
    assert all(isinstance(s, InspectionSummary) for s in all_resp.data)

    # status filter: rows all match the requested status.
    status_resp = api.list_inspections(status=status)
    assert isinstance(status_resp, ApiResponse)
    assert status_resp.data, "at least the created reports share this status"
    assert all(s.status == status for s in status_resp.data)

    # text filter: only ids/recommendations containing the needle.
    text_resp = api.list_inspections(text="api-list-1")
    assert isinstance(text_resp, ApiResponse)
    assert {s.inspection_id for s in text_resp.data} == {"api-list-1"}

    # limit: caps the number of rows returned.
    limit_resp = api.list_inspections(limit=1)
    assert isinstance(limit_resp, ApiResponse)
    assert len(limit_resp.data) == 1


def test_list_inspections_invalid_status_returns_structured_error(api) -> None:
    """An unknown status filter is a structured validation error (Req 10.5)."""
    resp = api.list_inspections(status="bogus")
    assert isinstance(resp, ApiError)
    assert resp.error.code == "validation"


def test_list_inspections_invalid_limit_returns_structured_error(api) -> None:
    """A non-positive limit is a structured validation error."""
    resp = api.list_inspections(limit=0)
    assert isinstance(resp, ApiError)
    assert resp.error.code == "validation"


# ---------------------------------------------------------------------------
# evaluation records - empty, non-fabricated (Req 25.4, 27.2)
# ---------------------------------------------------------------------------


def test_list_evaluations_empty_when_none_exist(api) -> None:
    """With no eval store wired, the listing is an empty (not fabricated) list."""
    resp = api.list_evaluations()
    assert isinstance(resp, ApiResponse)
    _assert_versioned_local(resp)
    assert resp.data == []


def test_get_evaluation_missing_returns_structured_error(api) -> None:
    """Fetching a non-existent eval record returns a structured not_found error."""
    resp = api.get_evaluation("no-such-run")
    assert isinstance(resp, ApiError)
    assert resp.error.code == "not_found"
    assert resp.error.field == "eval_run_id"


def test_evaluation_store_is_honored_when_injected(adapter) -> None:
    """An injected eval store surfaces its measured records unchanged (no faking)."""
    from shelfsentry.data.contracts import EvaluationResult

    record = EvaluationResult(
        eval_run_id="run-001",
        is_demo=True,
        sample_count=2,
        agent_decision_correctness=0.5,
        latency_ms_p50=10.0,
        latency_ms_p95=20.0,
        documented_limitations=["synthetic sample only"],
    )

    class _Store:
        def list_results(self):
            return [record]

        def get_result(self, eval_run_id):
            return record if eval_run_id == "run-001" else None

    api = build_local_api(adapter, eval_store=_Store())

    listed = api.list_evaluations()
    assert isinstance(listed, ApiResponse)
    assert [r.eval_run_id for r in listed.data] == ["run-001"]

    got = api.get_evaluation("run-001")
    assert isinstance(got, ApiResponse)
    assert got.data.eval_run_id == "run-001"
    assert got.data.measured_only is True


# ---------------------------------------------------------------------------
# annotated evidence media fetch (Req 12, Req 9 boundary)
# ---------------------------------------------------------------------------


def test_get_annotated_evidence_returns_stored_bytes(api) -> None:
    """The Results-page media fetch returns the stored annotated evidence bytes."""
    created = api.create_inspection(example_mode=True, inspection_id="api-ev-001")
    assert isinstance(created, ApiResponse)
    finding_id = created.data.initial_finding_id

    resp = api.get_annotated_evidence("api-ev-001", finding_id)
    assert isinstance(resp, ApiResponse)
    _assert_versioned_local(resp)
    assert resp.data.ref.startswith("media://")
    assert resp.data.byte_length > 0
    assert resp.data.data, "evidence bytes must be present"


def test_get_annotated_evidence_missing_returns_structured_error(api) -> None:
    """A missing evidence slot returns a 404-style structured error."""
    api.create_inspection(example_mode=True, inspection_id="api-ev-404")
    resp = api.get_annotated_evidence("api-ev-404", "no-such-finding")
    assert isinstance(resp, ApiError)
    assert resp.error.code == "not_found"
    assert resp.error.field == "finding_id"
