"""Tests for the local persistence layer (task 10.1; Req 15.4, 9).

Covers:

* media put/get round-trip,
* report save/get round-trip,
* ``list_reports`` status filter + text search + limit,
* job save/get round-trip,
* crafted traversal refs (``..``, absolute path, URL, drive letter, NUL byte)
  are rejected / safely scoped and cannot read outside the base dir (Req 9),
* the :class:`StorageMediaResolver` shim satisfies the MCP ``MediaResolver``
  Protocol and round-trips image / frame / evidence,
* the bundled synthetic sample is deterministic own-work (Req 15.4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shelfsentry.data import (
    HistoryQuery,
    InspectionReport,
    InspectionSummary,
    InvalidMediaRef,
    JobRecord,
    LocalStorageAdapter,
    RecordNotFound,
    StorageAdapter,
    StorageMediaResolver,
)
from shelfsentry.mcp import MediaResolver


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def adapter(tmp_path: Path) -> LocalStorageAdapter:
    return LocalStorageAdapter(tmp_path / "store")


def _report(
    inspection_id: str,
    *,
    status: str = "complete",
    recommendation: str = "Shelf looks adequately stocked.",
    created_at: str = "2024-01-01T00:00:00Z",
    requires_human_review: bool = False,
    mode: str = "local",
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
# Protocol conformance
# ---------------------------------------------------------------------------


def test_local_adapter_satisfies_storage_adapter_protocol(adapter: LocalStorageAdapter):
    assert isinstance(adapter, StorageAdapter)


# ---------------------------------------------------------------------------
# Media round-trip
# ---------------------------------------------------------------------------


def test_put_get_media_round_trip(adapter: LocalStorageAdapter):
    data = b"\x89PNG\r\n\x1a\n some bytes \x00\x01\x02"
    ref = adapter.put_media("insp-1", data, "image")
    assert ref.startswith("media://")
    assert adapter.get_media(ref) == data


def test_put_media_mints_distinct_refs(adapter: LocalStorageAdapter):
    ref_a = adapter.put_media("insp-1", b"a", "image")
    ref_b = adapter.put_media("insp-1", b"b", "frame:1")
    assert ref_a != ref_b
    assert adapter.get_media(ref_a) == b"a"
    assert adapter.get_media(ref_b) == b"b"


def test_get_missing_media_raises_record_not_found(adapter: LocalStorageAdapter):
    # A well-formed but never-stored token resolves to no file.
    good_token = "0" * 32
    with pytest.raises(RecordNotFound):
        adapter.get_media(f"media://{good_token}")


# ---------------------------------------------------------------------------
# Report round-trip
# ---------------------------------------------------------------------------


def test_save_get_report_round_trip(adapter: LocalStorageAdapter):
    report = _report("insp-42", recommendation="Two gaps flagged for review.")
    adapter.save_report(report)
    loaded = adapter.get_report("insp-42")
    assert loaded is not None
    assert loaded == report
    assert loaded.canonical_json() == report.canonical_json()


def test_get_missing_report_returns_none(adapter: LocalStorageAdapter):
    assert adapter.get_report("does-not-exist") is None


def test_save_report_overwrites(adapter: LocalStorageAdapter):
    adapter.save_report(_report("insp-1", status="needs_review"))
    adapter.save_report(_report("insp-1", status="complete"))
    loaded = adapter.get_report("insp-1")
    assert loaded is not None
    assert loaded.status == "complete"


# ---------------------------------------------------------------------------
# list_reports: status filter + text search + limit
# ---------------------------------------------------------------------------


def test_list_reports_empty_store(adapter: LocalStorageAdapter):
    assert adapter.list_reports(HistoryQuery()) == []


def test_list_reports_status_filter(adapter: LocalStorageAdapter):
    adapter.save_report(_report("a", status="complete", created_at="2024-01-01T00:00:00Z"))
    adapter.save_report(_report("b", status="needs_review", created_at="2024-01-02T00:00:00Z"))
    adapter.save_report(_report("c", status="failed", created_at="2024-01-03T00:00:00Z"))

    needs = adapter.list_reports(HistoryQuery(status="needs_review"))
    assert [s.inspection_id for s in needs] == ["b"]
    assert all(isinstance(s, InspectionSummary) for s in needs)


def test_list_reports_text_search_matches_id_and_recommendation(adapter: LocalStorageAdapter):
    adapter.save_report(
        _report("alpha-1", recommendation="Restock the cereal aisle.", created_at="2024-01-01T00:00:00Z")
    )
    adapter.save_report(
        _report("beta-2", recommendation="No action needed.", created_at="2024-01-02T00:00:00Z")
    )

    # Matches on recommendation text (case-insensitive).
    by_text = adapter.list_reports(HistoryQuery(text="CEREAL"))
    assert [s.inspection_id for s in by_text] == ["alpha-1"]

    # Matches on inspection id.
    by_id = adapter.list_reports(HistoryQuery(text="beta"))
    assert [s.inspection_id for s in by_id] == ["beta-2"]


def test_list_reports_limit_and_newest_first(adapter: LocalStorageAdapter):
    adapter.save_report(_report("old", created_at="2024-01-01T00:00:00Z"))
    adapter.save_report(_report("mid", created_at="2024-02-01T00:00:00Z"))
    adapter.save_report(_report("new", created_at="2024-03-01T00:00:00Z"))

    limited = adapter.list_reports(HistoryQuery(limit=2))
    assert [s.inspection_id for s in limited] == ["new", "mid"]


def test_list_reports_combined_filter_search_limit(adapter: LocalStorageAdapter):
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
    assert [s.inspection_id for s in result] == ["x2"]  # newest complete+gap


# ---------------------------------------------------------------------------
# Job round-trip
# ---------------------------------------------------------------------------


def test_save_get_job_round_trip(adapter: LocalStorageAdapter):
    job = _job("job-abc", status="running")
    adapter.save_job(job)
    loaded = adapter.get_job("job-abc")
    assert loaded is not None
    assert loaded == job


def test_get_missing_job_returns_none(adapter: LocalStorageAdapter):
    assert adapter.get_job("no-such-job") is None


def test_save_job_overwrites_status(adapter: LocalStorageAdapter):
    adapter.save_job(_job("j", status="queued"))
    adapter.save_job(_job("j", status="succeeded"))
    loaded = adapter.get_job("j")
    assert loaded is not None
    assert loaded.status == "succeeded"


# ---------------------------------------------------------------------------
# Security boundary: crafted traversal refs are rejected / safely scoped
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
        "file:///etc/passwd",  # wrong scheme
        "/etc/passwd",  # no scheme
        "../../etc/passwd",
    ],
)
def test_crafted_media_ref_is_rejected(adapter: LocalStorageAdapter, crafted: str):
    with pytest.raises(InvalidMediaRef):
        adapter.get_media(crafted)


def test_crafted_ref_cannot_read_outside_base_dir(adapter: LocalStorageAdapter, tmp_path: Path):
    # Plant a secret file OUTSIDE the store base dir.
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"TOP SECRET")

    # Build refs whose token tries to climb out of media/ to reach the secret.
    rel = "..\\..\\secret.txt" if "\\" in str(tmp_path) else "../../secret.txt"
    for crafted in (
        f"media://{rel}",
        f"media://{secret}",  # absolute path as a token
        "media://../secret.txt",
    ):
        with pytest.raises(InvalidMediaRef):
            adapter.get_media(crafted)

    # Sanity: the secret still exists and was never surfaced via the adapter.
    assert secret.read_bytes() == b"TOP SECRET"


@pytest.mark.parametrize(
    "bad_id",
    ["../escape", "..\\escape", "a/b", "a\\b", "with:colon", "", "with\x00nul"],
)
def test_crafted_inspection_id_for_report_is_rejected(adapter: LocalStorageAdapter, bad_id: str):
    report = _report("placeholder")
    # Force the unsafe id onto the model via a copy so validation at the storage
    # boundary (not the contract) is what rejects it.
    object.__setattr__(report, "inspection_id", bad_id)
    with pytest.raises(InvalidMediaRef):
        adapter.save_report(report)


def test_crafted_job_id_is_rejected(adapter: LocalStorageAdapter):
    job = _job("placeholder")
    object.__setattr__(job, "job_id", "../../evil")
    with pytest.raises(InvalidMediaRef):
        adapter.save_job(job)


# ---------------------------------------------------------------------------
# MediaResolver shim: satisfies the Protocol and round-trips media
# ---------------------------------------------------------------------------


def test_resolver_shim_satisfies_media_resolver_protocol(adapter: LocalStorageAdapter):
    resolver = StorageMediaResolver(adapter)
    assert isinstance(resolver, MediaResolver)


def test_resolver_image_round_trip(adapter: LocalStorageAdapter):
    resolver = StorageMediaResolver(adapter)
    resolver.put_image("insp-1", b"image-bytes")
    assert resolver.get_image("insp-1") == b"image-bytes"


def test_resolver_frame_round_trip(adapter: LocalStorageAdapter):
    resolver = StorageMediaResolver(adapter)
    resolver.put_frame("insp-1", "f0", b"frame-zero")
    resolver.put_frame("insp-1", "f1", b"frame-one")
    assert resolver.get_frame("insp-1", "f0") == b"frame-zero"
    assert resolver.get_frame("insp-1", "f1") == b"frame-one"


def test_resolver_evidence_round_trip(adapter: LocalStorageAdapter):
    resolver = StorageMediaResolver(adapter)
    assert resolver.get_evidence_ref("insp-1", "finding-1") is None
    ref = resolver.put_evidence("insp-1", "finding-1", b"annotated-png")
    assert ref.startswith("media://")
    assert resolver.get_evidence_ref("insp-1", "finding-1") == ref
    assert adapter.get_media(ref) == b"annotated-png"


def test_resolver_missing_image_raises(adapter: LocalStorageAdapter):
    resolver = StorageMediaResolver(adapter)
    with pytest.raises(RecordNotFound):
        resolver.get_image("unknown")


def test_resolver_missing_frame_raises(adapter: LocalStorageAdapter):
    resolver = StorageMediaResolver(adapter)
    with pytest.raises(RecordNotFound):
        resolver.get_frame("insp-1", "no-frame")


def test_resolver_media_is_inspection_scoped(adapter: LocalStorageAdapter):
    resolver = StorageMediaResolver(adapter)
    resolver.put_image("insp-a", b"a-img")
    resolver.put_image("insp-b", b"b-img")
    assert resolver.get_image("insp-a") == b"a-img"
    assert resolver.get_image("insp-b") == b"b-img"


# ---------------------------------------------------------------------------
# Bundled sample inputs (Req 15.4)
# ---------------------------------------------------------------------------


def test_bundled_sample_is_deterministic_own_work():
    from shelfsentry.samples import ensure_samples, generate_sample_shelf_png

    path = ensure_samples()
    assert path.exists()
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG signature
    # Regeneration is byte-identical (deterministic, own-work).
    assert data == generate_sample_shelf_png()


def test_bundled_sample_has_provenance_note():
    from shelfsentry.samples import SAMPLES_DIR

    note = SAMPLES_DIR / "PROVENANCE.md"
    assert note.exists()
    text = note.read_text(encoding="utf-8").lower()
    assert "own work" in text or "public domain" in text
