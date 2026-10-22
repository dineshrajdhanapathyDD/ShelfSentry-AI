"""Framework-agnostic API handlers backing the frontend (task 11.1; Req 10.4, 16.1).

This module is the thin request-handling layer between the Frontend and the
mode-agnostic core. Each handler validates its request, routes to the
end-to-end pipeline (:func:`shelfsentry.pipeline.run_local_inspection`),
persists/reads through a :class:`~shelfsentry.data.storage.StorageAdapter`, and
returns a **versioned, mode-stamped envelope** (:mod:`shelfsentry.api.envelope`;
Req 10.4). Validation and missing-record failures are *returned* as structured
error envelopes rather than raised across the boundary (Req 10.5).

Design seam for AWS mode (Req 16.1). The handlers depend only on injected
collaborators:

* a :class:`StorageAdapter` (the persistence seam; local JSON/file store today,
  the ``AwsStorageAdapter`` over S3 + DynamoDB in AWS mode - unchanged here), and
* a ``runner`` callable with the signature of
  :func:`shelfsentry.pipeline.run_local_inspection` (the local deterministic
  pipeline today; an AWS/Bedrock-backed runner later), and
* an optional ``eval_store`` for :class:`EvaluationResult` retrieval.

Because those collaborators are injected, the same handler set targets AWS
resources in AWS mode simply by constructing it with the AWS adapter + runner
and ``mode="aws"`` - no handler code changes. The local factory
:func:`build_local_api` wires the local defaults and stamps ``mode="local"``.

Import-light / no AWS. This module imports no ``boto3`` and opens no socket;
the only I/O is through the injected adapter (local filesystem by default). The
pipeline is imported lazily inside the factory's default runner so merely
importing the handler surface stays cheap.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from pydantic import ValidationError

from shelfsentry.api.envelope import (
    ApiError,
    ApiMode,
    ApiResponse,
    error_from_input_validation,
    error_from_validation_error,
    not_found_error,
)
from shelfsentry.data.contracts import EvaluationResult, InspectionReport
from shelfsentry.data.storage import (
    HistoryQuery,
    InspectionSummary,
    LocalStorageAdapter,
    StorageAdapter,
    StorageMediaResolver,
)
from shelfsentry.vision.errors import InputValidationError

__all__ = [
    "InspectionRunner",
    "EvalStore",
    "LocalApi",
    "build_local_api",
]


class InspectionRunner(Protocol):
    """The injected pipeline entry point (matches ``run_local_inspection``).

    Keeping the runner behind a Protocol is the AWS seam (Req 16.1): local mode
    injects the deterministic local pipeline; AWS mode injects a runner that
    drives the same flow against Bedrock + AWS storage.
    """

    def __call__(
        self,
        data: bytes,
        content_type: str,
        *,
        adapter: Any,
        inspection_id: str | None = ...,
        mode: Any = ...,
    ) -> InspectionReport: ...


class EvalStore(Protocol):
    """Read-only retrieval of measured :class:`EvaluationResult` records.

    The :class:`StorageAdapter` seam does not persist evaluation records (those
    are produced by the eval runner, task 14), so the API layer takes an
    optional eval store. When none is wired, evaluation listings return an
    **empty** envelope - never a fabricated record (Req 25.4, 27.2).
    """

    def list_results(self) -> list[EvaluationResult]: ...

    def get_result(self, eval_run_id: str) -> EvaluationResult | None: ...


class _EmptyEvalStore:
    """Default eval store: no records yet, so every listing is empty.

    Returning an empty list honestly reflects that no measured evaluation has
    been recorded; it never invents a record (Req 25.4, 27.2).
    """

    def list_results(self) -> list[EvaluationResult]:
        return []

    def get_result(self, eval_run_id: str) -> EvaluationResult | None:
        return None


class _AnnotatedEvidence:
    """A small, serialization-friendly descriptor for stored annotated evidence.

    The Results page needs the annotated image for a finding. Rather than stream
    raw bytes through the typed envelope, the media-fetch handler returns this
    descriptor carrying the opaque storage ``ref``, the ``byte_length``, and the
    bytes themselves (as ``data``) so a transport adapter can serve them while a
    test can assert on length/ref without decoding. It is intentionally a plain
    object (not a contract) because it is transport data, not a core contract.
    """

    __slots__ = ("inspection_id", "finding_id", "ref", "data", "byte_length")

    def __init__(
        self, inspection_id: str, finding_id: str, ref: str, data: bytes
    ) -> None:
        self.inspection_id = inspection_id
        self.finding_id = finding_id
        self.ref = ref
        self.data = data
        self.byte_length = len(data)


class LocalApi:
    """The handler set backing the frontend in local mode (Req 10.4, 16.1).

    Construct via :func:`build_local_api` for local defaults, or directly with a
    custom adapter/runner/eval store (and ``mode`` label) to target AWS
    resources without changing handler code.
    """

    def __init__(
        self,
        adapter: StorageAdapter,
        *,
        runner: InspectionRunner,
        eval_store: EvalStore | None = None,
        mode: ApiMode = "local",
        sample_provider: Callable[[], tuple[bytes, str]] | None = None,
    ) -> None:
        """Wire the handler set to its injected collaborators.

        :param adapter: persistence seam (local or AWS) used to read/write
            reports and resolve stored media.
        :param runner: the pipeline entry point (``run_local_inspection`` locally).
        :param eval_store: optional measured-only evaluation record source;
            defaults to an empty store that never fabricates records.
        :param mode: the operating-mode label stamped on every envelope and
            passed through to the runner.
        :param sample_provider: returns ``(bytes, content_type)`` for the bundled
            example input when a request uses example mode; defaults to the
            synthetic shelf PNG from :mod:`shelfsentry.samples`.
        """
        self._adapter = adapter
        self._runner = runner
        self._eval_store = eval_store or _EmptyEvalStore()
        self._mode: ApiMode = mode
        self._sample_provider = sample_provider or _default_sample_provider
        # A resolver over the adapter lets the evidence handler fetch stored
        # media by inspection-scoped id without handling a path (Req 9).
        self._resolver = (
            StorageMediaResolver(adapter)
            if isinstance(adapter, LocalStorageAdapter)
            else None
        )

    # -- create_inspection ---------------------------------------------------

    def create_inspection(
        self,
        *,
        media: bytes | None = None,
        content_type: str | None = None,
        example_mode: bool = False,
        inspection_id: str | None = None,
    ) -> ApiResponse[InspectionReport] | ApiError:
        """Run and persist an inspection, returning the report in an envelope.

        In ``example_mode`` the bundled synthetic sample is used and ``media`` /
        ``content_type`` are ignored. Otherwise the request must carry non-empty
        ``media`` bytes and a ``content_type``; a missing/empty request is a
        structured ``validation`` error, and the pipeline's own
        :class:`InputValidationError` (bad type, empty, too large, undecodable)
        is mapped to the same structured envelope (Req 10.5) - never raised.
        """
        if example_mode:
            media, content_type = self._sample_provider()
        else:
            if not isinstance(media, (bytes, bytearray)) or len(media) == 0:
                return self._validation_error(
                    "Request media is required and must be non-empty bytes.",
                    field="media",
                )
            media = bytes(media)
            if not content_type or not str(content_type).strip():
                return self._validation_error(
                    "Request content_type is required.", field="content_type"
                )

        try:
            report = self._runner(
                media,
                content_type,
                adapter=self._adapter,
                inspection_id=inspection_id,
                mode=self._mode,
            )
        except InputValidationError as exc:  # structured input failure (Req 1)
            return error_from_input_validation(exc, mode=self._mode)
        except ValidationError as exc:  # boundary schema failure (Req 10.5)
            return error_from_validation_error(exc, mode=self._mode)

        return self._ok(report)

    # -- get_inspection ------------------------------------------------------

    def get_inspection(
        self, inspection_id: str
    ) -> ApiResponse[InspectionReport] | ApiError:
        """Return the stored report, or a 404-style structured error.

        A missing report (unknown or malformed id) returns a ``not_found`` error
        envelope rather than raising, so the Results page can render an honest
        empty/error state (Req 10.5).
        """
        if not inspection_id or not str(inspection_id).strip():
            return self._validation_error(
                "inspection_id is required.", field="inspection_id"
            )
        report = self._adapter.get_report(inspection_id)
        if report is None:
            return not_found_error(
                f"No inspection found for id {inspection_id!r}.",
                mode=self._mode,
                field="inspection_id",
            )
        return self._ok(report)

    # -- list_inspections ----------------------------------------------------

    def list_inspections(
        self,
        *,
        status: str | None = None,
        text: str | None = None,
        limit: int | None = None,
    ) -> ApiResponse[list[InspectionSummary]] | ApiError:
        """History listing with status/text/limit filters (Req 13.1, 13.2).

        The query is validated into a :class:`HistoryQuery`; an invalid filter
        (unknown status, non-positive limit) returns a structured ``validation``
        error. Results are newest-first summaries from the adapter.
        """
        try:
            query = HistoryQuery(status=status, text=text, limit=limit)
        except ValidationError as exc:
            return error_from_validation_error(exc, mode=self._mode)
        summaries = self._adapter.list_reports(query)
        return self._ok(summaries)

    # -- evaluation records --------------------------------------------------

    def list_evaluations(self) -> ApiResponse[list[EvaluationResult]] | ApiError:
        """Return measured evaluation records, or an empty list when none exist.

        Never fabricates a record: with no eval store wired (or no runs yet) the
        envelope carries an empty list (Req 25.4, 27.2).
        """
        return self._ok(self._eval_store.list_results())

    def get_evaluation(
        self, eval_run_id: str
    ) -> ApiResponse[EvaluationResult] | ApiError:
        """Return one measured evaluation record, or a 404-style error."""
        if not eval_run_id or not str(eval_run_id).strip():
            return self._validation_error(
                "eval_run_id is required.", field="eval_run_id"
            )
        result = self._eval_store.get_result(eval_run_id)
        if result is None:
            return not_found_error(
                f"No evaluation record found for id {eval_run_id!r}.",
                mode=self._mode,
                field="eval_run_id",
            )
        return self._ok(result)

    # -- annotated evidence (media fetch for the Results page) ---------------

    def get_annotated_evidence(
        self, inspection_id: str, finding_id: str
    ) -> ApiResponse[_AnnotatedEvidence] | ApiError:
        """Return stored annotated-evidence bytes for a finding (Req 12).

        Resolves the evidence by inspection-scoped id through the storage
        resolver (never a path; Req 9) and returns a small descriptor carrying
        the opaque ref, byte length, and the bytes. A missing evidence slot
        returns a ``not_found`` structured error.
        """
        if not inspection_id or not str(inspection_id).strip():
            return self._validation_error(
                "inspection_id is required.", field="inspection_id"
            )
        if not finding_id or not str(finding_id).strip():
            return self._validation_error(
                "finding_id is required.", field="finding_id"
            )
        if self._resolver is None:
            return not_found_error(
                "Annotated evidence retrieval is not available for this adapter.",
                mode=self._mode,
                field="finding_id",
            )
        ref = self._resolver.get_evidence_ref(inspection_id, finding_id)
        if ref is None:
            return not_found_error(
                f"No annotated evidence for finding {finding_id!r} in "
                f"inspection {inspection_id!r}.",
                mode=self._mode,
                field="finding_id",
            )
        try:
            data = self._adapter.get_media(ref)  # type: ignore[attr-defined]
        except Exception:  # missing bytes behind a stale ref -> not found
            return not_found_error(
                f"Annotated evidence bytes are unavailable for finding "
                f"{finding_id!r}.",
                mode=self._mode,
                field="finding_id",
            )
        return self._ok(
            _AnnotatedEvidence(inspection_id, finding_id, ref, data)
        )

    # -- internal envelope helpers -------------------------------------------

    def _ok(self, data: Any) -> ApiResponse[Any]:
        """Wrap ``data`` in a versioned, mode-stamped success envelope."""
        return ApiResponse(mode=self._mode, data=data)

    def _validation_error(self, message: str, *, field: str | None) -> ApiError:
        from shelfsentry.api.envelope import ApiErrorDetail

        return ApiError(
            mode=self._mode,
            error=ApiErrorDetail(code="validation", message=message, field=field),
        )


# ---------------------------------------------------------------------------
# Lazy pipeline / sample wiring (keeps import light; no boto3)
# ---------------------------------------------------------------------------


def _default_runner(
    data: bytes,
    content_type: str,
    *,
    adapter: Any,
    inspection_id: str | None = None,
    mode: Any = "local",
) -> InspectionReport:
    """Default runner: the deterministic local pipeline.

    Imported lazily so importing the handler surface does not pull in the vision
    stack until an inspection is actually run.
    """
    from shelfsentry.pipeline import run_local_inspection

    return run_local_inspection(
        data,
        content_type,
        adapter=adapter,
        inspection_id=inspection_id,
        mode=mode,
    )


def _default_sample_provider() -> tuple[bytes, str]:
    """Return the bundled synthetic sample image bytes + its content type.

    Uses :func:`shelfsentry.samples.generate_sample_shelf_png` so example mode is
    fully offline, deterministic, and own-work (Req 15.1, 15.2).
    """
    from shelfsentry.samples import generate_sample_shelf_png

    return generate_sample_shelf_png(), "image/png"


def build_local_api(
    adapter: StorageAdapter,
    *,
    runner: InspectionRunner | None = None,
    eval_store: EvalStore | None = None,
    sample_provider: Callable[[], tuple[bytes, str]] | None = None,
) -> LocalApi:
    """Build the local handler set over ``adapter`` (Req 16.1).

    Wires the local defaults: the deterministic local pipeline as the runner,
    an empty (non-fabricating) eval store, the bundled synthetic sample for
    example mode, and ``mode="local"``. Inject a custom ``runner``/``eval_store``
    to target other backends; an AWS build simply constructs :class:`LocalApi`
    with the AWS adapter/runner and ``mode="aws"`` instead.
    """
    return LocalApi(
        adapter,
        runner=runner or _default_runner,
        eval_store=eval_store,
        mode="local",
        sample_provider=sample_provider,
    )
