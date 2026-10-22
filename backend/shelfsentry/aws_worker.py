"""AWS-mode inspection runner + async worker entry point (task 16.2).

This module wires the AWS side of the system (Req 16.2, 16.3, 16.4, 19.4, 19.5)
onto the *same* mode-agnostic orchestration seam the local pipeline uses
(:mod:`shelfsentry.pipeline`). It adds nothing to the perception/decision logic;
it swaps the two injectable dependencies the pipeline was always designed to
accept and layers AWS-specific job tracking and retry/failure handling on top:

* storage: an :class:`~shelfsentry.data.aws_storage.AwsStorageAdapter` (S3 for
  media, DynamoDB for records) in place of the local adapter; and
* decision model: a :class:`~shelfsentry.agent.bedrock.BedrockDecisionModel` as
  the PRIMARY decision path (Req 16.2) in place of the deterministic policy.

Because the AWS adapter satisfies the identical ``StorageAdapter`` Protocol and
supports :meth:`get_media_ref_for`, the existing
:class:`~shelfsentry.data.storage.StorageMediaResolver` shim backs the MCP
dispatcher unchanged - so the runner reuses the pipeline's real seeding and
tool-client helpers (``_seed_from_image`` / ``_seed_from_video`` /
``DispatcherToolClient``) rather than re-implementing the perception loop.

Primary Bedrock path with recorded fallback (Req 16.2, 7.1, 7.2).
    The runner passes the ``BedrockDecisionModel`` as the loop's ``model``.
    When Bedrock answers, it drives the action selection. When it is
    unavailable, misconfigured, or returns something unparseable, it raises a
    :class:`BedrockDecisionError`; the loop's built-in ``fallback_on_error``
    seam catches that, engages the deterministic
    :class:`~shelfsentry.agent.fallback.FallbackPolicy` for that iteration, and
    records ``used_fallback=True`` with a reason on the decision. The runner
    does not implement its own fallback - it relies on the loop's documented
    seam so the provenance of the switch stays honest.

Never a mock result in AWS mode (Req 16.4).
    The runner stamps ``mode="aws"`` on the report and always runs the real
    OpenCV 5 vision through the MCP dispatcher and the real bounded loop. There
    is no code path here that substitutes a canned/mock finding or report; a
    primary-model outage degrades only to the *deterministic policy* decision
    (still a real, finding-derived decision), never to fabricated results.

Retry / failure handling (Req 16.3).
    A configured :class:`~shelfsentry.agent.retry.RetryPolicy` wraps the
    fallible AWS operations performed during an inspection (seeding, which
    reads/writes S3 + DynamoDB, and persisting the final report). A transient
    failure that recovers within the attempt budget is retried transparently;
    when the budget is exhausted the runner raises
    :class:`~shelfsentry.agent.retry.RetryExhausted`, which the async worker
    turns into a FAILED job status (and which callers of
    :func:`run_aws_inspection` may catch to surface a failed
    ``InspectionReport``). The policy's sleeper is injectable so tests assert
    the attempt count without real sleeping.

Async worker entry point (Req 19.4, 19.5).
    :func:`run_async_job` is the worker entry point for longer video jobs. It
    records a :class:`~shelfsentry.data.storage.JobRecord` in DynamoDB and
    transitions it ``queued`` -> ``running`` -> ``succeeded``/``failed``,
    persisting *each* transition through ``AwsStorageAdapter.save_job`` so the
    status can be polled (``get_job``). The actual inspection work is delegated
    to :func:`run_aws_inspection`; a failure (including exhausted retries) is
    recorded as a ``failed`` job carrying the error, never swallowed.

Import-light / lazy. This module imports only typed contracts, the storage
Protocol types, the loop, and the pipeline helpers at module top - no ``boto3``
and no ``strands``. The AWS adapter and the Bedrock model both lazy-import
``boto3`` only when a real client is actually needed, so importing
:mod:`shelfsentry.aws_worker` never requires AWS libraries or credentials.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from shelfsentry.agent.bedrock import BedrockDecisionModel
from shelfsentry.agent.decision import DecisionModel
from shelfsentry.agent.loop import run_agent_loop
from shelfsentry.agent.retry import RetryExhausted, RetryPolicy
from shelfsentry.config.settings import AgentConfig, VisionConfig
from shelfsentry.data.contracts import InspectionReport, VisionFinding
from shelfsentry.data.storage import JobRecord, StorageMediaResolver
from shelfsentry.mcp import VisionToolDispatcher
from shelfsentry.pipeline import (
    DispatcherToolClient,
    PipelineError,
    _seed_from_image,
    _seed_from_video,
)
from shelfsentry.vision.validate import media_kind_for_content_type, validate_input

__all__ = [
    "AwsInspectionError",
    "run_aws_inspection",
    "run_async_job",
]


class AwsInspectionError(RuntimeError):
    """An AWS-mode inspection could not produce a persisted report.

    Distinct from the structured input-validation error (which surfaces to the
    caller unchanged) and from :class:`~shelfsentry.agent.retry.RetryExhausted`
    (a transient-failure budget exhaustion). Used for orchestration-level AWS
    failures that are not simply "retries ran out".
    """


# ---------------------------------------------------------------------------
# The AWS-mode inspection runner (Req 16.2, 16.3, 16.4)
# ---------------------------------------------------------------------------


def run_aws_inspection(
    data: bytes,
    content_type: str,
    *,
    inspection_id: str | None = None,
    adapter: Any,
    cfg_vision: VisionConfig | None = None,
    cfg_agent: AgentConfig | None = None,
    model: DecisionModel | None = None,
    tool_client: Any | None = None,
    retry_policy: RetryPolicy | None = None,
) -> InspectionReport:
    """Run a full AWS-mode inspection and persist its report (mode ``"aws"``).

    Mirrors the local pipeline's orchestration (validate -> store media ->
    perceive via MCP -> bounded loop -> save report) but uses the AWS storage
    adapter and the Bedrock decision model as the PRIMARY decision path
    (Req 16.2). It NEVER substitutes a mock result: the report is stamped
    ``mode="aws"`` and every stage runs real OpenCV 5 + the real loop; a Bedrock
    outage degrades only to the loop's recorded deterministic fallback, not to
    fabricated output (Req 16.4).

    :param data: raw media bytes (image or short video).
    :param content_type: declared content type (e.g. ``"image/png"`` /
        ``"video/mp4"``); used for validation and to pick the image vs video
        seeding path.
    :param inspection_id: id for this inspection; a fresh id is generated when
        omitted.
    :param adapter: an :class:`~shelfsentry.data.aws_storage.AwsStorageAdapter`
        (or any ``StorageAdapter`` the :class:`StorageMediaResolver` accepts -
        tests may inject a fake adapter or the real one with fake S3/DDB
        clients).
    :param cfg_vision: vision config; defaults to :class:`VisionConfig`.
    :param cfg_agent: agent config; defaults to :class:`AgentConfig`.
    :param model: the PRIMARY decision model. Defaults to a
        :class:`BedrockDecisionModel` (lazy boto3; built only when a decision is
        actually requested). Tests inject an ``InjectedDecisionModel`` to
        simulate Bedrock-available, or a raising model to force the loop's
        recorded fallback.
    :param tool_client: optional pre-built :class:`ToolClient`; defaults to a
        :class:`DispatcherToolClient` over the MCP dispatcher for this
        inspection. Mainly an injection seam for tests.
    :param retry_policy: the :class:`RetryPolicy` applied to the fallible AWS
        operations (seeding + report persistence). Defaults to a 3-attempt
        exponential-backoff policy; inject one with a fake sleeper in tests.
    :returns: the persisted :class:`InspectionReport` (``mode == "aws"``).
    :raises shelfsentry.vision.errors.InputValidationError: on invalid input.
    :raises RetryExhausted: when a wrapped AWS operation never recovers within
        the retry budget (Req 16.3). The async worker converts this to a failed
        job.
    :raises AwsInspectionError: if a video yields no analyzable frame.
    """
    cfg_vision = cfg_vision or VisionConfig()
    cfg_agent = cfg_agent or AgentConfig()
    inspection_id = inspection_id or _new_inspection_id()
    retry_policy = retry_policy or RetryPolicy()
    # Default PRIMARY path is Bedrock (Req 16.2). Lazy: no boto3 at import, and
    # no client built until the loop actually asks for a decision.
    model = model if model is not None else BedrockDecisionModel()

    # 1. Validate the raw input (Req 1). A validation error is not a transient
    #    AWS failure, so it is NOT wrapped in the retry policy - it surfaces
    #    immediately and unchanged.
    validate_input(data, content_type, cfg_vision)

    # The AWS adapter satisfies the same StorageAdapter Protocol and supports
    # get_media_ref_for, so the existing resolver shim + MCP dispatcher back the
    # perception seam unchanged (no AWS-specific resolver needed).
    resolver = StorageMediaResolver(adapter)
    dispatcher = VisionToolDispatcher(resolver, cfg_vision)

    # 2 + 3. Store media and produce the initial finding via the MCP dispatcher.
    #    These touch S3/DynamoDB, so they run under the retry policy (Req 16.3).
    kind = media_kind_for_content_type(content_type)
    if kind == "video":
        initial_finding, frame_ids = _retry_seed_video(
            retry_policy,
            data,
            dispatcher,
            resolver,
            inspection_id=inspection_id,
            cfg=cfg_vision,
        )
    else:
        initial_finding = _retry_seed_image(
            retry_policy, data, dispatcher, resolver, inspection_id=inspection_id
        )
        frame_ids = []

    # 4. Run the bounded agent loop with Bedrock as the primary model and the
    #    loop's built-in deterministic fallback recording used_fallback on a
    #    Bedrock error (Req 16.2 primary path, 7.2 recorded fallback). The tool
    #    client maps the two tool-backed actions onto the MCP dispatcher.
    tools = tool_client or DispatcherToolClient(
        dispatcher, inspection_id=inspection_id, frame_ids=frame_ids
    )
    report = run_agent_loop(
        initial_finding,
        model,
        tools,
        cfg_agent,
        inspection_id=inspection_id,
        mode="aws",
    )

    # 5. Persist the final report (also an AWS write -> under the retry policy).
    retry_policy.run(
        lambda: adapter.save_report(report), description="save_report (DynamoDB)"
    )
    return report


# ---------------------------------------------------------------------------
# Retry-wrapped seeding helpers (reuse the pipeline's real seeding logic)
# ---------------------------------------------------------------------------


def _retry_seed_image(
    retry_policy: RetryPolicy,
    data: bytes,
    dispatcher: VisionToolDispatcher,
    resolver: StorageMediaResolver,
    *,
    inspection_id: str,
) -> VisionFinding:
    """Seed the initial image finding under the retry policy (Req 16.3).

    Delegates to the pipeline's :func:`_seed_from_image` (store original image
    in S3 + analyze via MCP), wrapped so a transient S3/DynamoDB failure is
    retried and an exhausted budget surfaces as ``RetryExhausted``.
    """
    return retry_policy.run(
        lambda: _seed_from_image(
            data, dispatcher, resolver, inspection_id=inspection_id
        ),
        description="seed image (S3 put + analyze)",
    )


def _retry_seed_video(
    retry_policy: RetryPolicy,
    data: bytes,
    dispatcher: VisionToolDispatcher,
    resolver: StorageMediaResolver,
    *,
    inspection_id: str,
    cfg: VisionConfig,
) -> tuple[VisionFinding, list[str]]:
    """Seed the initial video finding + frame ids under the retry policy.

    Delegates to the pipeline's :func:`_seed_from_video` (sample + store frames
    in S3, seed from frame one), wrapped for retry. A video that yields no
    analyzable frame is a :class:`PipelineError`, which is NOT a transient AWS
    failure; it is re-raised as :class:`AwsInspectionError` without consuming
    the retry budget.
    """

    def _seed() -> tuple[VisionFinding, list[str]]:
        try:
            return _seed_from_video(
                data, dispatcher, resolver, inspection_id=inspection_id, cfg=cfg
            )
        except PipelineError as exc:
            raise AwsInspectionError(str(exc)) from exc

    # AwsInspectionError is not an AWS transient; the default predicate would
    # retry it, so use a policy variant that does not retry it.
    non_retry = (AwsInspectionError,)
    return _run_except(retry_policy, _seed, non_retry, "seed video (S3 puts + analyze)")


def _run_except(
    retry_policy: RetryPolicy,
    operation: Callable[[], Any],
    non_retryable: tuple[type[BaseException], ...],
    description: str,
) -> Any:
    """Run ``operation`` under ``retry_policy`` but never retry ``non_retryable``.

    A small shim so a deterministic orchestration error (e.g. "no analyzable
    frame") is raised immediately rather than being retried as if it were a
    transient AWS fault. Transient faults still get the configured backoff.
    """
    base_predicate = retry_policy._retry_on  # noqa: SLF001 - intentional reuse

    def _predicate(exc: BaseException) -> bool:
        if isinstance(exc, non_retryable):
            return False
        return base_predicate(exc)

    scoped = RetryPolicy(
        max_attempts=retry_policy.max_attempts,
        base_delay=retry_policy.base_delay,
        multiplier=retry_policy.multiplier,
        max_delay=retry_policy.max_delay,
        sleep=retry_policy._sleep,  # noqa: SLF001 - preserve the injected sleeper
        retry_on=_predicate,
    )
    return scoped.run(operation, description=description)


# ---------------------------------------------------------------------------
# The async worker entry point (Req 19.4, 19.5)
# ---------------------------------------------------------------------------


def run_async_job(
    job_id: str,
    inspection_id: str,
    data: bytes,
    content_type: str,
    *,
    adapter: Any,
    cfg_vision: VisionConfig | None = None,
    cfg_agent: AgentConfig | None = None,
    model: DecisionModel | None = None,
    retry_policy: RetryPolicy | None = None,
    now: Callable[[], str] | None = None,
) -> JobRecord:
    """Run one async inspection job, recording its status in DynamoDB (Req 19.4).

    This is the worker entry point for longer video jobs. It records a
    :class:`JobRecord` and transitions it through its lifecycle, persisting EACH
    transition via ``adapter.save_job`` so the status is pollable with
    ``adapter.get_job`` (Req 19.5):

    * ``queued`` - persisted immediately on entry;
    * ``running`` - persisted just before the inspection work starts;
    * ``succeeded`` - persisted after :func:`run_aws_inspection` returns a
      report; or
    * ``failed`` - persisted (with the error detail) if the inspection raises,
      including :class:`RetryExhausted` when the retry budget is exhausted
      (Req 16.3). The failure is recorded, never swallowed.

    :param job_id: the id of the job record to create/track.
    :param inspection_id: the inspection this job runs; stamped on the report.
    :param data: raw media bytes for the inspection.
    :param content_type: declared content type.
    :param adapter: the AWS storage adapter (or a compatible fake) used for both
        the inspection and the job-record persistence.
    :param cfg_vision: vision config forwarded to the inspection.
    :param cfg_agent: agent config forwarded to the inspection.
    :param model: the primary decision model forwarded to the inspection
        (defaults to Bedrock inside :func:`run_aws_inspection`).
    :param retry_policy: the retry policy forwarded to the inspection (Req 16.3).
    :param now: an injectable UTC-ISO timestamp factory (defaults to the real
        clock); injected in tests for deterministic timestamps.
    :returns: the terminal :class:`JobRecord` (``succeeded`` or ``failed``).
    """
    clock = now or _utc_now_iso

    created = clock()
    job = JobRecord(
        job_id=job_id,
        inspection_id=inspection_id,
        status="queued",
        created_at=created,
        updated_at=created,
        error=None,
    )
    adapter.save_job(job)  # queued persisted (Req 19.5)

    job = job.model_copy(update={"status": "running", "updated_at": clock()})
    adapter.save_job(job)  # running persisted (Req 19.5)

    try:
        run_aws_inspection(
            data,
            content_type,
            inspection_id=inspection_id,
            adapter=adapter,
            cfg_vision=cfg_vision,
            cfg_agent=cfg_agent,
            model=model,
            retry_policy=retry_policy,
        )
    except Exception as exc:  # noqa: BLE001 - any failure -> recorded failed job
        # Surface a FAILED status (Req 16.3) carrying the error; exhausted
        # retries arrive here as RetryExhausted and are recorded like any other
        # failure. The report (if any) was not persisted, so the job's failure
        # is the authoritative outcome.
        detail = _error_detail(exc)
        job = job.model_copy(
            update={"status": "failed", "updated_at": clock(), "error": detail}
        )
        adapter.save_job(job)  # failed persisted (Req 19.5)
        return job

    job = job.model_copy(
        update={"status": "succeeded", "updated_at": clock(), "error": None}
    )
    adapter.save_job(job)  # succeeded persisted (Req 19.5)
    return job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_detail(exc: BaseException) -> str:
    """A compact, non-secret error string for a failed job record.

    For :class:`RetryExhausted` it names the exhausting underlying error so the
    polled job explains *why* it failed after retries (Req 16.3). Carries only
    type + message text; it never logs or embeds a credential.
    """
    if isinstance(exc, RetryExhausted):
        return (
            f"retries exhausted after {exc.attempts} attempt(s): "
            f"{type(exc.last_error).__name__}: {exc.last_error}"
        )
    return f"{type(exc).__name__}: {exc}"


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (the production clock)."""
    return datetime.now(timezone.utc).isoformat()


def _new_inspection_id() -> str:
    """Generate a fresh, record-id-safe inspection id (mirrors the pipeline)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"insp-{stamp}-{uuid.uuid4().hex[:12]}"
