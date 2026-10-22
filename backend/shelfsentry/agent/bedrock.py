"""The Bedrock-backed primary decision path (Req 7.1, 7.2; task 7.3).

:class:`BedrockDecisionModel` is the PRIMARY decision path in AWS Mode
(design.md "Decision model abstraction", Req 7.1). It slots behind the
:class:`shelfsentry.agent.decision.DecisionModel` protocol exactly like the
deterministic :class:`~shelfsentry.agent.fallback.FallbackPolicy` and the
test-only :class:`~shelfsentry.agent.decision.InjectedDecisionModel`, so the
bounded loop (:func:`shelfsentry.agent.loop.run_agent_loop`) consumes it as its
``model`` argument with no loop change.

Recorded-fallback provenance (the important design point). This model does NOT
silently fall back. When Bedrock is unavailable, misconfigured, or returns
something that cannot be parsed into a valid action, :meth:`decide` raises a
clear :class:`BedrockDecisionError`. The *loop* owns the fallback: its
``fallback_on_error`` seam catches that error, engages the deterministic
policy, and records ``used_fallback=True`` with a reason on the resulting
``AgentDecision`` (Req 7.2). Swallowing the error inside this model would hide
the provenance of the switch, so we deliberately let it surface.

Import-light and credential-free at import time. ``boto3`` / ``botocore`` are
imported lazily *inside* methods, never at module top. Importing
``shelfsentry.agent`` (and therefore this module) requires no ``boto3``, no
``strands``, and no AWS credentials. A caller may inject a client for tests
(:class:`BedrockDecisionModel(client=...)`); only when no client is supplied
*and* :meth:`decide` is actually invoked does the model try to construct a real
``bedrock-runtime`` client, failing loudly if the library or configuration is
missing. No credentials are ever embedded: region and model id come from the
constructor or from the environment.

Transport: the stable boto3 ``bedrock-runtime`` Converse API. The model is sent
a structured prompt built from the current :class:`VisionFinding` (its candidate
regions, per-region uncertainty, and the vision stage's ``suggested_follow_up``)
plus the running decision trace, and is asked to return a single JSON object
selecting one of the fixed five actions. The returned JSON is parsed and then
validated through :func:`shelfsentry.agent.tools.validate_tool_args`, so the
same action/argument contract the loop enforces before execution also gates what
this model may emit. The produced :class:`AgentDecision` records
``decided_by="bedrock"``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from shelfsentry.agent.decision import is_allowed_action
from shelfsentry.agent.tools import ToolArgumentError, validate_tool_args
from shelfsentry.config.settings import AgentConfig
from shelfsentry.data.contracts import (
    AGENT_ACTIONS,
    AgentDecision,
    CandidateRegion,
    TraceEntry,
    VisionFinding,
)

__all__ = [
    "BedrockDecisionError",
    "BedrockDecisionModel",
    "DEFAULT_BEDROCK_MODEL_ID",
]

#: Environment variable names read when the corresponding constructor argument
#: is not supplied. No credentials are read here - only non-secret config
#: (region and model id). AWS credentials are resolved by boto3 itself from the
#: standard credential chain, never embedded or read into context by this code.
_ENV_MODEL_ID: str = "SHELFSENTRY_BEDROCK_MODEL_ID"
_ENV_REGION: tuple[str, ...] = (
    "SHELFSENTRY_BEDROCK_REGION",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
)

#: A reasonable default Bedrock model id (Anthropic Claude on Bedrock). Only
#: used when neither the constructor nor the environment supplies one; still
#: overridable so no model choice is hard-wired into the pipeline.
DEFAULT_BEDROCK_MODEL_ID: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"


class BedrockDecisionError(RuntimeError):
    """Raised when the Bedrock path is unavailable, misconfigured, or unusable.

    The loop's ``fallback_on_error`` seam catches this (like any model error),
    engages the deterministic :class:`~shelfsentry.agent.fallback.FallbackPolicy`,
    and records ``used_fallback=True`` with a reason (Req 7.2). Raising rather
    than falling back inside the model keeps the provenance of the switch
    honest: the recorded decision shows the fallback was engaged *because the
    primary path failed*.

    :param code: a stable classifier, one of ``"unavailable"`` (boto3/botocore
        missing), ``"misconfigured"`` (no model id/region or no credentials),
        ``"api_error"`` (the Converse call raised), or ``"unparseable"`` (the
        response could not be turned into a valid action).
    """

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"[{self.code}] {super().__str__()}"


class BedrockDecisionModel:
    """Primary :class:`DecisionModel` backed by Amazon Bedrock (Req 7.1).

    :param client: an optional pre-built ``bedrock-runtime`` client (anything
        exposing a ``converse(**kwargs)`` method). Injecting one makes the model
        fully testable with no AWS dependency. When ``None``, a real client is
        built lazily on the first :meth:`decide` call.
    :param model_id: the Bedrock model id to invoke. Falls back to the
        ``SHELFSENTRY_BEDROCK_MODEL_ID`` env var, then
        :data:`DEFAULT_BEDROCK_MODEL_ID`.
    :param region: the AWS region for the runtime client. Falls back to the
        ``SHELFSENTRY_BEDROCK_REGION`` / ``AWS_REGION`` / ``AWS_DEFAULT_REGION``
        env vars. Required (from one of those sources) only when a client has to
        be built; ignored when a ``client`` is injected.

    No credentials are accepted or embedded: when building a real client, boto3
    resolves credentials from its standard chain. If that chain yields nothing,
    :meth:`decide` raises :class:`BedrockDecisionError` so the loop falls back
    rather than failing silently.
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        model_id: str | None = None,
        region: str | None = None,
    ) -> None:
        self._client = client
        self._model_id = model_id
        self._region = region

    # -- public API -------------------------------------------------------

    def decide(
        self,
        finding: VisionFinding,
        trace: list[TraceEntry],
        cfg: AgentConfig,
    ) -> AgentDecision:
        """Select the next action via Bedrock and return an ``AgentDecision``.

        Builds a structured prompt from ``finding`` + ``trace``, calls the
        Converse API, parses the model's JSON action selection, validates it
        through :func:`validate_tool_args`, and returns an ``AgentDecision``
        with ``decided_by="bedrock"``.

        :raises BedrockDecisionError: if boto3/botocore are missing, required
            configuration/credentials are absent, the Converse call errors, or
            the response cannot be parsed into a valid action. The loop converts
            this into a recorded fallback (Req 7.2); this method never falls back
            on its own.
        """
        iteration = self._next_iteration(trace)
        client = self._resolve_client()
        model_id = self._resolve_model_id()

        system_prompt, user_prompt = self._build_prompt(finding, trace, cfg)
        response = self._invoke(client, model_id, system_prompt, user_prompt)
        text = self._extract_text(response)
        action, args, rationale = self._parse_selection(text)

        # Build the decision, then validate its arguments against the same
        # action/argument contract the loop enforces before execution (Req 6.4).
        # An unparseable-into-valid selection is a Bedrock failure for fallback
        # purposes (Req 7.2), so re-raise it as a BedrockDecisionError.
        decision = AgentDecision(
            decision_id=f"bedrock-{iteration}",
            iteration=iteration,
            action=action,  # validated allowed below / by the literal
            rationale=rationale,
            args=args,
            decided_by="bedrock",
        )
        try:
            validate_tool_args(decision)
        except ToolArgumentError as exc:
            raise BedrockDecisionError(
                f"Bedrock selected action {action!r} with invalid arguments: "
                f"{exc}",
                code="unparseable",
            ) from exc
        return decision

    # -- client / config resolution --------------------------------------

    def _resolve_client(self) -> Any:
        """Return the injected client or lazily build a real one.

        Lazy-imports ``boto3`` so importing this module never requires it. A
        missing library or a client-construction failure is surfaced as a
        :class:`BedrockDecisionError` (never a bare import error) so the loop's
        fallback seam handles it uniformly.
        """
        if self._client is not None:
            return self._client

        try:
            import boto3  # noqa: PLC0415 - intentional lazy import
        except Exception as exc:  # noqa: BLE001 - missing OR unimportable -> fallback
            # Not just ImportError: a transitive dependency can raise other
            # errors (or warnings escalated to errors) at import time. Any such
            # failure means the Bedrock client stack is unusable, so surface it
            # as a clean 'unavailable' error and let the loop fall back rather
            # than letting an opaque import-time error propagate.
            raise BedrockDecisionError(
                "the boto3 Bedrock client stack could not be imported "
                f"({type(exc).__name__}: {exc}); install/repair boto3 or inject "
                "a client. Engaging fallback.",
                code="unavailable",
            ) from exc

        region = self._resolve_region()
        try:
            self._client = boto3.client("bedrock-runtime", region_name=region)
        except Exception as exc:  # noqa: BLE001 - any construction failure -> fallback
            raise BedrockDecisionError(
                f"failed to construct a bedrock-runtime client for region "
                f"{region!r}: {type(exc).__name__}: {exc}",
                code="misconfigured",
            ) from exc
        return self._client

    def _resolve_model_id(self) -> str:
        """Resolve the model id from constructor, env, or the default."""
        model_id = self._model_id or os.environ.get(_ENV_MODEL_ID)
        model_id = (model_id or DEFAULT_BEDROCK_MODEL_ID).strip()
        if not model_id:  # pragma: no cover - default is non-empty
            raise BedrockDecisionError(
                "no Bedrock model id configured (constructor, "
                f"{_ENV_MODEL_ID}, or default).",
                code="misconfigured",
            )
        return model_id

    def _resolve_region(self) -> str:
        """Resolve the AWS region from constructor or the env var chain.

        Raises when none is available: a region is required to build a real
        client, and failing loudly here lets the loop fall back rather than
        letting boto3 raise an opaque error deeper in.
        """
        if self._region and self._region.strip():
            return self._region.strip()
        for name in _ENV_REGION:
            value = os.environ.get(name)
            if value and value.strip():
                return value.strip()
        raise BedrockDecisionError(
            "no AWS region configured for Bedrock (constructor or one of "
            f"{', '.join(_ENV_REGION)}).",
            code="misconfigured",
        )

    # -- Converse invocation ---------------------------------------------

    def _invoke(
        self,
        client: Any,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Call the Converse API, translating any failure into our error type.

        Uses the stable ``converse`` shape: a ``messages`` list of role/content
        blocks plus a ``system`` list. Any exception (including a credential or
        access error from botocore) becomes a :class:`BedrockDecisionError` so
        the loop's fallback engages (Req 7.2).
        """
        request: dict[str, Any] = {
            "modelId": model_id,
            "messages": [
                {"role": "user", "content": [{"text": user_prompt}]}
            ],
            "system": [{"text": system_prompt}],
            "inferenceConfig": {"temperature": 0.0, "maxTokens": 512},
        }
        try:
            return client.converse(**request)
        except BedrockDecisionError:
            raise
        except Exception as exc:  # noqa: BLE001 - any API error engages fallback
            raise BedrockDecisionError(
                f"Bedrock Converse call failed: {type(exc).__name__}: {exc}",
                code="api_error",
            ) from exc

    @staticmethod
    def _extract_text(response: dict[str, Any]) -> str:
        """Pull the assistant's text out of a Converse-style response.

        Expected shape (boto3 Converse)::

            {"output": {"message": {"content": [{"text": "..."}, ...]}}}

        Concatenates every ``text`` block in order. A response missing this
        structure, or carrying no text, is a parse failure -> fallback.
        """
        if not isinstance(response, dict):
            raise BedrockDecisionError(
                f"Bedrock response was not a mapping, got "
                f"{type(response).__name__}.",
                code="unparseable",
            )
        output = response.get("output")
        message = output.get("message") if isinstance(output, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            raise BedrockDecisionError(
                "Bedrock response is missing output.message.content; cannot "
                "extract an action selection.",
                code="unparseable",
            )
        texts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        text = "".join(texts).strip()
        if not text:
            raise BedrockDecisionError(
                "Bedrock response contained no text content to parse.",
                code="unparseable",
            )
        return text

    # -- prompt construction ---------------------------------------------

    def _build_prompt(
        self,
        finding: VisionFinding,
        trace: list[TraceEntry],
        cfg: AgentConfig,
    ) -> tuple[str, str]:
        """Build the ``(system, user)`` prompt pair sent to Bedrock.

        The system prompt states the task, the fixed five-action set, the
        required JSON response shape, and the honesty guardrails (never assert
        out-of-stock as proven; empty-looking regions carry caveats, Req 27.4).
        The user prompt carries the structured finding (candidate regions with
        their bounding boxes, empty-space ratio, uncertainty, and caveat), the
        vision stage's ``suggested_follow_up`` hint, and a compact rendering of
        the decision trace so far.
        """
        system_prompt = (
            "You are the decision model for ShelfSentry, an agentic retail "
            "shelf inspection system. Given a computer-vision finding and the "
            "decision trace so far, choose exactly ONE next action.\n\n"
            "Allowed actions (choose exactly one):\n"
            "- inspect_crop: re-inspect a specific uncertain region more "
            "closely. REQUIRES args.region_id (string) and args.bounding_box "
            "{x,y,w,h} (non-negative integers, w>=1, h>=1) naming that region.\n"
            "- analyze_another_frame: sample and analyze another video frame "
            "(video sources only). Optional args.frame_id hint.\n"
            "- request_better_image: ask for a higher-quality image when the "
            "input is too poor to judge.\n"
            "- request_human_review: escalate to a human when evidence is "
            "conflicting or cannot be resolved. Optional args.reason.\n"
            "- finalize_report: stop and report when evidence is sufficient.\n\n"
            "Honesty guardrails: an empty-looking region is an ESTIMATE and may "
            "be shadow, occlusion, dividers, or poor image quality; never "
            "assert an out-of-stock condition as proven. Prefer gathering more "
            "evidence over guessing when uncertainty is high.\n\n"
            "Respond with ONLY a single JSON object and no other text, of the "
            'form: {"action": <one of the allowed actions>, "args": {<action '
            'arguments, or empty object>}, "rationale": <short string '
            "explaining the choice>}. Do not wrap it in markdown fences."
        )

        user_prompt = (
            "max_iterations="
            + str(cfg.max_iterations)
            + "; sufficient_uncertainty="
            + str(cfg.sufficient_uncertainty)
            + "\n\nVISION FINDING:\n"
            + self._render_finding(finding)
            + "\n\nDECISION TRACE SO FAR:\n"
            + self._render_trace(trace)
            + "\n\nChoose the next action as a single JSON object."
        )
        return system_prompt, user_prompt

    @staticmethod
    def _render_finding(finding: VisionFinding) -> str:
        """Render a finding into a compact, deterministic text block."""
        lines = [
            f"finding_id={finding.finding_id}",
            f"source={finding.source}",
            f"frame_id={finding.frame_id}",
            f"suggested_follow_up={finding.suggested_follow_up}",
            f"candidate_regions={len(finding.candidate_regions)}:",
        ]
        for region in finding.candidate_regions:
            lines.append("  " + BedrockDecisionModel._render_region(region))
        if finding.uncertainty_interpretations:
            lines.append(
                "uncertainty_interpretations="
                + json.dumps(
                    finding.uncertainty_interpretations, sort_keys=True
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _render_region(region: CandidateRegion) -> str:
        box = region.bounding_box
        ratio = (
            f"{region.empty_space_ratio:.3f}"
            if region.empty_space_ratio is not None
            else "n/a"
        )
        return (
            f"region_id={region.region_id} "
            f"bounding_box={{x:{box.x},y:{box.y},w:{box.w},h:{box.h}}} "
            f"empty_space_ratio={ratio} "
            f"heuristic_score={region.heuristic_score:.3f} "
            f"uncertainty_value={region.uncertainty_value:.3f} "
            f"frame_id={region.frame_id} "
            f"caveat={region.visual_caveat!r}"
        )

    @staticmethod
    def _render_trace(trace: list[TraceEntry]) -> str:
        """Render the trace compactly; ``(none)`` when empty (first iteration)."""
        if not trace:
            return "(none)"
        lines: list[str] = []
        for entry in trace:
            if isinstance(entry, AgentDecision):
                lines.append(
                    f"[decision i={entry.iteration}] action={entry.action} "
                    f"args={json.dumps(entry.args, sort_keys=True)}"
                )
            else:  # ToolExecution
                lines.append(
                    f"[tool i={entry.iteration}] tool={entry.tool} "
                    f"status={entry.status} result_ref={entry.result_ref}"
                )
        return "\n".join(lines)

    # -- response parsing -------------------------------------------------

    def _parse_selection(self, text: str) -> tuple[str, dict[str, Any], str]:
        """Parse the model text into ``(action, args, rationale)``.

        Accepts a bare JSON object or one embedded in surrounding prose / a
        fenced code block (the first balanced ``{...}`` is used). Validates that
        ``action`` is one of the fixed five (Req 6.1) and that ``args`` is a
        mapping. A missing/unknown action, a non-object payload, or malformed
        JSON is a parse failure -> :class:`BedrockDecisionError` -> loop
        fallback (Req 7.2). Argument *shape* is validated later via
        :func:`validate_tool_args`.
        """
        payload = self._extract_json_object(text)

        action = payload.get("action")
        if not isinstance(action, str) or not is_allowed_action(action):
            raise BedrockDecisionError(
                f"Bedrock did not select a valid action; got {action!r}, "
                f"expected one of {sorted(AGENT_ACTIONS)}.",
                code="unparseable",
            )

        raw_args = payload.get("args", {})
        if raw_args is None:
            raw_args = {}
        if not isinstance(raw_args, dict):
            raise BedrockDecisionError(
                f"Bedrock 'args' must be a JSON object, got "
                f"{type(raw_args).__name__}.",
                code="unparseable",
            )

        rationale = payload.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = f"Bedrock selected {action}."
        else:
            rationale = rationale.strip()

        return action, dict(raw_args), rationale

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        """Return the first JSON object found in ``text``.

        Tries a direct parse first; if that fails, strips a ``json`` code fence
        and retries; finally falls back to the first balanced ``{...}`` span.
        Anything that still will not parse into a dict is a parse failure.
        """
        candidates: list[str] = [text]

        fenced = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if fenced:
            candidates.append(fenced.group(1))

        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            candidates.append(brace.group(0))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed

        raise BedrockDecisionError(
            "Bedrock response did not contain a parseable JSON action object.",
            code="unparseable",
        )

    # -- misc -------------------------------------------------------------

    @staticmethod
    def _next_iteration(trace: list[TraceEntry]) -> int:
        """Iteration index for the decision about to be recorded.

        Equal to the count of :class:`AgentDecision` entries already in the
        trace, matching the loop's zero-based iteration counter.
        """
        return sum(1 for entry in trace if isinstance(entry, AgentDecision))
