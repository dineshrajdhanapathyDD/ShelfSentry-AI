"""Tests for the Bedrock primary decision path (task 7.3).

These tests require NO AWS: a fake client whose ``converse`` returns a
configured response is injected into :class:`BedrockDecisionModel`, so the
Converse request/response handling, JSON parsing, and action validation are all
exercised offline. The credential-free path is also covered by asserting that a
model with no injected client and no boto3/credentials raises a clear error, and
that :func:`run_agent_loop` turns that error into a recorded fallback
(``used_fallback=True``) - proving Req 7.1 (Bedrock primary) composes with
Req 7.2 (recorded safe fallback) in the loop.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from shelfsentry.agent import (
    BedrockDecisionError,
    BedrockDecisionModel,
    FallbackPolicy,
    ToolResult,
    run_agent_loop,
)
from shelfsentry.config.settings import AgentConfig
from shelfsentry.data.contracts import (
    AgentDecision,
    BoundingBox,
    CandidateRegion,
    InspectionReport,
    VisionFinding,
)


# ---------------------------------------------------------------------------
# Builders / fakes
# ---------------------------------------------------------------------------


def _region(
    region_id: str = "r0",
    *,
    empty_space_ratio: float | None = 0.8,
    uncertainty_value: float = 0.3,
    x: int = 1,
    y: int = 2,
    w: int = 3,
    h: int = 4,
) -> CandidateRegion:
    return CandidateRegion(
        region_id=region_id,
        bounding_box=BoundingBox(x=x, y=y, w=w, h=h),
        empty_space_ratio=empty_space_ratio,
        heuristic_score=0.5,
        uncertainty_value=uncertainty_value,
        visual_caveat="estimate only; may be shadow/occlusion/dividers/quality",
    )


def _finding(
    finding_id: str = "f0",
    *,
    regions: list[CandidateRegion] | None = None,
    source: str = "image",
    frame_id: str | None = None,
) -> VisionFinding:
    return VisionFinding(
        finding_id=finding_id,
        inspection_id="insp-1",
        source=source,  # type: ignore[arg-type]
        frame_id=frame_id,
        opencv_version="5.0.0",
        candidate_regions=regions if regions is not None else [_region()],
        uncertainty_interpretations={"r0": "lower is less confident"},
        annotated_output_ref="ref://annotated",
        suggested_follow_up="inspect_crop",
    )


def _cfg(**overrides: Any) -> AgentConfig:
    return AgentConfig(**overrides)


def _converse_response(text: str) -> dict[str, Any]:
    """Build a Converse-style response wrapping ``text`` as assistant output."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
    }


class FakeBedrockClient:
    """A fake ``bedrock-runtime`` client exposing ``converse``.

    Returns a pre-built response and records the request kwargs so tests can
    assert the prompt was constructed. When ``raises`` is set, ``converse``
    raises it (simulating an API/credential error).
    """

    def __init__(
        self,
        *,
        response: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.response = response
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        assert self.response is not None
        return self.response


# ---------------------------------------------------------------------------
# well-formed response -> valid AgentDecision (decided_by="bedrock")
# ---------------------------------------------------------------------------


def test_finalize_report_selection_returns_bedrock_decision() -> None:
    text = json.dumps(
        {"action": "finalize_report", "args": {}, "rationale": "evidence sufficient"}
    )
    client = FakeBedrockClient(response=_converse_response(text))
    model = BedrockDecisionModel(client=client, model_id="m", region="us-east-1")

    decision = model.decide(_finding(), [], _cfg())

    assert isinstance(decision, AgentDecision)
    assert decision.action == "finalize_report"
    assert decision.decided_by == "bedrock"
    assert decision.rationale == "evidence sufficient"
    assert decision.args == {}
    assert decision.iteration == 0
    # The model was actually invoked via converse with a modelId + messages.
    assert len(client.calls) == 1
    assert client.calls[0]["modelId"] == "m"
    assert "messages" in client.calls[0] and client.calls[0]["messages"]


def test_inspect_crop_selection_with_bounding_box_is_valid() -> None:
    text = json.dumps(
        {
            "action": "inspect_crop",
            "args": {
                "region_id": "r0",
                "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
            },
            "rationale": "high uncertainty on r0",
        }
    )
    client = FakeBedrockClient(response=_converse_response(text))
    model = BedrockDecisionModel(client=client)

    decision = model.decide(_finding(), [], _cfg())

    assert decision.action == "inspect_crop"
    assert decision.decided_by == "bedrock"
    assert decision.args["region_id"] == "r0"
    assert decision.args["bounding_box"] == {"x": 1, "y": 2, "w": 3, "h": 4}


def test_selection_embedded_in_markdown_fence_is_parsed() -> None:
    text = (
        "Here is my decision:\n```json\n"
        + json.dumps({"action": "request_human_review", "args": {}})
        + "\n```\nThanks."
    )
    client = FakeBedrockClient(response=_converse_response(text))
    model = BedrockDecisionModel(client=client)

    decision = model.decide(_finding(), [], _cfg())
    assert decision.action == "request_human_review"
    assert decision.decided_by == "bedrock"


def test_iteration_tracks_existing_decisions_in_trace() -> None:
    text = json.dumps({"action": "finalize_report", "args": {}})
    client = FakeBedrockClient(response=_converse_response(text))
    model = BedrockDecisionModel(client=client)

    prior = AgentDecision(
        decision_id="d0",
        iteration=0,
        action="analyze_another_frame",
        rationale="first",
        args={},
        decided_by="bedrock",
    )
    decision = model.decide(_finding(), [prior], _cfg())
    assert decision.iteration == 1


# ---------------------------------------------------------------------------
# garbage / unparseable / invalid-args -> raises (loop will fall back)
# ---------------------------------------------------------------------------


def test_unparseable_text_raises() -> None:
    client = FakeBedrockClient(response=_converse_response("not json at all"))
    model = BedrockDecisionModel(client=client)
    with pytest.raises(BedrockDecisionError) as exc:
        model.decide(_finding(), [], _cfg())
    assert exc.value.code == "unparseable"


def test_unknown_action_raises() -> None:
    text = json.dumps({"action": "teleport", "args": {}})
    client = FakeBedrockClient(response=_converse_response(text))
    model = BedrockDecisionModel(client=client)
    with pytest.raises(BedrockDecisionError):
        model.decide(_finding(), [], _cfg())


def test_inspect_crop_missing_bounding_box_raises() -> None:
    # Valid action, but args fail validate_tool_args (no bounding_box).
    text = json.dumps({"action": "inspect_crop", "args": {"region_id": "r0"}})
    client = FakeBedrockClient(response=_converse_response(text))
    model = BedrockDecisionModel(client=client)
    with pytest.raises(BedrockDecisionError) as exc:
        model.decide(_finding(), [], _cfg())
    assert exc.value.code == "unparseable"
    assert "bounding_box" in str(exc.value)


def test_args_not_an_object_raises() -> None:
    text = json.dumps({"action": "finalize_report", "args": [1, 2, 3]})
    client = FakeBedrockClient(response=_converse_response(text))
    model = BedrockDecisionModel(client=client)
    with pytest.raises(BedrockDecisionError):
        model.decide(_finding(), [], _cfg())


def test_missing_output_structure_raises() -> None:
    client = FakeBedrockClient(response={"output": {}})
    model = BedrockDecisionModel(client=client)
    with pytest.raises(BedrockDecisionError) as exc:
        model.decide(_finding(), [], _cfg())
    assert exc.value.code == "unparseable"


def test_converse_api_error_raises_api_error() -> None:
    client = FakeBedrockClient(raises=RuntimeError("AccessDeniedException"))
    model = BedrockDecisionModel(client=client)
    with pytest.raises(BedrockDecisionError) as exc:
        model.decide(_finding(), [], _cfg())
    assert exc.value.code == "api_error"
    assert "AccessDeniedException" in str(exc.value)


# ---------------------------------------------------------------------------
# no client + no creds -> clear error; loop records used_fallback (Req 7.1+7.2)
# ---------------------------------------------------------------------------


def test_no_client_no_region_raises_misconfigured(monkeypatch: Any) -> None:
    # Strip any region env so building a real client fails loudly before boto3
    # would try the network. (If boto3 is absent, we get 'unavailable' instead;
    # both are clear BedrockDecisionErrors, which is what matters.)
    for name in ("SHELFSENTRY_BEDROCK_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    model = BedrockDecisionModel()  # no client, no region
    with pytest.raises(BedrockDecisionError) as exc:
        model.decide(_finding(), [], _cfg())
    assert exc.value.code in {"misconfigured", "unavailable"}


class _UnavailableBedrockModel(BedrockDecisionModel):
    """A BedrockDecisionModel that always fails to resolve a client.

    Simulates 'Bedrock unavailable / no creds' deterministically regardless of
    whether boto3 happens to be installed in the test environment, so the
    loop-integration assertion (Req 7.2 recorded fallback) is stable.
    """

    def _resolve_client(self) -> Any:  # type: ignore[override]
        raise BedrockDecisionError(
            "simulated: no credentials / Bedrock unavailable", code="misconfigured"
        )


class _FakeToolClient:
    def call(self, action: str, args: dict[str, Any]) -> ToolResult:  # type: ignore[override]
        return ToolResult("ref://unused")


def test_loop_records_used_fallback_when_bedrock_unavailable() -> None:
    # A clear, sufficient finding so the deterministic fallback finalizes.
    clear = _finding(
        regions=[_region("r0", empty_space_ratio=0.1, uncertainty_value=0.95)]
    )
    model = _UnavailableBedrockModel()  # primary path (Req 7.1) that errors

    report = run_agent_loop(
        clear,
        model,
        _FakeToolClient(),
        _cfg(),
        inspection_id="insp-1",
        mode="aws",
        # default fallback_on_error=FallbackPolicy() engages on the error
    )

    assert isinstance(report, InspectionReport)
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert decisions, "expected at least one recorded decision"
    first = decisions[0]
    # Req 7.2: the loop engaged the fallback and recorded it with a reason.
    assert first.used_fallback is True
    assert first.decided_by == "fallback"
    assert first.fallback_reason is not None
    assert "BedrockDecisionError" in first.fallback_reason
    assert report.status == "complete"


def test_loop_with_explicit_fallback_policy_also_records_it() -> None:
    clear = _finding(
        regions=[_region("r0", empty_space_ratio=0.1, uncertainty_value=0.95)]
    )
    report = run_agent_loop(
        clear,
        _UnavailableBedrockModel(),
        _FakeToolClient(),
        _cfg(),
        inspection_id="insp-1",
        mode="aws",
        fallback_on_error=FallbackPolicy(),
    )
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert decisions[0].used_fallback is True


def test_bedrock_primary_drives_loop_when_available() -> None:
    # Positive integration: a working Bedrock model drives the loop end to end
    # with decided_by="bedrock" and no fallback.
    text = json.dumps({"action": "finalize_report", "args": {}})
    client = FakeBedrockClient(response=_converse_response(text))
    model = BedrockDecisionModel(client=client)

    report = run_agent_loop(
        _finding(),
        model,
        _FakeToolClient(),
        _cfg(),
        inspection_id="insp-1",
        mode="aws",
    )
    decisions = [e for e in report.decision_trace if isinstance(e, AgentDecision)]
    assert len(decisions) == 1
    assert decisions[0].decided_by == "bedrock"
    assert decisions[0].used_fallback is False
    assert report.status == "complete"
