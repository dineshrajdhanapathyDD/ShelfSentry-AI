"""Tests for the DecisionModel seam and the InjectedDecisionModel (task 6.1).

Covers Req 6.1 (fixed five-action set) and Req 7.3 (an injected/offline model
drives decisions with no AWS credentials): queued decisions and a callable are
returned in order with ``decided_by="injected"``, and an out-of-set action is
rejected.
"""

from __future__ import annotations

import pytest

from shelfsentry.agent import (
    AGENT_ACTIONS,
    DecisionModel,
    InjectedDecisionModel,
    is_allowed_action,
)
from shelfsentry.config.settings import AgentConfig
from shelfsentry.data.contracts import AgentDecision, VisionFinding


def _finding(finding_id: str = "f1") -> VisionFinding:
    return VisionFinding(
        finding_id=finding_id,
        inspection_id="insp-1",
        source="image",
        opencv_version="5.0.0",
        candidate_regions=[],
        uncertainty_interpretations={},
        annotated_output_ref="ref://annotated",
        suggested_follow_up="finalize_report",
    )


def _cfg() -> AgentConfig:
    return AgentConfig()


def test_fixed_five_action_set_matches_contract() -> None:
    assert set(AGENT_ACTIONS) == {
        "inspect_crop",
        "analyze_another_frame",
        "request_better_image",
        "request_human_review",
        "finalize_report",
    }
    assert is_allowed_action("inspect_crop")
    assert not is_allowed_action("delete_everything")


def test_injected_model_satisfies_protocol() -> None:
    model = InjectedDecisionModel(fn=lambda f, t, c: _decision("finalize_report"))
    assert isinstance(model, DecisionModel)


def _decision(action: str, decision_id: str = "d", iteration: int = 0) -> AgentDecision:
    return AgentDecision(
        decision_id=decision_id,
        iteration=iteration,
        action=action,  # type: ignore[arg-type]
        rationale="r",
        decided_by="injected",
    )


def test_queue_mode_returns_specs_in_order_with_injected_provenance() -> None:
    model = InjectedDecisionModel(
        decisions=[
            {"action": "inspect_crop", "args": {"region_id": "r1"}},
            {"action": "analyze_another_frame"},
            {"action": "finalize_report"},
        ]
    )
    cfg = _cfg()

    d0 = model.decide(_finding(), [], cfg)
    d1 = model.decide(_finding(), [d0], cfg)
    d2 = model.decide(_finding(), [d0, d1], cfg)

    assert [d.action for d in (d0, d1, d2)] == [
        "inspect_crop",
        "analyze_another_frame",
        "finalize_report",
    ]
    assert all(d.decided_by == "injected" for d in (d0, d1, d2))
    assert [d.iteration for d in (d0, d1, d2)] == [0, 1, 2]
    assert d0.args == {"region_id": "r1"}


def test_queue_mode_accepts_ready_made_decisions_unchanged() -> None:
    supplied = _decision("request_human_review", decision_id="custom-id", iteration=7)
    model = InjectedDecisionModel(decisions=[supplied])

    out = model.decide(_finding(), [], _cfg())

    assert out is supplied
    assert out.decision_id == "custom-id"
    assert out.decided_by == "injected"


def test_queue_exhaustion_raises() -> None:
    model = InjectedDecisionModel(decisions=[{"action": "finalize_report"}])
    cfg = _cfg()
    model.decide(_finding(), [], cfg)
    with pytest.raises(IndexError):
        model.decide(_finding(), [], cfg)


def test_callable_mode_branches_on_finding() -> None:
    def fn(finding: VisionFinding, trace, cfg) -> AgentDecision:
        action = "inspect_crop" if finding.finding_id == "ambiguous" else "finalize_report"
        return _decision(action, decision_id=f"d-{finding.finding_id}")

    model = InjectedDecisionModel(fn=fn)
    cfg = _cfg()

    a = model.decide(_finding("ambiguous"), [], cfg)
    b = model.decide(_finding("clear"), [], cfg)

    # Different findings produce different actions (causality, Req 7.3 enables).
    assert a.action == "inspect_crop"
    assert b.action == "finalize_report"
    assert a.decided_by == "injected"


def test_out_of_set_action_is_rejected_in_callable_mode() -> None:
    # Build an AgentDecision that bypasses the literal by validating a spec with
    # a bad action would fail at the contract level, so return a decision whose
    # action was mutated to an out-of-set value.
    bad = _decision("finalize_report")
    object.__setattr__(bad, "action", "launch_missiles")

    model = InjectedDecisionModel(fn=lambda f, t, c: bad)
    with pytest.raises(ValueError, match="allowed action set"):
        model.decide(_finding(), [], _cfg())


def test_out_of_set_action_is_rejected_in_queue_mode() -> None:
    # A spec with an unknown action fails when materialized against the
    # AgentDecision contract (its action is the fixed AgentAction literal).
    model = InjectedDecisionModel(decisions=[{"action": "launch_missiles"}])
    with pytest.raises(Exception):
        model.decide(_finding(), [], _cfg())


def test_requires_exactly_one_of_queue_or_callable() -> None:
    with pytest.raises(ValueError):
        InjectedDecisionModel()
    with pytest.raises(ValueError):
        InjectedDecisionModel(decisions=[{"action": "finalize_report"}], fn=lambda f, t, c: _decision("finalize_report"))
