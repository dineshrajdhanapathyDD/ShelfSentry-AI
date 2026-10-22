"""Property-based and example tests for the typed data contracts.

Covers task 1.3 of the ShelfSentry AI plan:

* **Property 21: Equivalent typed objects serialize identically**
  (**Validates: Requirements 10.3**). Two independently constructed contract
  instances built from the *same* field values produce byte-for-byte identical
  ``canonical_json()``, and round-tripping a model through its canonical JSON /
  dict and back is stable.
* Example-based tests that malformed boundary data is rejected with a
  field-identifying structured error (Req 10.2, 10.5): the raised
  ``pydantic.ValidationError`` names the offending field in its error
  ``loc``.

Property tests run under the Hypothesis "ci" profile registered in
``tests/conftest.py`` (>= 100 examples each). Smart strategies constrain
generated values to each contract's valid input space so the
"equivalent serialize identically" and round-trip properties exercise
realistic, acceptable data.

Design references: design.md "Data Models" and "Correctness Properties"
(Property 21); contracts in ``shelfsentry.data.contracts``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from shelfsentry.data.contracts import (
    AGENT_ACTIONS,
    MCP_TOOL_NAMES,
    SCHEMA_VERSION,
    AgentDecision,
    BoundingBox,
    CandidateRegion,
    CanonicalModel,
    EvalExample,
    EvaluationResult,
    InspectionReport,
    InspectionRequest,
    ToolExecution,
    VisionFinding,
    canonical_json,
)

# ---------------------------------------------------------------------------
# Strategy helpers
#
# These build *valid* field values within each contract's documented input
# space. Because the base model config strips surrounding whitespace on string
# fields, non-empty string strategies avoid relying on leading/trailing
# whitespace being preserved; this keeps both the "equivalent serialize
# identically" and the round-trip-stability properties meaningful.
# ---------------------------------------------------------------------------

# Printable, non-whitespace-only text with no surrounding whitespace, so a
# value survives ``str_strip_whitespace`` unchanged and round-trips stably.
_non_empty_text = (
    st.text(
        alphabet=st.characters(
            min_codepoint=0x20,
            max_codepoint=0x7E,
            blacklist_characters="",
        ),
        min_size=1,
        max_size=24,
    )
    .map(str.strip)
    .filter(lambda s: len(s) > 0)
)

# Finite floats are required: Pydantic/JSON reject NaN/inf under strict JSON.
_finite_floats = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
)
_unit_floats = st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1.0)
_opt_unit_floats = st.one_of(st.none(), _unit_floats)

# JSON-native scalar values for free-form dict fields (args / alt_metric /
# config_overrides). Restricted to JSON round-trippable scalars so a dict
# serializes stably.
_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    _finite_floats,
    _non_empty_text,
)
_json_dicts = st.dictionaries(keys=_non_empty_text, values=_json_scalars, max_size=4)
_opt_json_dicts = st.one_of(st.none(), _json_dicts)


@st.composite
def bounding_boxes(draw: st.DrawFn) -> BoundingBox:
    return BoundingBox(
        x=draw(st.integers(min_value=0, max_value=4096)),
        y=draw(st.integers(min_value=0, max_value=4096)),
        w=draw(st.integers(min_value=1, max_value=4096)),
        h=draw(st.integers(min_value=1, max_value=4096)),
    )


@st.composite
def candidate_region_kwargs(draw: st.DrawFn) -> dict[str, Any]:
    """Keyword args for a CandidateRegion, drawn once so two instances match."""
    return {
        "region_id": draw(_non_empty_text),
        "bounding_box": draw(bounding_boxes()),
        "empty_space_ratio": draw(_opt_unit_floats),
        "alt_metric": draw(_opt_json_dicts),
        "heuristic_score": draw(_finite_floats),
        "uncertainty_value": draw(_unit_floats),
        "frame_id": draw(st.one_of(st.none(), _non_empty_text)),
        "visual_caveat": draw(_non_empty_text),
    }


@st.composite
def vision_finding_kwargs(draw: st.DrawFn) -> dict[str, Any]:
    regions = draw(st.lists(candidate_region_kwargs(), max_size=3))
    return {
        "schema_version": SCHEMA_VERSION,
        "finding_id": draw(_non_empty_text),
        "inspection_id": draw(_non_empty_text),
        "source": draw(st.sampled_from(["image", "video_frame", "crop", "frame_compare"])),
        "frame_id": draw(st.one_of(st.none(), _non_empty_text)),
        "opencv_version": draw(_non_empty_text),
        "candidate_regions": regions,
        "uncertainty_interpretations": draw(
            st.dictionaries(keys=_non_empty_text, values=_non_empty_text, max_size=3)
        ),
        "annotated_output_ref": draw(_non_empty_text),
        "suggested_follow_up": draw(st.sampled_from(list(AGENT_ACTIONS))),
    }


@st.composite
def agent_decision_kwargs(draw: st.DrawFn) -> dict[str, Any]:
    used_fallback = draw(st.booleans())
    return {
        "schema_version": SCHEMA_VERSION,
        "entry_type": "agent_decision",
        "decision_id": draw(_non_empty_text),
        "iteration": draw(st.integers(min_value=0, max_value=16)),
        "action": draw(st.sampled_from(list(AGENT_ACTIONS))),
        "rationale": draw(_non_empty_text),
        "args": draw(_json_dicts),
        "decided_by": draw(st.sampled_from(["bedrock", "fallback", "injected"])),
        "used_fallback": used_fallback,
        "fallback_reason": draw(_non_empty_text) if used_fallback else None,
    }


@st.composite
def tool_execution_kwargs(draw: st.DrawFn) -> dict[str, Any]:
    status = draw(st.sampled_from(["ok", "rejected", "error"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "entry_type": "tool_execution",
        "execution_id": draw(_non_empty_text),
        "iteration": draw(st.integers(min_value=0, max_value=16)),
        "tool": draw(st.sampled_from(list(MCP_TOOL_NAMES))),
        "args": draw(_json_dicts),
        "result_ref": draw(_non_empty_text),
        "status": status,
        "error": None if status == "ok" else draw(_non_empty_text),
    }


@st.composite
def inspection_request_kwargs(draw: st.DrawFn) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "inspection_id": draw(_non_empty_text),
        "media_kind": draw(st.sampled_from(["image", "video"])),
        "content_type": draw(_non_empty_text),
        "size_bytes": draw(st.integers(min_value=0, max_value=50_000_000)),
        "media_ref": draw(_non_empty_text),
        "config_overrides": draw(_opt_json_dicts),
    }


@st.composite
def inspection_report_kwargs(draw: st.DrawFn) -> dict[str, Any]:
    trace: list[dict[str, Any]] = draw(
        st.lists(
            st.one_of(agent_decision_kwargs(), tool_execution_kwargs()),
            max_size=4,
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "inspection_id": draw(_non_empty_text),
        "mode": draw(st.sampled_from(["local", "aws"])),
        "final_recommendation": draw(_non_empty_text),
        "requires_human_review": draw(st.booleans()),
        "initial_finding_id": draw(_non_empty_text),
        "decision_trace": trace,
        "supporting_evidence_refs": draw(st.lists(_non_empty_text, max_size=4)),
        "created_at": draw(_non_empty_text),
        "status": draw(st.sampled_from(["complete", "failed", "needs_review"])),
    }


@st.composite
def eval_example_kwargs(draw: st.DrawFn) -> dict[str, Any]:
    return {
        "sample_id": draw(_non_empty_text),
        "inspection_id": draw(st.one_of(st.none(), _non_empty_text)),
        "expected": draw(_non_empty_text),
        "actual": draw(_non_empty_text),
        "note": draw(st.one_of(st.none(), _non_empty_text)),
    }


@st.composite
def evaluation_result_kwargs(draw: st.DrawFn) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_run_id": draw(_non_empty_text),
        "is_demo": draw(st.booleans()),
        "sample_count": draw(st.integers(min_value=0, max_value=1000)),
        "false_positives": draw(st.lists(eval_example_kwargs(), max_size=3)),
        "false_negatives": draw(st.lists(eval_example_kwargs(), max_size=3)),
        "agent_decision_correctness": draw(_unit_floats),
        "latency_ms_p50": draw(st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6)),
        "latency_ms_p95": draw(st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e6)),
        "documented_limitations": draw(st.lists(_non_empty_text, max_size=3)),
    }


# Map each contract class to the strategy producing a kwargs dict for it. Each
# kwargs dict is drawn once per example and used to construct TWO independent
# instances (Property 21).
_CONTRACT_KWARGS = [
    (BoundingBox, st.fixed_dictionaries(
        {
            "x": st.integers(min_value=0, max_value=4096),
            "y": st.integers(min_value=0, max_value=4096),
            "w": st.integers(min_value=1, max_value=4096),
            "h": st.integers(min_value=1, max_value=4096),
        }
    )),
    (CandidateRegion, candidate_region_kwargs()),
    (VisionFinding, vision_finding_kwargs()),
    (AgentDecision, agent_decision_kwargs()),
    (ToolExecution, tool_execution_kwargs()),
    (InspectionRequest, inspection_request_kwargs()),
    (InspectionReport, inspection_report_kwargs()),
    (EvalExample, eval_example_kwargs()),
    (EvaluationResult, evaluation_result_kwargs()),
]


def _kwargs_with_boxes_rebuilt(cls: type[CanonicalModel], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a fresh kwargs dict safe to pass to the model constructor twice.

    Nested ``BoundingBox`` models are value-equal and immutable-enough for our
    purposes, but to keep the two constructed instances *fully* independent we
    rebuild any nested model from its dumped data. This makes the "two
    independently-constructed instances" claim literal rather than shared-object.
    """
    rebuilt: dict[str, Any] = {}
    for key, value in kwargs.items():
        if isinstance(value, CanonicalModel):
            rebuilt[key] = value.model_dump()
        elif isinstance(value, list):
            rebuilt[key] = [
                v.model_dump() if isinstance(v, CanonicalModel) else v for v in value
            ]
        else:
            rebuilt[key] = value
    return rebuilt


# ---------------------------------------------------------------------------
# Property 21: Equivalent typed objects serialize identically
# Validates: Requirements 10.3
# ---------------------------------------------------------------------------


class TestProperty21EquivalentSerializeIdentically:
    """Feature: shelfsentry-ai-active-perception,
    Property 21: Equivalent typed objects serialize identically."""

    @pytest.mark.parametrize("cls, kwargs_strategy", _CONTRACT_KWARGS, ids=lambda a: getattr(a, "__name__", ""))
    def test_equivalent_objects_serialize_identically(self, cls, kwargs_strategy):
        @given(kwargs=kwargs_strategy)
        def check(kwargs: dict[str, Any]) -> None:
            first = cls(**_kwargs_with_boxes_rebuilt(cls, kwargs))
            second = cls(**_kwargs_with_boxes_rebuilt(cls, kwargs))

            # Two independently constructed, equal objects...
            assert first == second
            # ...serialize to byte-for-byte identical canonical JSON (Req 10.3).
            assert first.canonical_json() == second.canonical_json()
            # The module-level helper agrees with the method.
            assert canonical_json(first) == first.canonical_json()

        check()

    @pytest.mark.parametrize("cls, kwargs_strategy", _CONTRACT_KWARGS, ids=lambda a: getattr(a, "__name__", ""))
    def test_canonical_json_is_order_independent(self, cls, kwargs_strategy):
        @given(kwargs=kwargs_strategy)
        def check(kwargs: dict[str, Any]) -> None:
            instance = cls(**_kwargs_with_boxes_rebuilt(cls, kwargs))
            produced = instance.canonical_json()
            # Canonical output must have sorted keys at every level: reparsing
            # and re-dumping with sort_keys yields the identical string.
            reparsed = json.loads(produced)
            assert json.dumps(reparsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False) == produced

        check()

    @pytest.mark.parametrize("cls, kwargs_strategy", _CONTRACT_KWARGS, ids=lambda a: getattr(a, "__name__", ""))
    def test_roundtrip_through_json_is_stable(self, cls, kwargs_strategy):
        @given(kwargs=kwargs_strategy)
        def check(kwargs: dict[str, Any]) -> None:
            instance = cls(**_kwargs_with_boxes_rebuilt(cls, kwargs))
            canonical = instance.canonical_json()

            # model -> canonical_json -> model_validate_json -> canonical_json
            from_json = cls.model_validate_json(canonical)
            assert from_json.canonical_json() == canonical

            # model -> dict -> model_validate(dict) -> canonical_json
            from_dict = cls.model_validate(instance.model_dump(mode="json"))
            assert from_dict.canonical_json() == canonical

            # Equality is preserved across the round trip.
            assert from_json == instance
            assert from_dict == instance

        check()


# ---------------------------------------------------------------------------
# Example-based boundary-validation tests.
# Malformed data is rejected with a field-identifying structured error
# (Req 10.2, 10.5).
# ---------------------------------------------------------------------------


def _error_locs(exc: ValidationError) -> list[tuple[Any, ...]]:
    """All ``loc`` tuples reported by a ValidationError."""
    return [tuple(err["loc"]) for err in exc.errors()]


def _field_named(exc: ValidationError, field: str) -> bool:
    """True if any reported error location names ``field`` at any depth."""
    return any(field in loc for loc in _error_locs(exc))


class TestBoundaryValidationRejectsMalformedData:
    """Malformed boundary data raises a field-identifying ValidationError
    (Req 10.2, 10.5)."""

    def test_negative_bounding_box_width_is_rejected_naming_w(self):
        with pytest.raises(ValidationError) as excinfo:
            BoundingBox(x=0, y=0, w=-5, h=10)
        assert _field_named(excinfo.value, "w")

    def test_zero_bounding_box_height_is_rejected_naming_h(self):
        with pytest.raises(ValidationError) as excinfo:
            BoundingBox(x=0, y=0, w=10, h=0)
        assert _field_named(excinfo.value, "h")

    def test_negative_bounding_box_origin_is_rejected_naming_x(self):
        with pytest.raises(ValidationError) as excinfo:
            BoundingBox(x=-1, y=0, w=10, h=10)
        assert _field_named(excinfo.value, "x")

    def test_empty_space_ratio_above_one_is_rejected_naming_field(self):
        with pytest.raises(ValidationError) as excinfo:
            CandidateRegion(
                region_id="r1",
                bounding_box=BoundingBox(x=0, y=0, w=10, h=10),
                empty_space_ratio=1.5,
                heuristic_score=0.3,
                uncertainty_value=0.2,
                visual_caveat="possible shadow",
            )
        assert _field_named(excinfo.value, "empty_space_ratio")

    def test_empty_space_ratio_below_zero_is_rejected_naming_field(self):
        with pytest.raises(ValidationError) as excinfo:
            CandidateRegion(
                region_id="r1",
                bounding_box=BoundingBox(x=0, y=0, w=10, h=10),
                empty_space_ratio=-0.01,
                heuristic_score=0.3,
                uncertainty_value=0.2,
                visual_caveat="possible shadow",
            )
        assert _field_named(excinfo.value, "empty_space_ratio")

    def test_unknown_extra_field_is_rejected_naming_that_field(self):
        # extra="forbid" => the unexpected field is named in the error loc.
        with pytest.raises(ValidationError) as excinfo:
            BoundingBox(x=0, y=0, w=10, h=10, surprise=1)
        assert _field_named(excinfo.value, "surprise")

    def test_extra_field_on_vision_finding_is_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            VisionFinding(
                finding_id="f1",
                inspection_id="i1",
                source="image",
                opencv_version="5.0.0",
                annotated_output_ref="ref",
                suggested_follow_up="finalize_report",
                not_a_real_field=True,
            )
        assert _field_named(excinfo.value, "not_a_real_field")

    def test_empty_visual_caveat_is_rejected_naming_field(self):
        # Honesty guardrail: every region must carry a non-empty caveat (Req 4.7).
        with pytest.raises(ValidationError) as excinfo:
            CandidateRegion(
                region_id="r1",
                bounding_box=BoundingBox(x=0, y=0, w=10, h=10),
                empty_space_ratio=0.5,
                heuristic_score=0.3,
                uncertainty_value=0.2,
                visual_caveat="",
            )
        assert _field_named(excinfo.value, "visual_caveat")

    def test_missing_discriminator_in_trace_union_is_rejected(self):
        # A trace entry with no entry_type tag cannot be routed to a union member.
        bad_trace_entry = {
            "decision_id": "d1",
            "iteration": 0,
            "action": "finalize_report",
            "rationale": "done",
            "decided_by": "injected",
        }
        with pytest.raises(ValidationError) as excinfo:
            InspectionReport(
                inspection_id="i1",
                mode="local",
                final_recommendation="ok",
                requires_human_review=False,
                initial_finding_id="f1",
                decision_trace=[bad_trace_entry],
                created_at="2024-01-01T00:00:00Z",
                status="complete",
            )
        # The error points at the offending decision_trace element.
        assert _field_named(excinfo.value, "decision_trace")

    def test_unknown_discriminator_in_trace_union_is_rejected(self):
        bad_trace_entry = {
            "entry_type": "not_a_known_entry",
            "decision_id": "d1",
            "iteration": 0,
            "action": "finalize_report",
            "rationale": "done",
            "decided_by": "injected",
        }
        with pytest.raises(ValidationError) as excinfo:
            InspectionReport(
                inspection_id="i1",
                mode="local",
                final_recommendation="ok",
                requires_human_review=False,
                initial_finding_id="f1",
                decision_trace=[bad_trace_entry],
                created_at="2024-01-01T00:00:00Z",
                status="complete",
            )
        assert _field_named(excinfo.value, "decision_trace")

    def test_wrong_action_literal_is_rejected_naming_action(self):
        with pytest.raises(ValidationError) as excinfo:
            AgentDecision(
                decision_id="d1",
                iteration=0,
                action="teleport_shelf",  # not one of the five actions
                rationale="invalid",
                decided_by="injected",
            )
        assert _field_named(excinfo.value, "action")

    def test_wrong_tool_literal_is_rejected_naming_tool(self):
        with pytest.raises(ValidationError) as excinfo:
            ToolExecution(
                execution_id="e1",
                iteration=0,
                tool="rm_rf_slash",  # not one of the four tools
                result_ref="ref",
                status="ok",
            )
        assert _field_named(excinfo.value, "tool")

    def test_missing_required_field_is_rejected_naming_field(self):
        with pytest.raises(ValidationError) as excinfo:
            CandidateRegion(
                # region_id omitted
                bounding_box=BoundingBox(x=0, y=0, w=10, h=10),
                heuristic_score=0.3,
                uncertainty_value=0.2,
                visual_caveat="possible divider",
            )
        assert _field_named(excinfo.value, "region_id")

    def test_evaluation_correctness_out_of_range_is_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            EvaluationResult(
                eval_run_id="run1",
                is_demo=True,
                sample_count=3,
                agent_decision_correctness=1.2,  # outside [0, 1]
                latency_ms_p50=10.0,
                latency_ms_p95=20.0,
            )
        assert _field_named(excinfo.value, "agent_decision_correctness")

    def test_validate_boundary_rejects_mapping_with_bad_field(self):
        # The documented boundary entry point raises the same structured error.
        with pytest.raises(ValidationError) as excinfo:
            BoundingBox.validate_boundary({"x": 0, "y": 0, "w": 0, "h": 5})
        assert _field_named(excinfo.value, "w")


class TestBoundaryValidationAcceptsValidData:
    """A well-formed mapping validates cleanly through the boundary entry point."""

    def test_validate_boundary_accepts_valid_candidate_region(self):
        region = CandidateRegion.validate_boundary(
            {
                "region_id": "r1",
                "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
                "empty_space_ratio": 0.25,
                "heuristic_score": 0.7,
                "uncertainty_value": 0.4,
                "visual_caveat": "possible shadow near divider",
            }
        )
        assert region.region_id == "r1"
        assert region.bounding_box.w == 3
        # Canonical JSON is produced without error.
        assert region.canonical_json().startswith("{")
