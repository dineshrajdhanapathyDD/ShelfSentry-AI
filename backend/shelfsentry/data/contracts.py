"""The seven typed Pydantic v2 data contracts for ShelfSentry AI.

This module defines the system's exchange schemas (Req 10.1), their boundary
validation with field-identifying structured errors (Req 10.2, 10.5), and a
canonical JSON serialization so equivalent objects serialize identically
(Req 10.3).

Design references: see design.md "Data Models". The seven contracts are
``InspectionRequest``, ``VisionFinding``, ``CandidateRegion``,
``AgentDecision``, ``ToolExecution``, ``InspectionReport`` (whose
``decision_trace`` is a discriminated union of ``AgentDecision | ToolExecution``),
and ``EvaluationResult``. Shared value types ``BoundingBox`` and the
``AgentAction`` literal set are defined here as well.

Honesty guardrails (Req 27) are encoded structurally: ``VisionFinding`` always
labels results as estimates (``is_estimate`` is a ``Literal[True]``), every
``CandidateRegion`` carries a non-empty ``visual_caveat`` (Req 4.7, 27.4), the
``heuristic_score`` is documented as an ordinal rule output kept distinct from a
calibrated probability (Req 2.6), and ``EvaluationResult`` is pinned to
measured-only data (``measured_only`` is a ``Literal[True]``; Req 25.4, 27.2).
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = [
    "SCHEMA_VERSION",
    "AGENT_ACTIONS",
    "MCP_TOOL_NAMES",
    "AgentAction",
    "McpToolName",
    "BoundingBox",
    "CandidateRegion",
    "VisionFinding",
    "AgentDecision",
    "ToolExecution",
    "TraceEntry",
    "InspectionRequest",
    "InspectionReport",
    "EvalExample",
    "EvaluationResult",
    "CanonicalModel",
    "canonical_json",
]

# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------

#: Current contract schema version carried by every model (Req 10). Bump this
#: when a contract changes shape in a non-backward-compatible way.
SCHEMA_VERSION: str = "1.0"


# ---------------------------------------------------------------------------
# Shared literal sets
# ---------------------------------------------------------------------------

#: The fixed five-action set the agent may choose from (Req 6.1). Kept as a
#: tuple so other modules can iterate/validate against it without importing a
#: typing construct.
AGENT_ACTIONS: tuple[str, ...] = (
    "inspect_crop",
    "analyze_another_frame",
    "request_better_image",
    "request_human_review",
    "finalize_report",
)

#: The names of the four restricted MCP vision tools (Req 8.2).
MCP_TOOL_NAMES: tuple[str, ...] = (
    "analyze_shelf_image",
    "inspect_region_crop",
    "compare_shelf_frames",
    "get_annotated_evidence",
)

#: One of the fixed five agent actions (Req 6.1).
AgentAction = Literal[
    "inspect_crop",
    "analyze_another_frame",
    "request_better_image",
    "request_human_review",
    "finalize_report",
]

#: One of the four restricted MCP vision tool names (Req 8.2).
McpToolName = Literal[
    "analyze_shelf_image",
    "inspect_region_crop",
    "compare_shelf_frames",
    "get_annotated_evidence",
]


# ---------------------------------------------------------------------------
# Canonical base model
# ---------------------------------------------------------------------------

_Model = TypeVar("_Model", bound="CanonicalModel")


class CanonicalModel(BaseModel):
    """Base class for every ShelfSentry contract.

    Provides:

    * Strict boundary validation. ``extra="forbid"`` makes unexpected fields a
      field-identifying error (Req 10.2, 10.5); ``validate_assignment`` keeps
      mutation honest; ``str_strip_whitespace`` normalizes incoming strings so
      equivalent objects serialize identically (Req 10.3).
    * Canonical JSON serialization via :meth:`canonical_json` (Req 10.3).
    * A :meth:`validate_boundary` helper that always raises a structured,
      field-identifying error on malformed data (Req 10.2, 10.5).
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        frozen=False,
    )

    def canonical_json(self) -> str:
        """Return a canonical JSON string for this object.

        Serialization is stable and order-independent: the model is first
        dumped in JSON mode (so enums/literals, etc. become JSON-native
        values), then re-serialized with ``sort_keys=True`` and compact,
        fixed separators. Two objects that are equal therefore produce
        byte-for-byte identical JSON regardless of the order in which their
        fields were assigned (Req 10.3).
        """
        payload = self.model_dump(mode="json")
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def validate_boundary(cls: type[_Model], data: Any) -> _Model:
        """Validate ``data`` at a component boundary (Req 10.2, 10.5).

        Accepts a mapping (or any input Pydantic can validate) and returns a
        validated instance. On failure it raises :class:`pydantic.ValidationError`,
        whose entries name the specific failing field(s) and the reason, which
        satisfies the "error identifying the failing field" requirement.
        """
        return cls.model_validate(data)


def canonical_json(obj: CanonicalModel) -> str:
    """Module-level convenience wrapper for :meth:`CanonicalModel.canonical_json`."""
    return obj.canonical_json()


# ---------------------------------------------------------------------------
# Shared value type: BoundingBox
# ---------------------------------------------------------------------------


class BoundingBox(CanonicalModel):
    """An integer-pixel bounding box in one documented coordinate system.

    Origin is top-left; coordinates are ``(x, y, w, h)`` relative to the
    analyzed input's dimensions (design.md Determinism note, Req 5.2). ``x`` and
    ``y`` are non-negative pixel offsets; ``w`` and ``h`` are at least one pixel
    so a box always covers real area. Out-of-range values raise a
    field-identifying error (Req 10.2, 10.5).
    """

    x: int = Field(..., ge=0, description="Left edge, pixels from the origin (top-left).")
    y: int = Field(..., ge=0, description="Top edge, pixels from the origin (top-left).")
    w: int = Field(..., ge=1, description="Width in pixels; at least 1.")
    h: int = Field(..., ge=1, description="Height in pixels; at least 1.")


# ---------------------------------------------------------------------------
# CandidateRegion (Req 4, 5, 27)
# ---------------------------------------------------------------------------


class CandidateRegion(CanonicalModel):
    """A single region of interest within an analyzed input (Req 4.2)."""

    region_id: str = Field(..., min_length=1, description="Stable identifier for the region.")
    bounding_box: BoundingBox = Field(
        ..., description="(x, y, w, h) integer pixels, top-left origin (Req 5.2)."
    )
    empty_space_ratio: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Measured fraction of the box estimated to be empty shelf space "
            "(Req 4.2). A measured visual metric reported as an estimate, NOT a "
            "calibrated out-of-stock probability. None when a documented "
            "alternative metric is used instead (see alt_metric)."
        ),
    )
    alt_metric: dict[str, Any] | None = Field(
        default=None,
        description="Documented alternative visual metric used when empty_space_ratio is N/A.",
    )
    heuristic_score: float = Field(
        ...,
        description=(
            "Rule-derived ordinal score used to rank regions (Req 2.6). "
            "Explicitly NOT a probability and not calibrated; it combines "
            "empty-space ratio, region size, and edge sparsity for ordering only."
        ),
    )
    uncertainty_value: float = Field(
        ...,
        description=(
            "Confidence/uncertainty over the measurement, interpreted via the "
            "finding's uncertainty_interpretations map (Req 4.2, 4.3). Never a "
            "calibrated out-of-stock probability."
        ),
    )
    frame_id: str | None = Field(
        default=None,
        description="Source frame identifier when the region derives from a video frame (Req 4.4).",
    )
    visual_caveat: str = Field(
        ...,
        min_length=1,
        description=(
            "Human-readable caveat (shadow / occlusion / dividers / image "
            "quality) attached to every region so an empty-looking region is "
            "never asserted as a proven out-of-stock condition (Req 4.7, 27.4)."
        ),
    )


# ---------------------------------------------------------------------------
# VisionFinding (Req 2, 4, 5)
# ---------------------------------------------------------------------------


class VisionFinding(CanonicalModel):
    """A structured, uncertainty-quantified result from the vision pipeline."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    finding_id: str = Field(..., min_length=1)
    inspection_id: str = Field(..., min_length=1)
    source: Literal["image", "video_frame", "crop", "frame_compare"]
    frame_id: str | None = Field(
        default=None, description="Present for video-derived findings (Req 4.4)."
    )
    opencv_version: str = Field(
        ...,
        min_length=1,
        description="Exact runtime OpenCV version used for the analysis (Req 2.1).",
    )
    candidate_regions: list[CandidateRegion] = Field(
        default_factory=list, description="Zero or more candidate regions (Req 4.1)."
    )
    uncertainty_interpretations: dict[str, str] = Field(
        default_factory=dict,
        description="Documented meaning for each uncertainty value in this finding (Req 4.3).",
    )
    annotated_output_ref: str = Field(
        ..., min_length=1, description="Storage reference to the Annotated_Output (Req 4.5)."
    )
    suggested_follow_up: AgentAction = Field(
        ..., description="Suggested next agent action hint (Req 4.6)."
    )
    is_estimate: Literal[True] = Field(
        default=True, description="Classical CV results are always labeled estimates (Req 2.5)."
    )


# ---------------------------------------------------------------------------
# AgentDecision (Req 6, 7)
# ---------------------------------------------------------------------------


class AgentDecision(CanonicalModel):
    """A record of one action the agent selected, including its rationale."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    #: Discriminator tag for the TraceEntry union. Fixed literal value.
    entry_type: Literal["agent_decision"] = Field(
        default="agent_decision",
        description="Discriminator for the InspectionReport.decision_trace union.",
    )
    decision_id: str = Field(..., min_length=1)
    iteration: int = Field(..., ge=0)
    action: AgentAction = Field(..., description="One of the fixed five actions (Req 6.1).")
    rationale: str = Field(..., min_length=1, description="Why this action was selected (Req 6.3).")
    args: dict[str, Any] = Field(
        default_factory=dict, description="Action arguments, validated before use (Req 6.4)."
    )
    decided_by: Literal["bedrock", "fallback", "injected"]
    used_fallback: bool = Field(
        default=False, description="True when the deterministic fallback policy was used (Req 7.2)."
    )
    fallback_reason: str | None = Field(
        default=None, description="Reason the fallback engaged, when used_fallback is True."
    )


# ---------------------------------------------------------------------------
# ToolExecution (Req 6)
# ---------------------------------------------------------------------------


class ToolExecution(CanonicalModel):
    """A record of one MCP tool call the agent made and its result."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    #: Discriminator tag for the TraceEntry union. Fixed literal value.
    entry_type: Literal["tool_execution"] = Field(
        default="tool_execution",
        description="Discriminator for the InspectionReport.decision_trace union.",
    )
    execution_id: str = Field(..., min_length=1)
    iteration: int = Field(..., ge=0)
    tool: McpToolName = Field(..., description="One of the four restricted MCP tools (Req 8.2).")
    args: dict[str, Any] = Field(default_factory=dict)
    result_ref: str = Field(
        ..., min_length=1, description="Reference to the resulting finding/evidence."
    )
    status: Literal["ok", "rejected", "error"]
    error: str | None = Field(
        default=None, description="Error detail when status is 'rejected' or 'error'."
    )


# ---------------------------------------------------------------------------
# TraceEntry discriminated union (AgentDecision | ToolExecution)
# ---------------------------------------------------------------------------

#: An ordered Decision_Trace entry: either an AgentDecision or a ToolExecution,
#: discriminated on the ``entry_type`` tag (Req 6.10). Using a Field
#: discriminator lets Pydantic route raw dicts to the correct model and produce
#: a field-identifying error when the tag is missing or unknown.
TraceEntry = Annotated[
    AgentDecision | ToolExecution,
    Field(discriminator="entry_type"),
]


# ---------------------------------------------------------------------------
# InspectionRequest (Req 1, 11)
# ---------------------------------------------------------------------------


class InspectionRequest(CanonicalModel):
    """A typed input describing an image or video to inspect."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    inspection_id: str = Field(..., min_length=1)
    media_kind: Literal["image", "video"]
    content_type: str = Field(
        ..., min_length=1, description="Checked against the accepted set at validation (Req 1.5)."
    )
    size_bytes: int = Field(
        ..., ge=0, description="File size in bytes; checked against the max (Req 1.6)."
    )
    media_ref: str = Field(
        ...,
        min_length=1,
        description="Storage reference resolved by the StorageAdapter; never a raw path or URL.",
    )
    config_overrides: dict[str, Any] | None = Field(default=None)


# ---------------------------------------------------------------------------
# InspectionReport (Req 6, 12, 16)
# ---------------------------------------------------------------------------


class InspectionReport(CanonicalModel):
    """The evidence-backed final output of an inspection."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    inspection_id: str = Field(..., min_length=1)
    mode: Literal["local", "aws"] = Field(
        ..., description="Operating mode that produced this report (Req 17)."
    )
    final_recommendation: str = Field(..., min_length=1)
    requires_human_review: bool = Field(..., description="Set when human review is required (Req 6.9).")
    initial_finding_id: str = Field(..., min_length=1)
    decision_trace: list[TraceEntry] = Field(
        default_factory=list,
        description="Ordered AgentDecision/ToolExecution entries, the Decision_Trace (Req 6.10).",
    )
    supporting_evidence_refs: list[str] = Field(
        default_factory=list,
        description="Trace/evidence references that back the final recommendation (Req 6.10).",
    )
    created_at: str = Field(..., min_length=1, description="ISO-8601 creation timestamp.")
    status: Literal["complete", "failed", "needs_review"]


# ---------------------------------------------------------------------------
# EvaluationResult (Req 14, 25, 27)
# ---------------------------------------------------------------------------


class EvalExample(CanonicalModel):
    """A single labeled example from the evaluation set (measured only; Req 25.2)."""

    sample_id: str = Field(..., min_length=1, description="Identifier of the evaluation sample.")
    inspection_id: str | None = Field(
        default=None, description="Inspection that produced this measured outcome, when applicable."
    )
    expected: str = Field(..., min_length=1, description="The ground-truth label for the sample.")
    actual: str = Field(..., min_length=1, description="The measured system output for the sample.")
    note: str | None = Field(default=None, description="Optional human-readable note.")


class EvaluationResult(CanonicalModel):
    """A record of measured performance against a labeled evaluation set."""

    schema_version: str = Field(default=SCHEMA_VERSION)
    eval_run_id: str = Field(..., min_length=1)
    is_demo: bool = Field(..., description="Distinguishes demo records from real records (Req 14.5).")
    sample_count: int = Field(..., ge=0)
    false_positives: list[EvalExample] = Field(
        default_factory=list, description="Measured false positives (Req 25.2)."
    )
    false_negatives: list[EvalExample] = Field(
        default_factory=list, description="Measured false negatives (Req 25.2)."
    )
    agent_decision_correctness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Measured-only decision correctness in [0, 1] (Req 14.4, 27.2).",
    )
    latency_ms_p50: float = Field(..., ge=0.0)
    latency_ms_p95: float = Field(..., ge=0.0)
    documented_limitations: list[str] = Field(
        default_factory=list, description="Documented limitations reported alongside results (Req 25.3)."
    )
    measured_only: Literal[True] = Field(
        default=True,
        description="Results are measured only and never fabricated (Req 25.4, 27.2).",
    )

    @field_validator("false_positives", "false_negatives")
    @classmethod
    def _examples_are_examples(cls, v: list[EvalExample]) -> list[EvalExample]:
        # Pydantic already enforces element types; this validator exists so a
        # misuse (e.g. a non-list) surfaces against the named field.
        return v
