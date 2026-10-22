"""Versioned API response envelopes (task 11.1; Req 10.4, 10.5).

Every value the API layer returns to a caller is wrapped in one of two
envelopes, and **both** carry the two cross-cutting identifiers the contract
requires on every response (Req 10.4 and design.md "report (apiVersion, mode)"):

* ``apiVersion`` - a stable API version identifier, independent of the per-model
  ``schema_version`` carried inside the contract payloads. Bump it when the
  envelope/handler surface changes shape.
* ``mode`` - the operating mode that produced the response, ``"local"`` or
  ``"aws"`` (design.md mode labeling; Req 17 selects it). The local factory
  stamps ``"local"``; an AWS wiring stamps ``"aws"`` without changing this type.

Two shapes:

* :class:`ApiResponse` - a success envelope ``{apiVersion, mode, data}`` whose
  ``data`` is the typed contract payload (an :class:`InspectionReport`, a list of
  :class:`InspectionSummary`, a list of :class:`EvaluationResult`, raw evidence
  bytes rendered as a small descriptor, etc.).
* :class:`ApiError` - a structured error envelope
  ``{apiVersion, mode, error: {code, message, field?}}`` (Req 10.5). Input
  validation failures and boundary schema failures are *mapped* onto this shape
  rather than raised as exceptions across the handler boundary, so a caller
  always receives a well-formed envelope it can branch on.

These envelopes are intentionally framework-agnostic plain Pydantic models: they
describe the response contract, not an HTTP transport. A FastAPI/Lambda adapter
can serialize them directly; the local tests consume them as objects.
"""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from shelfsentry.vision.errors import InputValidationError

__all__ = [
    "API_VERSION",
    "ApiMode",
    "ErrorCode",
    "ApiErrorDetail",
    "ApiError",
    "ApiResponse",
    "error_from_input_validation",
    "error_from_validation_error",
    "not_found_error",
]

#: The API version identifier stamped on every envelope (Req 10.4). This is the
#: *envelope/handler* surface version and is deliberately separate from the
#: per-contract ``schema_version`` (contracts.SCHEMA_VERSION).
API_VERSION: str = "v1"

#: Operating mode label carried on every response (design.md; Req 17).
ApiMode = Literal["local", "aws"]

#: Stable, machine-readable error codes the API layer may return. ``validation``
#: covers both input-validation and schema-boundary failures (Req 1.3-1.6,
#: 10.5); ``not_found`` is the 404-style structured error for a missing record;
#: ``bad_request`` covers a malformed call that is not a schema failure.
ErrorCode = Literal["validation", "not_found", "bad_request"]

_DataT = TypeVar("_DataT")


class ApiErrorDetail(BaseModel):
    """The ``error`` object inside a structured error envelope (Req 10.5).

    ``code`` is a stable machine-readable identifier; ``message`` is an honest
    human-readable explanation; ``field`` names the offending field/condition
    when one applies (for input validation it carries the validation
    ``condition`` code; for a schema failure it carries the dotted field path).
    """

    model_config = ConfigDict(extra="forbid")

    code: ErrorCode
    message: str = Field(..., min_length=1)
    field: str | None = Field(
        default=None,
        description="Offending field/condition when applicable (Req 10.5).",
    )
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured context (never raw media bytes).",
    )


class ApiError(BaseModel):
    """Structured error envelope: ``{apiVersion, mode, error}`` (Req 10.4, 10.5).

    Returned (not raised) by handlers so a caller always receives a well-formed,
    versioned, mode-stamped envelope it can branch on.
    """

    model_config = ConfigDict(extra="forbid")

    apiVersion: str = Field(default=API_VERSION)
    mode: ApiMode
    error: ApiErrorDetail

    @property
    def ok(self) -> bool:
        """Always ``False``; lets callers test ``resp.ok`` uniformly."""
        return False


class ApiResponse(BaseModel, Generic[_DataT]):
    """Success envelope: ``{apiVersion, mode, data}`` (Req 10.4).

    ``data`` is the typed contract payload. The envelope is generic so the
    payload type is preserved for typing, but Pydantic validates the concrete
    value at construction.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    apiVersion: str = Field(default=API_VERSION)
    mode: ApiMode
    data: _DataT

    @property
    def ok(self) -> bool:
        """Always ``True``; lets callers test ``resp.ok`` uniformly."""
        return True


# ---------------------------------------------------------------------------
# Error mapping helpers (exceptions -> structured error envelope; Req 10.5)
# ---------------------------------------------------------------------------


def error_from_input_validation(
    exc: InputValidationError, *, mode: ApiMode
) -> ApiError:
    """Map an :class:`InputValidationError` onto an error envelope (Req 1, 10.5).

    The structured ``condition`` code (``empty_file``, ``undecodable``,
    ``unsupported_content_type``, ``size_exceeded``) is surfaced as the
    envelope's ``field`` so a caller can branch on the exact failing condition,
    and the error's structured ``details`` are carried through unchanged.
    """
    return ApiError(
        mode=mode,
        error=ApiErrorDetail(
            code="validation",
            message=exc.message,
            field=exc.condition,
            details=dict(exc.details) or None,
        ),
    )


def error_from_validation_error(exc: ValidationError, *, mode: ApiMode) -> ApiError:
    """Map a Pydantic :class:`ValidationError` onto an error envelope (Req 10.5).

    The first error's dotted location becomes ``field`` so the offending field
    is identified (Req 10.2, 10.5); the human message summarizes the failure.
    """
    errors = exc.errors()
    first = errors[0] if errors else None
    field = None
    if first is not None:
        loc = first.get("loc", ())
        field = ".".join(str(p) for p in loc) if loc else None
    message = (
        f"{first.get('msg', 'validation error')}"
        if first is not None
        else "schema validation failed"
    )
    return ApiError(
        mode=mode,
        error=ApiErrorDetail(
            code="validation",
            message=message,
            field=field,
            details={"error_count": len(errors)} if errors else None,
        ),
    )


def not_found_error(message: str, *, mode: ApiMode, field: str | None = None) -> ApiError:
    """Build a 404-style structured error envelope for a missing record."""
    return ApiError(
        mode=mode,
        error=ApiErrorDetail(code="not_found", message=message, field=field),
    )
