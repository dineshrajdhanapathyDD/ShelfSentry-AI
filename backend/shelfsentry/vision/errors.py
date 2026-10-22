"""Structured vision-pipeline errors (Req 1.3-1.6, 10.5).

Input validation rejects a bad :class:`~shelfsentry.data.InspectionRequest`
with a *structured* error that names the specific failing condition rather than
a bare string, so callers (the API layer, the frontend) can branch on a stable
machine-readable code and still show an honest human message.

The single error type is :class:`InputValidationError`. Its
:attr:`~InputValidationError.condition` is one of :data:`VALIDATION_CONDITIONS`:

* ``empty_file`` - the submitted file had zero bytes (Req 1.3).
* ``undecodable`` - the bytes could not be decoded as a valid image or video
  (the corrupt-or-invalid condition, Req 1.4).
* ``unsupported_content_type`` - the content type is not in the configured
  accepted set (Req 1.5).
* ``size_exceeded`` - the file is larger than the configured maximum (Req 1.6).
"""

from __future__ import annotations

from typing import Any, Literal

__all__ = [
    "VALIDATION_CONDITIONS",
    "ValidationCondition",
    "InputValidationError",
]

#: The machine-readable validation condition codes, one per rejectable
#: condition in Requirement 1. Kept as a tuple so other modules (tests, the API
#: layer) can validate/iterate against it without importing a typing construct.
VALIDATION_CONDITIONS: tuple[str, ...] = (
    "empty_file",
    "undecodable",
    "unsupported_content_type",
    "size_exceeded",
)

#: One of the fixed validation condition codes above.
ValidationCondition = Literal[
    "empty_file",
    "undecodable",
    "unsupported_content_type",
    "size_exceeded",
]


class InputValidationError(ValueError):
    """Raised when an inspection input fails validation (Req 1.3-1.6).

    Carries a machine-readable :attr:`condition` code and a human-readable
    message. Extends :class:`ValueError` so generic ``except ValueError``
    handlers still catch it, while code that wants to branch on the specific
    condition can inspect :attr:`condition`.

    Attributes:
        condition: One of :data:`VALIDATION_CONDITIONS`.
        message: Human-readable explanation naming the specific condition.
        details: Optional structured context (for example the offending size
            and the configured limit) for logging or API responses. Never
            contains the raw file bytes.
    """

    def __init__(
        self,
        condition: ValidationCondition,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        if condition not in VALIDATION_CONDITIONS:
            # Fail loudly on an unknown code rather than silently storing it;
            # the condition set is a closed contract (Req 1.3-1.6).
            raise ValueError(
                f"Unknown validation condition {condition!r}; expected one of "
                f"{VALIDATION_CONDITIONS}."
            )
        super().__init__(message)
        self.condition: ValidationCondition = condition
        self.message: str = message
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable structured representation of the error.

        Shaped for an API error envelope (Req 10.5): a stable ``condition``
        code plus a human ``message`` and any structured ``details``.
        """
        return {
            "error": "input_validation_error",
            "condition": self.condition,
            "message": self.message,
            "details": self.details,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"{type(self).__name__}(condition={self.condition!r}, "
            f"message={self.message!r}, details={self.details!r})"
        )
