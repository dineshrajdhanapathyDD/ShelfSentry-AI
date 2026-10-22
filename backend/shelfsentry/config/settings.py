"""Typed configuration for ShelfSentry AI (Pydantic v2).

Two frozen, validated config objects drive the mode-agnostic core:

* :class:`VisionConfig` - inputs the OpenCV 5 vision pipeline needs: accepted
  content types, size/frame limits, and the fixed kernel sizes and thresholds
  that keep classical CV deterministic (Req 5.1).
* :class:`AgentConfig` - inputs the bounded active-perception loop needs, most
  importantly ``max_iterations`` with a DEFAULT of 4 (Req 6.6).

Every field carries a documented default and is overridable: construct with
keyword arguments, or use :meth:`VisionConfig.from_overrides` /
:meth:`AgentConfig.from_overrides` to apply a partial override mapping (for
example from ``InspectionRequest.config_overrides`` or environment-derived
settings). The models are ``frozen`` so a resolved config cannot be mutated
mid-pipeline, which protects the determinism guarantees (Req 5.1).
"""

from __future__ import annotations

from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Accepted upload content types, shared default for both the server-side
# validator (Req 1.5) and the frontend's mirrored client-side check (Req 11.2).
DEFAULT_ACCEPTED_CONTENT_TYPES: tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",  # .avi
)

# 25 MiB default upload ceiling (Req 1.6). Large AWS-mode uploads go through a
# presigned URL, but the configured max still bounds what the pipeline accepts.
DEFAULT_MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024

# Default cap on sampled video frames (Req 3.1, 3.4).
DEFAULT_MAX_FRAME_COUNT: int = 16

# Default bounded agent-loop iteration cap (Req 6.6).
DEFAULT_MAX_ITERATIONS: int = 4


class VisionConfig(BaseModel):
    """Typed, validated inputs for the OpenCV 5 vision pipeline.

    Kernel sizes and thresholds are config-driven and fixed per run so the
    pipeline is deterministic for a given input and config (Req 5.1). All
    fields have documented defaults and are overridable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # --- input validation (Req 1.5, 1.6, 3.1) ---
    accepted_content_types: tuple[str, ...] = Field(
        default=DEFAULT_ACCEPTED_CONTENT_TYPES,
        description="Content types accepted for inspection input (Req 1.5).",
    )
    max_file_size_bytes: int = Field(
        default=DEFAULT_MAX_FILE_SIZE_BYTES,
        gt=0,
        description="Maximum accepted upload size in bytes (Req 1.6).",
    )
    max_frame_count: int = Field(
        default=DEFAULT_MAX_FRAME_COUNT,
        gt=0,
        description="Maximum number of frames sampled from a video (Req 3.1, 3.4).",
    )

    # --- deterministic CV kernel sizes (odd, positive) ---
    gaussian_blur_ksize: int = Field(
        default=5,
        gt=0,
        description="Gaussian blur kernel size (odd) used in preprocessing.",
    )
    morphology_ksize: int = Field(
        default=5,
        gt=0,
        description="Structuring-element size for morphologyEx open/close.",
    )

    # --- lighting normalization (CLAHE), fixed for determinism ---
    clahe_clip_limit: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "CLAHE contrast clip limit for lighting normalization in "
            "preprocessing. Fixed per run for determinism (Req 5.1)."
        ),
    )
    clahe_tile_grid_size: int = Field(
        default=8,
        gt=0,
        description=(
            "CLAHE tile grid size (square NxN) for lighting normalization. "
            "Fixed per run for determinism (Req 5.1)."
        ),
    )

    # --- deterministic CV thresholds ---
    canny_threshold_low: int = Field(
        default=50,
        ge=0,
        description="Lower hysteresis threshold for cv2.Canny edge detection.",
    )
    canny_threshold_high: int = Field(
        default=150,
        ge=0,
        description="Upper hysteresis threshold for cv2.Canny edge detection.",
    )
    binary_threshold: int = Field(
        default=127,
        ge=0,
        le=255,
        description="Fixed threshold for binary thresholding fallback (0-255).",
    )
    min_contour_area: float = Field(
        default=100.0,
        ge=0.0,
        description="Minimum contour area (px^2) for a region to be a candidate.",
    )
    empty_space_ratio_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Empty-space ratio at/above which a region is flagged empty.",
    )

    @field_validator("accepted_content_types", mode="before")
    @classmethod
    def _coerce_content_types(cls, value: Any) -> Any:
        """Accept any iterable of content types and normalize to a tuple."""
        if isinstance(value, (list, set, tuple)):
            return tuple(value)
        return value

    @field_validator("gaussian_blur_ksize", "morphology_ksize")
    @classmethod
    def _require_odd_kernel(cls, value: int) -> int:
        """OpenCV kernel sizes used here must be odd."""
        if value % 2 == 0:
            raise ValueError("kernel size must be an odd positive integer")
        return value

    @field_validator("accepted_content_types")
    @classmethod
    def _require_non_empty_content_types(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        """At least one accepted content type must be configured."""
        if not value:
            raise ValueError("accepted_content_types must not be empty")
        return value

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, Any] | None = None) -> "VisionConfig":
        """Build a config from defaults, applying any provided overrides.

        Unknown keys are rejected (``extra="forbid"``) so a misconfigured
        override fails loudly rather than being silently ignored.
        """
        return cls(**dict(overrides or {}))


class AgentConfig(BaseModel):
    """Typed, validated inputs for the bounded active-perception agent loop.

    ``max_iterations`` defaults to 4 (Req 6.6) and is overridable. All fields
    have documented defaults.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_iterations: int = Field(
        default=DEFAULT_MAX_ITERATIONS,
        gt=0,
        description="Maximum active-perception iterations; default 4 (Req 6.6).",
    )
    sufficient_uncertainty: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Measurement-confidence level at/above which the loop may treat "
            "evidence as sufficient and finalize."
        ),
    )
    conflict_margin: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Disagreement margin across trace evidence that triggers "
            "request_human_review in the deterministic fallback policy."
        ),
    )

    @classmethod
    def from_overrides(cls, overrides: Mapping[str, Any] | None = None) -> "AgentConfig":
        """Build a config from defaults, applying any provided overrides."""
        return cls(**dict(overrides or {}))
