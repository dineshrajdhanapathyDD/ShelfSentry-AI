"""Vision pipeline: OpenCV 5 validation, preprocessing, analysis, scoring,
annotation, and video frame sampling. No LLM dependency."""

from shelfsentry.vision.analyze import (
    AnalysisResult,
    RegionCandidate,
    ShelfBand,
    analyze_image_arrays,
    analyze_image_bytes,
    detect_shelf_lines,
)
from shelfsentry.vision.annotate import (
    build_annotated_finding,
    render_annotation,
)
from shelfsentry.vision.entrypoints import (
    FRAME_COMPARE_INTERPRETATIONS,
    BoxLike,
    RegionInspectionError,
    compare_frames,
    inspect_region,
)
from shelfsentry.vision.errors import (
    VALIDATION_CONDITIONS,
    InputValidationError,
    ValidationCondition,
)
from shelfsentry.vision.preprocess import (
    PreprocessError,
    PreprocessResult,
    decode_bgr,
    preprocess_image,
)
from shelfsentry.vision.scoring import (
    UNCERTAINTY_INTERPRETATIONS,
    build_candidate_region,
    build_finding,
    build_visual_caveat,
    compute_edge_density,
    compute_empty_space_ratio,
    compute_heuristic_score,
    compute_uncertainty_value,
    suggested_follow_up,
)
from shelfsentry.vision.validate import (
    media_kind_for_content_type,
    validate_input,
)
from shelfsentry.vision.video import (
    AnnotatedRefCallback,
    VideoDecodeError,
    analyze_video_bytes,
    frame_id_for_index,
    sample_frame_indices,
)
from shelfsentry.vision.version import (
    REQUIRED_OPENCV_MAJOR,
    OpenCVVersionError,
    opencv_version,
    verify_opencv_version,
)

__all__ = [
    "opencv_version",
    "verify_opencv_version",
    "OpenCVVersionError",
    "REQUIRED_OPENCV_MAJOR",
    "validate_input",
    "media_kind_for_content_type",
    "InputValidationError",
    "ValidationCondition",
    "VALIDATION_CONDITIONS",
    "preprocess_image",
    "decode_bgr",
    "PreprocessResult",
    "PreprocessError",
    "analyze_image_arrays",
    "analyze_image_bytes",
    "detect_shelf_lines",
    "AnalysisResult",
    "RegionCandidate",
    "ShelfBand",
    "build_finding",
    "build_candidate_region",
    "build_visual_caveat",
    "compute_empty_space_ratio",
    "compute_edge_density",
    "compute_heuristic_score",
    "compute_uncertainty_value",
    "suggested_follow_up",
    "UNCERTAINTY_INTERPRETATIONS",
    "sample_frame_indices",
    "analyze_video_bytes",
    "frame_id_for_index",
    "VideoDecodeError",
    "AnnotatedRefCallback",
    "render_annotation",
    "build_annotated_finding",
    "inspect_region",
    "compare_frames",
    "RegionInspectionError",
    "BoxLike",
    "FRAME_COMPARE_INTERPRETATIONS",
]
