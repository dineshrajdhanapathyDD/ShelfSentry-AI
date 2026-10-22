"""Configuration: typed VisionConfig/AgentConfig, the Local/AWS mode switch,
and OpenCV 5 startup version verification."""

from shelfsentry.config.mode import (
    DEFAULT_LOCAL_STORAGE_DIR,
    ENV_AWS_REGION,
    ENV_DDB_TABLE,
    ENV_LOCAL_STORAGE_DIR,
    ENV_MODE,
    ENV_S3_BUCKET,
    AwsSettings,
    LocalSettings,
    Mode,
    ModeConfigurationError,
    RuntimeConfig,
    resolve_mode,
    resolve_runtime_config,
)
from shelfsentry.config.settings import (
    DEFAULT_ACCEPTED_CONTENT_TYPES,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    DEFAULT_MAX_FRAME_COUNT,
    DEFAULT_MAX_ITERATIONS,
    AgentConfig,
    VisionConfig,
)

__all__ = [
    "VisionConfig",
    "AgentConfig",
    "DEFAULT_ACCEPTED_CONTENT_TYPES",
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "DEFAULT_MAX_FRAME_COUNT",
    "DEFAULT_MAX_ITERATIONS",
    # Local/AWS mode configuration switch (task 11.2; Req 16.4, 17.1-17.3)
    "Mode",
    "ModeConfigurationError",
    "RuntimeConfig",
    "LocalSettings",
    "AwsSettings",
    "resolve_mode",
    "resolve_runtime_config",
    "ENV_MODE",
    "ENV_S3_BUCKET",
    "ENV_DDB_TABLE",
    "ENV_AWS_REGION",
    "ENV_LOCAL_STORAGE_DIR",
    "DEFAULT_LOCAL_STORAGE_DIR",
]
