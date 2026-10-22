"""The explicit Local/AWS mode configuration switch (task 11.2).

A single ``Config_Switch`` selects the system topology (Req 17.1): the
``SHELFSENTRY_MODE`` environment variable chooses ``"local"`` (the default) or
``"aws"``. The switch is resolved **once at startup** into a frozen
:class:`RuntimeConfig`, and the resolution rules are deliberately strict:

* LOCAL mode (the default) requires **no** AWS configuration. It resolves to a
  local runtime carrying a local storage base directory and nothing cloud
  specific (Req 15.x, 17.1).
* AWS mode requires the non-secret AWS integration keys the deployed topology
  needs (Req 16.1): an S3 bucket name, a DynamoDB table name, and an AWS
  region. If AWS mode is selected and **any** of those keys is missing, startup
  **fails loudly** with a :class:`ModeConfigurationError` that names **every**
  missing key (Req 17.2).
* AWS mode **never** silently falls back to local/mock behavior when its
  configuration is incomplete (Req 16.4, 17.3). The resolver's only responses
  to an incompletely-configured AWS selection are (a) a fully-resolved AWS
  config or (b) a raised error - never a local substitute.

Secrets are intentionally out of scope here. Only non-secret resource
*names*/identifiers (bucket, table, region) are read from the environment;
AWS credentials are resolved later by the standard boto3 provider chain, so
this module never reads, stores, or logs a credential (Req 20.2).
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Mode",
    "ModeConfigurationError",
    "ENV_MODE",
    "ENV_S3_BUCKET",
    "ENV_DDB_TABLE",
    "ENV_AWS_REGION",
    "ENV_LOCAL_STORAGE_DIR",
    "DEFAULT_LOCAL_STORAGE_DIR",
    "AwsSettings",
    "LocalSettings",
    "RuntimeConfig",
    "resolve_mode",
    "resolve_runtime_config",
]


# ---------------------------------------------------------------------------
# Environment variable names (the Config_Switch surface)
# ---------------------------------------------------------------------------

#: Selects the mode. Values: ``"local"`` (default) or ``"aws"`` (Req 17.1).
ENV_MODE = "SHELFSENTRY_MODE"

#: Required AWS-mode keys (non-secret resource identifiers only; Req 16.1).
ENV_S3_BUCKET = "SHELFSENTRY_S3_BUCKET"
ENV_DDB_TABLE = "SHELFSENTRY_DDB_TABLE"
ENV_AWS_REGION = "SHELFSENTRY_AWS_REGION"

#: Optional LOCAL-mode override for the on-disk storage base directory.
ENV_LOCAL_STORAGE_DIR = "SHELFSENTRY_LOCAL_STORAGE_DIR"

#: Default local storage base directory when the override is unset (Req 15.4).
DEFAULT_LOCAL_STORAGE_DIR = ".shelfsentry-local"


class Mode(str, Enum):
    """The two mutually-exclusive runtime topologies (Req 17.1)."""

    LOCAL = "local"
    AWS = "aws"


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class ModeConfigurationError(Exception):
    """Startup configuration is invalid for the selected mode.

    Raised when an unknown mode value is supplied, or when AWS mode is selected
    but one or more required AWS configuration keys are missing (Req 17.2). The
    message names each offending key so the operator can fix every problem in
    one pass; the missing keys are also available on :attr:`missing_keys`.

    Because this is raised instead of resolving, an incompletely-configured AWS
    selection can never silently become a local/mock runtime (Req 16.4, 17.3).
    """

    def __init__(self, message: str, *, missing_keys: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        #: The environment-variable names that were required but absent (if any).
        self.missing_keys: tuple[str, ...] = tuple(missing_keys)


# ---------------------------------------------------------------------------
# Mode-specific resolved settings
# ---------------------------------------------------------------------------


class LocalSettings(BaseModel):
    """Resolved LOCAL-mode settings: no AWS configuration required (Req 17.1).

    Carries only the local storage base directory used by the on-disk
    :class:`~shelfsentry.data.storage.LocalStorageAdapter` (Req 15.4).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    storage_dir: str = Field(
        default=DEFAULT_LOCAL_STORAGE_DIR,
        min_length=1,
        description="Base directory for local on-disk storage (Req 15.4).",
    )


class AwsSettings(BaseModel):
    """Resolved AWS-mode settings: the non-secret integration keys (Req 16.1).

    These are the resource identifiers the AWS integration (task 16) needs to
    address deployed infrastructure. Credentials are **not** here: they are
    resolved by the standard boto3 provider chain at call time, so no secret is
    read or stored by the mode switch (Req 20.2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    s3_bucket: str = Field(
        ..., min_length=1, description="S3 bucket name for media/data (Req 16.1)."
    )
    ddb_table: str = Field(
        ..., min_length=1, description="DynamoDB table name for records (Req 16.1)."
    )
    region: str = Field(
        ..., min_length=1, description="AWS region of the deployed stack (Req 16.1)."
    )


class RuntimeConfig(BaseModel):
    """The once-at-startup resolved application configuration (the switch result).

    Exactly one of :attr:`local`/:attr:`aws` is populated, matching
    :attr:`mode`. The object is frozen so the resolved mode cannot be mutated
    mid-run, keeping the active topology unambiguous (Req 17.1).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Mode = Field(..., description="The resolved active mode (Req 17.1).")
    local: LocalSettings | None = Field(
        default=None, description="Populated iff mode is LOCAL."
    )
    aws: AwsSettings | None = Field(
        default=None, description="Populated iff mode is AWS."
    )

    @property
    def is_local(self) -> bool:
        """True iff the resolved mode is LOCAL."""
        return self.mode is Mode.LOCAL

    @property
    def is_aws(self) -> bool:
        """True iff the resolved mode is AWS."""
        return self.mode is Mode.AWS


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

# The AWS-mode required keys, paired with the attribute each feeds, in a fixed
# order so error messages and resolution are deterministic.
_AWS_REQUIRED_KEYS: tuple[tuple[str, str], ...] = (
    (ENV_S3_BUCKET, "s3_bucket"),
    (ENV_DDB_TABLE, "ddb_table"),
    (ENV_AWS_REGION, "region"),
)


def _env_mapping(env: Mapping[str, str] | None) -> Mapping[str, str]:
    """Return the environment mapping to read from (process env by default)."""
    return os.environ if env is None else env


def _get_clean(env: Mapping[str, str], key: str) -> str | None:
    """Return ``env[key]`` stripped of surrounding whitespace, or ``None``.

    A key that is absent, or present but blank/whitespace-only, is treated as
    missing - a blank required value must fail just as loudly as an absent one
    (Req 17.2).
    """
    raw = env.get(key)
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned or None


def resolve_mode(env: Mapping[str, str] | None = None) -> Mode:
    """Resolve the selected :class:`Mode` from the ``SHELFSENTRY_MODE`` switch.

    Reads ``env`` (the process environment by default). An unset switch defaults
    to :attr:`Mode.LOCAL`. The value is case-insensitive and whitespace-trimmed.
    An unrecognized value fails loudly with :class:`ModeConfigurationError`
    rather than defaulting silently (Req 17.1).
    """
    mapping = _env_mapping(env)
    raw = _get_clean(mapping, ENV_MODE)
    if raw is None:
        return Mode.LOCAL
    normalized = raw.lower()
    try:
        return Mode(normalized)
    except ValueError as exc:
        valid = ", ".join(repr(m.value) for m in Mode)
        raise ModeConfigurationError(
            f"Invalid {ENV_MODE}={raw!r}: expected one of {valid}."
        ) from exc


def resolve_runtime_config(env: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Resolve the full :class:`RuntimeConfig` once at startup (the Config_Switch).

    * LOCAL mode (default) requires no AWS configuration and resolves to a
      :class:`LocalSettings` carrying the local storage base directory
      (overridable via ``SHELFSENTRY_LOCAL_STORAGE_DIR``) (Req 17.1).
    * AWS mode requires every key in ``SHELFSENTRY_S3_BUCKET``,
      ``SHELFSENTRY_DDB_TABLE``, and ``SHELFSENTRY_AWS_REGION``. If any are
      missing, this raises :class:`ModeConfigurationError` naming **each**
      missing key (Req 17.2).

    This function never substitutes a local/mock runtime for an
    incompletely-configured AWS selection: when AWS mode is selected it returns
    an AWS config or raises - there is no local fallback path (Req 16.4, 17.3).
    """
    mapping = _env_mapping(env)
    mode = resolve_mode(mapping)

    if mode is Mode.LOCAL:
        storage_dir = _get_clean(mapping, ENV_LOCAL_STORAGE_DIR) or DEFAULT_LOCAL_STORAGE_DIR
        return RuntimeConfig(mode=Mode.LOCAL, local=LocalSettings(storage_dir=storage_dir))

    # mode is Mode.AWS: validate required keys and fail loudly if any are
    # missing. No branch here may fall back to LOCAL (Req 16.4, 17.3).
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for env_key, attr in _AWS_REQUIRED_KEYS:
        value = _get_clean(mapping, env_key)
        if value is None:
            missing.append(env_key)
        else:
            resolved[attr] = value

    if missing:
        raise ModeConfigurationError(
            f"{ENV_MODE}=aws is selected but required AWS configuration is "
            f"missing: {', '.join(missing)}. Set each of these environment "
            f"variables (non-secret resource names only; AWS credentials come "
            f"from the standard provider chain). AWS mode will not fall back to "
            f"local mode.",
            missing_keys=tuple(missing),
        )

    return RuntimeConfig(mode=Mode.AWS, aws=AwsSettings(**resolved))
