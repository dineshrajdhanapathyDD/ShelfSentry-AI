"""Tests for the explicit Local/AWS mode configuration switch (task 11.2).

Covers Req 16.4, 17.1, 17.2, 17.3:

* default (no env) resolves to LOCAL with no AWS requirements;
* ``SHELFSENTRY_MODE=local`` resolves to LOCAL requiring no AWS config;
* ``SHELFSENTRY_MODE=aws`` with all required keys resolves an AWS config;
* ``SHELFSENTRY_MODE=aws`` with missing keys raises ``ModeConfigurationError``
  naming every missing key, and never yields a local fallback;
* an invalid ``SHELFSENTRY_MODE`` value fails loudly.

All environment manipulation goes through ``monkeypatch`` so the process env is
restored between tests and nothing leaks across cases.
"""

from __future__ import annotations

import pytest

from shelfsentry.config import (
    ENV_AWS_REGION,
    ENV_DDB_TABLE,
    ENV_LOCAL_STORAGE_DIR,
    ENV_MODE,
    ENV_S3_BUCKET,
    DEFAULT_LOCAL_STORAGE_DIR,
    Mode,
    ModeConfigurationError,
    resolve_mode,
    resolve_runtime_config,
)

# The three env vars AWS mode requires; used to clear the environment and to
# assert on missing-key reporting.
_AWS_KEYS = (ENV_S3_BUCKET, ENV_DDB_TABLE, ENV_AWS_REGION)
_ALL_MODE_ENV = (ENV_MODE, ENV_LOCAL_STORAGE_DIR, *_AWS_KEYS)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every mode-related env var so each test starts from a blank slate."""
    for key in _ALL_MODE_ENV:
        monkeypatch.delenv(key, raising=False)


def _set_aws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_MODE, "aws")
    monkeypatch.setenv(ENV_S3_BUCKET, "my-bucket")
    monkeypatch.setenv(ENV_DDB_TABLE, "my-table")
    monkeypatch.setenv(ENV_AWS_REGION, "us-east-1")


# ---------------------------------------------------------------------------
# Default + LOCAL mode (Req 17.1)
# ---------------------------------------------------------------------------


def test_default_no_env_resolves_local(clean_env: None) -> None:
    """With no env set, both resolvers default to LOCAL (Req 17.1)."""
    assert resolve_mode() is Mode.LOCAL

    cfg = resolve_runtime_config()
    assert cfg.mode is Mode.LOCAL
    assert cfg.is_local and not cfg.is_aws
    # LOCAL requires no AWS config.
    assert cfg.aws is None
    assert cfg.local is not None
    assert cfg.local.storage_dir == DEFAULT_LOCAL_STORAGE_DIR


def test_explicit_local_requires_no_aws_config(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SHELFSENTRY_MODE=local resolves LOCAL and ignores absent AWS keys."""
    monkeypatch.setenv(ENV_MODE, "local")

    cfg = resolve_runtime_config()
    assert cfg.mode is Mode.LOCAL
    assert cfg.aws is None
    assert cfg.local is not None


def test_local_is_case_insensitive_and_trimmed(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mode value is normalized (case-insensitive, whitespace-trimmed)."""
    monkeypatch.setenv(ENV_MODE, "  LoCaL  ")
    assert resolve_mode() is Mode.LOCAL
    assert resolve_runtime_config().mode is Mode.LOCAL


def test_local_storage_dir_override(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provided local storage dir override is carried into LocalSettings."""
    monkeypatch.setenv(ENV_MODE, "local")
    monkeypatch.setenv(ENV_LOCAL_STORAGE_DIR, "/tmp/shelf-store")

    cfg = resolve_runtime_config()
    assert cfg.local is not None
    assert cfg.local.storage_dir == "/tmp/shelf-store"


def test_blank_local_storage_dir_falls_back_to_default(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank storage-dir override is treated as unset (uses the default)."""
    monkeypatch.setenv(ENV_MODE, "local")
    monkeypatch.setenv(ENV_LOCAL_STORAGE_DIR, "   ")

    cfg = resolve_runtime_config()
    assert cfg.local is not None
    assert cfg.local.storage_dir == DEFAULT_LOCAL_STORAGE_DIR


# ---------------------------------------------------------------------------
# AWS mode fully configured (Req 16.1, 17.1)
# ---------------------------------------------------------------------------


def test_aws_mode_with_all_keys_resolves(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AWS mode with every required key present resolves an AWS config."""
    _set_aws(monkeypatch)

    assert resolve_mode() is Mode.AWS

    cfg = resolve_runtime_config()
    assert cfg.mode is Mode.AWS
    assert cfg.is_aws and not cfg.is_local
    assert cfg.local is None  # no local fallback object on the AWS path
    assert cfg.aws is not None
    assert cfg.aws.s3_bucket == "my-bucket"
    assert cfg.aws.ddb_table == "my-table"
    assert cfg.aws.region == "us-east-1"


def test_aws_mode_is_case_insensitive(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AWS selection is normalized like the local one."""
    _set_aws(monkeypatch)
    monkeypatch.setenv(ENV_MODE, "AWS")
    assert resolve_runtime_config().mode is Mode.AWS


def test_aws_values_are_trimmed(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surrounding whitespace is stripped from AWS config values."""
    monkeypatch.setenv(ENV_MODE, "aws")
    monkeypatch.setenv(ENV_S3_BUCKET, "  bucket-x  ")
    monkeypatch.setenv(ENV_DDB_TABLE, "  table-x  ")
    monkeypatch.setenv(ENV_AWS_REGION, "  eu-west-1  ")

    cfg = resolve_runtime_config()
    assert cfg.aws is not None
    assert cfg.aws.s3_bucket == "bucket-x"
    assert cfg.aws.ddb_table == "table-x"
    assert cfg.aws.region == "eu-west-1"


# ---------------------------------------------------------------------------
# AWS mode missing config -> fail loud, name each key (Req 17.2)
# ---------------------------------------------------------------------------


def test_aws_mode_all_keys_missing_names_every_key(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AWS mode with no AWS config raises and names every missing key."""
    monkeypatch.setenv(ENV_MODE, "aws")

    with pytest.raises(ModeConfigurationError) as excinfo:
        resolve_runtime_config()

    err = excinfo.value
    assert set(err.missing_keys) == set(_AWS_KEYS)
    message = str(err)
    for key in _AWS_KEYS:
        assert key in message


@pytest.mark.parametrize("present_key", _AWS_KEYS)
def test_aws_mode_one_missing_key_is_named(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, present_key: str
) -> None:
    """With only one AWS key present, the other two are reported missing."""
    monkeypatch.setenv(ENV_MODE, "aws")
    monkeypatch.setenv(present_key, "value")

    with pytest.raises(ModeConfigurationError) as excinfo:
        resolve_runtime_config()

    err = excinfo.value
    expected_missing = {k for k in _AWS_KEYS if k != present_key}
    assert set(err.missing_keys) == expected_missing
    for key in expected_missing:
        assert key in str(err)
    assert present_key not in err.missing_keys


def test_aws_mode_blank_key_counts_as_missing(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present-but-blank required key is reported missing (Req 17.2)."""
    monkeypatch.setenv(ENV_MODE, "aws")
    monkeypatch.setenv(ENV_S3_BUCKET, "my-bucket")
    monkeypatch.setenv(ENV_DDB_TABLE, "   ")  # blank -> missing
    monkeypatch.setenv(ENV_AWS_REGION, "us-east-1")

    with pytest.raises(ModeConfigurationError) as excinfo:
        resolve_runtime_config()

    assert excinfo.value.missing_keys == (ENV_DDB_TABLE,)


# ---------------------------------------------------------------------------
# AWS mode never silently falls back to local (Req 16.4, 17.3)
# ---------------------------------------------------------------------------


def test_aws_mode_incomplete_never_returns_local_fallback(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incompletely-configured AWS selection raises; it never yields LOCAL.

    This is the Req 16.4 / 17.3 guarantee: the resolver must not substitute a
    local/mock runtime when AWS config is incomplete.
    """
    monkeypatch.setenv(ENV_MODE, "aws")
    monkeypatch.setenv(ENV_S3_BUCKET, "only-bucket")
    # DDB table and region intentionally absent.

    # resolve_mode still reports AWS was selected (no silent downgrade).
    assert resolve_mode() is Mode.AWS

    # and the full resolution raises rather than returning a LOCAL config.
    with pytest.raises(ModeConfigurationError):
        resolve_runtime_config()


# ---------------------------------------------------------------------------
# Invalid mode value -> clear error (Req 17.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", ["cloud", "prod", "LOCALHOST", "awss", "mock"])
def test_invalid_mode_value_raises_clear_error(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, bad_value: str
) -> None:
    """An unrecognized mode value fails loudly instead of defaulting."""
    monkeypatch.setenv(ENV_MODE, bad_value)

    with pytest.raises(ModeConfigurationError) as mode_err:
        resolve_mode()
    assert ENV_MODE in str(mode_err.value)
    assert bad_value in str(mode_err.value)

    with pytest.raises(ModeConfigurationError):
        resolve_runtime_config()


# ---------------------------------------------------------------------------
# Resolver reads an explicit mapping (injectable env)
# ---------------------------------------------------------------------------


def test_resolver_accepts_explicit_env_mapping() -> None:
    """Passing an explicit mapping bypasses the process environment."""
    assert resolve_mode(env={}) is Mode.LOCAL
    assert resolve_mode(env={ENV_MODE: "aws"}) is Mode.AWS

    cfg = resolve_runtime_config(
        env={
            ENV_MODE: "aws",
            ENV_S3_BUCKET: "b",
            ENV_DDB_TABLE: "t",
            ENV_AWS_REGION: "r",
        }
    )
    assert cfg.mode is Mode.AWS
    assert cfg.aws is not None and cfg.aws.s3_bucket == "b"


def test_resolved_runtime_config_is_frozen() -> None:
    """The resolved config is immutable so the active mode can't change."""
    cfg = resolve_runtime_config(env={})
    with pytest.raises(Exception):
        cfg.mode = Mode.AWS  # type: ignore[misc]
