"""Unit tests for OpenCV 5 runtime version verification (Req 2.1, 26.4).

Covers ``shelfsentry.vision.version``:

* ``opencv_version()`` returns the exact ``cv2.__version__`` string and the
  runtime is OpenCV 5 (major version starts with ``"5."``) (Req 2.1).
* A non-5 runtime (``cv2.__version__ == "4.9.0"``) makes both
  ``verify_opencv_version()`` and ``opencv_version()`` raise
  ``OpenCVVersionError`` with a message naming the found version (Req 2.1, 26.4).
* An unparseable version string raises ``OpenCVVersionError``.

The ``version`` module does ``import cv2`` and reads ``cv2.__version__`` at
call time, so these tests patch the attribute on the shared ``cv2`` module
object via ``monkeypatch`` (automatically restored after each test).
"""

from __future__ import annotations

import cv2
import pytest

from shelfsentry.vision import version as version_module
from shelfsentry.vision.version import (
    REQUIRED_OPENCV_MAJOR,
    OpenCVVersionError,
    opencv_version,
    verify_opencv_version,
)


def test_required_major_is_five() -> None:
    """The system is pinned to and validated against OpenCV major version 5."""
    assert REQUIRED_OPENCV_MAJOR == 5


def test_opencv_version_returns_exact_runtime_version_and_is_opencv_5() -> None:
    """opencv_version() returns the exact runtime string and it is OpenCV 5 (Req 2.1)."""
    result = opencv_version()

    # Exact runtime version string, not a normalized/derived value.
    assert result == cv2.__version__
    # Runtime must be OpenCV 5.
    assert result.startswith("5."), f"expected an OpenCV 5.x runtime, got {result!r}"
    # verify_opencv_version() agrees and returns the same exact string.
    assert verify_opencv_version() == cv2.__version__


def test_non_five_version_raises_and_names_found_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-5 runtime fails loudly and the message names the found version (Req 2.1, 26.4)."""
    monkeypatch.setattr(cv2, "__version__", "4.9.0", raising=False)

    with pytest.raises(OpenCVVersionError) as verify_exc:
        verify_opencv_version()
    assert "4.9.0" in str(verify_exc.value)

    # opencv_version() is the public entry point and must fail the same way.
    with pytest.raises(OpenCVVersionError) as public_exc:
        opencv_version()
    assert "4.9.0" in str(public_exc.value)


def test_non_five_version_patched_via_module_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching cv2 as imported by the version module also triggers the failure.

    ``version_module.cv2`` is the same module object as the top-level ``cv2``;
    this asserts the module reads the attribute off that reference.
    """
    monkeypatch.setattr(version_module.cv2, "__version__", "4.9.0", raising=False)

    with pytest.raises(OpenCVVersionError) as exc:
        opencv_version()
    assert "4.9.0" in str(exc.value)


@pytest.mark.parametrize("bad_version", ["", "not-a-version", "v5.0.0", ".5.0"])
def test_unparseable_version_raises(
    monkeypatch: pytest.MonkeyPatch, bad_version: str
) -> None:
    """An unparseable version string raises OpenCVVersionError."""
    monkeypatch.setattr(cv2, "__version__", bad_version, raising=False)

    with pytest.raises(OpenCVVersionError):
        verify_opencv_version()
