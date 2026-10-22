"""OpenCV 5 runtime version verification (Req 2.1, 26.4).

The system performs substantive analysis with OpenCV 5. The exact runtime
version is recorded in every ``VisionFinding`` (Req 2.1), and the pinned
package is documented in ``pyproject.toml`` / ``requirements.txt`` (Req 26.4).

This module reads ``cv2.__version__`` and verifies that the major version is
5 **at startup**. If a non-5 OpenCV is loaded, it FAILS LOUDLY by raising a
descriptive error that names the version actually found. The system never
silently accepts an OpenCV 4 (or any non-5) runtime, because the analysis
code is written and validated against the OpenCV 5 API surface.
"""

from __future__ import annotations

import cv2

#: The major OpenCV version this system requires and is validated against.
REQUIRED_OPENCV_MAJOR = 5


class OpenCVVersionError(RuntimeError):
    """Raised when the loaded OpenCV runtime is not OpenCV 5.

    The message names the exact version found so the failure is actionable
    and honest about what is actually installed.
    """


def _runtime_version() -> str:
    """Return the raw ``cv2.__version__`` string for the loaded runtime."""
    return str(cv2.__version__)


def _major_of(version: str) -> int:
    """Parse the leading integer (major) component of a version string.

    Raises:
        OpenCVVersionError: if the version string has no parseable leading
            integer component.
    """
    head = version.strip().split(".", 1)[0]
    try:
        return int(head)
    except ValueError as exc:  # pragma: no cover - defensive, unparseable version
        raise OpenCVVersionError(
            f"Could not parse an OpenCV major version from cv2.__version__="
            f"{version!r}. ShelfSentry requires OpenCV "
            f"{REQUIRED_OPENCV_MAJOR}.x."
        ) from exc


def verify_opencv_version() -> str:
    """Verify the loaded OpenCV runtime is version 5 and return it.

    Reads ``cv2.__version__``, confirms its major component equals
    :data:`REQUIRED_OPENCV_MAJOR`, and returns the exact runtime version
    string on success.

    Returns:
        The exact ``cv2.__version__`` string (for example ``"5.0.0"``).

    Raises:
        OpenCVVersionError: if the loaded OpenCV is not a 5.x release. The
            error names the version that was actually found.
    """
    version = _runtime_version()
    major = _major_of(version)
    if major != REQUIRED_OPENCV_MAJOR:
        raise OpenCVVersionError(
            f"ShelfSentry requires OpenCV {REQUIRED_OPENCV_MAJOR}.x, but the "
            f"loaded runtime reports cv2.__version__={version!r} "
            f"(major version {major}). Refusing to run: the vision pipeline "
            f"is written and validated against the OpenCV "
            f"{REQUIRED_OPENCV_MAJOR} API and must not silently use a "
            f"different major version. Install the pinned OpenCV 5 wheel "
            f"(see requirements.txt) and retry."
        )
    return version


def opencv_version() -> str:
    """Return the exact, verified OpenCV runtime version string.

    This is the public entry point used throughout the system (including when
    recording the OpenCV version in each ``VisionFinding``, Req 2.1). It
    verifies OpenCV 5 on every call and fails loudly otherwise, so callers can
    both validate at startup and trust the returned string.

    Returns:
        The exact ``cv2.__version__`` string (for example ``"5.0.0"``).

    Raises:
        OpenCVVersionError: if the loaded OpenCV is not a 5.x release.
    """
    return verify_opencv_version()
