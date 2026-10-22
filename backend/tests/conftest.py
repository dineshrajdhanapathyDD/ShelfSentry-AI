"""Shared pytest configuration for the ShelfSentry test suite.

Registers Hypothesis settings profiles and selects a profile that runs every
property-based test at >= 100 generated iterations (per the implementation
plan). Individual properties may raise ``max_examples`` with their own
``@settings`` decorator, but the active profile floor keeps the suite at or
above 100 examples by default.

pytest itself runs once and exits (single-run, no watch mode); this file only
configures how many examples Hypothesis generates per property.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

# >= 100 iterations per property test.
_PROPERTY_ITERATIONS = 100

settings.register_profile(
    "ci",
    max_examples=_PROPERTY_ITERATIONS,
    deadline=None,  # vision/image work can be slow; avoid flaky per-example timeouts
    suppress_health_check=[HealthCheck.too_slow],
)

# "dev" matches "ci" here so local and CI runs use the same iteration floor.
settings.register_profile(
    "dev",
    max_examples=_PROPERTY_ITERATIONS,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

# Load the CI profile by default. Override with HYPOTHESIS_PROFILE if needed,
# but note no registered profile drops below the 100-example floor.
settings.load_profile("ci")
