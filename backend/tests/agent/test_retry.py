"""Unit tests for the deterministic retry/backoff helper (task 16.2; Req 16.3).

These tests pin the retry policy's behavior with an INJECTED fake sleeper (so no
test actually sleeps) and a call-counting fake operation:

* a transient failure that recovers within the attempt budget eventually
  succeeds, and the recorded backoff schedule is exactly the deterministic
  ``base * multiplier**(k-1)`` sequence;
* a failure that never recovers surfaces :class:`RetryExhausted` after exactly
  ``max_attempts`` attempts, wrapping the last underlying error;
* a non-transient error (``retry_on`` returns ``False``) is raised immediately
  without consuming further attempts.
"""

from __future__ import annotations

import pytest

from shelfsentry.agent.retry import RetryExhausted, RetryPolicy


class _Recorder:
    """A fake sleeper that records the delays it was asked to sleep for."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)


class _FlakyOp:
    """A call-counting operation that fails ``fail_times`` then returns ``value``."""

    def __init__(self, *, fail_times: int, value: object = "ok", exc: Exception | None = None) -> None:
        self.calls = 0
        self._fail_times = fail_times
        self._value = value
        self._exc = exc or RuntimeError("transient")

    def __call__(self) -> object:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return self._value


# ---------------------------------------------------------------------------
# Success-within-budget
# ---------------------------------------------------------------------------


def test_recovers_within_budget_returns_value_and_counts_attempts():
    sleeper = _Recorder()
    op = _FlakyOp(fail_times=2, value=42)
    policy = RetryPolicy(max_attempts=3, base_delay=0.5, multiplier=2.0, sleep=sleeper)

    result = policy.run(op, description="transient op")

    assert result == 42
    # Three attempts total: two failures + one success.
    assert op.calls == 3
    # Two backoff sleeps happened (before retry 1 and retry 2), none after
    # the final successful attempt.
    assert sleeper.delays == [0.5, 1.0]


def test_first_attempt_success_does_not_sleep():
    sleeper = _Recorder()
    op = _FlakyOp(fail_times=0, value="immediate")
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, sleep=sleeper)

    assert policy.run(op) == "immediate"
    assert op.calls == 1
    assert sleeper.delays == []  # never slept


# ---------------------------------------------------------------------------
# Exhaustion
# ---------------------------------------------------------------------------


def test_never_recovers_raises_retry_exhausted_after_configured_attempts():
    sleeper = _Recorder()
    boom = ValueError("still broken")
    op = _FlakyOp(fail_times=99, exc=boom)  # always fails
    policy = RetryPolicy(max_attempts=4, base_delay=0.25, multiplier=2.0, sleep=sleeper)

    with pytest.raises(RetryExhausted) as excinfo:
        policy.run(op, description="doomed op")

    # Exactly max_attempts attempts were made.
    assert op.calls == 4
    assert excinfo.value.attempts == 4
    # It wraps the last underlying error.
    assert excinfo.value.last_error is boom
    assert isinstance(excinfo.value.__cause__, ValueError)
    # Slept before each of the 3 retries (not after the final failure).
    assert sleeper.delays == [0.25, 0.5, 1.0]
    assert "doomed op" in str(excinfo.value)


def test_max_attempts_one_means_no_retries():
    sleeper = _Recorder()
    op = _FlakyOp(fail_times=99)
    policy = RetryPolicy(max_attempts=1, sleep=sleeper)

    with pytest.raises(RetryExhausted):
        policy.run(op)
    assert op.calls == 1
    assert sleeper.delays == []  # no retry -> no sleep


# ---------------------------------------------------------------------------
# Non-transient errors are not retried
# ---------------------------------------------------------------------------


def test_non_transient_error_raised_immediately_without_consuming_attempts():
    sleeper = _Recorder()

    class _Fatal(Exception):
        pass

    def op() -> None:
        op.calls += 1  # type: ignore[attr-defined]
        raise _Fatal("do not retry me")

    op.calls = 0  # type: ignore[attr-defined]

    # retry_on returns False for _Fatal, so it must surface on the first try.
    policy = RetryPolicy(
        max_attempts=5,
        base_delay=0.1,
        sleep=sleeper,
        retry_on=lambda exc: not isinstance(exc, _Fatal),
    )

    with pytest.raises(_Fatal):
        policy.run(op)
    assert op.calls == 1  # type: ignore[attr-defined]
    assert sleeper.delays == []


# ---------------------------------------------------------------------------
# Backoff schedule is deterministic (pure helper)
# ---------------------------------------------------------------------------


def test_delay_schedule_is_deterministic_and_capped():
    policy = RetryPolicy(base_delay=1.0, multiplier=3.0, max_delay=10.0)
    assert policy.delay_for_retry(1) == 1.0
    assert policy.delay_for_retry(2) == 3.0
    assert policy.delay_for_retry(3) == 9.0
    # 27.0 would be next but is capped at max_delay.
    assert policy.delay_for_retry(4) == 10.0
    # Guard: a zero/negative index yields no delay.
    assert policy.delay_for_retry(0) == 0.0


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_attempts": 0},
        {"base_delay": -1.0},
        {"multiplier": 0.5},
        {"max_delay": -0.1},
    ],
)
def test_invalid_policy_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)
