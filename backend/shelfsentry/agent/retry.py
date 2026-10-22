"""A small, deterministic retry/backoff helper for transient AWS failures.

AWS Mode must apply the *configured* failure handling and retry behavior to an
operation that fails during an inspection, and surface a FAILED status when the
retries are exhausted (Req 16.3). This module provides exactly that policy as a
tiny, dependency-free, fully testable helper - it is used by the AWS-mode runner
(:mod:`shelfsentry.aws_worker`) around the fallible S3/DynamoDB/Bedrock calls.

Design points the task called out:

* **Injected sleeper.** :class:`RetryPolicy` never imports or calls
  ``time.sleep`` itself; the caller injects a ``sleep`` callable. Tests pass a
  fake that merely records the delays, so the suite runs without actually
  sleeping and can assert the exact backoff schedule. The production default
  (:func:`default_sleeper`) is the only place :func:`time.sleep` is referenced,
  and it is created lazily so importing this module has no side effect.

* **Deterministic schedule.** With ``base_delay`` ``b`` and ``multiplier`` ``m``
  the delay before retry *k* (1-indexed) is ``b * m**(k-1)``, optionally capped
  at ``max_delay``. No jitter is applied, so a given policy produces an exactly
  reproducible sequence of delays - which is what makes the retry behavior
  testable (and what keeps a demo run reproducible).

* **Only transient errors are retried.** The caller supplies a ``retry_on``
  predicate (default: any :class:`Exception`); an error for which the predicate
  returns ``False`` is raised immediately without consuming further attempts.
  When the attempt budget is exhausted, :class:`RetryExhausted` is raised
  wrapping the last underlying error, so the runner can turn it into a FAILED
  status (Req 16.3).
"""

from __future__ import annotations

from typing import Callable, TypeVar

__all__ = [
    "RetryPolicy",
    "RetryExhausted",
    "default_sleeper",
]

T = TypeVar("T")

#: A sleeper is any callable taking a non-negative delay in seconds. The real
#: one wraps :func:`time.sleep`; tests inject a recorder that does not sleep.
Sleeper = Callable[[float], None]

#: A predicate deciding whether a raised exception is transient (retryable).
RetryPredicate = Callable[[BaseException], bool]


class RetryExhausted(RuntimeError):
    """Raised when a retried operation never succeeded within its attempt budget.

    Wraps the final underlying error on :attr:`last_error` and records how many
    attempts were made on :attr:`attempts`, so the AWS-mode runner can surface a
    FAILED status that names the exhausting error (Req 16.3).
    """

    def __init__(self, message: str, *, attempts: int, last_error: BaseException) -> None:
        super().__init__(message)
        #: Total attempts made (including the first, non-retry attempt).
        self.attempts = attempts
        #: The last exception raised by the operation before giving up.
        self.last_error = last_error


def default_sleeper(delay: float) -> None:
    """Sleep for ``delay`` seconds using the standard library (production path).

    Imported lazily so that importing this module (and therefore the AWS worker)
    never imports :mod:`time` eagerly or sleeps as a side effect. Tests inject
    their own no-op sleeper instead of using this.
    """
    import time  # noqa: PLC0415 - lazy by design; only the real sleeper touches time

    if delay > 0:
        time.sleep(delay)


class RetryPolicy:
    """A deterministic max-attempts + exponential-backoff retry policy (Req 16.3).

    :param max_attempts: total attempts allowed, including the first. Must be
        ``>= 1``. ``1`` means "no retries" (run once, re-raise on failure).
    :param base_delay: delay in seconds before the *first* retry (``>= 0``).
    :param multiplier: factor the delay grows by on each successive retry
        (``>= 1``); ``1.0`` gives a constant delay.
    :param max_delay: optional ceiling applied to every computed delay.
    :param sleep: the injected sleeper. Defaults to :func:`default_sleeper`;
        tests pass a recorder so the suite never actually sleeps.
    :param retry_on: predicate deciding whether a raised exception is transient
        and should be retried. Defaults to "retry any ``Exception``" (but never
        a non-``Exception`` ``BaseException`` such as ``KeyboardInterrupt``).
    """

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        base_delay: float = 0.5,
        multiplier: float = 2.0,
        max_delay: float | None = None,
        sleep: Sleeper | None = None,
        retry_on: RetryPredicate | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if base_delay < 0:
            raise ValueError("base_delay must be >= 0")
        if multiplier < 1:
            raise ValueError("multiplier must be >= 1")
        if max_delay is not None and max_delay < 0:
            raise ValueError("max_delay must be >= 0 when provided")

        self.max_attempts = int(max_attempts)
        self.base_delay = float(base_delay)
        self.multiplier = float(multiplier)
        self.max_delay = float(max_delay) if max_delay is not None else None
        self._sleep: Sleeper = sleep if sleep is not None else default_sleeper
        self._retry_on: RetryPredicate = (
            retry_on if retry_on is not None else _default_retry_on
        )

    def delay_for_retry(self, retry_index: int) -> float:
        """Return the backoff delay (seconds) before retry ``retry_index`` (1-based).

        Retry 1 waits :attr:`base_delay`; each subsequent retry multiplies by
        :attr:`multiplier`; the result is capped at :attr:`max_delay` when set.
        Exposed (and pure) so tests can assert the schedule directly.
        """
        if retry_index < 1:
            return 0.0
        delay = self.base_delay * (self.multiplier ** (retry_index - 1))
        if self.max_delay is not None:
            delay = min(delay, self.max_delay)
        return delay

    def run(self, operation: Callable[[], T], *, description: str = "operation") -> T:
        """Call ``operation`` with retry/backoff; return its result or raise.

        The operation is attempted up to :attr:`max_attempts` times. A raised
        exception for which ``retry_on`` returns ``True`` triggers a backoff
        sleep (via the injected sleeper) and another attempt, until either the
        operation succeeds or the budget is exhausted - at which point
        :class:`RetryExhausted` is raised wrapping the last error (Req 16.3). A
        non-transient exception (``retry_on`` returns ``False``) is re-raised
        immediately without consuming the remaining budget.

        :param operation: a zero-argument callable performing the fallible work.
        :param description: a short label used in the exhaustion message.
        """
        last_error: BaseException | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001 - classify via the predicate
                last_error = exc
                if not self._retry_on(exc):
                    raise
                if attempt >= self.max_attempts:
                    break
                self._sleep(self.delay_for_retry(attempt))

        assert last_error is not None  # the loop only breaks after an exception
        raise RetryExhausted(
            f"{description} failed after {self.max_attempts} attempt(s): "
            f"{type(last_error).__name__}: {last_error}",
            attempts=self.max_attempts,
            last_error=last_error,
        ) from last_error


def _default_retry_on(exc: BaseException) -> bool:
    """Default transience predicate: retry any ``Exception``, nothing above it.

    A plain :class:`Exception` (the base of ordinary AWS/botocore client errors)
    is treated as transient by default; a :class:`BaseException` that is not an
    ``Exception`` (``KeyboardInterrupt``, ``SystemExit``, ...) is never retried.
    """
    return isinstance(exc, Exception)
