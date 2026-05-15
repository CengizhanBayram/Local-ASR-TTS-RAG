"""
Async retry with exponential backoff + simple circuit breaker.
"""

import asyncio
import logging
import random
import time
from enum import Enum
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


async def retry_async(
    coro_fn: Callable,
    *args,
    max_retries: int = 2,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    **kwargs,
):
    """
    Call ``coro_fn(*args, **kwargs)`` up to ``max_retries`` extra times on failure.
    Waits exponentially between attempts (with optional ±50% jitter).
    Raises the last exception if all attempts fail.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay *= 0.5 + random.random() * 0.5
            logger.warning(
                f"[retry] attempt {attempt + 1}/{max_retries} failed ({type(exc).__name__}), "
                f"retrying in {delay:.1f}s: {exc}"
            )
            await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


class _State(Enum):
    CLOSED = "closed"        # normal — all calls pass through
    OPEN = "open"            # failing — reject immediately
    HALF_OPEN = "half_open"  # recovery probe — allow one call


class CircuitBreaker:
    """
    Async circuit breaker.

    States:
    - CLOSED  → calls pass through; consecutive failures increment counter
    - OPEN    → calls are rejected immediately (CircuitOpenError) until
                ``recovery_timeout`` seconds have elapsed
    - HALF_OPEN → one probe call is allowed; success → CLOSED, failure → OPEN
    """

    class CircuitOpenError(RuntimeError):
        pass

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        name: str = "circuit",
    ):
        self._threshold = failure_threshold
        self._timeout = recovery_timeout
        self._name = name
        self._state = _State.CLOSED
        self._failures = 0
        self._opened_at: Optional[float] = None

    async def call(self, coro_fn: Callable, *args, **kwargs):
        """Wrap an async call with circuit-breaker logic."""
        if self._state == _State.OPEN:
            elapsed = time.monotonic() - self._opened_at  # type: ignore[operator]
            if elapsed >= self._timeout:
                self._state = _State.HALF_OPEN
                logger.info(f"[{self._name}] circuit HALF_OPEN — probing recovery")
            else:
                raise self.CircuitOpenError(
                    f"Circuit [{self._name}] is OPEN ({self._timeout - elapsed:.0f}s remaining)"
                )

        try:
            result = await coro_fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        if self._state != _State.CLOSED:
            logger.info(f"[{self._name}] circuit CLOSED — service recovered")
        self._failures = 0
        self._state = _State.CLOSED

    def _on_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._threshold or self._state == _State.HALF_OPEN:
            self._state = _State.OPEN
            self._opened_at = time.monotonic()
            logger.error(
                f"[{self._name}] circuit OPEN after {self._failures} consecutive failures"
            )

    @property
    def state(self) -> str:
        return self._state.value

    @property
    def failure_count(self) -> int:
        return self._failures
