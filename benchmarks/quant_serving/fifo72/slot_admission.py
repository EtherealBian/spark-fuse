"""Opt-in bounded FIFO admission over an owned SlotRouter. No worker thread or GPU.

A caller that owns a backend request must call finish exactly once after the
backend stops using that slot (even on cancellation). close() only drains admission;
it cannot stop a running backend. Scope is supplied by the trusted serving layer.
"""

from collections import deque
from math import isfinite
from threading import Condition, Lock, TIMEOUT_MAX
from time import monotonic
from slot_policy import SlotRouter


class AdmissionBusy(RuntimeError):
    """Waiting batch or token limit reached before admission; no reservation exists."""


class AdmissionClosed(RuntimeError):
    """This broker no longer grants reservations."""


class _Ticket:
    __slots__ = ("size", "tokens", "wake")

    def __init__(self, size, tokens, lock):
        self.size, self.tokens, self.wake = size, tokens, Condition(lock)


class BatchAdmission:
    """FIFO, all-or-none. A large head can block a later small request by design.

    Limits concern queued batches/tokens, not upstream handlers, Python heap bytes,
    active requests or backend KV size. `timeout_s` bounds waiting for slots from
    method entry, not running Python validation, locking or an in-progress grant.
    """

    def __init__(
        self,
        slots,
        context,
        *,
        max_waiters,
        max_waiting_tokens,
        min_prefix=16,
        batch_matcher="scan",
    ):
        if (
            any(type(x) is not int or x < 0 for x in (max_waiters, max_waiting_tokens))
            or type(slots) is not int
            or not 1 <= slots <= 64
        ):
            raise ValueError("invalid admission capacity")
        self._router = SlotRouter(
            slots, context, min_prefix=min_prefix, batch_matcher=batch_matcher
        )
        self._slots, self._ctx = slots, context
        self._max_waiters, self._max_tokens = max_waiters, max_waiting_tokens
        self._lock, self._q, self._active = Lock(), deque(), set()
        self._tokens, self._closed = 0, False
        self._counts = dict(
            granted_batches=0,
            queued_batches=0,
            timeouts=0,
            overloads=0,
            closed_waiters=0,
            wait_calls=0,
            router_calls=0,
        )

    def _freeze(self, requests):
        if type(requests) not in (tuple, list) or not 1 <= len(requests) <= self._slots:
            raise ValueError("expected one nonempty batch no larger than the backend")
        rows = []
        for row in tuple(requests):
            if type(row) not in (tuple, list) or len(row) != 2:
                raise ValueError("expected (scope, ids)")
            scope, ids = row
            if (
                type(scope) is not str
                or not scope
                or type(ids) not in (list, tuple)
                or not 1 <= len(ids) <= self._ctx
            ):
                raise ValueError("invalid scope or input length")
            ids = tuple(ids)
            if not 1 <= len(ids) <= self._ctx or any(type(x) is not int or x < 0 for x in ids):
                raise ValueError("invalid token IDs")
            rows.append((scope, ids))
        if not 1 <= len(rows) <= self._slots:
            raise ValueError("batch changed during copy")
        return tuple(rows)

    def _signal(self):
        if self._q and self._q[0].size <= self._slots - len(self._active):
            self._q[0].wake.notify()

    def _grant(self, rows):
        self._counts["router_calls"] += 1
        leases = self._router.acquire_batch(rows)
        if leases is None or len(leases) != len(rows):
            raise RuntimeError("owned router occupancy disagrees")
        self._active.update(leases)
        self._counts["granted_batches"] += 1
        return tuple(leases)

    def reserve(self, requests, *, timeout_s=None):
        if timeout_s is not None and (
            type(timeout_s) not in (int, float)
            or not 0 <= timeout_s <= TIMEOUT_MAX
            or not isfinite(timeout_s)
        ):
            raise ValueError("invalid timeout")
        deadline = None if timeout_s is None else monotonic() + timeout_s
        rows = self._freeze(requests)
        n = len(rows)
        tokens = sum(len(ids) for _, ids in rows)
        with self._lock:
            if self._closed:
                raise AdmissionClosed("admission closed")
            if not self._q and n <= self._slots - len(self._active):
                return self._grant(rows)
            if deadline is not None and monotonic() >= deadline:
                self._counts["timeouts"] += 1
                raise TimeoutError("no immediate FIFO capacity")
            if len(self._q) >= self._max_waiters or self._tokens + tokens > self._max_tokens:
                self._counts["overloads"] += 1
                raise AdmissionBusy("waiting budget exhausted")
            t = _Ticket(n, tokens, self._lock)
            self._q.append(t)
            self._tokens += tokens
            self._counts["queued_batches"] += 1
            try:
                while True:
                    if self._closed:
                        self._counts["closed_waiters"] += 1
                        raise AdmissionClosed("admission closed")
                    left = None if deadline is None else deadline - monotonic()
                    if left is not None and left <= 0:
                        self._counts["timeouts"] += 1
                        raise TimeoutError("FIFO wait expired")
                    if self._q[0] is t and n <= self._slots - len(self._active):
                        return self._grant(rows)
                    self._counts["wait_calls"] += 1
                    t.wake.wait(left)
            finally:
                self._q.remove(t)
                self._tokens -= tokens
                self._signal()

    def finish(self, lease, *, success):
        with self._lock:
            if lease not in self._active:
                raise ValueError("foreign or already finished lease")
            self._router.finish(lease, success=success)
            self._active.remove(lease)
            self._signal()

    def close(self):
        with self._lock:
            self._closed = True
            for t in self._q:
                t.wake.notify()

    def snapshot(self):
        """Counters only: no prompt IDs, tenant labels, hashes or queue contents."""
        with self._lock:
            return dict(
                closed=self._closed,
                active_slots=len(self._active),
                waiting_batches=len(self._q),
                waiting_tokens=self._tokens,
                **self._counts,
            )
