"""Thread-safe reservations for a bounded set of KV slots owned by a single backend.

A scope identifies a model/tokenizer/adapter/tenant epoch; the caller must use
cache_prompt=False on misses. Only confirmed prompt tokens enter the index.
No logits, text, KV tensors or generated continuations are retained here.
"""

from dataclasses import dataclass, field
from threading import Lock


def common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


@dataclass(frozen=True, eq=False)
class Lease:
    slot: int
    prefix: int
    reuse: bool


@dataclass
class _Slot:
    scope: str = ""
    ids: tuple = field(default=(), repr=False)
    age: int = 0
    pending: object = field(default=None, repr=False)


class SlotRouter:
    """Choose the longest eligible idle prefix; cold misses take the least recent idle slot."""

    def __init__(self, slots, context, min_prefix=16, policy="affinity"):
        if (
            any(type(x) is not int or x < 1 for x in (slots, context, min_prefix))
            or min_prefix > context
        ):
            raise ValueError("invalid dimensions")
        if policy not in ("affinity", "round_robin"):
            raise ValueError("unknown policy")
        self._slots = [_Slot() for _ in range(slots)]
        self._ctx = context
        self._min = min_prefix
        self._policy = policy
        self._clock = 0
        self._cursor = 0
        self._lock = Lock()

    def acquire(self, scope, ids):
        if (
            not isinstance(scope, str)
            or not scope
            or not isinstance(ids, (tuple, list))
            or not (0 < len(ids) <= self._ctx)
        ):
            raise ValueError("invalid scope or prompt length")
        ids = tuple(ids)
        if any(type(x) is not int or x < 0 for x in ids):
            raise ValueError("invalid token ID")
        with self._lock:
            candidates = []
            for i, s in enumerate(self._slots):
                if s.pending is not None:
                    continue
                match = common_prefix(s.ids, ids) if s.scope == scope else 0
                gain = match if match >= self._min else 0
                score = (
                    (-gain, s.age, i)
                    if self._policy == "affinity"
                    else ((i - self._cursor) % len(self._slots),)
                )
                candidates.append((score, i, gain))
            if not candidates:
                return None
            _, i, gain = min(candidates)
            lease = Lease(i, gain, bool(gain))
            self._slots[i].pending = (lease, scope, ids)
            self._cursor = (i + 1) % len(self._slots)
            return lease

    def finish(self, lease, *, success):
        if (
            not isinstance(lease, Lease)
            or type(lease.slot) is not int
            or not (0 <= lease.slot < len(self._slots))
            or type(success) is not bool
        ):
            raise ValueError("invalid lease or completion status")
        with self._lock:
            s = self._slots[lease.slot]
            if s.pending is None or s.pending[0] is not lease:
                raise ValueError("stale or foreign lease")
            _, scope, ids = s.pending
            self._clock += 1
            s.age = self._clock
            s.pending = None
            s.scope, s.ids = (scope, ids) if success else ("", ())

    def reset(self):
        """Call after restarting/clearing the backend. Old completions cannot repopulate it."""
        with self._lock:
            self._slots = [_Slot() for _ in self._slots]
            self._cursor = self._clock = 0
