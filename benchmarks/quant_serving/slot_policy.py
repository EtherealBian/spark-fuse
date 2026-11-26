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


def minimum_assignment(costs):
    """Rectangular integer assignment with one distinct column per row (O(rows² * cols)).

    An augmenting path and column potentials avoid enumerating slot permutations.
    Deterministic for ties; no dependency on SciPy or a GPU. Costs may be negative.
    """
    n = len(costs)
    m = len(costs[0]) if n else 0
    if not 0 < n <= m <= 64 or any(
        len(row) != m or any(type(x) is not int for x in row) for row in costs
    ):
        raise ValueError("expected nonempty rectangular integer costs, rows <= columns <= 64")
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    owner = [0] * (m + 1)
    back = [0] * (m + 1)
    for i in range(1, n + 1):
        owner[0] = i
        j = 0
        best = [None] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j] = True
            r = owner[j]
            delta = None
            nxt = 0
            for k in range(1, m + 1):
                if used[k]:
                    continue
                reduced = costs[r - 1][k - 1] - u[r] - v[k]
                if best[k] is None or reduced < best[k]:
                    best[k] = reduced
                    back[k] = j
                if delta is None or best[k] < delta:
                    delta = best[k]
                    nxt = k
            for k in range(m + 1):
                if used[k]:
                    u[owner[k]] += delta
                    v[k] -= delta
                elif best[k] is not None:
                    best[k] -= delta
            j = nxt
            if not owner[j]:
                break
        while j:
            k = back[j]
            owner[j] = owner[k]
            j = k
    chosen = [None] * n
    for k, i in enumerate(owner[1:]):
        if i:
            chosen[i - 1] = k
    return chosen


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

    def _prompt(self, scope, ids):
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
        return ids

    def acquire(self, scope, ids):
        ids = self._prompt(scope, ids)
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

    def acquire_batch(self, requests, *, strategy="joint"):
        """Reserve all or none, returning leases in input order. Does not wait to form a batch.

        Joint mode maximizes eligible confirmed prefix length across all idle slots, then
        favors less recently used slots on ties. This optimizes a token-work estimate,
        not measured latency: actual reuse and output equality must be checked downstream.
        """
        if self._policy != "affinity" or strategy not in ("greedy", "joint"):
            raise ValueError("batch assignment requires an affinity router and a known strategy")
        if (
            not isinstance(requests, (tuple, list))
            or not 0 < len(requests) <= len(self._slots) <= 64
        ):
            raise ValueError("requires 1..64 slots and no more requests than slots")
        items = []
        for spec in requests:
            if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                raise ValueError("expected (scope, token IDs)")
            scope, ids = spec
            items.append((scope, self._prompt(scope, ids)))
        with self._lock:
            free = sorted(
                (i for i, s in enumerate(self._slots) if s.pending is None),
                key=lambda i: (self._slots[i].age, i),
            )
            if len(free) < len(items):
                return None
            weights = []
            for scope, ids in items:
                line = []
                for j in free:
                    slot = self._slots[j]
                    match = common_prefix(slot.ids, ids) if slot.scope == scope else 0
                    line.append(match if match >= self._min else 0)
                weights.append(line)
            if strategy == "joint":
                # All secondary rank terms together cannot outweigh one reused token.
                scale = len(items) * len(free) + 1
                choices = minimum_assignment(
                    [[-w * scale + rank for rank, w in enumerate(row)] for row in weights]
                )
            else:
                left = set(range(len(free)))
                choices = []
                for row in weights:
                    j = min(left, key=lambda k: (-row[k], k))
                    choices.append(j)
                    left.remove(j)
            leases = []
            for i, (scope, ids) in enumerate(items):
                j = choices[i]
                idx = free[j]
                gain = weights[i][j]
                lease = Lease(idx, gain, bool(gain))
                leases.append(lease)
                self._slots[idx].pending = (lease, scope, ids)
            self._cursor = (leases[-1].slot + 1) % len(self._slots)
            return leases

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
