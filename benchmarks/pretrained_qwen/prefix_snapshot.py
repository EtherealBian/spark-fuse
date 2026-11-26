"""In-process snapshot for a frozen batch-one Runner. One owner, single-threaded API.

No cross-process handles, cross-tenant cache, paged allocator or persistent storage.
Prepare and restore run on the same PyTorch stream. A restore copies the matching
prefix into the original graph's KV addresses, never swaps or aliases those buffers.
"""

import torch


def validate_ids(ids, limit, vocab):
    if (
        type(ids) is not tuple
        or not 0 < len(ids) < limit
        or not all(type(t) is int and 0 <= t < vocab for t in ids)
    ):
        raise ValueError("invalid token IDs or capacity")


def banks(r):
    c = r.cache
    if not c.key_cache or len(c.key_cache) != len(c.value_cache):
        raise ValueError("incomplete static KV banks")
    vs = tuple(t for pair in zip(c.key_cache, c.value_cache, strict=True) for t in pair)
    t = vs[0]
    shape = t.shape
    if (
        t.ndim != 4
        or shape[0] != 1
        or shape[-2] != r.pos.numel()
        or any(
            x.shape != shape
            or x.device != r.pos.device
            or x.dtype != t.dtype
            or not x.is_contiguous()
            for x in vs
        )
        or len({x.data_ptr() for x in vs}) != len(vs)
    ):
        raise ValueError("not disjoint, uniform, batch-one static storage")
    return vs


class PrefixFork:
    def __init__(self, runner, scope, prefix, vocab):
        if type(scope) is not str or not scope or type(vocab) is not int or vocab < 2:
            raise ValueError("nonempty scope and vocabulary required")
        validate_ids(prefix, runner.pos.numel(), vocab)
        self._owner, self._scope, self._prefix, self._vocab = runner, scope, prefix, vocab
        self._stream = torch.cuda.current_stream(runner.pos.device) if runner.pos.is_cuda else None
        vs = banks(runner)
        self._ptrs = tuple(x.data_ptr() for x in vs)
        runner.cache.reset()
        ids = torch.tensor([prefix], device=runner.pos.device, dtype=torch.long)
        runner.forward(ids, runner.pos[: len(prefix)])
        # clone(contiguous_format) guarantees independent, compact storage, even for a single KV head.
        self._state = tuple(
            x[:, :, : len(prefix), :].clone(memory_format=torch.contiguous_format) for x in vs
        )
        self.storage_bytes = sum(x.untyped_storage().nbytes() for x in self._state)
        if self.storage_bytes != sum(x.numel() * x.element_size() for x in self._state):
            raise ValueError("snapshot unexpectedly retained excess storage")

    def restore(self, runner, scope, target):
        # Reject foreign owners/keys, full-prefix-only requests, and changed backing storage
        # before resetting or writing any part of the destination.
        validate_ids(target, runner.pos.numel(), self._vocab)
        p = len(self._prefix)
        if (
            runner is not self._owner
            or scope != self._scope
            or type(scope) is not str
            or len(target) <= p
            or target[:p] != self._prefix
        ):
            raise ValueError("snapshot does not own this continuation")
        if (
            self._stream is not None
            and torch.cuda.current_stream(runner.pos.device) != self._stream
        ):
            raise ValueError("unexpected CUDA stream")
        vs = banks(runner)
        if tuple(x.data_ptr() for x in vs) != self._ptrs or any(
            x.dtype != y.dtype or x.shape[:2] + x.shape[3:] != y.shape[:2] + y.shape[3:]
            for x, y in zip(vs, self._state, strict=True)
        ):
            raise ValueError("destination storage changed")
        runner.cache.reset()
        for dst, src in zip(vs, self._state, strict=True):
            dst[:, :, :p, :].copy_(src)
        return p
