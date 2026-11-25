"""One bounded symbolic KV axis, two CUDA binaries, exact canonical output ownership."""

import ctypes as C
from dataclasses import dataclass
import math
import sympy
import torch
from torch.utils import _pytree as tree
from .decode_attention import _bounded, _policy, _require, _sdpa_options
from .decode_codegen import emit
from .ir import Unsupported
from .runtime import build


def _expr(x):
    return x.node.expr if isinstance(x, torch.SymInt) else x


def _pattern(ep):
    ns = list(ep.graph.nodes)
    ins = [n for n in ns if n.op == "placeholder"]
    n = len(ins)
    _require(
        n in (3, 4) and len(ns) == n + 3 and ns[:n] == ins and len(ep.range_constraints) == 1,
        "expected single-symbol SDPA and explicit contiguous clone",
    )
    op, clone, out = ns[-3:]
    _require(
        all(x.kind.name == "USER_INPUT" for x in ep.graph_signature.input_specs)
        and len(ep.graph_signature.output_specs) == 1
        and ep.graph_signature.output_specs[0].kind.name == "USER_OUTPUT"
        and ep.call_spec.in_spec == tree.tree_flatten((tuple(range(n)), {}))[1]
        and ep.call_spec.out_spec.is_leaf(),
        "expected positional, stateless tensor export",
    )
    _require(
        op.op == "call_function"
        and op.target is torch.ops.aten.scaled_dot_product_attention.default
        and clone.op == "call_function"
        and clone.target is torch.ops.aten.clone.default
        and clone.args == (op,)
        and clone.kwargs == {"memory_format": torch.contiguous_format}
        and out.op == "output"
        and out.args == ((clone,),),
        "canonical clone must explicitly define fresh result strides",
    )
    symbol, bounds = next(iter(ep.range_constraints.items()))
    _require(
        isinstance(symbol, sympy.Symbol)
        and isinstance(bounds.lower, sympy.Integer)
        and isinstance(bounds.upper, sympy.Integer)
        and 2 <= bounds.lower < bounds.upper <= 32768,
        "need a finite bounded KV axis in [2,32768]",
    )
    opt = _sdpa_options(op, ins)
    ms = [x.meta.get("val") for x in ins]
    _require(
        all(isinstance(m, torch.Tensor) and m.ndim == 4 for m in ms), "expected rank four inputs"
    )
    shapes = [tuple(map(_expr, m.shape)) for m in ms]
    b, h, one, d = shapes[0]
    g = shapes[1][1]
    mask = shapes[-1][0] if n == 4 else 0
    _require(
        all(type(i) is int for i in (b, h, one, d, g, mask))
        and 1 <= b <= 32
        and 1 <= g <= h <= 256
        and h % g == 0
        and (h == g or opt["enable_gqa"])
        and one == 1
        and d in (64, 128)
        and shapes[1] == shapes[2] == (b, g, symbol, d)
        and (n == 3 or (mask in (1, b) and shapes[3] == (mask, 1, 1, symbol))),
        "not a single shared KV length",
    )
    for i, (m, sh) in enumerate(zip(ms, shapes)):
        _require(
            m.layout == torch.strided
            and m.dtype == ms[0].dtype
            and m.device == ms[0].device
            and not m.is_neg()
            and not m.is_conj()
            and tuple(map(_expr, m.stride())) == tuple(math.prod(sh[j + 1 :]) for j in range(4)),
            "need canonical dense narrow inputs",
        )
    scale = 1 / math.sqrt(d) if opt["scale"] is None else opt["scale"]
    _require(
        ms[0].dtype in (torch.bfloat16, torch.float16)
        and ms[0].device.type == "cuda"
        and type(scale) in (int, float)
        and math.isfinite(scale)
        and 0 < scale <= 32,
        "CUDA narrow inputs and constant scale required",
    )
    return int(bounds.lower), int(bounds.upper), b, h, g, d, mask, ms[0].dtype, ms[0].device, scale


@dataclass(frozen=True)
class LengthPlan:
    lower: int
    upper: int
    batch: int
    heads: int
    kv_heads: int
    width: int
    mask_batch: int
    dtype: torch.dtype
    device: torch.device
    scale: float
    tile: int

    def workspace_bytes(self, s):
        if type(s) is not int or not self.lower <= s <= self.upper:
            raise ValueError("length outside exported range")
        c = (s + self.tile - 1) // self.tile
        return 0 if c == 1 else 4 * self.batch * self.heads * c * (self.width + 2)


class LengthDecode:
    def _check_policy(self):
        if not torch.is_inference_mode_enabled() or _policy() != self._policy:
            raise ValueError("qualified inference policy required")

    def _guard(self, xs):
        self._check_policy()
        p = self.plan
        if len(xs) != 3 + bool(p.mask_batch) or any(
            type(x) is not torch.Tensor
            or x.device != p.device
            or x.dtype != p.dtype
            or x.layout != torch.strided
            or x.ndim != 4
            or x.is_conj()
            or x.is_neg()
            or x.data_ptr() % 16
            for x in xs
        ):
            raise ValueError("invalid tensor descriptor")
        s = xs[1].shape[2]
        p.workspace_bytes(s)
        kv = (p.batch, p.kv_heads, s, p.width)
        shapes = [(p.batch, p.heads, 1, p.width), kv, kv] + (
            [(p.mask_batch, 1, 1, s)] if p.mask_batch else []
        )
        for x, sh in zip(xs, shapes):
            if x.shape != sh or x.stride() != tuple(math.prod(sh[j + 1 :]) for j in range(4)):
                raise ValueError("wrong shape or stride")
        return s

    def _run(self, xs, s):
        p = self.plan
        c = (s + p.tile - 1) // p.tile
        with torch.cuda.device(p.device):
            stream = torch.cuda.current_stream(p.device)
            y = torch.empty_like(xs[0], memory_format=torch.contiguous_format)
            z = (
                y
                if c == 1
                else torch.empty(
                    (p.batch, p.heads, c, p.width + 2), device=p.device, dtype=torch.float32
                )
            )
            args = (*xs, z)
            ptr = (C.c_void_p * len(args))(*(x.data_ptr() for x in args))
            # Record ownership before either launch, even if the second launch fails.
            for x in args:
                x.record_stream(stream)
            if c != 1:
                y.record_stream(stream)
            err = self._libs[0].launch(ptr, s, stream.cuda_stream)
            if not err and c != 1:
                err = self._libs[1].launch(
                    (C.c_void_p * 2)(z.data_ptr(), y.data_ptr()), s, stream.cuda_stream
                )
            if err:
                raise RuntimeError(f"length decode launch error {err}")
            return y

    def __call__(self, *xs):
        s = self._guard(xs)
        _bounded(xs)
        return self._native(*xs) if self._native is not None else self._run(xs, s)

    def run_bounded(self, *xs):
        """Caller asserts finite |Q/K/V|<=64 and bounded additive bias, or -infinity."""
        if self._native is not None:
            self._check_policy()
            return self._native(*xs)
        return self._run(xs, self._guard(xs))


def compile_length_decode(
    ep, example, audit, *, tile=32, max_workspace_bytes=32 << 20, host_backend="ctypes"
):
    """Lower a symbolic, single-token SDPA+contiguous clone. No per-length JIT."""
    p = LengthPlan(*_pattern(ep), tile)
    policy = _policy()
    _require(
        torch.is_inference_mode_enabled()
        and not policy[0]
        and host_backend in ("ctypes", "native")
        and type(tile) is int
        and tile in (32, 64, 128, 256)
        and type(max_workspace_bytes) is int
        and p.workspace_bytes(p.upper) <= max_workspace_bytes,
        "inference policy, tile or scratch budget rejected",
    )
    result = LengthDecode()
    result.plan = p
    result._policy = policy
    xs, ys = tuple(example), tuple(audit)
    s, t = result._guard(xs), result._guard(ys)
    _require(
        s != t
        and {x.untyped_storage().data_ptr() for x in xs}.isdisjoint(
            y.untyped_storage().data_ptr() for y in ys
        ),
        "two independent lengths and storages required",
    )
    for batch in (xs, ys):
        _bounded(batch)
    a, m = emit(
        p.batch,
        p.heads,
        p.kv_heads,
        p.upper,
        p.width,
        "bf16" if p.dtype == torch.bfloat16 else "f16",
        p.tile,
        p.scale,
        p.mask_batch,
        narrow_bias=bool(p.mask_batch),
        runtime_length=True,
    )
    builds = [build(src, p.device) for src in (a, m) if src is not None]
    result._libs, result.paths = zip(*builds)
    result._native = None
    if host_backend == "native":
        from .native import _extension

        entries = [C.cast(lib.launch, C.c_void_p).value for lib in result._libs]
        first = entries[0]
        second = entries[1] if len(entries) > 1 else 0
        result._native = (
            _extension()
            ._LengthIO(
                result._libs,
                first,
                second,
                p.device.index,
                p.lower,
                p.upper,
                p.batch,
                p.heads,
                p.kv_heads,
                p.width,
                p.mask_batch,
                p.tile,
                p.dtype == torch.bfloat16,
            )
            .run
        )
    ref = ep.module()
    at, rt = (0.012, 0.02) if p.dtype == torch.bfloat16 else (0.002, 0.004)
    for batch in (xs, ys):
        e = ref(*batch)
        y = result.run_bounded(*batch)
        _require(
            e._base is None
            and e.storage_offset() == 0
            and e.untyped_storage().nbytes() == y.numel() * y.element_size()
            and all(
                e.untyped_storage().data_ptr() != x.untyped_storage().data_ptr() for x in batch
            ),
            "unexpected exported output ownership",
        )
        try:
            torch.testing.assert_close(y, e, atol=at, rtol=rt, check_stride=True)
        except AssertionError as exc:
            raise Unsupported("dynamic decode qualification failed") from exc
    return result
