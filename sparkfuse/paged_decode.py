"""Recognize and fuse an explicit read-only physical-page gather with canonical one-token GQA."""

from dataclasses import dataclass
import ctypes
import math
import torch
from torch.utils import _pytree as tree
from .decode_attention import _bounded, _policy, _require
from .decode_codegen import emit
from .index_program import IndexProgram
from .native import native_dispatch, _extension


@dataclass(frozen=True)
class PagePlan:
    batch: int
    heads: int
    groups: int
    width: int
    physical: int
    page: int
    sequence: int
    mask: int
    dtype: torch.dtype
    device: torch.device
    ragged: bool = False

    @property
    def pages(self):
        return (self.sequence + self.page - 1) // self.page

    @property
    def scratch(self):
        return 0 if self.pages == 1 else self.batch * self.heads * self.pages * (self.width + 2) * 4


def recognize(ep):
    ns = list(ep.graph.nodes)
    ins = [n for n in ns if n.op == "placeholder"]
    n = len(ins)
    ragged = n == 6
    mi = 5 if ragged else 4
    _require(
        n in (4, 5, 6)
        and ns[:n] == ins
        and not ep.range_constraints
        and all(x.kind.name == "USER_INPUT" for x in ep.graph_signature.input_specs)
        and len(ep.graph_signature.output_specs) == 1
        and ep.graph_signature.output_specs[0].kind.name == "USER_OUTPUT"
        and ep.call_spec.in_spec == tree.tree_flatten((tuple(range(n)), {}))[1]
        and ep.call_spec.out_spec.is_leaf(),
        "expected static stateless single-output export",
    )
    ms = [q.meta.get("val") for q in ins]
    _require(
        all(
            isinstance(m, torch.Tensor)
            and m.layout == torch.strided
            and m.device == ms[0].device
            and not m.is_conj()
            and not m.is_neg()
            and all(type(k) is int and k > 0 for k in m.shape)
            and m.stride() == tuple(math.prod(m.shape[i + 1 :]) for i in range(m.ndim))
            for m in ms
        ),
        "canonical inputs required",
    )
    _require(
        ms[0].device.type == "cuda"
        and ms[0].dtype in (torch.float16, torch.bfloat16)
        and all(m.dtype == ms[0].dtype for m in [ms[1], ms[2], *ms[mi:]])
        and ms[3].dtype == torch.int32,
        "expected CUDA half/BF16 inputs and an int32 page table",
    )
    _require(ms[0].ndim == ms[1].ndim == ms[2].ndim == 4 and ms[3].ndim == 2, "invalid page ranks")
    b, h, one, d = ms[0].shape
    p, g, t, d2 = ms[1].shape
    bt, c = ms[3].shape
    _require(
        ms[2].shape == ms[1].shape
        and b == bt
        and 1 <= b <= 32
        and 1 <= g <= h <= 256
        and h % g == 0
        and d == d2
        and d in (64, 128)
        and one == 1
        and t in (32, 64)
        and p * g * t * d < 2**29,
        "unsupported page geometry",
    )
    ops = iter(ns[n:])
    aten = torch.ops.aten

    def take(op, args, kwargs=None):
        a = next(ops, None)
        _require(
            a is not None
            and a.op == "call_function"
            and a.target == op
            and a.args == args
            and a.kwargs == (kwargs or {}),
            "expected " + str(op),
        )
        return a

    tab = take(aten.clamp.default, (ins[3], 0, p - 1))
    ids = take(aten.reshape.default, (tab, [-1]))
    seq = None
    dense = []
    for bank in ins[1:3]:
        z = take(aten.index_select.default, (bank, 0, ids))
        z = take(aten.reshape.default, (z, [b, c, g, t, d]))
        z = take(aten.permute.default, (z, [0, 2, 1, 3, 4]))
        z = take(aten.reshape.default, (z, [b, g, c * t, d]))
        a = next(ops, None)
        _require(
            a is not None
            and a.op == "call_function"
            and a.target == aten.narrow.default
            and not a.kwargs
            and len(a.args) == 4
            and a.args[:3] == (z, 2, 0)
            and type(a.args[3]) is int
            and (c - 1) * t < a.args[3] <= min(c * t, 32768),
            "expected precisely one read-only logical tail",
        )
        _require(seq in (None, a.args[3]), "K/V valid lengths differ")
        seq = a.args[3]
        dense.append(a)
    mask = ms[mi].shape[0] if n > mi and ms[mi].ndim == 4 else 0
    _require(
        n == 4 or (mask in (1, b) and ms[mi].shape == (mask, 1, 1, seq)), "invalid additive mask"
    )
    bias = ins[mi:]
    if ragged:
        _require(
            ms[4].shape == (b,) and ms[4].dtype == torch.int32, "per-row int32 lengths required"
        )
        pos = take(
            aten.arange.default,
            (seq,),
            {"dtype": torch.int32, "device": ms[0].device, "pin_memory": False},
        )
        length = take(aten.reshape.default, (ins[4], [b, 1, 1, 1]))
        inactive = take(aten.ge.Tensor, (pos, length))
        bias = [take(aten.masked_fill.Scalar, (ins[5], inactive, -math.inf))]
    sd = take(
        aten.scaled_dot_product_attention.default, (ins[0], *dense, *bias), {"enable_gqa": True}
    )
    y = take(aten.clone.default, (sd,), {"memory_format": torch.contiguous_format})
    out = next(ops, None)
    _require(
        out is not None
        and out.op == "output"
        and out.args == ((y,),)
        and not out.kwargs
        and next(ops, None) is None,
        "gather or auxiliary result escapes",
    )
    return PagePlan(b, h, g, d, p, t, seq, mask, ms[0].dtype, ms[0].device, ragged), tuple(
        tuple(m.shape) for m in ms
    )


class PagedDecode:
    def _guard(self, xs):
        p = self.plan
        if (
            not torch.is_inference_mode_enabled()
            or _policy() != self.policy
            or len(xs) != len(self.shapes)
        ):
            raise ValueError("qualified inference policy and operands required")
        # The partial loads 128 bits at a time. Check the actual storage offset every call.
        integer = (3, 4) if p.ragged else (3,)
        for i, (x, sh) in enumerate(zip(xs, self.shapes, strict=True)):
            if (
                type(x) is not torch.Tensor
                or x.layout != torch.strided
                or x.device != p.device
                or x.shape != sh
                or x.dtype != (torch.int32 if i in integer else p.dtype)
                or x.is_conj()
                or x.is_neg()
                or not x.is_contiguous()
                or x.data_ptr() % (4 if i in integer else 16)
            ):
                raise ValueError("invalid page operand descriptor/alignment")

    def _run(self, xs):
        (z,) = self.first(*xs)
        return (
            self.merge(z, *((xs[4],) if self.plan.ragged else ()))[0]
            if self.merge is not None
            else z
        )

    def __call__(self, *xs):
        self._guard(xs)
        _bounded((*xs[:3], *xs[4 + int(self.plan.ragged) :]))
        return self.page_io(*xs) if self.page_io is not None else self._run(xs)

    def run_bounded(self, *xs):
        """Caller asserts the documented Q/K/V and bias domain. All int32 page IDs are valid."""
        if self.page_io is not None:
            if _policy() != self.policy:
                raise ValueError("qualified inference policy required")
            return self.page_io(*xs)
        self._guard(xs)
        return self._run(xs)


def compile_paged_decode(
    ep, example, audit, *, warp_heads=4, max_workspace_bytes=32 << 20, host_backend="split"
):
    _require(host_backend in ("split", "native"), "unknown host backend")
    p, shapes = recognize(ep)
    z = PagedDecode()
    z.page_io = None
    z.policy = _policy()
    z.plan = p
    z.shapes = shapes
    _require(
        torch.is_inference_mode_enabled()
        and not z.policy[0]
        and type(warp_heads) is int
        and warp_heads in (1, 4)
        and type(max_workspace_bytes) is int
        and max_workspace_bytes >= p.scratch,
        "unsupported inference policy, schedule, or workspace",
    )
    a, b = tuple(example), tuple(audit)
    for xs in (a, b):
        z._guard(xs)
        _bounded((*xs[:3], *xs[4 + int(p.ragged) :]))
    _require(
        {v.untyped_storage().data_ptr() for v in a}.isdisjoint(
            v.untyped_storage().data_ptr() for v in b
        ),
        "independent qualification storages required",
    )
    dt = "bf16" if p.dtype == torch.bfloat16 else "f16"
    code = emit(
        p.batch,
        p.heads,
        p.groups,
        p.sequence,
        p.width,
        dt,
        p.page,
        1 / math.sqrt(p.width),
        p.mask,
        warp_heads=warp_heads,
        narrow_bias=bool(p.mask),
        page_pool=p.physical,
        ragged=p.ragged,
    )
    scratch = (p.batch, p.heads, p.pages, p.width + 2)
    first = IndexProgram(
        shapes,
        [scratch if p.pages > 1 else shapes[0]],
        dt,
        p.device,
        code[0],
        input_dtypes=[dt, dt, dt, "i32", *(["i32"] if p.ragged else []), *([dt] if p.mask else [])],
        output_dtypes=["f32" if p.pages > 1 else dt],
    )
    z.programs = [first]
    z.first = native_dispatch(first)
    z.merge = None
    if code[1]:
        second = IndexProgram(
            [scratch, *([(p.batch,)] if p.ragged else [])],
            [shapes[0]],
            dt,
            p.device,
            code[1],
            input_dtypes=["f32", *(["i32"] if p.ragged else [])],
        )
        z.programs.append(second)
        z.merge = native_dispatch(second)
    if host_backend == "native":
        libs = tuple(k.lib for k in z.programs)
        addr = [ctypes.cast(v.launch, ctypes.c_void_p).value for v in libs]
        z.page_io = (
            _extension()
            ._PageIO(
                libs,
                addr[0],
                addr[1] if len(addr) > 1 else 0,
                p.device.index,
                p.batch,
                p.heads,
                p.groups,
                p.width,
                p.physical,
                p.page,
                p.sequence,
                p.mask,
                p.ragged,
                p.dtype == torch.bfloat16,
            )
            .run
        )
    m = ep.module()
    for xs in (a, b):
        ref = m(*xs)
        val = z.run_bounded(*xs)
        _require(
            val.stride() == ref.stride()
            and val._base is None
            and val.storage_offset() == 0
            and val.untyped_storage().nbytes() == val.numel() * val.element_size(),
            "unexpected result layout",
        )
        limit = (0.004, 0.016) if p.dtype == torch.bfloat16 else (0.0006, 0.003)
        _require(
            bool(
                ((val.float() - ref.float()).abs() <= limit[0] + limit[1] * ref.float().abs()).all()
            ),
            "page decode failed numerical qualification",
        )
    return z
