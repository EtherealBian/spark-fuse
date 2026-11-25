"""Qualified whole-export lowering for one-token, static scaled dot-product attention."""

from dataclasses import dataclass
import math
import torch
from torch.utils import _pytree as tree
from .decode_codegen import emit
from .index_program import IndexProgram
from .ir import Unsupported
from .native import native_dispatch


def _require(condition, why):
    if not condition:
        raise Unsupported(why)


# The private runtime switches are version-specific. Resolve the getters once, but
# query their live values for every dispatch; fail closed on an unfamiliar build.
_POLICY_KEYS = (
    "_get_flash_sdp_enabled",
    "_get_math_sdp_enabled",
    "_get_mem_efficient_sdp_enabled",
    "_get_cudnn_sdp_enabled",
    "_get_math_sdp_allow_fp16_bf16_reduction",
    "_get_fa3_sdp_enabled",
    "_get_overrideable_sdp_enabled",
)
_GETTERS = tuple(getattr(torch._C, k, None) for k in _POLICY_KEYS)
_PRIORITY = getattr(torch._C, "_get_sdp_priority_order", None)
_KNOWN = _PRIORITY is not None and all(f is not None for f in _GETTERS)


def _policy():
    if not _KNOWN:
        raise Unsupported("cannot fingerprint this runtime's SDP backend policy")
    return (
        torch.is_autocast_enabled("cuda"),
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.enabled,
        torch.get_float32_matmul_precision(),
        tuple(_PRIORITY()),
        *(f() for f in _GETTERS),
    )


def _bounded(xs):
    # NaN and +/-inf fail bounded comparisons; -inf is a legal additive mask.
    for i, x in enumerate(xs):
        valid = x.abs() <= (10000 if i == 3 else 64)
        if i == 3:
            valid |= torch.isneginf(x)
        if not bool(valid.all()):
            raise ValueError("outside declared decode numerical domain")


def _pattern(ep):
    nodes = list(ep.graph.nodes)
    ins = [n for n in nodes if n.op == "placeholder"]
    n = len(ins)
    _require(
        n in (3, 4) and not ep.range_constraints and len(nodes) == n + 2 and nodes[:n] == ins,
        "expected isolated static SDPA export",
    )
    _require(
        all(s.kind.name == "USER_INPUT" for s in ep.graph_signature.input_specs)
        and len(ep.graph_signature.output_specs) == 1
        and ep.graph_signature.output_specs[0].kind.name == "USER_OUTPUT",
        "state or mutation not supported",
    )
    op, out = nodes[-2:]
    _require(
        op.op == "call_function"
        and op.target is torch.ops.aten.scaled_dot_product_attention.default
        and out.op == "output"
        and out.args == ((op,),),
        "expected one SDPA output",
    )
    _require(
        ep.call_spec.in_spec == tree.tree_flatten((tuple(range(n)), {}))[1]
        and ep.call_spec.out_spec.is_leaf(),
        "expected positional tensor inputs and a single tensor result",
    )
    return ins, _sdpa_options(op, ins)


def _sdpa_options(op, ins):
    n = len(ins)
    # Bind the operator schema, including explicitly supplied default arguments.
    schema = op.target._schema.arguments
    options = {x.name: x.default_value for x in schema}
    options.update({x.name: v for x, v in zip(schema, op.args)})
    options.update(op.kwargs)
    _require(
        [options[k] for k in ("query", "key", "value")] == ins[:3]
        and options["attn_mask"] is (ins[3] if n == 4 else None),
        "input order or mask not recognized",
    )
    _require(
        options["dropout_p"] == 0.0
        and options["is_causal"] is False
        and type(options["enable_gqa"]) is bool,
        "causal, dynamic options and dropout are unsupported",
    )
    return options


@dataclass(frozen=True)
class DecodePlan:
    batch: int
    heads: int
    kv_heads: int
    sequence: int
    width: int
    tile: int
    chunks: int
    workspace_bytes: int
    output_stride: tuple
    scale: float
    head_group: int


class QualifiedDecode:
    def __init__(self, plan, first, second, policy, raw_stride):
        self.plan, self._first, self._second, self._policy = plan, first, second, policy
        self._fix = raw_stride != plan.output_stride

    def _guard(self, args):
        if not torch.is_inference_mode_enabled() or _policy() != self._policy:
            raise ValueError("inference mode and qualified SDP backend policy required")
        if any(
            type(x) is not torch.Tensor
            or x.layout != torch.strided
            or x.device.type != "cuda"
            or x.is_neg()
            or x.is_conj()
            or x.data_ptr() % 16
            for x in args
        ):
            raise ValueError("expected aligned, regular CUDA tensors")

    def _execute(self, args):
        y = self._first(*args)[0]
        if self._second is not None:
            y = self._second(y)[0]
        if self._fix:
            y.as_strided_(y.shape, self.plan.output_stride)
        return y

    def __call__(self, *args):
        """Checked API; scans the numerical domain each call (synchronizes)."""
        self._guard(args)
        # Validate descriptors before performing any input-sized checks.
        if tuple((x.shape, x.stride(), x.dtype, x.device) for x in args) != self._descriptors:
            raise ValueError("decode input descriptor mismatch")
        _bounded(args)
        return self._execute(args)

    def run_bounded(self, *args):
        """Fast inference entry: caller guarantees |Q,K,V|<=64, mask in [-1e4,1e4] or -inf."""
        self._guard(args)
        return self._execute(args)


def compile_decode_attention(
    ep, example, audit, *, tile=32, head_group=1, max_workspace_bytes=32 << 20
):
    """Qualify one exported ATen SDPA against two distinct batches on the target GPU."""
    ins, opt = _pattern(ep)
    mod = ep.module()
    return _build_site(
        ins,
        opt,
        example,
        audit,
        reference=lambda i, batch: mod(*batch),
        tile=tile,
        head_group=head_group,
        max_workspace_bytes=max_workspace_bytes,
    )


def _build_site(ins, opt, example, audit, *, reference, tile, head_group, max_workspace_bytes):
    """Use the original operator output, not a compact-GQA eager surrogate."""
    policy = _policy()
    _require(
        torch.is_inference_mode_enabled() and not policy[0],
        "qualify in inference mode without autocast",
    )
    _require(
        type(tile) is int
        and tile in (32, 64, 128, 256)
        and type(head_group) is int
        and head_group in (1, 2, 4)
        and type(max_workspace_bytes) is int
        and max_workspace_bytes >= 0,
        "invalid tile or scratch budget",
    )
    xs = tuple(example)
    ys = tuple(audit)
    n = len(ins)
    _require(len(xs) == len(ys) == n, "wrong number of inputs")
    for batch in (xs, ys):
        for input_index, (node, x) in enumerate(zip(ins, batch)):
            m = node.meta.get("val")
            _require(
                type(x) is torch.Tensor
                and isinstance(m, torch.Tensor)
                and x.layout == torch.strided
                and x.device.type == "cuda"
                and x.device == m.device
                and x.dtype == m.dtype
                and not x.is_conj()
                and not x.is_neg()
                and all(type(a) is int for a in m.shape)
                and x.shape == m.shape
                and x.is_contiguous()
                and all(a == b or k == 1 for a, b, k in zip(x.stride(), m.stride(), x.shape))
                and x.stride() == xs[input_index].stride()
                and all(a > 0 for a in x.stride())
                and x.data_ptr() % 16 == 0,
                "expected fixed, aligned input matching export metadata",
            )
    _require(
        {x.untyped_storage().data_ptr() for x in xs}.isdisjoint(
            y.untyped_storage().data_ptr() for y in ys
        ),
        "audit must use independent input storage",
    )
    q, k, v = xs[:3]
    _require(
        q.ndim == k.ndim == v.ndim == 4
        and k.shape == v.shape
        and q.dtype in (torch.bfloat16, torch.float16)
        and all(x.device == q.device for x in xs)
        and k.dtype == v.dtype == q.dtype,
        "invalid QKV descriptors",
    )
    b, h, query_length, d = q.shape
    bk, g, s, dk = k.shape
    _require(
        b == bk
        and query_length == 1
        and d == dk
        and d in (64, 128)
        and 1 <= b <= 32
        and 1 <= g <= h <= 256
        and h % (g * head_group) == 0
        and (h == g or opt["enable_gqa"])
        and 1 <= s <= 32768,
        "unsupported single-token decode geometry",
    )
    mask_b = 0
    if n == 4:
        m = xs[3]
        _require(
            m.dtype in (q.dtype, torch.float32)
            and m.ndim == 4
            and m.shape[0] in (1, b)
            and tuple(m.shape[1:]) == (1, 1, s)
            and (m.dtype == q.dtype or h > g),
            "expected shared-head additive mask (FP32 only for GQA)",
        )
        mask_b = m.shape[0]
    scale = opt["scale"] if opt["scale"] is not None else 1 / math.sqrt(d)
    _require(
        type(scale) in (int, float) and math.isfinite(scale) and 0 < scale <= 32,
        "invalid constant scale",
    )
    chunks = (s + tile - 1) // tile
    scratch = (b, h, chunks, d + 2)
    size = 0 if chunks == 1 else 4 * math.prod(scratch)
    _require(
        max(b * g * s * d, b * h * d, math.prod(scratch)) < 2**29 and size <= max_workspace_bytes,
        "index range or workspace budget exceeded",
    )
    stride = None
    refs = []
    for batch in (xs, ys):
        _bounded(batch)
        y = reference(len(refs), batch)
        _require(
            y.shape == q.shape
            and y.dtype == q.dtype
            and y.device == q.device
            and y.is_contiguous()
            and y.storage_offset() == 0
            and y.untyped_storage().nbytes() == y.numel() * y.element_size()
            and y._base is None
            and all(
                y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr() for x in batch
            ),
            "unrecognized output layout/ownership",
        )
        _require(stride is None or stride == y.stride(), "inconsistent eager output stride")
        stride = y.stride()
        refs.append(y)
    dt = "bf16" if q.dtype == torch.bfloat16 else "f16"
    a, m = emit(
        b,
        h,
        g,
        s,
        d,
        dt,
        tile,
        scale,
        mask_b,
        head_group=head_group,
        narrow_bias=bool(mask_b and xs[3].dtype == q.dtype),
    )
    p1 = IndexProgram(
        [tuple(x.shape) for x in xs],
        [scratch if m else tuple(q.shape)],
        dt,
        q.device,
        a,
        input_strides=[x.stride() for x in xs],
        input_dtypes=[dt] * 3 + ([dt if xs[3].dtype == q.dtype else "f32"] if mask_b else []),
        output_dtypes=["f32" if m else dt],
    )
    p2 = (
        IndexProgram([scratch], [tuple(q.shape)], dt, q.device, m, input_dtypes=["f32"])
        if m
        else None
    )
    first = native_dispatch(p1)
    second = native_dispatch(p2) if p2 else None
    result = QualifiedDecode(
        DecodePlan(b, h, g, s, d, tile, chunks, size, stride, scale, head_group),
        first,
        second,
        policy,
        (h * d, d, d, 1),
    )
    result.programs = (p1, p2) if p2 else (p1,)
    result._descriptors = tuple((x.shape, x.stride(), x.dtype, x.device) for x in xs)
    # Qualification is approximate, not the bitwise contract used for integer GEMMs.
    for batch, ref in zip((xs, ys), refs):
        y = result.run_bounded(*batch)
        atol, rtol = (0.012, 0.02) if dt == "bf16" else (0.002, 0.004)
        try:
            torch.testing.assert_close(y, ref, atol=atol, rtol=rtol, check_stride=True)
        except AssertionError as e:
            raise Unsupported("qualified decode disagrees with eager") from e
    return result
