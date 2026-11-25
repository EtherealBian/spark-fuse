"""Fixed-address CUDA execution with private weights, request staging and caller-owned results."""

import copy
from dataclasses import asdict, dataclass
import operator
import threading

import torch
from torch import fx, nn
from torch.export._unlift import GuardsFn
from torch.utils import _pytree as tree

from .contextual import _bytes, _exact
from .ir import Unsupported
from .index_program import IndexRegion
from .partition import _RegionModule, partition_export

# Closed functional ATen boundary. No custom operators, RNG, CPU scalar extraction,
# in-place writes, data-dependent output shapes, control flow, or arbitrary Python.
_OPS = frozenset(
    """add.Tensor add.Scalar addmm.default alias.default as_strided.default bmm.default
cat.default chunk.default clone.default contiguous.default detach.default div.Tensor div.Scalar
expand.default flatten.using_ints gelu.default layer_norm.default linear.default matmul.default
mean.dim mm.default mul.Tensor mul.Scalar native_layer_norm.default neg.default permute.default
pow.Tensor_Scalar relu.default reshape.default rms_norm.default rsqrt.default select.int
sigmoid.default silu.default slice.Tensor softmax.int _softmax.default split.Tensor split_with_sizes.default
square.default squeeze.dim squeeze.default sub.Tensor t.default tanh.default to.dtype _to_copy.default
transpose.int unbind.int unsqueeze.default view.default view_as.default _assert_tensor_metadata.default""".split()
)
_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int64, torch.bool}
_CAPTURE = threading.Lock()


def _validate_export(ep, *, quantized=False):
    if ep.range_constraints or any(
        s.kind.name != "USER_OUTPUT" for s in ep.graph_signature.output_specs
    ):
        raise Unsupported("resident execution requires a static, functional user-output export")
    added = set()
    if quantized:
        from .quant_state import certify

        added = certify(ep)
    for n in ep.graph.nodes:
        if n.op in ("placeholder", "output"):
            continue
        if n.op == "call_function" and n.target is operator.getitem:
            continue
        if (
            n.op == "call_function"
            and n.target == torch.ops.aten.dropout.default
            and len(n.args) == 3
            and n.args[2] is False
        ):
            continue
        if (
            n.op != "call_function"
            or (str(n.target).removeprefix("aten.") not in _OPS and n not in added)
            or not isinstance(n.target, torch._ops.OpOverload)
            or n.target.namespace != "aten"
            or n.target._schema.is_mutable
        ):
            raise Unsupported(f"outside resident functional allowlist: {n.op} {n.target}")
        if any(
            isinstance(v, torch.device) and v.type != "cuda"
            for v in tree.tree_leaves((n.args, n.kwargs))
        ):
            raise Unsupported("explicit non-CUDA tensor device")


def _store(t):
    return (t.device, t.untyped_storage().data_ptr())


def _geometry(t):
    return (t.dtype, tuple(t.shape), tuple(t.stride()), t.storage_offset())


def _plain(t, *, int8=False):
    return (
        type(t) in (torch.Tensor, nn.Parameter)
        and t.layout == torch.strided
        and (t.dtype in _DTYPES or (int8 and t.dtype is torch.int8))
        and t.numel() > 0
        and not t.is_conj()
        and not t.is_neg()
    )


def _whole(t):
    return (
        t.storage_offset() == 0
        and t.is_contiguous()
        and t.untyped_storage().nbytes() == t.numel() * t.element_size()
    )


def _request(args, dev):
    if not isinstance(args, tuple) or not args:
        raise TypeError("expected a nonempty positional tuple of tensors")
    stores = set()
    layout = []
    for t in args:
        if not _plain(t) or t.device != dev or not _whole(t):
            raise Unsupported("request requires whole contiguous tensors on the capture device")
        key = _store(t)
        if key in stores:
            raise Unsupported("request inputs share storage")
        stores.add(key)
        layout.append(_geometry(t))
    return tuple(layout)


def _get(root, path):
    for piece in path.split("."):
        root = getattr(root, piece)
    return root


def _snapshot(gm, dev, args, max_bytes, shared, rhs_layout, adopted=None):
    # Create a separate module hierarchy. Rebinding/mutating the caller's Parameters
    # after setup cannot free or overwrite cuBLAS/CUDA addresses captured here.
    root = nn.Module()
    copied = {}
    state_stores = set()
    used = sum(x.untyped_storage().nbytes() for x in args)
    if used > max_bytes:
        raise ValueError("request staging + weight snapshot exceeds byte limit")
    graph = copy.deepcopy(gm.graph)
    # An export may lift an unused buffer from another shape-specific branch.
    # Do not count or retain dead state as a dependency of this executable.
    for n in list(graph.nodes):
        if n.op == "get_attr" and not n.users:
            graph.erase_node(n)
    names = {n.target for n in graph.nodes if n.op == "get_attr"}
    if shared is not None and names != set(shared):
        raise Unsupported("shape family changed its named state")
    bindings, inventory = {}, []
    if rhs_layout is not None:
        from .quant_state import frozen_weights

        bindings, inventory = frozen_weights(graph, gm, dev, rhs_layout, _get, _store, _geometry)
    private_keys = set()
    for n in list(graph.nodes):
        if n.op == "call_module":
            if "." in n.target:
                raise Unsupported("unexpected nested mixed module")
            mod = gm.get_submodule(n.target)
            if n.target == "_guards_fn" and type(mod) is GuardsFn and not n.users:
                # Original ExportedProgram guards still run for both qualification
                # requests. Replay enforces their static tensor layout separately.
                graph.erase_node(n)
                continue
            if type(mod) not in (_RegionModule, IndexRegion):
                raise Unsupported("unknown mixed submodule")
            kernel = mod.kernel if type(mod) is _RegionModule else mod.program
            if (
                any(t is not None for t in kernel.bind)
                or mod._forward_hooks
                or mod._forward_pre_hooks
            ):
                raise Unsupported("mixed submodule has hidden state or hooks")
            root.add_module(n.target, mod)
        elif n.op == "get_attr":
            t = _get(gm, n.target)
            packed = bindings.get(n.target)
            if (
                not _plain(t, int8=packed is not None)
                or t.device != dev
                or (packed is None and not _whole(t))
            ):
                raise Unsupported(
                    "resident state must be whole contiguous tensors on the capture device"
                )
            key = _store(t)
            state_stores.add(key)
            if any(key == _store(x) for x in args):
                raise Unsupported("request aliases model state")
            if key not in copied:
                used += t.untyped_storage().nbytes()
                if used > max_bytes:
                    raise ValueError("request staging + weight snapshot exceeds byte limit")
                if adopted is not None and n.target in adopted:
                    # Only the final allocation minted by prepare() is adoptable.
                    if (
                        t is not adopted[n.target]
                        or packed != (1, t.shape[0])
                        or t._base is not None
                    ):
                        raise Unsupported("invalid internally packed weight")
                    copied[key] = t
                elif shared is None:
                    # Allocate a single private RHS in its final physical order; no retained source clone.
                    copied[key] = (
                        t.detach().clone()
                        if packed is None
                        else torch.empty_strided(
                            t.shape, packed, device=t.device, dtype=t.dtype
                        ).copy_(t.detach())
                    )
                else:
                    candidate = shared[n.target]
                    if (
                        candidate.device != t.device
                        or _geometry(candidate) != _geometry(t)
                        or not torch.equal(_bytes(t), _bytes(candidate))
                    ):
                        raise Unsupported(
                            "shape family state contents or layout changed: " + n.target
                        )
                    if _store(candidate) in private_keys:
                        raise Unsupported("shape family state alias groups changed")
                    private_keys.add(_store(candidate))
                    copied[key] = candidate
            if shared is not None and _store(copied[key]) != _store(shared[n.target]):
                raise Unsupported("shape family state alias groups changed")
            desired = _geometry(t) if packed is None else (t.dtype, tuple(t.shape), packed, 0)
            if _geometry(copied[key]) != desired:
                raise Unsupported("incompatible tied state layouts")
            parent = root
            parts = n.target.split(".")
            for piece in parts[:-1]:
                if not hasattr(parent, piece):
                    parent.add_module(piece, nn.Module())
                parent = getattr(parent, piece)
            parent.register_buffer(parts[-1], copied[key])
    owned = fx.GraphModule(root, graph)
    for name in ("_in_spec", "_out_spec"):
        if hasattr(gm, name):
            setattr(owned, name, getattr(gm, name))
    return owned, state_stores, {_store(x) for x in copied.values()}, used, inventory


@dataclass(frozen=True)
class _Binding:
    # base is an input index, a full-size fresh output index, or an already-built identical leaf.
    source: str
    base: int
    shape: tuple
    stride: tuple
    offset: int


def _recipe(args, out, forbidden, *, int8=False):
    ys, spec = tree.tree_flatten(out)
    if not ys or not all(_plain(y, int8=int8) and y.device == args[0].device for y in ys):
        raise Unsupported("expected nonempty same-device tensor outputs")
    by_input = {_store(x): i for i, x in enumerate(args)}
    leaders = {}
    for i, y in enumerate(ys):
        k = _store(y)
        if k in forbidden:
            raise Unsupported("output aliases captured model state")
        if k not in by_input and _whole(y):
            leaders.setdefault(k, i)
    bindings = []
    seen = {}
    for i, y in enumerate(ys):
        key = _store(y)
        if id(y) in seen:
            source, base = "repeat", seen[id(y)]
        elif key in by_input:
            base = by_input[key]
            src = args[base]
            source = "input" if y is src else "input_view"
            if src.dtype != y.dtype:
                raise Unsupported("dtype-reinterpreting output view")
        elif key in leaders:
            base = leaders[key]
            source = "copy" if i == base else "output_view"
            if ys[base].dtype != y.dtype:
                raise Unsupported("dtype-reinterpreting output view")
        else:
            raise Unsupported("fresh output storage has no full contiguous visible owner")
        seen[id(y)] = i
        bindings.append(
            _Binding(source, base, tuple(y.shape), tuple(y.stride()), y.storage_offset())
        )
    return spec, bindings


class ResidentExecutable:
    """Obtained from capture_export(). Use/close on the creating thread and CUDA stream."""

    def __init__(self, dev, owner, fn, static, outputs, recipe, graph, layout, metadata):
        self._dev = dev
        self._owner = owner
        self._thread = threading.current_thread()
        self._fn = fn
        self._static = static
        self._outputs = tree.tree_leaves(outputs)
        self._spec, self._bindings = recipe
        self._graph = graph
        self._layout = layout
        self.metadata = metadata
        self._failed = False

    def _guard(self):
        if threading.current_thread() is not self._thread:
            raise RuntimeError("resident executable belongs to a different thread")
        if (
            torch.cuda.current_device() != self._dev.index
            or torch.cuda.current_stream(self._dev) != self._owner
        ):
            raise RuntimeError("resident executable belongs to a different CUDA device or stream")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("nested CUDA graph capture is unsupported")

    @torch.inference_mode()
    def __call__(self, *args):
        self._guard()
        if self._graph is None or self._failed:
            raise RuntimeError("closed or failed resident executable")
        if torch.is_autocast_enabled("cuda") or _request(args, self._dev) != self._layout:
            raise ValueError("resident request layout or precision mode changed")
        try:
            for src, dst in zip(args, self._static, strict=True):
                src.record_stream(self._owner)
                dst.copy_(src)
            self._graph.replay()
            fresh = {
                b.base: self._outputs[b.base].clone() for b in self._bindings if b.source == "copy"
            }
            values = []
            for b in self._bindings:
                if b.source == "input":
                    v = args[b.base]
                elif b.source == "copy":
                    v = fresh[b.base]
                elif b.source == "repeat":
                    v = values[b.base]
                else:
                    base = args[b.base] if b.source == "input_view" else fresh[b.base]
                    v = base.as_strided(b.shape, b.stride, b.offset)
                values.append(v)
            return tree.tree_unflatten(values, self._spec)
        except Exception:
            self._failed = True
            raise

    def __copy__(self):
        raise TypeError("resident handles cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("resident handles cannot be copied")

    def __del__(self):
        # Explicit scoped close is preferred. If the last Python owner drops the handle
        # (even on another thread), retain capture bindings until queued work finishes.
        try:
            if getattr(self, "_graph", None) is not None:
                self._owner.synchronize()
        except Exception:
            pass  # Interpreter shutdown / CUDA context teardown.

    def close(self):
        self._guard()
        if self._graph is not None:
            self._owner.synchronize()
            self._graph.reset()
            self._graph = None
            self._outputs = []
            self._static = ()
            self._fn = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@torch.inference_mode()
def capture_export(
    ep,
    example,
    audit,
    *,
    backend="mixed",
    partition_options=None,
    max_static_bytes=1024**3,
    int8_rhs_layout=None,
):
    """Capture a deterministic ATen export or its SparkFuse partition, with a private weight snapshot.

    Validate two content-distinct requests against the original export before returning. Limit covers
    staging inputs and copied model state, not graph-private intermediates, external workspace or outputs.
    Caller provides stream dependencies for inputs, and excludes concurrent process-wide capture/setup.
    int8_rhs_layout="preserve" or "column_major" admits a signed W8A8 projection with direct,
    frozen full-storage RHS buffers without changing the GEMM. "packed_fanout" instead
    combines 2–4 frozen weights into one library GEMM and materializes live compact branch outputs.
    """
    return _capture_export(
        ep,
        example,
        audit,
        backend,
        partition_options,
        max_static_bytes,
        None,
        int8_rhs_layout=int8_rhs_layout,
    )


@torch.inference_mode()
def _capture_export(
    ep,
    example,
    audit,
    backend,
    partition_options,
    max_static_bytes,
    shared,
    *,
    int8_rhs_layout=None,
):
    if backend not in ("aten", "mixed") or (backend == "aten" and partition_options):
        raise ValueError("choose aten without partition options, or mixed")
    if type(max_static_bytes) is not int or max_static_bytes < 1:
        raise ValueError("positive byte limit required")
    if int8_rhs_layout not in (None, "preserve", "column_major", "packed_fanout") or (
        shared is not None and int8_rhs_layout is not None
    ):
        raise ValueError(
            "choose a single-export W8A8 snapshot layout: preserve, column_major, or packed_fanout"
        )
    fanout = int8_rhs_layout == "packed_fanout"
    if fanout and (
        backend != "mixed"
        or not all(
            (partition_options or {}).get(k) is True for k in ("token_quant", "group_epilogues")
        )
    ):
        raise ValueError("packed_fanout requires mixed token_quant and group_epilogues")
    quantized = int8_rhs_layout is not None
    _validate_export(ep, quantized=quantized)
    if (
        not isinstance(example, tuple)
        or not example
        or not isinstance(example[0], torch.Tensor)
        or not example[0].is_cuda
    ):
        raise Unsupported("CUDA request expected")
    dev = example[0].device
    if (
        torch.cuda.current_device() != dev.index
        or torch.cuda.is_current_stream_capturing()
        or torch.is_autocast_enabled("cuda")
    ):
        raise Unsupported("setup requires owning device, no nested capture or autocast")
    layout = _request(example, dev)
    if _request(audit, dev) != layout:
        raise ValueError("example and audit layout mismatch")
    if not _CAPTURE.acquire(blocking=False):
        raise RuntimeError("another resident setup is in progress")
    try:
        if all(
            torch.equal(x.reshape(-1).view(torch.uint8), y.reshape(-1).view(torch.uint8))
            for x, y in zip(example, audit, strict=True)
        ):
            raise ValueError("a content-distinct audit request is required")
        raw = ep.module()
        if fanout:
            from .packed_projection import preflight, prepare

            source_keys, _, source_size = preflight(raw, dev, example, audit, max_static_bytes)
        elif quantized:
            # Reject malformed or aliased source state before even the reference BLAS call.
            # Check again on the partitioned/snapshotted graph, whose users may differ.
            from .quant_state import frozen_weights

            frozen_weights(raw.graph, raw, dev, int8_rhs_layout, _get, _store, _geometry)
        reference = raw(*example)  # Check the export input contract before running generated code.
        gm, recs = (
            partition_export(ep, device=dev, **dict(partition_options or {}))
            if backend == "mixed"
            else (raw, [])
        )
        adopted = None
        if fanout:
            gm, recs, adopted, keys, size = prepare(gm, recs, dev, example, audit, max_static_bytes)
            if keys != source_keys or size != source_size:
                raise Unsupported("mixed packed fanout changed live model state")
        fn, original_state, private_state, used, inventory = _snapshot(
            gm,
            dev,
            example,
            max_static_bytes,
            shared,
            "column_major" if fanout else int8_rhs_layout,
            adopted,
        )
        if fanout:
            original_state |= source_keys
            if used != source_size:
                raise Unsupported("packed snapshot changed staging and state accounting")
        if any(_store(x) in original_state for x in audit):
            raise Unsupported("audit request aliases model state")
        static = tuple(x.clone() for x in example)
        owner = torch.cuda.current_stream(dev)
        side = torch.cuda.Stream(device=dev)
        side.wait_stream(owner)
        for t in (*static, *fn.buffers()):
            t.record_stream(side)
        with torch.cuda.stream(side):
            for _ in range(3):
                warm = fn(*static)
        owner.wait_stream(side)
        for t in tree.tree_leaves(warm):
            t.record_stream(owner)
        recipe = _recipe(example, reference, original_state, int8=quantized)
        if not any(b.source == "copy" for b in recipe[1]):
            raise Unsupported("alias-only outputs have no resident CUDA work to serve")
        _exact(warm, reference, "warm result")
        if _recipe(static, warm, private_state, int8=quantized) != recipe:
            raise Unsupported("snapshot changed output alias contract")
        del warm, reference
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side):
            outputs = fn(*static)
        owner.wait_stream(side)
        for t in tree.tree_leaves(outputs):
            t.record_stream(owner)
        if _recipe(static, outputs, private_state, int8=quantized) != recipe:
            raise Unsupported("capture changed output alias contract")
        exe = ResidentExecutable(
            dev,
            owner,
            fn,
            static,
            outputs,
            recipe,
            graph,
            layout,
            dict(
                backend=backend,
                staging_and_state_bytes=used,
                int8_rhs_layout=int8_rhs_layout,
                frozen_int8_weights=inventory,
                input_bytes=sum(t.nbytes for t in static),
                fresh_output_bytes=sum(
                    t.nbytes
                    for t, b in zip(tree.tree_leaves(outputs), recipe[1], strict=True)
                    if b.source == "copy"
                ),
                output_bindings=[asdict(b) for b in recipe[1]],
                regions=recs,
            ),
        )
        try:
            for tag, args in [("example", example), ("audit", audit)]:
                want = raw(*args)
                got = exe(*args)
                _exact(got, want, "resident " + tag)
                if _recipe(args, got, original_state, int8=quantized) != _recipe(
                    args, want, original_state, int8=quantized
                ):
                    raise Unsupported("replay changed public aliases")
        except Exception:
            exe.close()
            raise
        return exe
    finally:
        _CAPTURE.release()
