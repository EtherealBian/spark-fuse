"""CUDA for owned outputs of one horizontally packed signed integer matrix product."""

import copy
from dataclasses import dataclass
from hashlib import sha256
import operator

import torch
from torch import fx
from .index_program import IndexProgram, IndexRegion
from .ir import Unsupported
from .quant_state import frozen_weights
from .rotary import _is


def emit(branches):
    """Inputs: row[M,1], packed_acc[M,sum(widths)], separate channel scales. No public band views."""
    roleset = {"acc", "cast", "row_scaled", "output"}
    if not 2 <= len(branches) <= 4:
        raise ValueError("two to four branches")
    m = branches[0][0]
    total = sum(n for _, n, _ in branches)
    if not (type(m) is int and m > 0 and total * m <= 2**28):
        raise ValueError("static packed matrix shape")
    args = ["const float* row", "const int* packed"]
    call = ["(const float*)p[0]", "(const int*)p[1]"]
    outs = []
    body = []
    start = 0
    offset = 0
    for j, (rows, n, roles) in enumerate(branches):
        if (
            rows != m
            or type(n) is not int
            or n <= 0
            or not roles
            or len(set(roles)) != len(roles)
            or set(roles) - roleset
        ):
            raise ValueError("invalid packed projection branch")
        args.append(f"const float* c{j}")
        call.append(f"(const float*)p[{j + 2}]")
        blocks = (m * n + 127) // 128
        body.extend(
            [
                f"    if (blockIdx.x < {start + blocks}u) {{",
                f"        const int i=(int(blockIdx.x)-{start})*128+int(threadIdx.x);",
                f"        if (i >= {m * n}) return;",
                f"        const int r=i/{n}, c=i%{n};",
                f"        const int v=packed[r*{total}+{offset}+c];",
                "        const float a=__int2float_rn(v);",
                "        const float b=__fmul_rn(a,row[r]);",
            ]
        )
        expr = {"acc": "v", "cast": "a", "row_scaled": "b", "output": f"__fmul_rn(b,c{j}[c])"}
        for role in roles:
            dt = "int" if role == "acc" else "float"
            outs.append((dt, f"{role}{j}"))
            body.append(f"        {role}{j}[i]={expr[role]};")
        body += ["        return;", "    }"]
        start += blocks
        offset += n
    for dt, name in outs:
        call.append(f"({dt}*)p[{len(args)}]")
        args.append(f"{dt}* {name}")
    return (
        "#include <cuda_runtime.h>\n__global__ void packed_int32_fanout("
        + ",".join(args)
        + ") {\n"
        + "\n".join(body)
        + "\n}\n"
        + f'extern "C" int launch(void** p,int,void* stream) {{\n    packed_int32_fanout<<<{start},128,0,(cudaStream_t)stream>>>('
        + ",".join(call)
        + ");\n    return int(cudaGetLastError());\n}\n"
    )


def preflight(gm, dev, example, audit, cap, *, multiple=False):
    """Refuse aliasing, nonowned state, unbounded or tied matrices before packing or raw BLAS."""
    from .resident import _geometry, _get, _plain, _store, _whole

    bindings, inv = frozen_weights(gm.graph, gm, dev, "column_major", _get, _store, _geometry)
    if multiple and not 4 <= len(inv) <= 32:
        raise Unsupported("packed groups require 4..32 distinct frozen RHS matrices")
    if (not multiple and not 2 <= len(inv) <= 4) or len({v["storage_group"] for v in inv}) != len(
        inv
    ):
        raise Unsupported(
            "packed layout needs distinct, untied frozen RHS matrices"
            if multiple
            else "packed fanout requires two to four distinct, untied frozen RHS matrices"
        )
    stores = {}
    sizes = {}
    for node in gm.graph.nodes:
        if node.op != "get_attr" or not node.users:
            continue
        t = _get(gm, node.target)
        weight = node.target in bindings
        if not _plain(t, int8=weight) or t.device != dev or (not weight and not _whole(t)):
            raise Unsupported("packed capture requires whole, same-device model state")
        if weight and len(node.users) != 1:
            raise Unsupported("packed RHS must serve exactly one projection")
        if weight:
            mm = next(iter(node.users))
            k = mm.args[0].meta["val"].shape[1]
            n = mm.meta["val"].shape[1]
            if tuple(t.shape) != (k, n):
                raise Unsupported("source weight does not match certified signed GEMM shape")
        key = _store(t)
        if key in stores and stores[key] != _geometry(t):
            raise Unsupported("packed capture rejects incompatible tied state geometry")
        stores[key] = _geometry(t)
        sizes[key] = t.untyped_storage().nbytes()
    if any(_store(x) in stores for x in (*example, *audit)):
        raise Unsupported("request or audit aliases source state")
    bytes_needed = sum(x.nbytes for x in example) + sum(sizes.values())
    if bytes_needed > cap:
        raise ValueError("request staging + weight snapshot exceeds byte limit")
    return set(stores), inv, bytes_needed


@dataclass
class _Group:
    record: dict
    node: fx.Node
    accs: tuple
    weights: list
    guards: list
    items: dict
    old: list
    branches: list
    inventory: list
    shape: tuple
    weight_name: str
    module_name: str
    affine: bool


def _plan_group(private, grouped, record, position, by_name, dev, suffix):
    """No graph edits, device allocations or launches. Prove each group's own boundary."""
    if type(private.get_submodule(grouped.target)) is not IndexRegion:
        raise Unsupported("unexpected grouped projection module")
    row, *tail = grouped.args
    affine = record["kind"] == "grouped_int32_affine"
    arity = 3 if affine else 2
    accs = tail[::arity]
    if (
        not 2 <= len(accs) <= 4
        or len(set(accs)) != len(accs)
        or not all(_is(a, torch.ops.aten._int_mm.default, 2) for a in accs)
    ):
        raise Unsupported("group must consume two to four distinct signed GEMMs")
    lhs = accs[0].args[0]
    if any(a.args[0] is not lhs for a in accs):
        raise Unsupported("packed projections must share the same signed codes tensor")
    # Export inserts metadata-only assertions before casts. The generated region's native
    # descriptors enforce exactly these properties. Never erase size/stride or unknown guards.
    guards = []
    for a in accs:
        for u in list(a.users):
            if (
                u.op == "call_function"
                and u.target == torch.ops.aten._assert_tensor_metadata.default
                and u.args == (a,)
                and not u.users
                and u.kwargs == {"dtype": torch.int32, "device": dev, "layout": torch.strided}
            ):
                guards.append(u)
    if any(
        position[u] < position[grouped]
        for a in accs
        for u in a.users
        if u is not grouped and u not in guards
    ):
        raise Unsupported("early accumulator reader prevents packed projection placement")
    weights = [a.args[1] for a in accs]
    if any(w.op != "get_attr" or set(w.users) != {a} for w, a in zip(weights, accs, strict=True)):
        raise Unsupported("each packed RHS must be distinct direct frozen state")
    items = {}
    for u in grouped.users:
        if (
            u.op != "call_function"
            or u.target is not operator.getitem
            or type(u.args[1]) is not int
            or u.args[1] in items
        ):
            raise Unsupported("unknown grouped epilogue consumer")
        items[u.args[1]] = u
    branches = []
    old = []
    columns = 0
    packed_inv = []
    m, k = map(int, lhs.meta["val"].shape)
    j = 0
    for a, w, r in zip(accs, weights, record["branches"], strict=True):
        n = by_name[w.target]["shape"][1]
        if r["shape"] != [m, n]:
            raise Unsupported("frozen RHS and exported accumulator geometry disagree")
        roles = []
        if any(u is not grouped and u not in guards for u in a.users):
            roles.append("acc")
            old.append(a)
        for role in r["output_roles"]:
            if j not in items:
                raise Unsupported("missing grouped output")
            roles.append(role)
            old.append(items[j])
            j += 1
        branches.append((m, n, roles))
        packed_inv.append({**by_name[w.target], "packed_column_offset": columns})
        columns += n
    if set(items) != set(range(j)) or max(m * columns, k * columns) > 2**28:
        raise Unsupported("unsupported packed output arity or matrix extent")
    weight_name = "_sparkfuse_packed_rhs" + suffix
    module_name = "_sparkfuse_packed_projection" + suffix
    if hasattr(private, weight_name) or hasattr(private, module_name):
        raise Unsupported("packed projection internal name is already bound")
    return _Group(
        record,
        grouped,
        tuple(accs),
        weights,
        guards,
        items,
        old,
        branches,
        packed_inv,
        (m, k, columns),
        weight_name,
        module_name,
        affine,
    )


def _apply_group(private, plan, dev, stores, reuse):
    from .resident import _get

    graph = private.graph
    record, grouped = plan.record, plan.node
    accs, weights = plan.accs, plan.weights
    packed_inv, branches, affine = plan.inventory, plan.branches, plan.affine
    m, k, columns = plan.shape
    weight_name, module_name = plan.weight_name, plan.module_name
    # An earlier rewrite can have replaced inputs. Read them from the live graph.
    lhs = accs[0].args[0]
    row, *tail = grouped.args
    side_inputs = [v for i, v in enumerate(tail) if i % (3 if affine else 2)]
    for guard in plan.guards:
        graph.erase_node(guard)
    if affine:
        from .projection_affine import emit as emit_affine

        dts = [r["narrow_dtype"] for r in record["branches"]]
        src = emit_affine([(*b, dt) for b, dt in zip(branches, dts, strict=True)], packed=True)
    else:
        dts = [None] * len(branches)
        src = emit(branches)
    out_shapes = [(m, n) for _, n, rs in branches for _ in rs]
    types = [
        "i32" if role == "acc" else (dt if role == "narrow" else "f32")
        for (_, _, rs), dt in zip(branches, dts, strict=True)
        for role in rs
    ]
    kernel = IndexProgram(
        [(m, 1), (m, columns), *(tuple(c.meta["val"].shape) for c in side_inputs)],
        out_shapes,
        "f32",
        dev,
        src,
        input_dtypes=["f32", "i32"] + ["f32"] * len(side_inputs),
        output_dtypes=types,
    )
    private.add_module(module_name, IndexRegion(kernel, record["host_backend"]))
    if reuse is None:
        packed = torch.empty_strided((k, columns), (1, k), dtype=torch.int8, device=dev)
    else:
        previous, packed = reuse
        # A shape-specialized export can change branch order even when widths and bytes match.
        # Validate logical source identity and layout BEFORE accepting any shared address.
        if previous != packed_inv:
            raise Unsupported("shape family changed packed RHS descriptors or branch order")
        if (
            type(packed) is not torch.Tensor
            or packed.device != dev
            or packed.dtype != torch.int8
            or packed.shape != (k, columns)
            or packed.stride() != (1, k)
            or packed._base is not None
            or packed.storage_offset()
            or packed.untyped_storage().nbytes() != k * columns
            or (packed.device, packed.untyped_storage().data_ptr()) in stores
        ):
            raise Unsupported("invalid private packed RHS for shape family")
    for w, desc in zip(weights, packed_inv, strict=True):
        band = packed.narrow(1, desc["packed_column_offset"], desc["shape"][1])
        source = _get(private, w.target).detach()
        if reuse is None:
            band.copy_(source)
        elif not torch.equal(band, source):
            # Never 'refresh' this buffer: preceding graphs retain its device address.
            raise Unsupported("shape family changed packed RHS contents: " + w.target)
    private.register_buffer(weight_name, packed)
    with graph.inserting_before(grouped):
        rhs = graph.get_attr(weight_name)
        mm = graph.call_function(torch.ops.aten._int_mm.default, (lhs, rhs))
        call = graph.call_module(module_name, (row, mm, *side_inputs))
        for i, prev in enumerate(plan.old):
            y = graph.call_function(operator.getitem, (call, i))
            y.meta = dict(prev.meta)
            for user in list(prev.users):
                if user is not grouped:
                    user.replace_input_with(prev, y)
    for n in (*plan.items.values(), grouped, *accs, *weights):
        graph.erase_node(n)
    entry = dict(
        kind="packed_int32_projection",
        module=module_name,
        weight=weight_name,
        gemm=str(mm.target),
        shape=[m, k, columns],
        snapshot_stride=[1, k],
        branches=[
            {**dict(shape=[m, n], output_roles=roles), **({"narrow_dtype": dt} if affine else {})}
            for (_, n, roles), dt in zip(branches, dts, strict=True)
        ],
        source_weights=packed_inv,
        ops=[a.name for a in accs] + record["ops"],
        host_backend=record["host_backend"],
        cuda_sha256=sha256(src.encode()).hexdigest(),
        schedule=dict(threads=128),
    )
    return entry, packed


def prepare(gm, records, dev, example, audit, cap, *, reuse=None, multiple=False):
    """Privately pack each certified group once. No partial or best-effort rewrites."""
    if multiple and reuse is not None:
        raise ValueError("shape-family reuse is not implemented for packed groups")
    stores, inventory, size = preflight(gm, dev, example, audit, cap, multiple=multiple)
    private = fx.GraphModule(gm, copy.deepcopy(gm.graph))
    for key in ("_in_spec", "_out_spec"):
        if hasattr(gm, key):
            setattr(private, key, getattr(gm, key))
    nodes = list(private.graph.nodes)
    position = {n: i for i, n in enumerate(nodes)}
    signed_mm = {n for n in nodes if _is(n, torch.ops.aten._int_mm.default, 2)}
    groups = [
        r for r in records if r.get("kind") in ("grouped_int32_epilogue", "grouped_int32_affine")
    ]
    if multiple and not 2 <= len(groups) <= 8:
        raise Unsupported("packed groups require two to eight grouped ordered epilogues")
    if not multiple and len(groups) != 1:
        raise Unsupported("packed fanout requires exactly one grouped ordered epilogue")
    table = {r["module"]: r for r in groups}
    calls = [n for n in nodes if n.op == "call_module" and n.target in table]
    if (
        len(table) != len(groups)
        or len(calls) != len(groups)
        or len({n.target for n in calls}) != len(calls)
    ):
        raise Unsupported("ambiguous grouped epilogue binding")
    by_name = {w["name"]: w for w in inventory}
    plans = [
        _plan_group(
            private, n, table[n.target], position, by_name, dev, f"_{i}" if multiple else ""
        )
        for i, n in enumerate(calls)
    ]
    accs = [a for plan in plans for a in plan.accs]
    if len(accs) != len(inventory) or len(accs) != len(set(accs)) or set(accs) != signed_mm:
        raise Unsupported("group must consume all, and only, distinct signed GEMMs")
    if any(hasattr(gm, n) for p in plans for n in (p.weight_name, p.module_name)):
        raise Unsupported("packed projection internal name is already bound")
    replaced, adopted = {}, {}
    for plan in plans:
        entry, packed = _apply_group(private, plan, dev, stores, reuse)
        replaced[plan.node.target] = entry
        adopted[plan.weight_name] = packed
    private.graph.lint()
    private.recompile()
    return private, [replaced.get(r.get("module"), r) for r in records], adopted, stores, size
