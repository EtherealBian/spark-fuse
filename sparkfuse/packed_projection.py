"""CUDA for owned outputs of one horizontally packed signed integer matrix product."""

import copy
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


def preflight(gm, dev, example, audit, cap):
    """Refuse aliasing, nonowned state, unbounded or tied matrices before packing or raw BLAS."""
    from .resident import _geometry, _get, _plain, _store, _whole

    bindings, inv = frozen_weights(gm.graph, gm, dev, "column_major", _get, _store, _geometry)
    if not 2 <= len(inv) <= 4 or len({v["storage_group"] for v in inv}) != len(inv):
        raise Unsupported("packed fanout requires two to four distinct, untied frozen RHS matrices")
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


def prepare(gm, records, dev, example, audit, cap, *, reuse=None):
    """Rewrite a *separate* graph, adopting one private final-layout weight, materialized once per family."""
    from .resident import _get

    stores, inventory, size = preflight(gm, dev, example, audit, cap)
    private = fx.GraphModule(gm, copy.deepcopy(gm.graph))
    for key in ("_in_spec", "_out_spec"):
        if hasattr(gm, key):
            setattr(private, key, getattr(gm, key))
    graph = private.graph
    nodes = list(graph.nodes)
    position = {n: i for i, n in enumerate(nodes)}
    groups = [r for r in records if r.get("kind") == "grouped_int32_epilogue"]
    if len(groups) != 1:
        raise Unsupported("packed fanout requires exactly one grouped ordered epilogue")
    record = groups[0]
    grouped = next(n for n in nodes if n.op == "call_module" and n.target == record["module"])
    if type(private.get_submodule(grouped.target)) is not IndexRegion:
        raise Unsupported("unexpected grouped projection module")
    row, *tail = grouped.args
    accs, channels = tail[::2], tail[1::2]
    all_mm = {n for n in nodes if _is(n, torch.ops.aten._int_mm.default, 2)}
    if len(accs) != len(inventory) or len(accs) != len(set(accs)) or set(accs) != all_mm:
        raise Unsupported("group must consume all, and only, distinct signed GEMMs")
    lhs = accs[0].args[0]
    if any(a.args[0] is not lhs for a in accs):
        raise Unsupported("packed projections must share the same signed codes tensor")
    # Export inserts metadata-only assertions before casts. The generated region's native
    # descriptors enforce exactly these properties. Never erase size/stride or unknown guards.
    for a in accs:
        for u in list(a.users):
            if (
                u.op == "call_function"
                and u.target == torch.ops.aten._assert_tensor_metadata.default
                and u.args == (a,)
                and not u.users
                and u.kwargs == {"dtype": torch.int32, "device": dev, "layout": torch.strided}
            ):
                graph.erase_node(u)
    if any(position[u] < position[grouped] for a in accs for u in a.users if u is not grouped):
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
    by_name = {w["name"]: w for w in inventory}
    m, k = map(int, lhs.meta["val"].shape)
    j = 0
    for a, w, r in zip(accs, weights, record["branches"], strict=True):
        n = by_name[w.target]["shape"][1]
        if r["shape"] != [m, n]:
            raise Unsupported("frozen RHS and exported accumulator geometry disagree")
        roles = []
        if any(u is not grouped for u in a.users):
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
    weight_name = "_sparkfuse_packed_rhs"
    module_name = "_sparkfuse_packed_projection"
    if hasattr(gm, weight_name) or hasattr(gm, module_name):
        raise Unsupported("packed projection internal name is already bound")
    src = emit(branches)
    out_shapes = [(m, n) for _, n, rs in branches for _ in rs]
    types = ["i32" if role == "acc" else "f32" for _, _, rs in branches for role in rs]
    kernel = IndexProgram(
        [(m, 1), (m, columns), *(tuple(c.meta["val"].shape) for c in channels)],
        out_shapes,
        "f32",
        dev,
        src,
        input_dtypes=["f32", "i32"] + ["f32"] * len(channels),
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
        source = _get(gm, w.target).detach()
        if reuse is None:
            band.copy_(source)
        elif not torch.equal(band, source):
            # Never 'refresh' this buffer: preceding graphs retain its device address.
            raise Unsupported("shape family changed packed RHS contents: " + w.target)
    private.register_buffer(weight_name, packed)
    with graph.inserting_before(grouped):
        rhs = graph.get_attr(weight_name)
        mm = graph.call_function(torch.ops.aten._int_mm.default, (lhs, rhs))
        call = graph.call_module(module_name, (row, mm, *channels))
        for i, prev in enumerate(old):
            y = graph.call_function(operator.getitem, (call, i))
            y.meta = dict(prev.meta)
            for user in list(prev.users):
                if user is not grouped:
                    user.replace_input_with(prev, y)
    for n in (*items.values(), grouped, *accs, *weights):
        graph.erase_node(n)
    graph.lint()
    private.recompile()
    entry = dict(
        kind="packed_int32_projection",
        module=module_name,
        weight=weight_name,
        gemm=str(mm.target),
        shape=[m, k, columns],
        snapshot_stride=[1, k],
        branches=[dict(shape=[m, n], output_roles=roles) for _, n, roles in branches],
        source_weights=packed_inv,
        ops=[a.name for a in accs] + record["ops"],
        host_backend=record["host_backend"],
        cuda_sha256=sha256(src.encode()).hexdigest(),
        schedule=dict(threads=128),
    )
    return (
        private,
        [entry if r is record else r for r in records],
        {weight_name: packed},
        stores,
        size,
    )
