"""Ordered biased signed-projection tails with an explicit BF16/FP16 output boundary."""

from hashlib import sha256
import operator
import torch
from .group_epilogue import _placement
from .index_program import IndexProgram, IndexRegion
from .projection_epilogue import candidate
from .rotary import _is
from .token_quant import _dense

A = torch.ops.aten
TYPES = {torch.float16: "f16", torch.bfloat16: "bf16"}


def match(end, dev, pos):
    if not _is(end, A.to.dtype, 2) or end.args[1] not in TYPES or not end.users:
        return None
    bias_add = end.args[0]
    if not _is(bias_add, A.add.Tensor, 2):
        return None
    full, bias = bias_add.args
    site = candidate(full, dev, pos)
    if site is None:
        return None
    inputs, old, _, (m, n), channel_shape = site
    bv = getattr(bias, "meta", {}).get("val")
    bs = tuple(bv.shape) if isinstance(bv, torch.Tensor) else ()
    if (
        bs not in ((n,), (1, n))
        or not _dense(bias, bs, torch.float32, dev)
        or not _dense(bias_add, (m, n), torch.float32, dev)
        or not _dense(end, (m, n), end.args[1], dev)
    ):
        return None
    nodes = (*old, bias_add, end)
    members = set(nodes)
    if bias in members:
        return None
    guards = []
    for t in nodes[:-1]:
        for u in t.users:
            if u.op == "call_function" and u.target == A._assert_tensor_metadata.default:
                if (
                    u.args != (t,)
                    or u.users
                    or u.kwargs != {"dtype": torch.float32, "device": dev, "layout": torch.strided}
                ):
                    return None
                guards.append(u)
    members.update(guards)
    role_names = ("cast", "row_scaled", "output", "biased", "narrow")
    visible = [
        (t, role)
        for t, role in zip(nodes, role_names, strict=True)
        if any(u not in members for u in t.users)
    ]
    if not visible or visible[-1][0] is not end:
        return None
    # Keep end last: placement uses each branch's final value and all external consumers.
    return (
        (*inputs, bias),
        (*old, bias_add, *guards, end),
        visible,
        (m, n),
        (channel_shape, bs, TYPES[end.args[1]]),
    )


def choose(gm, dev):
    pos = {n: j for j, n in enumerate(gm.graph.nodes)}
    groups = {}
    for node in gm.graph.nodes:
        s = match(node, dev, pos)
        if s is not None:
            groups.setdefault(s[0][1], []).append(s)
    for sites in groups.values():
        for i, first in enumerate(sites):
            group = [first]
            anchor = _placement(group, pos)
            if anchor is None:
                continue
            for other in sites[i + 1 :]:
                at = _placement([*group, other], pos)
                if at is not None:
                    group.append(other)
                    anchor = at
                if len(group) == 4:
                    break
            return group, anchor
    return None


def emit(branches, *, packed=False):
    if not 1 <= len(branches) <= 4 or (packed and len(branches) < 2):
        raise ValueError("one to four affine branches; packing requires siblings")
    total = sum(n for _, n, _, _ in branches)
    m = branches[0][0]
    params = []
    casts = []
    out = []
    body = []
    start = offset = 0

    def arg(ty, name):
        casts.append(f"({ty}*)p[{len(params)}]")
        params.append(f"{ty}* {name}")

    arg("const float", "row")
    if packed:
        arg("const int", "packed")
    for j, (rows, n, roles, dt) in enumerate(branches):
        if (
            dt not in ("bf16", "f16")
            or not roles
            or len(set(roles)) != len(roles)
            or set(roles) - {"acc", "cast", "row_scaled", "output", "biased", "narrow"}
            or rows != m
            or min(m, n) < 1
            or m * total > 2**28
        ):
            raise ValueError("invalid affine projection shape or live role")
        if not packed:
            arg("const int", f"a{j}")
        arg("const float", f"c{j}")
        arg("const float", f"b{j}")
        blocks = (m * n + 127) // 128
        val = f"packed[(i/{n})*{total}+{offset}+i%{n}]" if packed else f"a{j}[i]"
        ty, cv = (
            ("__nv_bfloat16", "__float2bfloat16_rn")
            if dt == "bf16"
            else ("__half", "__float2half_rn")
        )
        body += [
            f"    if (blockIdx.x < {start + blocks}u) {{",
            f"        const int i=(int(blockIdx.x)-{start})*128+int(threadIdx.x);",
            f"        if (i >= {m * n}) return;",
            f"        const int v={val};",
            "        const float a=__int2float_rn(v);",
            f"        const float r=__fmul_rn(a,row[i/{n}]);",
            f"        const float o=__fmul_rn(r,c{j}[i%{n}]);",
            f"        const float b=__fadd_rn(o,b{j}[i%{n}]);",
        ]
        values = {
            "acc": ("int", "v"),
            "cast": ("float", "a"),
            "row_scaled": ("float", "r"),
            "output": ("float", "o"),
            "biased": ("float", "b"),
            "narrow": (ty, cv + "(b)"),
        }
        for role in roles:
            kind, expr = values[role]
            out.append((kind, f"{role}{j}"))
            body.append(f"        {role}{j}[i]={expr};")
        body += ["        return;", "    }"]
        start += blocks
        offset += n
    for a in out:
        arg(*a)
    kernel = "packed_affine_projection" if packed else "grouped_affine_projection"
    return (
        "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n"
        + f"__global__ void {kernel}("
        + ",".join(params)
        + ") {\n"
        + "\n".join(body)
        + "\n}\n"
        + f'extern "C" int launch(void** p,int,void* stream) {{\n    {kernel}<<<{start},128,0,(cudaStream_t)stream>>>('
        + ",".join(casts)
        + ");\n    return int(cudaGetLastError());\n}\n"
    )


def fuse_affine(gm, device, host_backend):
    dev = torch.device(device)
    if dev.type != "cuda":
        return []
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    records = []
    graph = gm.graph
    while (choice := choose(gm, dev)) is not None:
        sites, anchor = choice
        row = sites[0][0][1]
        ins = [row]
        ish = [(sites[0][3][0], 1)]
        idt = ["f32"]
        osh = []
        odt = []
        live = []
        branches = []
        members = set()
        for (acc, _, c, bias), nodes, vis, (m, n), (ch, bs, dt) in sites:
            ins.extend((acc, c, bias))
            ish.extend(((m, n), ch, bs))
            idt.extend(("i32", "f32", "f32"))
            osh.extend([(m, n)] * len(vis))
            odt.extend(dt if role == "narrow" else "f32" for _, role in vis)
            branches.append((m, n, [role for _, role in vis], dt))
            live.extend(v for v, _ in vis)
            members.update(nodes)
        src = emit(branches)
        p = IndexProgram(ish, osh, "f32", dev, src, input_dtypes=idt, output_dtypes=odt)
        label = f"_sparkfuse_affine_projection_{len(records)}"
        gm.add_module(label, IndexRegion(p, host_backend))
        with graph.inserting_before(anchor):
            call = graph.call_module(label, tuple(ins))
            for i, old in enumerate(live):
                y = graph.call_function(operator.getitem, (call, i))
                y.meta = dict(old.meta)
                for user in list(old.users):
                    if user not in members:
                        user.replace_input_with(old, y)
        ordered = [n for n in graph.nodes if n in members]
        for n in reversed(ordered):
            graph.erase_node(n)
        records.append(
            dict(
                kind="grouped_int32_affine",
                module=label,
                branches=[
                    dict(shape=[m, n], output_roles=rs, narrow_dtype=dt)
                    for m, n, rs, dt in branches
                ],
                ops=[t.name for t in ordered],
                live_ins=[v.name for v in ins],
                live_outs=[v.name for v in live],
                host_backend=host_backend,
                cuda_sha256=sha256(src.encode()).hexdigest(),
                schedule=dict(threads=128),
            )
        )
    if records:
        graph.lint()
        gm.recompile()
    return records
