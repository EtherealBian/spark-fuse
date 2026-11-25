"""Horizontally fuse independent ordered dequantizations that read the same row scale."""

from hashlib import sha256
import operator

import torch
from .index_program import IndexProgram, IndexRegion
from .projection_epilogue import candidate


def _placement(sites, pos):
    """One legal FX instruction position; do not introduce a cycle or move past a live reader."""
    members = {v for _, nodes, _, _, _ in sites for v in nodes}
    if len(members) != sum(len(s[1]) for s in sites):
        return None
    inputs = {v for args, *_ in sites for v in args}
    if members & inputs:
        return None
    readers = {u for t in members for u in t.users if u not in members}
    if not readers:
        return None
    anchor = min((*readers, max((s[1][-1] for s in sites), key=pos.get)), key=pos.get)
    return anchor if max(pos[n] for n in inputs) < pos[anchor] else None


def _choose(gm, dev):
    pos = {n: i for i, n in enumerate(gm.graph.nodes)}
    by_row = {}
    for n in gm.graph.nodes:
        s = candidate(n, dev, pos)
        if s is not None:
            by_row.setdefault(s[0][1], []).append(s)
    for choices in by_row.values():
        for j, first in enumerate(choices[:-1]):
            group = [first]
            anchor = None
            for other in choices[j + 1 :]:
                at = _placement(group + [other], pos)
                if at is not None:
                    group.append(other)
                    anchor = at
                if len(group) == 4:
                    break
            if len(group) > 1:
                return group, anchor
    return None


def emit(branches):
    # Each block belongs to a single branch; each output has its own compact, caller-owned storage.
    params = ["const float* row"]
    casts = ["(const float*)p[0]"]
    body = []
    out = []
    start = 0
    for j, (m, n, roles) in enumerate(branches):
        params.extend([f"const int* a{j}", f"const float* c{j}"])
        casts.extend([f"(const int*)p[{2 * j + 1}]", f"(const float*)p[{2 * j + 2}]"])
        blocks = (m * n + 127) // 128
        body.extend(
            [
                f"    if (blockIdx.x < {start + blocks}u) {{",
                f"        const int i=(int(blockIdx.x)-{start})*128 + int(threadIdx.x);",
                f"        if (i >= {m * n}) return;",
                f"        const float a=__int2float_rn(a{j}[i]);",
                f"        const float r=__fmul_rn(a,row[i/{n}]);",
            ]
        )
        values = {"cast": "a", "row_scaled": "r", "output": f"__fmul_rn(r,c{j}[i%{n}])"}
        for name in roles:
            body.append(f"        {name}{j}[i]={values[name]};")
            out.append(f"float* {name}{j}")
        body.extend(["        return;", "    }"])
        start += blocks
    casts.extend([f"(float*)p[{len(params) + i}]" for i in range(len(out))])
    params.extend(out)
    return (
        "#include <cuda_runtime.h>\n__global__ void grouped_int32_epilogue("
        + ",".join(params)
        + ") {\n"
        + "\n".join(body)
        + "\n}\n"
        + f'extern "C" int launch(void** p,int,void* stream) {{\n    grouped_int32_epilogue<<<{start},128,0,(cudaStream_t)stream>>>('
        + ",".join(casts)
        + ");\n    return int(cudaGetLastError());\n}\n"
    )


def fuse_grouped_epilogues(gm, device, host_backend):
    dev = torch.device(device)
    if dev.type != "cuda":
        return []
    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    records = []
    while (choice := _choose(gm, dev)) is not None:
        sites, anchor = choice
        row = sites[0][0][1]
        shapes = [(sites[0][3][0], 1)]
        dtypes = ["f32"]
        ins = [row]
        outs = []
        branches = []
        live = []
        members = set()
        for (a, _, c), nodes, vis, (m, n), ch in sites:
            shapes.extend([(m, n), ch])
            dtypes.extend(["i32", "f32"])
            ins.extend([a, c])
            outs.extend([(m, n)] * len(vis))
            branches.append((m, n, [name for _, name in vis]))
            live.extend(t for t, _ in vis)
            members.update(nodes)
        src = emit(branches)
        program = IndexProgram(shapes, outs, "f32", dev, src, input_dtypes=dtypes)
        label = f"_sparkfuse_grouped_epilogue_{len(records)}"
        gm.add_module(label, IndexRegion(program, host_backend))
        with gm.graph.inserting_before(anchor):
            call = gm.graph.call_module(label, tuple(ins))
            for i, old in enumerate(live):
                y = gm.graph.call_function(operator.getitem, (call, i))
                y.meta = dict(old.meta)
                for user in list(old.users):
                    if user not in members:
                        user.replace_input_with(old, y)
        ordered = [n for n in gm.graph.nodes if n in members]
        for n in reversed(ordered):
            gm.graph.erase_node(n)
        records.append(
            dict(
                kind="grouped_int32_epilogue",
                module=label,
                branches=[dict(shape=[m, n], output_roles=roles) for m, n, roles in branches],
                ops=[n.name for n in ordered],
                live_ins=[n.name for n in ins],
                live_outs=[n.name for n in live],
                cuda_sha256=sha256(src.encode()).hexdigest(),
                host_backend=host_backend,
                schedule=dict(threads=128),
            )
        )
    if records:
        gm.graph.lint()
        gm.recompile()
    return records
