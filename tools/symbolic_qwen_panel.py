"""One genuine strict export, one CUDA pair across four steps and eight initial KV lengths."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import fx
import transformers
from examples.qwen_sdpa_step import suite, trajectory
from sparkfuse.symbolic_sites import partition_symbolic_decode
from length_decode_panel import counted_inductor
from decode_sdpa_panel import cpu_reference, metrics
from qwen_sdpa_sites import compare, carry
from ordered_score_panel import binary
from paired_timing import measure
from serving_clocks import clocks
from provenance import capture
from token_quant_panel import digest


SEEDS = (3511217, 3522353)
PAST = (2, 29, 30, 31, 125, 253, 509, 2045)


def walk(f, stages):
    rows = []
    for xs in stages:
        args = carry(xs, rows[-1][1:]) if rows else xs
        rows.append(f(*args))
    return tuple(rows)


def case(seed, mask, smoke):
    m, rope = suite(seed)
    weights = {k: digest(x) for k, x in m.state_dict().items()}
    ex, au = [trajectory(m, rope, p, mask, seed + 17 * i)[0][0] for i, p in enumerate((17, 65))]
    past = torch.export.Dim("past", min=2, max=2048)
    dims = ({}, {}, {}, {2: past}, {2: past}) + (({3: past + 1},) if mask else ())
    ep = torch.export.export(m, ex, dynamic_shapes=dims, strict=True)
    graph = str(ep.graph)
    aten = ep.module()
    paths = {"hf_eager": m, "exported_aten": aten}
    parts = {}
    bins = {}
    records = {}
    for host in ("native", "ctypes"):
        p = partition_symbolic_decode(ep, ex, au, host_backend=host, domain="bounded")
        assert len(p.sites) == 1 and not p.skipped, p.skipped
        (rec,) = p.sites
        assert (
            len(rec["erased_repeats"]) == 2
            and rec["plan"]["lower"] == 3
            and rec["plan"]["upper"] == 2049
        )
        records[host] = rec
        parts[host] = p.module.get_submodule(rec["module"]).op
        paths[host] = p.module
        bins[host] = [
            binary(SimpleNamespace(path=x, source=x.read_text())) for x in parts[host].paths
        ]
    assert bins["native"] == bins["ctypes"]
    graphs = []
    fail = {}
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_dynamic"], graphs = counted_inductor(aten, True)
    sizes = PAST[:2] if smoke else PAST
    tr = [trajectory(m, rope, s, mask, seed + 1019 + 317 * i, steps=4) for i, s in enumerate(sizes)]
    requests = [(tuple(args for args, expected in t),) for t in tr]
    out_nodes = {n.name: n for n in aten.graph.nodes}
    rec = records["native"]
    gate_rows = []

    def input_hashes():
        return [[[digest(x) for x in xs] for xs, _ in t] for t in tr]

    hashes = input_hashes()
    work = []
    for t in tr:
        boundary = []
        ws = []
        for xs, _ in t:
            inter = fx.Interpreter(aten, garbage_collect_values=False)
            inter.run(*xs)
            env = inter.env
            args = tuple(env[out_nodes[name]] for name in rec["operands"])
            o = env[out_nodes[rec["name"]]].clone(memory_format=torch.contiguous_format)
            f = parts["native"]
            r = cpu_reference(args, f.plan.scale)
            y = f.run_bounded(*args)
            z = metrics(y, r, o, args)
            assert z["structure"] and z["outside_tolerance"] == 0
            assert z["sha256"] == digest(parts["ctypes"].run_bounded(*args))
            s = args[1].shape[2]
            ws.append(f.plan.workspace_bytes(s))
            boundary.append(
                dict(
                    sequence=s,
                    reference_sha256=digest(r),
                    cpu_error=z,
                    kv_pitch=list(args[1].stride()),
                    incoming_cache_pitch=list(xs[3].stride()),
                )
            )
        gate_rows.append(boundary)
        work.append(ws)
    gates = {}
    for key, fn in paths.items():
        try:
            all_out = [walk(fn, req[0]) for req in requests]
        except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as exc:
            assert key == "inductor_dynamic"
            fail[key] = type(exc).__name__
            continue
        got = []
        live = (
            len({y.untyped_storage().data_ptr() for row in all_out for stage in row for y in stage})
            == len(tr) * 4 * 3
        )
        for t, outputs in zip(tr, all_out, strict=True):
            zs = []
            for j, ((xs, truth), y) in enumerate(zip(t, outputs, strict=True)):
                args = xs if j == 0 else carry(xs, outputs[j - 1][1:])
                zs.append(compare(y, truth, args))
            got.append(zs)
        all_z = [z for row in got for z in row]
        ok = live and all(
            z["within_tolerance"] and z["geometry"] and z["fresh"] and z["prefix_bitwise"]
            for z in all_z
        )
        exact = ok and all(z["exact_cache"] for z in all_z)
        gates[key] = dict(admitted=ok, all_live=live, exact_caches=exact, steps=got)
        assert key == "inductor_dynamic" or exact, (key, got)
    assert gates["native"] == gates["ctypes"]
    warm_count = len(graphs)
    accepted = {
        k: (lambda stages, f=paths[k]: walk(f, stages)) for k, g in gates.items() if g["admitted"]
    }
    hot = {}
    mixed = None
    if not smoke:
        for i in (0, 4, 7):
            req = requests[i]
            hot[str(sizes[i])] = measure(
                {name: (lambda f=f: f(*req)) for name, f in accepted.items()}, seed + sizes[i] + 127
            )
        mixed = clocks(accepted, requests, seed + 3751)
    assert (
        warm_count == len(graphs)
        and graph == str(ep.graph)
        and hashes == input_hashes()
        and weights == {k: digest(x) for k, x in m.state_dict().items()}
    )
    return dict(
        seed=seed,
        masked=mask,
        qualification_past=[17, 65],
        initial_past=sizes,
        steps=4,
        weight_sha256=weights,
        input_sha256=hashes,
        export_graph=graph,
        range_constraints=str(ep.range_constraints),
        sites=records,
        binary=bins,
        site_fp64=gate_rows,
        workspace_bytes=work,
        gates=gates,
        build_failures=fail,
        inductor_backend_calls=warm_count,
        hot=hot,
        rotating=mixed,
    )


@torch.inference_mode()
def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", required=True, type=Path)
    a.add_argument("--smoke", action="store_true")
    args = a.parse_args()
    torch.set_num_threads(4)
    r = dict(
        smoke=args.smoke,
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        provenance=capture(args.output.parent / "sources"),
        cases=[],
    )
    with torch._dynamo.config.patch(recompile_limit=32):
        for mask in (True,) if args.smoke else (True, False):
            for seed in (3541751,) if args.smoke else SEEDS:
                c = case(seed, mask, args.smoke)
                r["cases"].append(c)
                print(
                    mask,
                    seed,
                    c["inductor_backend_calls"],
                    c["rotating"]["median_us"] if c["rotating"] else "verified",
                    flush=True,
                )
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(args.output)


if __name__ == "__main__":
    main()
