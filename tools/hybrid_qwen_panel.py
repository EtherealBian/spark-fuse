"""Ablate byte-qualified static islands, legal splits and host calls in a symbolic Qwen2 layer."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import fx
import transformers
from examples.qwen_sdpa_step import suite, trajectory
from sparkfuse.hybrid_decode import partition_hybrid_decode
from sparkfuse.symbolic_sites import partition_symbolic_decode
from symbolic_qwen_panel import walk, PAST
from length_decode_panel import counted_inductor
from decode_sdpa_panel import cpu_reference, metrics
from qwen_sdpa_sites import compare, carry
from ordered_score_panel import binary
from paired_timing import measure
from serving_clocks import clocks
from provenance import capture
from token_quant_panel import digest, raw


SEEDS = (3613901, 3624569)


def trace(f, xs):
    t = fx.Interpreter(f, garbage_collect_values=False)
    out = t.run(*xs)
    return out, {n.name: v for n, v in t.env.items()}


def case(seed, masked, smoke):
    m, rope = suite(seed)
    weights = {k: digest(x) for k, x in m.state_dict().items()}
    ex, au = [trajectory(m, rope, p, masked, seed + 89 * i)[0][0] for i, p in enumerate((17, 65))]
    past = torch.export.Dim("past", min=2, max=2048)
    dims = ({}, {}, {}, {2: past}, {2: past}) + (({3: past + 1},) if masked else ())
    ep = torch.export.export(m, ex, dynamic_shapes=dims, strict=True)
    graph = str(ep.graph)
    aten = ep.module()
    base = partition_symbolic_decode(ep, ex, au, host_backend="native", domain="bounded")
    assert len(base.sites) == 1
    paths = {"hf_eager": m, "exported_aten": aten, "attention_only": base.module}
    variants = {}
    binary_table = {}

    def register(pg):
        key = sha256(pg.path.with_suffix(".so").read_bytes()).hexdigest()
        if key not in binary_table:
            binary_table[key] = binary(pg)
        return key

    for key, split, host in [
        ("islands", False, False),
        ("split", True, False),
        ("split_host", True, True),
    ]:
        h = partition_hybrid_decode(
            ep, ex, au, domain="bounded", split_barriers=split, host_calls=host
        )
        assert h.attention == base.sites
        bs = {}
        for rec in h.regions:
            if "module" in rec:
                bs[rec["module"]] = register(h.module.get_submodule(rec["module"]).kernel)
        variants[key] = dict(regions=h.regions, host_calls=h.host_calls, binaries=bs)
        paths[key] = h.module
    (site,) = base.sites
    op = base.module.get_submodule(site["module"]).op
    attn_keys = [register(SimpleNamespace(path=p, source=p.read_text())) for p in op.paths]
    graphs = []
    fails = {}
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_dynamic"], graphs = counted_inductor(aten, True)
    sizes = PAST[:2] if smoke else PAST
    trajectories = [
        trajectory(m, rope, s, masked, seed + 2011 + 73 * i, steps=4) for i, s in enumerate(sizes)
    ]
    requests = [(tuple(xs for xs, _ in t),) for t in trajectories]

    def fingerprint():
        return [[[digest(x) for x in xs] for xs, _ in t] for t in trajectories]

    inp = fingerprint()
    details = []
    for t in trajectories:
        steps = []
        for xs, _ in t:
            _, refenv = trace(base.module, xs)
            _, env = trace(paths["split"], xs)
            mod = paths["split"]
            (node,) = [n for n in mod.graph.nodes if n.target == site["module"]]
            args = tuple(env[n.name] for n in node.args)
            fp64 = cpu_reference(args, op.plan.scale)
            z = metrics(env[node.name], fp64, refenv[node.name], args)
            assert z["structure"] and z["outside_tolerance"] == 0
            outs = {}
            for reg in variants["split"]["regions"]:
                if "module" not in reg:
                    continue
                name = reg["module"]
                (call,) = [n for n in mod.graph.nodes if n.target == name]
                ys = env[call.name]
                assert len(ys) == len(reg["live_outs"])
                for key, y in zip(reg["live_outs"], ys, strict=True):
                    ref = refenv[key]
                    outs[key] = dict(
                        different_bytes=int((raw(y) != raw(ref)).sum()),
                        sha256=digest(y),
                        ref_sha256=digest(ref),
                        stride_match=y.stride() == ref.stride(),
                        shape=list(y.shape),
                    )
                    assert outs[key]["stride_match"]
            steps.append(
                dict(
                    sequence=xs[3].shape[2] + 1, outputs=outs, attention=z, fp64_sha256=digest(fp64)
                )
            )
        details.append(steps)
    gates = {}
    for key, f in paths.items():
        try:
            all_out = [walk(f, req[0]) for req in requests]
        except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as e:
            assert key == "inductor_dynamic"
            fails[key] = type(e).__name__
            continue
        rows = []
        live = (
            len({y.untyped_storage().data_ptr() for r in all_out for s in r for y in s})
            == len(trajectories) * 12
        )
        for r, ys in zip(trajectories, all_out, strict=True):
            rows.append(
                [
                    compare(y, ref, xs if j == 0 else carry(xs, ys[j - 1][1:]))
                    for j, ((xs, ref), y) in enumerate(zip(r, ys, strict=True))
                ]
            )
        flat = [z for r in rows for z in r]
        ok = live and all(
            z["within_tolerance"] and z["geometry"] and z["fresh"] and z["prefix_bitwise"]
            for z in flat
        )
        exact = ok and all(z["exact_cache"] for z in flat)
        gates[key] = dict(admitted=ok, exact_caches=exact, all_live=live, steps=rows)
        if key in ("hf_eager", "attention_only", "exported_aten"):
            assert exact
    assert gates["split"] == gates["split_host"]
    warm = len(graphs)
    accepted = {
        k: (lambda stages, f=paths[k]: walk(f, stages)) for k, v in gates.items() if v["admitted"]
    }
    hot = {}
    mixed = None
    if not smoke:
        for i in (0, 7):
            req = requests[i]
            hot[str(sizes[i])] = measure(
                {k: (lambda f=f: f(*req)) for k, f in accepted.items()}, seed + sizes[i] + 123
            )
        mixed = clocks(accepted, requests, seed + 701)
    assert (
        warm == len(graphs)
        and str(ep.graph) == graph
        and inp == fingerprint()
        and weights == {k: digest(x) for k, x in m.state_dict().items()}
    )
    return dict(
        seed=seed,
        masked=masked,
        export_graph=graph,
        qualification_past=[17, 65],
        initial_past=sizes,
        steps=4,
        weight_sha256=weights,
        input_sha256=inp,
        site=site,
        attention_binaries=attn_keys,
        binaries=binary_table,
        variants=variants,
        checked_island_outputs=details,
        gates=gates,
        build_failures=fails,
        inductor_backend_calls=warm,
        hot=hot,
        rotating=mixed,
    )


@torch.inference_mode()
def main():
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--smoke", action="store_true")
    o = a.parse_args()
    torch.set_num_threads(4)
    r = dict(
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        cuda=torch.version.cuda,
        smoke=o.smoke,
        provenance=capture(o.output.parent / "sources"),
        cases=[],
    )
    with torch._dynamo.config.patch(recompile_limit=32):
        for mask in (True,) if o.smoke else (True, False):
            for seed in (3643919,) if o.smoke else SEEDS:
                c = case(seed, mask, o.smoke)
                r["cases"].append(c)
                print(mask, seed, {k: g["exact_caches"] for k, g in c["gates"].items()}, flush=True)
    t = o.output.with_suffix(".tmp")
    t.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    t.replace(o.output)


if __name__ == "__main__":
    main()
