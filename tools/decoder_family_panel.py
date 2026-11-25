"""Compare real Qwen2/Llama/Mistral decode layers with and without a byte-qualified RoPE pair."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import transformers
from examples.decoder_families import suite
from examples.qwen_sdpa_step import trajectory
from sparkfuse.hybrid_decode import partition_hybrid_decode
from sparkfuse.symbolic_sites import partition_symbolic_decode
from symbolic_qwen_panel import walk
from hybrid_qwen_panel import trace
from length_decode_panel import counted_inductor
from decode_sdpa_panel import cpu_reference, metrics
from qwen_sdpa_sites import compare, carry
from ordered_score_panel import binary
from paired_timing import measure
from serving_clocks import build, clocks
from provenance import capture
from token_quant_panel import digest, raw


SEEDS = (3734911, 3745561)
PAST = (2, 15, 30, 31, 125, 255, 513, 2045)


def state_gate(fn, trajectories):
    rows = []
    live = []
    for tr in trajectories:
        ys = walk(fn, [xs for xs, _ in tr])
        live.extend(x for out in ys for x in out)
        rows.append(
            [
                compare(y, ref, xs if j == 0 else carry(xs, ys[j - 1][1:]))
                for j, ((xs, ref), y) in enumerate(zip(tr, ys, strict=True))
            ]
        )
    unique = len({x.untyped_storage().data_ptr() for x in live}) == len(live)
    flat = [z for row in rows for z in row]
    eligible = unique and all(
        z["within_tolerance"] and z["prefix_bitwise"] and z["geometry"] and z["fresh"] for z in flat
    )
    return dict(
        steps=rows,
        all_live=unique,
        admitted=eligible,
        exact_caches=eligible and all(z["exact_cache"] for z in flat),
    )


def case(family, seed, masked, smoke):
    m, rope = suite(family, seed)
    weights = {k: digest(x) for k, x in m.state_dict().items()}
    ex, au = [trajectory(m, rope, p, masked, seed + 11 * i)[0][0] for i, p in enumerate((17, 65))]
    past = torch.export.Dim("past", min=2, max=2048)
    ds = ({}, {}, {}, {2: past}, {2: past}) + (({3: past + 1},) if masked else ())
    ep = torch.export.export(m, ex, dynamic_shapes=ds, strict=True)
    graph = str(ep.graph)
    aten = ep.module()
    paths = {"hf_eager": m, "exported_aten": aten}
    meta = {}
    binaries = {}

    def register(pg):
        k = sha256(pg.path.with_suffix(".so").read_bytes()).hexdigest()
        if k not in binaries:
            binaries[k] = binary(pg)
        return k

    (a, setup) = build(lambda: partition_symbolic_decode(ep, ex, au, domain="bounded"))
    (rec,) = a.sites
    paths["attention_only"] = a.module
    attn = a.module.get_submodule(rec["module"]).op
    meta["attention_only"] = dict(
        setup=setup,
        binaries=[register(SimpleNamespace(path=p, source=p.read_text())) for p in attn.paths],
    )
    pair = None
    for key, flag in [("hybrid", False), ("hybrid_rope", True)]:
        h, setup = build(
            lambda: partition_hybrid_decode(
                ep, ex, au, domain="bounded", host_calls=True, paired_rotary=flag
            )
        )
        assert h.attention == a.sites
        paths[key] = h.module
        bs = {
            r["module"]: register(h.module.get_submodule(r["module"]).kernel)
            for r in h.regions
            if "module" in r
        }
        if flag:
            (pair,) = h.rotary
            assert "module" in pair
            bs[pair["module"]] = register(h.module.get_submodule(pair["module"]).program)
        meta[key] = dict(
            setup=setup, regions=h.regions, rotary=h.rotary, host_calls=h.host_calls, binaries=bs
        )
    graphs = []
    fails = {}
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_dynamic"], graphs = counted_inductor(aten, True)
    sizes = PAST[:2] if smoke else PAST
    tr = [
        trajectory(m, rope, s, masked, seed + 1117 + 113 * i, steps=4) for i, s in enumerate(sizes)
    ]
    requests = [(tuple(xs for xs, _ in t),) for t in tr]

    def fingerprints():
        return [[[digest(x) for x in xs] for xs, _ in tt] for tt in tr]

    hashes = fingerprints()
    audits = []
    final = paths["hybrid_rope"]
    (pair_node,) = [n for n in final.graph.nodes if n.target == pair["module"]]
    (sdpa_node,) = [n for n in final.graph.nodes if n.target == rec["module"]]
    for tt in tr:
        checks = []
        for xs, _ in tt:
            _, base = trace(aten, xs)
            _, env = trace(final, xs)
            out = env[pair_node.name]
            rot = []
            for y, k, layout in zip(out, pair["live_outs"], pair["physical_order"], strict=True):
                if layout == "BSHD":
                    y = y.transpose(1, 2)
                ref = base[k]
                row = dict(
                    sha256=digest(y),
                    ref_sha256=digest(ref),
                    different_bytes=int((raw(y) != raw(ref)).sum()),
                    stride_match=y.stride() == ref.stride(),
                    shape=list(y.shape),
                )
                rot.append(row)
            operands = tuple(env[n.name] for n in sdpa_node.args)
            ref = cpu_reference(operands, attn.plan.scale)
            z = metrics(
                env[sdpa_node.name],
                ref,
                base[rec["name"]].clone(memory_format=torch.contiguous_format),
                operands,
            )
            assert z["structure"] and z["outside_tolerance"] == 0
            checks.append(
                dict(sequence=xs[3].shape[2] + 1, rotary=rot, fp64_sha256=digest(ref), attention=z)
            )
        audits.append(checks)
    gates = {}
    for k, f in paths.items():
        try:
            gates[k] = state_gate(f, tr)
        except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as e:
            assert k == "inductor_dynamic"
            fails[k] = type(e).__name__
    assert all(gates[k]["exact_caches"] for k in ("hf_eager", "exported_aten", "attention_only"))
    warm = len(graphs)
    timers = {
        k: (lambda stages, f=paths[k]: walk(f, stages)) for k, g in gates.items() if g["admitted"]
    }
    hot = {}
    mixed = None
    if not smoke:
        for i in (0, 7):
            req = requests[i]
            hot[str(sizes[i])] = measure(
                {k: (lambda f=f: f(*req)) for k, f in timers.items()}, seed + sizes[i] + 301
            )
        mixed = clocks(timers, requests, seed + 3911)
    assert (
        len(graphs) == warm
        and graph == str(ep.graph)
        and weights == {k: digest(x) for k, x in m.state_dict().items()}
        and hashes == fingerprints()
    )
    return dict(
        family=family,
        layer_type=type(m.layer).__name__,
        seed=seed,
        masked=masked,
        past=sizes,
        qualification=[17, 65],
        steps=4,
        export_graph=graph,
        weights=weights,
        inputs=hashes,
        site=rec,
        variants=meta,
        binaries=binaries,
        held_out=audits,
        gates=gates,
        build_failures=fails,
        inductor_backend_calls=warm,
        hot=hot,
        rotating=mixed,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    r = dict(
        smoke=a.smoke,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    with torch._dynamo.config.patch(recompile_limit=32):
        for f in ("llama", "mistral") if a.smoke else ("qwen2", "llama", "mistral"):
            for mask in (True,) if a.smoke else (True, False):
                for seed in (3767717,) if a.smoke else SEEDS:
                    c = case(f, seed, mask, a.smoke)
                    r["cases"].append(c)
                    print(
                        f,
                        mask,
                        seed,
                        {k: g["exact_caches"] for k, g in c["gates"].items()},
                        flush=True,
                    )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
