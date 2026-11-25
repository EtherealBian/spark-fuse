"""Same-device ablation of one variable-length append instead of two ATen concatenations."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import torch
import transformers
from examples.decoder_families import suite
from examples.qwen_sdpa_step import trajectory
from sparkfuse.hybrid_decode import partition_hybrid_decode
from decoder_family_panel import state_gate
from symbolic_qwen_panel import walk
from hybrid_qwen_panel import trace
from length_decode_panel import counted_inductor
from token_quant_panel import digest
from inspect_reduction_sass import op_counts
import subprocess
from serving_clocks import clocks, build
from paired_timing import measure
from provenance import capture


PAST = (2, 15, 31, 125, 513, 2045, 4093, 8189)
SEEDS = (3823703, 3836107)


def one(family, seed, masked, smoke):
    m, rope = suite(family, seed)
    weights = {k: digest(t) for k, t in m.state_dict().items()}
    ex, au = [trajectory(m, rope, v, masked, seed + 7 * i)[0][0] for i, v in enumerate((17, 257))]
    past = torch.export.Dim("past", min=2, max=8192)
    ds = ({}, {}, {}, {2: past}, {2: past}) + (({3: past + 1},) if masked else ())
    ep = torch.export.export(m, ex, dynamic_shapes=ds, strict=True)
    graph = str(ep.graph)
    aten = ep.module()
    paths = {"hf_eager": m}
    setups = {}
    meta = {}
    for name, enabled in [("separate", False), ("joined", True)]:
        pg, setup = build(
            lambda: partition_hybrid_decode(
                ep,
                ex,
                au,
                domain="bounded",
                host_calls=True,
                paired_rotary=True,
                paired_append=enabled,
            )
        )
        paths[name] = pg.module
        setups[name] = setup
        meta[name] = dict(
            attention=pg.attention,
            rotary=pg.rotary,
            regions=pg.regions,
            host_calls=pg.host_calls,
            appends=pg.appends,
        )
    assert all(
        meta["joined"][k] == meta["separate"][k] for k in ("attention", "rotary", "host_calls")
    )
    assert [r.get("ops") for r in meta["joined"]["regions"]] == [
        r.get("ops") for r in meta["separate"]["regions"]
    ]
    (rec,) = pg.appends
    append = pg.module.get_submodule(rec["module"])
    (entry,) = [n for n in pg.module.graph.nodes if n.target == rec["module"]]
    so = append.path.with_suffix(".so")
    machine = dict(
        source_sha256=sha256(append.source.encode()).hexdigest(),
        binary_sha256=sha256(so.read_bytes()).hexdigest(),
        sass=op_counts(subprocess.check_output(["cuobjdump", "--dump-sass", str(so)], text=True)),
    )
    graphs = []
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_dynamic"], graphs = counted_inductor(aten, True)
    sizes = (31, 2045) if smoke else PAST
    tr = [
        trajectory(m, rope, s, masked, seed + 1711 + i * 271, steps=4) for i, s in enumerate(sizes)
    ]
    requests = [(tuple(xs for xs, _ in tt),) for tt in tr]
    hashes = lambda: [[[digest(x) for x in xs] for xs, _ in tt] for tt in tr]
    before = hashes()
    gates = {k: state_gate(fn, tr) for k, fn in paths.items()}
    assert all(gates[n]["exact_caches"] for n in ("hf_eager", "separate", "joined"))
    warm = len(graphs)
    timers = {
        k: (lambda stages, f=fn: walk(f, stages)) for k, fn in paths.items() if gates[k]["admitted"]
    }
    hot = {}
    mixed = {}
    site = []
    for ix in (0, len(sizes) - 1):
        steps = requests[ix][0]
        prev = None
        for j in range(2):
            xs = steps[j] if prev is None else (*steps[j][:3], *prev[1:], *steps[j][5:])
            prev, env = trace(pg.module, xs)
            args = tuple(env[n.name] for n in entry.args)
            yy = env[entry.name]
            ref = lambda: (torch.cat(args[:2], 2), torch.cat(args[2:], 2))
            gold = ref()
            assert all(
                digest(y) == digest(z) and y.stride() == z.stride()
                for y, z in zip(yy, gold, strict=True)
            )
            old = args[0]
            layout = (
                "BHSD"
                if old.is_contiguous()
                else ("BSHD" if old.transpose(1, 2).is_contiguous() else "other")
            )
            assert layout == ("BSHD" if j == 0 else "BHSD")
            io = sum(x.numel() * x.element_size() for x in (*args, *yy))
            site.append(
                dict(
                    past=old.shape[2],
                    layout=layout,
                    strides=[list(x.stride()) for x in args],
                    result_sha256=[digest(x) for x in yy],
                    bytes_read_plus_written=io,
                    timing=None
                    if smoke
                    else measure(
                        {"two_cats": ref, "paired": lambda: append(*args)}, seed + 71 * j + ix
                    ),
                )
            )
        if not smoke:
            fs = {k: (lambda f=f: f(steps)) for k, f in timers.items()}
            hot[str(sizes[ix])] = dict(timing=measure(fs, seed + 111 + ix), alloc={})
            for k, f in fs.items():
                y, mem = build(f)
                arr = [x for row in y for x in row]
                mem["live_output_bytes"] = sum(x.untyped_storage().nbytes() for x in arr)
                mem["distinct_storages"] = len({x.untyped_storage().data_ptr() for x in arr})
                hot[str(sizes[ix])]["alloc"][k] = mem
                del arr, y
    if not smoke:
        mixed = clocks(timers, requests, seed + 5333)
    assert (
        warm == len(graphs)
        and str(ep.graph) == graph
        and hashes() == before
        and weights == {k: digest(t) for k, t in m.state_dict().items()}
    )
    return dict(
        family=family,
        layer_type=type(m.layer).__name__,
        seed=seed,
        masked=masked,
        weights=weights,
        graph_sha256=sha256(graph.encode()).hexdigest(),
        inputs=before,
        past=sizes,
        qualification=[17, 257],
        steps=4,
        variants=meta,
        setup=setups,
        binary=machine,
        inductor_backend_calls=warm,
        cache_sites=site,
        gates=gates,
        hot=hot,
        rotating=mixed,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    torch.set_num_threads(4)
    r = dict(
        smoke=a.smoke,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        transformers=transformers.__version__,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    with torch._dynamo.config.patch(recompile_limit=32):
        for family in ("mistral",) if a.smoke else ("qwen2", "llama", "mistral"):
            for mask in (True,) if a.smoke else (True, False):
                for seed in (3842153,) if a.smoke else SEEDS:
                    c = one(family, seed, mask, a.smoke)
                    r["cases"].append(c)
                    print(
                        family,
                        mask,
                        seed,
                        {n: z["exact_caches"] for n, z in c["gates"].items()},
                        flush=True,
                    )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
