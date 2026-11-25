"""Paired materialization vs direct segmented reads for *two* complete Qwen2 steps."""

import argparse
import json
from pathlib import Path
import torch
import transformers
from examples.qwen_sdpa_step import suite, trajectory
from sparkfuse.cache_append import fuse_appends
from sparkfuse.decode_sites import partition_decode_sites
from sparkfuse.segmented_decode import partition_segmented_decode, _bits
from qwen_sdpa_sites import compare, carry, ptr
from decode_sdpa_panel import cpu_reference, metrics
from token_quant_panel import digest
from ordered_score_panel import binary
from serving_clocks import clocks
from paired_timing import measure
from provenance import capture

SPECS = ((64, True), (256, False), (1024, True))
SEEDS = (3571619, 3581749)


def combined(f, g):
    def two(req0, req1):
        y = f(*req0)
        z = g(*carry(req1, y[1:]))
        return y, z

    return two


def allocator(f, req):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    o = f(*req)
    torch.cuda.synchronize()
    retained = torch.cuda.memory_allocated() - base
    peak = torch.cuda.max_memory_allocated() - base
    assert peak >= retained > 0
    return dict(
        retained_bytes=retained,
        peak_increment_bytes=peak,
        all_six_fresh=len({ptr(y) for ys in o for y in ys}) == 6,
    )


def case(past, masked, seed, smoke):
    m, rope = suite(seed)
    weights = {k: digest(v) for k, v in m.state_dict().items()}
    ts = [trajectory(m, rope, past, masked, seed + 103 * j) for j in range(4 if smoke else 10)]
    held = ts[2:]
    ids = [[[digest(z) for z in req] for req, _ in tr] for tr in ts]
    steps = []
    pair = []
    fail = {}
    exports = []
    for j in (0, 1):
        ex, au = [tr[j][0] for tr in ts[:2]]
        ep = torch.export.export(m, ex)
        exports.append(ep)
        at = ep.module()
        a = partition_decode_sites(ep, ex, au, domain="bounded")
        b = partition_decode_sites(ep, ex, au, domain="bounded")
        pa = fuse_appends(b.module, ex[0].device, "native")
        assert len(pa) == 1
        f = partition_segmented_decode(ep, ex, au, domain="bounded")
        (rec,) = f.sites
        assert rec["segmented"]["published"] == [True, True] and not f.skipped
        seg = f.module.get_submodule(rec["module"])
        info = dict(
            graph=str(ep.graph),
            compact=a.sites,
            paired=pa,
            segmented=rec,
            binary=[binary(seg.program)] + [binary(pg) for pg in seg.base.op.programs[1:]],
            held_intermediates=[],
        )
        paths = {
            "hf_eager": m,
            "exported_aten": at,
            "two_cats": a.module,
            "paired_append": b.module,
            "segmented": f.module,
        }
        if not smoke and not fail:
            try:
                fn = torch.compile(at, fullgraph=True, dynamic=False)
                fn(*ex)
            except (torch._dynamo.exc.Unsupported, torch._dynamo.exc.BackendCompilerFailed) as e:
                fail["inductor"] = type(e).__name__
            else:
                paths["inductor"] = fn
        for tr in held:
            xs = tr[j][0]
            ai = torch.fx.Interpreter(a.module, garbage_collect_values=False)
            ai.run(*xs)
            fi = torch.fx.Interpreter(f.module, garbage_collect_values=False)
            fi.run(*xs)
            site = next(n for n in a.module.graph.nodes if n.target == a.sites[0]["module"])
            fsite = next(n for n in f.module.graph.nodes if n.target == rec["module"])
            x0 = tuple(ai.env[n] for n in site.args)
            x1 = tuple(fi.env[n] for n in fsite.args)
            orig = a.module.get_submodule(site.target).op
            raw0 = orig._first(*x0)[0]
            raw1, *published = seg.dispatch(*x1)
            assert _bits(raw0, raw1) and all(
                _bits(u, v) for u, v in zip(published, x0[1:3], strict=True)
            )
            y = fi.env[fsite][0]
            reference = cpu_reference(x0, orig.plan.scale)
            err = metrics(y, reference, ai.env[site], x0)
            assert err["structure"] and err["outside_tolerance"] == 0 and _bits(y, ai.env[site])
            info["held_intermediates"].append(
                dict(
                    partial_sha256=[digest(raw0), digest(raw1)],
                    cache_sha256=[
                        [digest(o), digest(v)] for o, v in zip(published, x0[1:3], strict=True)
                    ],
                    attention_sha256=[digest(y), digest(ai.env[site])],
                    cpu_fp64_sha256=digest(reference),
                    cpu_error=err,
                )
            )
        steps.append(info)
        pair.append(paths)
    callers = {
        name: combined(pair[0][name], pair[1][name]) for name in sorted(set(pair[0]) & set(pair[1]))
    }
    requests = [(tr[0][0], tr[1][0]) for tr in held]
    gates = {}
    timed = {}
    mem = {}
    for name, f in callers.items():
        rr = []
        live = []
        compact_out = []
        for tr, req in zip(held, requests, strict=True):
            y, z = f(*req)
            base = callers["two_cats"](*req)
            rr.append([compare(y, tr[0][1], req[0]), compare(z, tr[1][1], carry(req[1], y[1:]))])
            live.extend((*y, *z))
            compact_out.append(
                [
                    _bits(u, v)
                    for us, vs in zip((y, z), base, strict=True)
                    for u, v in zip(us, vs, strict=True)
                ]
            )
        distinct = len({ptr(x) for x in live}) == 6 * len(held)
        ok = distinct and all(
            r["geometry"] and r["fresh"] and r["prefix_bitwise"] and r["within_tolerance"]
            for row in rr
            for r in row
        )
        exact = ok and all(r["exact_cache"] for row in rr for r in row)
        gates[name] = dict(
            admitted=ok,
            exact_cache=exact,
            all_live=distinct,
            requests=rr,
            bitwise_vs_compact=compact_out,
        )
        if name != "inductor":
            assert exact, (name, rr)
        if name in ("two_cats", "paired_append", "segmented"):
            assert all(all(v) for v in compact_out)
        if ok:
            timed[name] = f
            mem[name] = allocator(f, requests[0])
    hot = ordinary = None
    if not smoke:
        hot = measure({k: (lambda f=f: f(*requests[0])) for k, f in timed.items()}, seed + past)
        ordinary = clocks(timed, requests, seed + past + 7)
    assert (
        all(str(ep.graph) == rr["graph"] for ep, rr in zip(exports, steps, strict=True))
        and ids == [[[digest(z) for z in req] for req, _ in tr] for tr in ts]
        and weights == {k: digest(v) for k, v in m.state_dict().items()}
    )
    return dict(
        past=past,
        masked=masked,
        seed=seed,
        weights=weights,
        input_sha256=ids,
        steps=steps,
        gates=gates,
        allocator=mem,
        build_failures=fail,
        hot=hot,
        requests=ordinary,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    d = dict(
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        cases=[],
    )
    for past, masked in SPECS[:1] if a.smoke else SPECS:
        for seed in (3597901,) if a.smoke else SEEDS:
            c = case(past, masked, seed, a.smoke)
            d["cases"].append(c)
            print(
                past,
                seed,
                {k: v["exact_cache"] for k, v in c["gates"].items()},
                c["hot"]["median_us"] if c["hot"] else "qualified",
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
