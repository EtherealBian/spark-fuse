"""Whole-layer ablation with a real prefill and two independently carried decode steps."""

import argparse
import json
from pathlib import Path
import torch
import transformers
from examples.qwen_sdpa_step import suite, trajectory
from sparkfuse.decode_sites import partition_decode_sites
from decode_sdpa_panel import cpu_reference, metrics
from ordered_score_panel import binary
from paired_timing import measure
from serving_clocks import clocks
from token_quant_panel import digest, raw
from provenance import capture

SPECS = ((64, True), (256, False), (1024, True))
SEEDS = (3260273, 3271229)


def ptr(x):
    return x.untyped_storage().data_ptr()


def compare(out, ref, args):
    def desc(t):
        return (
            t.shape,
            t.stride(),
            t.storage_offset(),
            t.dtype,
            t.device,
            t.untyped_storage().nbytes(),
            t._base is None,
        )

    geo = all(desc(x) == desc(y) for x, y in zip(out, ref, strict=True))
    fresh = len({ptr(x) for x in out}) == 3 and not {ptr(x) for x in out} & {ptr(x) for x in args}
    ms = []
    for a, b in zip(out, ref, strict=True):
        err = (a.float() - b.float()).abs()
        ms.append(
            dict(
                max_abs=float(err.max()),
                outside=int((~(err <= 0.012 + 0.02 * b.float().abs())).sum()),
                different_bytes=int((raw(a) != raw(b)).sum()),
                sha256=digest(a),
                ref_sha256=digest(b),
            )
        )
    prefix = all(
        torch.equal(raw(x[:, :, : y.shape[2]]), raw(y))
        for x, y in zip(out[1:], args[3:5], strict=True)
    )
    return dict(
        geometry=geo,
        fresh=fresh,
        prefix_bitwise=prefix,
        outputs=ms,
        within_tolerance=all(x["outside"] == 0 for x in ms),
        exact_cache=all(x["different_bytes"] == 0 for x in ms[1:]),
    )


def carry(xs, cache):
    return (*xs[:3], *cache, *xs[5:])


def case(past, masked, seed, smoke):
    m, rope = suite(seed)
    weight_hash = {k: digest(x) for k, x in m.state_dict().items()}
    trajectories = [
        trajectory(m, rope, past, masked, seed + 113 * i) for i in range(4 if smoke else 10)
    ]
    held = trajectories[2:]
    inputs = [[[digest(t) for t in xs] for xs, r in tr] for tr in trajectories]
    steps = []
    path_pair = []
    exports = []
    records = []
    skip = {}
    inductor_error = None
    for j in (0, 1):
        ex, au = (tr[j][0] for tr in trajectories[:2])
        ep = torch.export.export(m, ex)
        exports.append(ep)
        graph = str(ep.graph)
        paths = {"hf_eager": m, "exported_aten": ep.module()}
        info = {}
        for tag, elide, r in [
            ("cuda_keep", False, 1),
            ("cuda_compact", True, 1),
            ("cuda_compact_g4", True, 4),
        ]:
            part = partition_decode_sites(
                ep, ex, au, domain="bounded", elide_kv_repeats=elide, tile=32, head_group=r
            )
            assert len(part.sites) == 1 and not part.skipped, (tag, part.skipped)
            rec = part.sites[0]
            assert (
                len(rec["erased_repeats"]) == 2 * elide
                and rec["removed_materialized_bytes"] == 2 * 2 * 8 * (past + 1 + j) * 64 * 2 * elide
            )
            plan = part.module.get_submodule(rec["module"]).op
            info[tag] = dict(site=rec, kernels=[binary(pg) for pg in plan.programs])
            paths[tag] = part.module
            # Capture original ATen vs replacement on independent real Qwen layer intermediates.
            # A CPU FP64 reference uses compact physical K/V for the compact paths.
            g0 = paths["exported_aten"]
            orig = next(
                n
                for n in g0.graph.nodes
                if n.target is torch.ops.aten.scaled_dot_product_attention.default
            )
            g1 = part.module
            site = next(
                n for n in g1.graph.nodes if n.op == "call_module" and n.target == rec["module"]
            )
            checks = []
            for tr in held:
                req0 = tr[j][0]
                i = torch.fx.Interpreter(g0, garbage_collect_values=False)
                i.run(*req0)
                k = torch.fx.Interpreter(g1, garbage_collect_values=False)
                k.run(*req0)
                args = tuple(k.env[n] for n in site.args)
                y = k.env[site]
                ref = cpu_reference(args, rec["plan"]["scale"])
                z = metrics(y, ref, i.env[orig], args)
                assert z["structure"] and not z["outside_tolerance"]
                checks.append(
                    dict(
                        metrics=z,
                        fp64_sha256=digest(ref),
                        original_stride=list(i.env[orig].stride()),
                        export_meta_stride=list(orig.meta["val"].stride()),
                        operands=[digest(x) for x in args],
                    )
                )
            info[tag]["site_checks"] = checks
        if not smoke and not inductor_error:
            try:
                # Same unchanged exported layer for both steps; no custom CUDA in the baseline.
                fn = torch.compile(paths["exported_aten"], dynamic=False, fullgraph=True)
                fn(*ex)
            except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as e:
                inductor_error = type(e).__name__
                skip["inductor"] = inductor_error
            else:
                paths["inductor"] = fn
        path_pair.append(paths)
        records.append(info)
        steps.append(dict(graph=graph, sites=info))
    paths = set(path_pair[0]) & set(path_pair[1])
    gates = {}
    timers = {}
    requests = [t[0][0] for t in held]
    for name in sorted(paths):
        results = []
        live = []
        for tr in held:
            xs, ref = tr[0]
            out0 = path_pair[0][name](*xs)
            a = compare(out0, ref, xs)
            xs2 = carry(tr[1][0], out0[1:])
            out1 = path_pair[1][name](*xs2)
            b = compare(out1, tr[1][1], xs2)
            results.append([a, b])
            live.extend((*out0, *out1))
        distinct = len({ptr(t) for t in live}) == 6 * len(held)
        good = (
            all(
                r["geometry"] and r["fresh"] and r["prefix_bitwise"] and r["within_tolerance"]
                for row in results
                for r in row
            )
            and distinct
        )
        exact = good and all(r["exact_cache"] for row in results for r in row)
        gates[name] = dict(
            numerically_admitted=good,
            exact_cache_admitted=exact,
            all_live=distinct,
            requests=results,
        )
        if name != "inductor":
            assert exact, (name, results)
        if good:
            timers[name] = path_pair[0][name]
    hot = ordinary = None
    if not smoke:
        hot = measure({k: (lambda f=f: f(*requests[0])) for k, f in timers.items()}, seed + past)
        ordinary = clocks(timers, requests, seed + past + 67)
    assert (
        inputs == [[[digest(t) for t in xs] for xs, r in tr] for tr in trajectories]
        and weight_hash == {k: digest(x) for k, x in m.state_dict().items()}
        and all(str(ep.graph) == step["graph"] for ep, step in zip(exports, steps, strict=True))
    )
    return dict(
        past=past,
        masked=masked,
        seed=seed,
        weights=weight_hash,
        inputs=inputs,
        steps=steps,
        gates=gates,
        build_failures=skip,
        hot=hot,
        requests=ordinary,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    r = dict(
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        cases=[],
    )
    for past, masked in SPECS[:1] if a.smoke else SPECS:
        for seed in (3296371,) if a.smoke else SEEDS:
            c = case(past, masked, seed, a.smoke)
            r["cases"].append(c)
            print(
                past,
                seed,
                {
                    k: (v["numerically_admitted"], v["exact_cache_admitted"])
                    for k, v in c["gates"].items()
                },
                c["hot"]["median_us"] if c["hot"] else "qualified",
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
