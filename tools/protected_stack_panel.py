"""Follow each implementation's own multilayer KV state. Distinguish closeness from byte identity."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import transformers
from examples.decoder_stack import suite, trajectory, export
from sparkfuse.ir import Unsupported
from sparkfuse.hybrid_decode import partition_hybrid_decode
from length_decode_panel import counted_inductor
from decode_sdpa_panel import cpu_reference, metrics
from hybrid_qwen_panel import trace
from ordered_score_panel import binary
from token_quant_panel import raw, digest
from serving_clocks import build, clocks
from paired_timing import measure
from provenance import capture

SEEDS = (3998173, 4011227)
PAST = (2, 15, 30, 31, 125, 255, 513, 2045)


def carry(xs, prev):
    return xs if prev is None else (*xs[:3], *prev[1:], *xs[3 + len(prev) - 1 :])


def walk(f, steps):
    all_steps = []
    prev = None
    for xs in steps:
        prev = f(*carry(xs, prev))
        all_steps.append(prev)
    return all_steps


def gate(f, tr):
    def ptr(x):
        return x.untyped_storage().data_ptr()

    def desc(x):
        return (
            x.shape,
            x.stride(),
            x.dtype,
            x.device,
            x.storage_offset(),
            x._base is None,
            x.untyped_storage().nbytes(),
        )

    rows = []
    live = []
    for steps in tr:
        row = []
        prev = None
        for xs, ref in steps:
            args = carry(xs, prev)
            out = f(*args)
            prev = out
            live.extend(out)
            ms = []
            for a, b in zip(out, ref, strict=True):
                err = (a.float() - b.float()).abs()
                ms.append(
                    dict(
                        outside=int((~(err <= 0.012 + 0.02 * b.float().abs())).sum()),
                        max_abs=float(err.max()),
                        different_bytes=int((raw(a) != raw(b)).sum()),
                        sha256=digest(a),
                        reference_sha256=digest(b),
                    )
                )
            pre = all(
                torch.equal(raw(a[:, :, : old.size(2)]), raw(old))
                for a, old in zip(out[1:], args[3 : 3 + len(out) - 1], strict=True)
            )
            row.append(
                dict(
                    outputs=ms,
                    geometry=all(desc(a) == desc(b) for a, b in zip(out, ref, strict=True)),
                    fresh=len(set(map(ptr, out))) == len(out)
                    and not set(map(ptr, out)).intersection(map(ptr, args)),
                    own_prefix=pre,
                )
            )
        rows.append(row)
    unique = len(set(map(ptr, live))) == len(live)
    ok = unique and all(
        z["geometry"]
        and z["fresh"]
        and z["own_prefix"]
        and all(not v["outside"] for v in z["outputs"])
        for row in rows
        for z in row
    )
    errors = [
        sum(z["outputs"][i]["different_bytes"] != 0 for row in rows for z in row)
        for i in range(1, len(rows[0][0]["outputs"]))
    ]
    return dict(
        steps=rows,
        all_live=unique,
        admitted=ok,
        differing_cache_tensors_by_output=errors,
        exact_caches=ok and not any(errors),
    )


def case(family, depth, seed, masked, smoke):
    model, rope = suite(family, depth, seed)
    weights = lambda: {k: digest(t) for k, t in model.state_dict().items()}
    before = weights()
    ex, au = [
        trajectory(model, rope, s, masked, seed + 67 * i, 1)[0][0] for i, s in enumerate((17, 65))
    ]
    ep = export(model, ex, masked)
    graph = str(ep.graph)
    aten = ep.module()
    paths = {"hf_eager": model, "exported_aten": aten}
    meta = {}
    fail = {}
    proofs = {}
    bins = {}
    for name, locked in [("unprotected", ()), ("protected", tuple(range(1, 1 + 2 * depth)))]:
        try:
            h, ms = build(
                lambda: partition_hybrid_decode(
                    ep,
                    ex,
                    au,
                    domain="bounded",
                    host_calls=True,
                    paired_rotary=True,
                    paired_append=True,
                    warp_heads=4,
                    exact_outputs=locked,
                )
            )
        except Unsupported as e:
            if name == "protected":
                raise
            fail[name] = str(e)
            continue
        paths[name] = h.module
        att = []
        for site in h.attention:
            op = h.module.get_submodule(site["module"]).op
            codes = []
            for p in op.paths:
                key = sha256(p.with_suffix(".so").read_bytes()).hexdigest()
                codes.append(key)
                if key not in bins:
                    bins[key] = binary(SimpleNamespace(path=p, source=p.read_text()))
            att.append(codes)
        meta[name] = dict(
            setup=ms,
            attention=h.attention,
            regions=h.regions,
            appends=h.appends,
            rotary=h.rotary,
            host_calls=h.host_calls,
            protection=h.protection,
            attention_binaries=att,
        )
        if locked:
            assert (
                len(h.attention) == 1
                and len(h.protection["skipped_attention"]) == depth - 1
                and all(
                    z["reason"] == "feeds a protected public output"
                    for z in h.protection["skipped_attention"]
                )
            )
            (last,) = h.attention
            (node,) = [n for n in h.module.graph.nodes if n.target == last["module"]]
            scale = op.plan.scale
    graphs = []
    torch._dynamo.reset()
    if not smoke:
        paths["inductor_dynamic"], graphs = counted_inductor(aten, True)
    sizes = (31, 125) if smoke else PAST
    tr = [trajectory(model, rope, s, masked, seed + 2019 + 113 * i) for i, s in enumerate(sizes)]
    req = [(tuple(xs for xs, _ in t),) for t in tr]
    fingerprints = lambda: [[[digest(t) for t in xs] for xs, _ in steps] for steps in tr]
    inputs = fingerprints()
    gates = {}
    for name, f in paths.items():
        try:
            gates[name] = gate(f, tr)
        except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as e:
            if name != "inductor_dynamic":
                raise
            fail[name] = type(e).__name__
    for name in ("protected", "hf_eager", "exported_aten"):
        assert gates[name]["exact_caches"], (
            name,
            gates[name]["differing_cache_tensors_by_output"],
            gates[name]["admitted"],
        )
    for i in (0, len(tr) - 1):
        prev = None
        checks = []
        for xs, _ in tr[i]:
            args = carry(xs, prev)
            prev, env = trace(h.module, args)
            _, base = trace(aten, args)
            operands = tuple(env[n.name] for n in node.args)
            gold = cpu_reference(operands, scale)
            z = metrics(
                env[node.name],
                gold,
                base[last["name"]].clone(memory_format=torch.contiguous_format),
                operands,
            )
            assert z["structure"] and not z["outside_tolerance"]
            checks.append(dict(fp64_sha256=digest(gold), attention=z))
        proofs[str(sizes[i])] = checks
    warm = len(graphs)
    timers = {n: (lambda xs, f=paths[n]: walk(f, xs)) for n, g in gates.items() if g["admitted"]}
    hot = {}
    mix = None
    if not smoke:
        for i in (0, len(tr) - 1):
            hot[str(sizes[i])] = measure(
                {n: (lambda f=f: f(*req[i])) for n, f in timers.items()}, seed + 229 + i
            )
        mix = clocks(timers, req, seed + 5117)
    assert (
        weights() == before
        and fingerprints() == inputs
        and str(ep.graph) == graph
        and len(graphs) == warm
    )
    return dict(
        family=family,
        depth=depth,
        layer_type=type(model.layers[0]).__name__,
        seed=seed,
        masked=masked,
        qualification=[17, 65],
        steps=4,
        past=sizes,
        weights=before,
        export_graph_sha256=sha256(graph.encode()).hexdigest(),
        inputs=inputs,
        gates=gates,
        build_failures=fail,
        variants=meta,
        binaries=bins,
        fp64=proofs,
        numerically_timed=list(timers) if not smoke else [],
        byte_exact_cache_paths=[n for n, g in gates.items() if g["exact_caches"]],
        hot=hot,
        rotating=mix,
        inductor_backend_calls=warm,
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
        for f, d in [("qwen2", 2)] if a.smoke else [("qwen2", 2), ("llama", 3)]:
            for m in (True,) if a.smoke else (True, False):
                for s in (4024931,) if a.smoke else SEEDS:
                    c = case(f, d, s, m, a.smoke)
                    r["cases"].append(c)
                    print(f, d, m, s, c["byte_exact_cache_paths"], c["build_failures"], flush=True)
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
