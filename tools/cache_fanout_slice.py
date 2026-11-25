"""Compare functional KV-cache append plus head replication on actual Qwen2 operands."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median

import torch
import transformers
from kernel_trace import kernel_census
from provenance import capture
from qwen_cached_decode import suite, trajectory
from qwen_score_slice import Tap, compare, desc, subgraph
from sparkfuse.measure import device_time, host_time
from sparkfuse.partition import partition_export

ROOT = Path(__file__).resolve().parents[1]
OFFSET = 80831
EXPORT = 60401 + OFFSET
SEEDS = tuple(x + OFFSET for x in (60527, 60637, 60733, 60869, 60943))


def address(t):
    return t.untyped_storage().data_ptr()


def evaluate(fn, live, ref):
    before = [x.clone(memory_format=torch.preserve_format) for x in live]
    result = fn(*live)
    metrics = [compare(x, y) for x, y in zip(result, ref, strict=True)]
    ptrs = {address(x) for x in result}
    fresh = len(ptrs) == len(ref) and not ptrs.intersection(map(address, live))
    unchanged = all(
        torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))
        for x, y in zip(live, before, strict=True)
    )
    return dict(
        outputs=metrics,
        fresh=fresh,
        unchanged=unchanged,
        passed=all(v["passed"] for v in metrics) and fresh and unchanged,
    )


def work(model, rope, past):
    export = trajectory(model, rope, past, EXPORT)
    held = [trajectory(model, rope, past, seed) for seed in SEEDS]
    rows = []
    for i, (args, _) in enumerate(export):
        ep = torch.export.export(model, args)
        original = ep.module()
        _, records = partition_export(ep, cache_fanout=True, host_backend="native")
        (full,) = [r for r in records if r.get("kind") == "cache_fanout"]
        ins = full["live_ins"]
        outs = full["live_outs"]
        rec = dict(live_ins=ins, live_outs=outs, ops=[k for chain in full["ops"] for k in chain])
        eager = subgraph(original, rec)
        qk, pv = [n for n in original.graph.nodes if n.target is torch.ops.aten.matmul.default]
        kk, vv = (n.args[1].name for n in (qk, pv))
        tap = Tap(original, [*ins, *outs, kk, vv])
        tap.run(*args)
        region = torch.export.export(eager, tuple(tap.seen[k] for k in ins))
        fns = {
            "eager": eager,
            "inductor": torch.compile(eager, fullgraph=True, dynamic=False, backend="inductor"),
        }
        builds = {}
        for name, fused in [("separate", False), ("fanout", True)]:
            gm, rr = partition_export(
                region,
                cache_fanout=fused,
                paired_appends=True,
                paired_head_repeats=True,
                host_backend="native",
            )
            rr = [r for r in rr if r.get("module")]
            assert [r["kind"] for r in rr] == (
                ["cache_fanout"] if fused else ["paired_append", "paired_head_repeat"]
            )
            fns[name] = gm
            builds[name] = rr
        assert builds["fanout"][0]["cuda_sha256"] == full["cuda_sha256"]
        samples = []
        for seed, steps in zip(SEEDS, held, strict=True):
            a, reference = steps[i]
            tap.seen.clear()
            y = tap.run(*a)
            whole = [compare(x, r) for x, r in zip(y, reference, strict=True)]
            assert all(v["passed"] for v in whole)
            live = tuple(tap.seen[k] for k in ins)
            ref = tuple(tap.seen[k] for k in outs)
            # First two outputs are exactly the returned caches. Remaining two feed the
            # original QK and PV matmuls (K is transposed at the consumer).
            controls = [
                compare(x, r)
                for x, r in zip(
                    (ref[0], ref[1], ref[2].transpose(2, 3), ref[3]),
                    (*reference[-2:], tap.seen[kk], tap.seen[vv]),
                    strict=True,
                )
            ]
            assert all(v["passed"] for v in controls)
            methods = {}
            for name, fn in fns.items():
                try:
                    methods[name] = evaluate(fn, live, ref)
                except Exception as exc:
                    if name != "inductor":
                        raise
                    methods[name] = dict(passed=False, error=type(exc).__name__)
            samples.append(
                dict(
                    seed=seed,
                    live_ins=[desc(x) for x in live],
                    outputs=[desc(x) for x in ref],
                    whole_layer=whole,
                    cache_and_gemm_operands=controls,
                    methods=methods,
                )
            )
        allowed = [k for k in fns if all(s["methods"][k]["passed"] for s in samples)]
        rng = random.Random(27341 + 2 * past + i)
        wins = []
        for _ in range(7):
            order = allowed.copy()
            rng.shuffle(order)
            w = dict(order=order, device={}, host={})
            for name in order:

                def run():
                    return fns[name](*live)

                w["device"][name] = device_time(run, groups=7, unroll=4, replay=3)
                w["host"][name] = host_time(run, samples=21)
            wins.append(w)
        med = {
            m: {name: median(w[m][name]["median_us"] for w in wins) for name in allowed}
            for m in ("host", "device")
        }
        cache = sum(x.nbytes for x in ref[:2])
        writes = sum(x.nbytes for x in ref)
        rows.append(
            dict(
                prefix=past,
                step=i + 1,
                export_graph_sha256=sha256(str(ep.graph).encode()).hexdigest(),
                region_graph=str(eager.graph),
                full_record=full,
                builds=builds,
                samples=samples,
                scheduled_payload_bytes=dict(
                    separate=dict(read=2 * cache, write=writes),
                    fanout=dict(read=cache, write=writes),
                ),
                windows=wins,
                median_us=med,
                profiles={k: kernel_census(lambda: fns[k](*live)) for k in allowed},
                passed={"eager", "separate", "fanout"} <= set(allowed),
            )
        )
        print(past, i + 1, med["device"], flush=True)
    return rows


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    model, rope = suite(OFFSET)
    d = dict(
        scope="Four materialized BF16 outputs on real Qwen2 two-step cache trajectories. Captured live-ins and the original ATen operators only; no GEMM or layer timing in this report.",
        protocol="Validate original graph against stateful HF and both K/V and exact operands of the two original matmuls. Admit a backend to timing only on five-trajectory exact BF16, stride, fresh-output storage and unmodified-input checks. Last seed in seven shuffled device/host windows. Logical payload counts model algorithmic per-packet reads/writes, not transactions or measured DRAM traffic.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        seeds=SEEDS,
        export_seed=EXPORT,
        weight_seed=60313 + OFFSET,
        timing_seed=SEEDS[-1],
        provenance=capture(ROOT / ".runtime/cache-fanout/sources"),
        cases=[],
    )
    for past in (31, 1024, 2047):
        d["cases"].extend(work(model, rope, past))
    d["passed"] = all(c["passed"] for c in d["cases"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)
    return 0 if d["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
