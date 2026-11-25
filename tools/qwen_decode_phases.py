"""Attribute captured Qwen2 decode kernels to exported FX nodes, with adjacent unprofiled timings."""

import argparse
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import torch
import transformers
from phase_attribution import map_ordinary, replay_intervals, require, summarize
from provenance import capture
from qwen_cached_decode import check, suite, trajectory
from sparkfuse.partition import ordered_collectives, partition_export

ROOT = Path(__file__).resolve().parents[1]
MODES = ("separate", "fanout")


class Labeled(torch.fx.Interpreter):
    def run_node(self, n):
        with torch.profiler.record_function("phase/" + n.name):
            return super().run_node(n)


def trace(fn):
    torch.cuda.synchronize()
    with TemporaryDirectory() as d:
        path = Path(d) / "trace.json"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as p:
            fn()
            torch.cuda.synchronize()
        p.export_chrome_trace(str(path))
        return json.loads(path.read_text(), parse_float=Decimal)["traceEvents"]


def fingerprint(x):
    return dict(
        shape=list(x.shape),
        stride=list(x.stride()),
        dtype=str(x.dtype),
        storage_offset=x.storage_offset(),
        sha256=sha256(
            x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
    )


def metadata(g, records):
    recs = {r["module"]: r for r in records if "module" in r}
    nodes = {}
    matmuls = []
    for n in g.graph.nodes:
        target = str(n.target)
        group = "other"
        desc = target
        if n.target == torch.ops.aten.linear.default:
            desc = str(n.args[1].target)
            group = "mlp_projection" if ".mlp." in desc else "attention_projection"
        elif n.target == torch.ops.aten.matmul.default:
            matmuls.append(n.name)
            group = {1: "qk", 2: "pv"}[len(matmuls)]
        elif n.op == "call_module" and target in recs:
            r = recs[target]
            desc = r.get("kind") or ",".join(r["ops"])
            group = (
                "cache_copy"
                if desc in ("paired_append", "paired_head_repeat", "cache_fanout")
                else "other"
            )
        nodes[n.name] = dict(target=target, detail=desc, group=group)
    require(matmuls == ["matmul", "matmul_1"], "unexpected attention graph")
    return nodes


def event_times(graph):
    pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(7)
    ]
    for a, b in pairs:
        a.record()
        for _ in range(8):
            graph.replay()
        b.record()
    torch.cuda.synchronize()
    return [a.elapsed_time(b) * 1000 / 8 for a, b in pairs]


def study(ep, args, ref, prefix, step):
    before = tuple(x.clone() for x in args[-2:])
    graphs = {}
    held = {}
    builds = {}
    ins = [fingerprint(x) for x in args]
    for name in MODES:
        g, rs = partition_export(
            ep,
            rotary_pairs=True,
            paired_appends=True,
            paired_head_repeats=True,
            cache_fanout=name == "fanout",
            ordered_scores="slots",
            schedule_policy=ordered_collectives,
            host_backend="native",
        )
        g.graph.lint()
        for _ in range(5):
            out = g(*args)
        ordinary = check(out, ref, args, before, True)
        require(ordinary["passed"], "ordinary parity failed")
        mapping = map_ordinary(trace(lambda: Labeled(g).run(*args)), metadata(g, rs))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = g(*args)
        for _ in range(4):
            graph.replay()
        strict = check(out, ref, args, before, True)
        require(strict["passed"], "replay parity failed")
        graphs[name] = graph
        held[name] = (g, out)
        builds[name] = dict(
            mapping=mapping,
            ordinary=ordinary,
            replay=strict,
            windows=[],
            fx_sha256=sha256(str(g.graph).encode()).hexdigest(),
        )
    order = [list(MODES), list(reversed(MODES))] * 4
    random.Random(prefix * 31 + step).shuffle(order)
    for pair in order:
        for name in pair:
            graph = graphs[name]
            b = builds[name]
            for _ in range(4):
                graph.replay()
            times = event_times(graph)
            ids, intervals = replay_intervals(
                trace(lambda: [graph.replay() for _ in range(6)]), b["mapping"], 6
            )
            require(
                b.setdefault("graph_node_ids", ids) == ids,
                "graph-node identity changed between windows",
            )
            b["windows"].append(dict(event_samples_us=times, intervals_ns=intervals))
    require(ins == [fingerprint(x) for x in args], "mutable timing inputs")
    for name, b in builds.items():
        require(
            check(held[name][1], ref, args, before, True)["passed"], "post-profile parity failed"
        )
        b["median_us"] = summarize(b["mapping"], b["windows"])
    return dict(
        prefix=prefix,
        step=step,
        inputs=ins,
        reference=[fingerprint(x) for x in ref],
        export_graph_sha256=sha256(str(ep.graph).encode()).hexdigest(),
        window_order=order,
        backends=builds,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed-offset", type=int, default=80831)
    a = p.parse_args()
    m, r = suite(a.seed_offset)
    es = 60401 + a.seed_offset
    ts = 60943 + a.seed_offset
    d = dict(
        protocol="One random-weight Qwen2 layer; each static step uses the reference trajectory as common input, not a generation benchmark. FX label ranges only in the ordinary discovery trace. Kineto replay on an uninstrumented captured graph. Eight balanced paired windows, six profiled replays each; immediately preceding CUDA-event samples are seven groups of eight replays of the same captured graph. Summaries: median of within-window medians, in microseconds. Kernel durations include profiler perturbation; no DRAM counters or causal attribution to a particular cache level.",
        weight_seed=60313 + a.seed_offset,
        export_seed=es,
        timing_seed=ts,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        provenance=capture(ROOT / ".runtime/decode-phase/sources"),
        cases=[],
    )
    for prefix in (1024, 2047):
        exported = trajectory(m, r, prefix, es)
        timed = trajectory(m, r, prefix, ts)
        for i, ((ex, _), (args, ref)) in enumerate(zip(exported, timed, strict=True), 1):
            row = study(torch.export.export(m, ex), args, ref, prefix, i)
            d["cases"].append(row)
            print(prefix, i, {k: v["median_us"] for k, v in row["backends"].items()}, flush=True)
    d["passed"] = True
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
