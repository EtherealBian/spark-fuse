"""Three fixed Qwen2 graphs, independent or shared immutable state; isolate allocator accounting."""

import argparse
from contextlib import ExitStack
import gc
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import torch
import transformers
from packed_mean_panel import digest
from provenance import capture
from qwen_packed_mean import Qwen2RotaryEmbedding, compile_modes, make_model, whole_check
from sparkfuse import Schedule
from sparkfuse.contextual import _alias_pattern, _layout
from sparkfuse.family import _key, capture_family
from sparkfuse.packed_mean import match
from sparkfuse.partition import ordered_collectives, partition_export
from sparkfuse.resident import _get, capture_export

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "reports/qwen-packed-mean-heldout-gb10.json"


def make(weight, seqs):
    model, _ = make_model(weight, 33)
    cfg = model.layer.self_attn.config
    rot = Qwen2RotaryEmbedding(cfg, device="cuda")

    def request(seq, i):
        seed = weight + seq * 317 + i * 941
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(2, seq, 1536, device="cuda", dtype=torch.bfloat16, generator=g)
        pos = torch.arange(seq, device="cuda")[None, :] + torch.tensor(
            [[i * 3], [13 + i * 5]], device="cuda"
        )
        c, s = rot(x, pos)
        mask = torch.full((2, 1, seq, seq), float("-inf"), device="cuda", dtype=x.dtype).triu(1)
        mask[1, :, 1:, 0] = float("-inf")
        k = i % 13 + 1
        mask[:, :, k + 1 :, k] = float("-inf")
        return x, c.contiguous(), s.contiguous(), mask

    pairs = [(request(s, 0), request(s, 1)) for s in seqs]
    req = [request(s, i) for i in (3, 5, 7, 9) for s in seqs]
    schedules = {
        c["sequence"]: Schedule(**c["kernels"]["packed"]["schedule"])
        for c in json.loads(FROZEN.read_bytes())["cases"]
    }
    opts = dict(
        host_backend="native",
        rotary_pairs=True,
        schedule_policy=lambda p: schedules[p.domain[1]] if match(p) else ordered_collectives(p),
    )
    return model, pairs, req, opts


def alloc():
    torch.cuda.synchronize()
    gc.collect()
    gc.collect()
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()


def provision(kind, model, pairs, opts, stack):
    if kind == "shared":
        family = stack.enter_context(capture_family(model, pairs, partition_options=opts))
        return family._handles, family
    runs = {}
    for ex, audit in pairs:
        ep = torch.export.export(model, ex)
        runs[_key(ex)] = stack.enter_context(capture_export(ep, ex, audit, partition_options=opts))
    return runs, None


def inventory(handles):
    tables = []
    unique = {}
    state_uses = 0
    stage = 0
    for h in handles.values():
        state = {n.target: _get(h._fn, n.target) for n in h._fn.graph.nodes if n.op == "get_attr"}
        table = {k: (t.untyped_storage().data_ptr(), t.nbytes) for k, t in state.items()}
        tables.append(table)
        size = dict(table.values())
        unique.update(size)
        state_uses += sum(size.values())
        stage += h.metadata["input_bytes"]
    assert all(t.keys() == tables[0].keys() for t in tables)
    return dict(
        named_state_per_shape=len(tables[0]),
        unique_weight_stores=len(unique),
        unique_weight_bytes=sum(unique.values()),
        weight_bytes_counting_each_shape=state_uses,
        input_staging_bytes=stage,
        logical_static_bytes=stage + sum(unique.values()),
    )


@torch.inference_mode()
def memory_worker(kind, weight):
    seqs = [33, 65, 129]
    m, pairs, r, opts = make(weight, seqs)
    hash_w = {n: digest(t) for n, t in m.named_parameters()}
    # Workspaces, native extension, export and cached kernels initialized symmetrically,
    # then release all warm graphs and temporary FX modules before recording a baseline.
    for ex, audit in pairs:
        with capture_export(torch.export.export(m, ex), ex, audit, partition_options=opts) as warm:
            warm(*audit)
    del warm
    base = alloc()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter_ns()
    with ExitStack() as stack:
        handles, f = provision(kind, m, pairs, opts, stack)
        retained = alloc()
        wall = (time.perf_counter_ns() - start) / 1e6
        peak = torch.cuda.max_memory_allocated()
        inv = inventory(handles)
        if f:
            assert inv["logical_static_bytes"] == f.metadata["static_bytes"]
        for args in r:
            got = handles[_key(args)](*args)
            want = m(*args)
            assert whole_check(got, want, args[-1])["passed"]
            del got, want
    del handles, f
    after = alloc()
    return dict(
        variant=kind,
        weight_seed=weight,
        weight_sha256=hash_w,
        baseline=base,
        after_build=retained,
        peak_during_build=peak,
        after_close=after,
        retained_delta=retained - base,
        setup_ms=wall,
        inventory=inv,
    )


def timing(fns, seed):
    rng = random.Random(seed)
    s = torch.cuda.current_stream()
    ws = []
    n = len(next(iter(fns.values())))
    for w in range(7):
        order = sorted(fns)
        rng.shuffle(order)
        windows = {}
        for name in order:
            calls = fns[name]
            sync = []
            sub = []
            queued = []
            for j in range(2 * n):
                f = calls[(j + w) % n]
                s.synchronize()
                start = time.perf_counter_ns()
                out = f()
                sent = time.perf_counter_ns()
                s.synchronize()
                end = time.perf_counter_ns()
                sub.append((sent - start) / 1000)
                sync.append((end - start) / 1000)
                del out
            for k in range(5):
                burst = [calls[(j + w + k) % n] for j in range(n)]
                s.synchronize()
                start = time.perf_counter_ns()
                out = [f() for f in burst]
                s.synchronize()
                end = time.perf_counter_ns()
                queued.append((end - start) / (1000 * n))
                del out
            windows[name] = dict(sync_us=sync, submit_us=sub, burst_us_per_forward=queued)
        ws.append(dict(order=order, samples=windows))
    return dict(
        seed=seed,
        windows=ws,
        median_us={
            p: {k: median(median(w["samples"][p][k]) for w in ws) for k in ws[0]["samples"][p]}
            for p in fns
        },
    )


def compare(y, w, args):
    g = whole_check(y, w, args[-1])
    g["public_layout"] = all(_layout(a) == _layout(b) for a, b in zip(y, w, strict=True))
    g["alias_match"] = _alias_pattern(args, y) == _alias_pattern(args, w)
    return g


@torch.inference_mode()
def serving(weight, smoke):
    seqs = [33, 65] if smoke else [33, 65, 129]
    torch.compiler.reset()
    m, pairs, r, opts = make(weight, seqs)
    want = [m(*a) for a in r]
    paths = {}
    gates = {}
    seen = {}
    with ExitStack() as stack:
        for kind in ("separate", "shared"):
            handles, family = provision(kind, m, pairs, opts, stack)
            seen[kind] = inventory(handles)
            paths[kind + "_direct"] = [(lambda f=handles[_key(a)], a=a: f(*a)) for a in r]
            if family:
                paths["shared_routed"] = [(lambda f=family, a=a: f(*a)) for a in r]
        gm = {_key(ex): partition_export(torch.export.export(m, ex), **opts)[0] for ex, _ in pairs}
        paths["mixed"] = [(lambda f=gm[_key(a)], a=a: f(*a)) for a in r]
        for n, model in {"eager": m, **({} if smoke else compile_modes(m))}.items():
            paths[n] = [(lambda f=model, a=a: f(*a)) for a in r]
        for n, calls in paths.items():
            out = [f() for f in calls]
            checks = [compare(y, w, a) for y, w, a in zip(out, want, r, strict=True)]
            ind = len({x.untyped_storage().data_ptr() for y in out for x in y}) == 2 * len(out)
            gates[n] = dict(
                checks=checks,
                distinct_outputs=ind,
                passed=ind
                and all(q["passed"] and q["public_layout"] and q["alias_match"] for q in checks),
            )
            del out
        admitted = {k: v for k, v in paths.items() if gates[k]["passed"]}
        assert {
            "separate_direct",
            "shared_direct",
            "shared_routed",
            "eager",
            "mixed",
        } <= admitted.keys()
        result = None if smoke else timing(admitted, weight + 19001)
        return dict(
            weight_seed=weight,
            weight_sha256={n: digest(t) for n, t in m.named_parameters()},
            sequences=seqs,
            request_shapes=[a[0].shape[1] for a in r],
            inputs=[[digest(x) for x in a] for a in r],
            references=[[digest(x) for x in y] for y in want],
            inventory=seen,
            gates=gates,
            timing=result,
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--memory-worker", choices=["shared", "separate"])
    p.add_argument("--weight", type=int)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.memory_worker:
        if a.weight is None or a.smoke:
            p.error("worker needs weight and no smoke")
        a.output.write_text(
            json.dumps(memory_worker(a.memory_worker, a.weight), allow_nan=False) + "\n"
        )
        return
    d = dict(
        protocol="Single BF16 Qwen2 layer, one fixed model per shape family. Memory experiment: fresh child per weight and topology, same separate-graph warmup, GC+synchronize before baseline and after build/close. PyTorch active allocator bytes only, not physical/driver memory. Unique weight storage census from actual tensor bindings. Serving: fixed mixed schedules, interleaved lengths, four distinct requests per shape, hold all results; numerical/alias/layout failures retained and not timed. Direct paths resolve callable before timers; routed pays shape lookup. Seven shuffled method windows, two full rotations of single synchronized requests and five full queued bursts. No outer graph.",
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        transformers=transformers.__version__,
        rms_report_sha256=sha256(FROZEN.read_bytes()).hexdigest(),
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for i, w in enumerate([899903] if a.smoke else [903383, 911293]):
        mem = {}
        if not a.smoke:
            with TemporaryDirectory() as tmp:
                for v in ("shared", "separate") if i else ("separate", "shared"):
                    dest = Path(tmp) / (v + ".json")
                    subprocess.run(
                        [
                            sys.executable,
                            __file__,
                            "--memory-worker",
                            v,
                            "--weight",
                            str(w),
                            "--output",
                            str(dest),
                        ],
                        check=True,
                        timeout=180,
                    )
                    mem[v] = json.loads(dest.read_bytes())
        c = serving(w, a.smoke)
        c["memory"] = mem
        d["cases"].append(c)
        if mem:
            print(
                w,
                {v: m["retained_delta"] for v, m in mem.items()},
                c["timing"]["median_us"],
                flush=True,
            )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
