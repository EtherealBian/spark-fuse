"""Time the actual extracted ATen score graph using captured Qwen2 decode live-ins."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median

import torch
import transformers
from kernel_trace import kernel_census
from ordered_score_panel import binary
from provenance import capture
from qwen_cached_decode import suite, trajectory
from sparkfuse.measure import device_time, host_time
from sparkfuse.partition import ordered_collectives, partition_export

ROOT = Path(__file__).resolve().parents[1]
OFFSET = 64219  # Match the companion whole-layer experiment, including weights and masks.
EXPORT = 60401 + OFFSET
SEEDS = tuple(s + OFFSET for s in (60527, 60637, 60733, 60869, 60943))


def digest(t):
    return sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def desc(t):
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype), sha256=digest(t))


def compare(x, y):
    if x.shape != y.shape or x.dtype != y.dtype:
        return dict(passed=False, error="shape or dtype mismatch")
    neq = int((x.contiguous().view(torch.int16) != y.contiguous().view(torch.int16)).sum())
    return dict(
        unequal_bf16=neq,
        stride_match=x.stride() == y.stride(),
        passed=not neq and x.stride() == y.stride(),
    )


class Tap(torch.fx.Interpreter):
    def __init__(self, gm, names):
        super().__init__(gm)
        self.names = set(names)
        self.seen = {}

    def run_node(self, n):
        x = super().run_node(n)
        if n.name in self.names:
            self.seen[n.name] = x
        return x


def subgraph(gm, rec):
    nodes = {n.name: n for n in gm.graph.nodes}
    graph = torch.fx.Graph()
    env = {nodes[k]: graph.placeholder(k) for k in rec["live_ins"]}
    for k in rec["ops"]:
        n = nodes[k]
        env[n] = graph.node_copy(n, env.__getitem__)
    graph.output(tuple(env[nodes[k]] for k in rec["live_outs"]))
    graph.lint()
    return torch.fx.GraphModule({}, graph)


def masked_input(live, kind):
    qk, mask = live
    if kind == "observed":
        return live
    changed = mask.clone()
    if kind == "periodic":
        changed[..., 2::64] = -float("inf")
    elif kind == "prefix":
        changed[..., : (mask.size(-1) + 63) // 64] = -float("inf")
    elif kind == "window":
        changed[..., : mask.size(-1) // 2] = -float("inf")
    else:
        raise ValueError(kind)
    return qk, changed


def mask_control(past, fns, held):
    # Change only the mask, never QK values, shape, compiled object, or reference semantics.
    checks = {}
    inputs = {}
    wins = []
    kinds = ("observed", "periodic", "prefix", "window")
    for kind in kinds:
        checks[kind] = []
        for seed, tensors in zip(SEEDS, held, strict=True):
            ls = masked_input(tensors, kind)
            (ref,) = fns["eager"](*ls)
            out = {k: compare(fns[k](*ls)[0], ref) for k in ("slots", "guarded")}
            checks[kind].append(
                dict(
                    seed=seed,
                    checks=out,
                    mask_sha256=digest(ls[1]),
                    masked_per_batch=ls[1].isneginf().flatten(1).sum(-1).tolist(),
                )
            )
        inputs[kind] = ls
    pairs = [
        (kind, name)
        for kind in kinds
        for name in ("slots", "guarded")
        if all(r["checks"][name]["passed"] for r in checks[kind])
    ]
    rng = random.Random(41519 + past)
    for _ in range(7):
        order = pairs.copy()
        rng.shuffle(order)
        timing = {}
        for kind, name in order:
            timing[kind + ":" + name] = device_time(
                lambda: fns[name](*inputs[kind]), groups=7, unroll=8, replay=3
            )
        wins.append(dict(order=[k + ":" + n for k, n in order], device=timing))
    return dict(
        scope="Counterfactual masks on the captured QK tensors; not the original HF trajectory except observed. Same compiled modules for all masks. Device-only shuffled windows. Prefix and periodic remove similar counts but different positions; not a measurement of GPU branch frequency.",
        checks=checks,
        windows=wins,
        median_us={k: median(w["device"][k]["median_us"] for w in wins) for k in wins[0]["order"]},
        passed=len(pairs) == 2 * len(kinds),
    )


def case(model, rope, past):
    args, _ = trajectory(model, rope, past, EXPORT)[0]
    ep = torch.export.export(model, args)
    gm = ep.module()
    builds, fns = {}, {}
    original = None
    for key in ("slots", "guarded"):
        opt, records = partition_export(
            ep,
            host_backend="native",
            rotary_pairs=True,
            paired_appends=True,
            paired_head_repeats=True,
            schedule_policy=ordered_collectives,
            ordered_scores=key,
        )
        (rec,) = [r for r in records if r.get("module") and "softmax" in r.get("ops", ())]
        mod = opt.get_submodule(rec["module"])
        fx = subgraph(gm, rec)
        if original is None:
            original = fx
        assert str(original.graph) == str(fx.graph)
        fns[key] = mod
        builds[key] = dict(record=rec, **binary(mod.kernel))
    ins, outs = rec["live_ins"], rec["live_outs"]
    assert len(ins) == 2 and len(outs) == 1
    tap = Tap(gm, [*ins, *outs])
    fns["eager"] = original
    fns["inductor"] = torch.compile(original, backend="inductor", fullgraph=True, dynamic=False)
    samples = []
    held = []
    for seed in SEEDS:
        args, ref = trajectory(model, rope, past, seed)[0]
        tap.seen.clear()
        whole = tap.run(*args)
        full = [compare(x, y) for x, y in zip(whole, ref, strict=True)]
        assert all(x["passed"] for x in full) and set(tap.seen) == set(ins + outs)
        live = tuple(tap.seen[k] for k in ins)
        held.append(live)
        expected = tap.seen[outs[0]]
        assert compare(expected, ref[1])["passed"]
        checks = {}
        for name, fn in fns.items():
            try:
                (x,) = fn(*live)
                checks[name] = compare(x, expected)
            except Exception as exc:
                if name != "inductor":
                    raise
                checks[name] = dict(error=type(exc).__name__, passed=False)
        samples.append(
            dict(
                seed=seed,
                live_ins=[desc(x) for x in live],
                probability=desc(expected),
                whole_layer=full,
                methods=checks,
            )
        )
    eligible = [k for k in fns if all(s["methods"][k]["passed"] for s in samples)]
    rng = random.Random(19273 + past)
    windows = []
    for _ in range(7):
        order = eligible.copy()
        rng.shuffle(order)
        win = dict(order=order, device={}, host={})
        for name in order:

            def fn():
                return fns[name](*live)

            win["device"][name] = device_time(fn, groups=7, unroll=8, replay=3)
            win["host"][name] = host_time(fn, samples=21)
        windows.append(win)
    med = {
        m: {k: median(w[m][k]["median_us"] for w in windows) for k in eligible}
        for m in ("device", "host")
    }
    print(past, med, flush=True)
    control = mask_control(past, fns, held)
    return dict(
        mask_control=control,
        prefix=past,
        export_graph_sha256=sha256(str(ep.graph).encode()).hexdigest(),
        fx=str(original.graph),
        live_ins=ins,
        live_outs=outs,
        builds=builds,
        samples=samples,
        windows=windows,
        median_us=med,
        profiles={k: kernel_census(lambda: fns[k](*live)) for k in eligible},
        passed=control["passed"] and {"slots", "guarded", "eager"} <= set(eligible),
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    model, rope = suite(OFFSET)
    doc = dict(
        scope="Score region only, first cached step. Actual QK matmul output and per-batch mask (B2/H12/Q1); not full-layer latency or trained-model token throughput.",
        protocol="Re-extract exact live-in/operation/live-out FX nodes from the original export. Each of five held-out trajectories must match all four stateful HF outputs and strides; the extracted probability must match the HF layer probability. Same layer weights, prefixes, export and held-out seeds as the companion whole-layer report. Reject a backend from timing at a prefix if any BF16 bit or stride differs. Time the last seed in seven shuffled windows; digest contiguous tensor bytes separately from strides.",
        weight_seed=60313 + OFFSET,
        export_seed=EXPORT,
        seeds=SEEDS,
        timing_seed=SEEDS[-1],
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        provenance=capture(ROOT / ".runtime/round-guard/sources"),
        cases=[case(model, rope, n) for n in (1024, 1535, 2047)],
    )
    doc["passed"] = all(c["passed"] for c in doc["cases"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)
    return 0 if doc["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
