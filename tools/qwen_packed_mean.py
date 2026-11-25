"""Change exactly one native schedule in an exported Qwen2 decoder layer; measure both scopes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random

import torch
import transformers
from packed_mean_panel import digest, key, metrics, timed_windows
from paired_timing import measure
from provenance import capture
from qwen_decoder import Block, Qwen2Config, Qwen2RotaryEmbedding, stats, witness
from sparkfuse import compile_export
from sparkfuse.native import native_dispatch
from sparkfuse.packed_mean import match
from sparkfuse.partition import ordered_collectives, partition_export
from sparkfuse.tuning import candidates


def make_model(weight_seed, seq):
    cfg = Qwen2Config(
        hidden_size=1536,
        intermediate_size=8960,
        num_attention_heads=12,
        num_key_value_heads=2,
        num_hidden_layers=1,
        sliding_window=None,
        use_sliding_window=False,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(weight_seed)
    model = Block(cfg).eval()
    for norm in (model.layer.input_layernorm, model.layer.post_attention_layernorm):
        norm.weight.normal_(1, 0.18)
    model.to("cuda", torch.bfloat16)
    rotary = Qwen2RotaryEmbedding(cfg, device="cuda")

    def make(seed):
        x = torch.randn(
            2,
            seq,
            1536,
            device="cuda",
            dtype=torch.bfloat16,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        positions = torch.arange(seq, device="cuda")[None, :] + torch.tensor(
            [[0], [13]], device="cuda"
        )
        c, s = rotary(x, positions)
        mask = torch.full((2, 1, seq, seq), float("-inf"), device="cuda", dtype=x.dtype).triu(1)
        mask[1, :, 1:, 0] = float("-inf")
        return x, c, s, mask

    return model, make


def observed(aten, hit, args):
    # No generated kernels run in this interpreter. Preserve real input strides and weights.
    names = set(hit["live_ins"] + hit["live_outs"])
    found = {}

    class Tap(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if node.name in names:
                found[node.name] = value
            return value

    layer = Tap(aten).run(*args)
    assert set(found) == names
    return (
        tuple(found[n] for n in hit["live_ins"]),
        tuple(found[n] for n in hit["live_outs"]),
        layer,
    )


def two(out, ref, names):
    assert len(out) == len(ref) == len(names) == 2
    return {name: stats(v, t) for name, v, t in zip(names, out, ref, strict=True)}


def exact(row):
    return all(v["finite"] and v["tolerance_pass"] and v["unequal_bits"] == 0 for v in row.values())


def whole_check(out, ref, mask):
    observed = two(out, ref, ("hidden", "probabilities"))
    m = mask.isneginf().expand_as(ref[1])
    assert bool(m[0].any()) and bool(m[1, :, 1:, 0].all())
    nz = int(out[1][m].count_nonzero())
    err = float((out[1].float().sum(-1) - 1).abs().max())
    return dict(
        values=observed,
        masked_nonzero=nz,
        max_row_sum_error=err,
        passed=exact(observed) and nz == 0 and err <= 0.004,
    )


def compile_modes(m):
    return {
        "inductor": torch.compile(m, fullgraph=True, dynamic=False),
        "inductor_casts": torch.compile(
            m, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
        ),
    }


def compare_records(left, right, target):
    def ignore_target_schedule(rec):
        return [
            {k: v for k, v in r.items() if not (r.get("module") == target and k == "schedule")}
            for r in rec
        ]

    assert len(left) == len(right) and ignore_target_schedule(left) == ignore_target_schedule(right)
    return sha256(json.dumps(ignore_target_schedule(left), sort_keys=True).encode()).hexdigest()


def case(seq, offset, fixed, verify):
    torch.compiler.reset()
    weight = 405173 + offset
    seed = 503119 + seq * 991 + offset
    model, make = make_model(weight, seq)
    ep = torch.export.export(model, make(seed))
    aten = ep.module()

    def partition(policy):
        return partition_export(
            ep,
            host_backend="native",
            schedule_policy=policy,
            split_barriers=True,
            rotary_pairs=True,
        )

    base, records = partition(ordered_collectives)
    (hit,) = [
        r
        for r in records
        if "module" in r
        and (k := getattr(base.get_submodule(r["module"]), "kernel", None)) is not None
        and match(k.plan) is not None
    ]
    assert (
        hit["split"]["reason"] == "fallback_barrier"
        and len(hit["live_ins"]) == 3
        and len(hit["live_outs"]) == 2
    )
    kernel = base.get_submodule(hit["module"]).kernel
    plan = kernel.plan
    region = torch.fx.GraphModule({}, kernel.ep.graph)
    roles = match(plan)
    assert roles.residual == 0 and roles.result == 1 and plan.domain == (2, seq, 1536)
    grid = {key(s): s for s in candidates(plan)}
    prior = {k for k, s in grid.items() if s.layout != "mean_packed_tile"}
    tile = grid.keys() - prior
    assert len(prior) == 10 and len(tile) == 9
    ex_hash = sha256(str(ep.graph).encode()).hexdigest()
    if fixed:
        assert (
            fixed["sequence"] == seq
            and fixed["export_sha256"] == ex_hash
            and fixed["region"] == hit
        )
        chosen = dict(fixed["selected"])
    elif verify:
        chosen = {"previous": key(kernel.schedule), "packed": min(tile)}
    else:
        chosen = None
    needed = sorted(grid if chosen is None else chosen.values())
    bins = {k: compile_export(kernel.ep, schedule=grid[k]) for k in needed}
    natives = {k: native_dispatch(v) for k, v in bins.items()}
    calibration = None
    if chosen is None:
        a, ref, _ = observed(aten, hit, make(seed + 733))
        paths = {k: (lambda f=f: f(*a)) for k, f in natives.items()}
        values = {k: two(f(), ref, ("residual", "normalized")) for k, f in paths.items()}
        assert all(map(exact, values.values()))
        windows = timed_windows(paths, random.Random(seed + 23), 3)
        med = metrics(windows)
        chosen = {
            "previous": min(sorted(prior), key=med.__getitem__),
            "packed": min(sorted(tile), key=med.__getitem__),
        }
        calibration = dict(
            seed=seed + 733,
            input_sha256=[digest(t) for t in a],
            values=values,
            windows=windows,
            medians=med,
        )
    assert chosen["previous"] in prior and chosen["packed"] in tile
    whole = {"eager": model, "aten": aten, "default": base}
    iso = {"eager": region, "default": native_dispatch(kernel)}
    variants = {}
    signatures = {}
    region_meta = {}
    for name, kid in chosen.items():
        gm, recs = partition(lambda p: grid[kid] if match(p) else ordered_collectives(p))
        gm.graph.lint()
        variants[name] = dict(records=recs, fx_sha256=sha256(str(gm.graph).encode()).hexdigest())
        signatures[name] = compare_records(records, recs, hit["module"])
        assert str(gm.graph) == str(base.graph)
        k = gm.get_submodule(hit["module"]).kernel
        assert k.source == bins[kid].source and k.schedule == grid[kid]
        whole[name] = gm
        iso[name] = natives[kid]
        region_meta[name] = dict(
            schedule=vars(k.schedule),
            cuda_sha256=sha256(k.source.encode()).hexdigest(),
            binary_sha256=sha256((k.path.parent / "kernel.so").read_bytes()).hexdigest(),
        )
    if not verify:
        whole.update(compile_modes(model))
        iso.update(compile_modes(region))
    trials = []
    for delta in (2917, 12007, 25013):
        req = make(seed + delta)
        a, ref, aten_out = observed(aten, hit, req)
        y = model(*req)
        assert whole_check(aten_out, y, req[-1])["passed"]
        values = {k: two(f(*a), ref, ("residual", "normalized")) for k, f in iso.items()}
        layers = {k: whole_check(f(*req), y, req[-1]) for k, f in whole.items()}
        wires = {
            k: witness(model, whole[k], hit["module"], req)
            for k in ("default", "previous", "packed")
        }
        for k in wires:
            assert exact(wires[k]) and exact(values[k]) and layers[k]["passed"], (
                seq,
                k,
                layers[k],
                wires[k],
                values[k],
            )
        trials.append(
            dict(
                seed=seed + delta,
                model_input_sha256=[digest(t) for t in req],
                live_input_sha256=[digest(t) for t in a],
                live_output_sha256=[digest(t) for t in ref],
                model_output_sha256=[digest(t) for t in y],
                isolated=values,
                whole=layers,
                wires=wires,
            )
        )
    allow_iso = [k for k in iso if all(exact(t["isolated"][k]) for t in trials)]
    allow_layer = [k for k in whole if all(t["whole"][k]["passed"] for t in trials)]
    assert {"eager", "default", "previous", "packed"} <= set(allow_iso) & set(allow_layer)
    ti = (
        measure({k: (lambda k=k: iso[k](*a)) for k in allow_iso}, seed + 67103)
        if not verify
        else None
    )
    tw = (
        measure({k: (lambda k=k: whole[k](*req)) for k in allow_layer}, seed + 69809)
        if not verify
        else None
    )
    if tw:
        assert (
            all(ti["profiles"][k]["count"] == 1 for k in ("previous", "packed"))
            and tw["profiles"]["previous"]["count"] == tw["profiles"]["packed"]["count"]
        )
        print(seq, chosen, "isolated", ti["median_us"], "whole", tw["median_us"], flush=True)
    return dict(
        sequence=seq,
        weight_seed=weight,
        export_seed=seed,
        export_sha256=ex_hash,
        weight_sha256={n: digest(w) for n, w in model.named_parameters()},
        region=hit,
        ir=plan.explain(),
        partition_records=records,
        variants=variants,
        aten_graph=str(ep.graph),
        mixed_fx=str(base.graph),
        unchanged_records_sha256=signatures,
        selected=chosen,
        kernels=region_meta,
        calibration=calibration,
        trials=trials,
        allowed={"isolated": allow_iso, "whole": allow_layer},
        isolated=ti,
        whole=tw,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq", type=int, action="append")
    p.add_argument("--selection", type=Path)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    seqs = args.seq or [33, 65, 129]
    if len(set(seqs)) != len(seqs) or not all(8 <= s <= 256 for s in seqs):
        p.error("expected distinct sequence lengths in [8,256]")
    raw = args.selection.read_bytes() if args.selection else None
    src = json.loads(raw) if raw else None
    if args.seed_offset < 0 or (
        raw
        and (
            not args.seed_offset
            or src["sequences"] != seqs
            or src["verify_only"]
            or src["selection_sha256"] is not None
            or src["seed_offset"] == args.seed_offset
        )
    ):
        p.error("frozen run requires an original timed panel on these sequences and fresh seeds")
    res = dict(
        protocol="Random-weight, BF16 Qwen2DecoderLayer, batch 2 with distinct positions and causal/padding masks. Isolated inputs/outputs from unmodified exported ATen. Independently calibrate ten prior and nine packed native schedules in three shuffled device windows. Same exported layer, identical other schedules/records/FX. Freeze selected schedules for new weights/requests. Each scope separately requires bitwise-equal visible outputs across three requests before timing fresh dispatch; whole also verifies dropped mask entries and row sums. Seven shuffled windows, independent device and synchronized host medians; profiler counts outside timers. Whole Inductor internal wires are not observable. No generation-throughput or trained-model claim.",
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        transformers=transformers.__version__,
        sequences=seqs,
        verify_only=args.verify_only,
        seed_offset=args.seed_offset,
        selection_sha256=sha256(raw).hexdigest() if raw else None,
        provenance=capture(args.output.parent / "sources"),
        cases=[],
    )
    for i, s in enumerate(seqs):
        res["cases"].append(
            case(s, args.seed_offset, src["cases"][i] if src else None, args.verify_only)
        )
    args.output.write_text(json.dumps(res, allow_nan=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
