"""Two real Qwen2 decoder steps with per-backend cache carry, tested against stateful Transformers."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median
import torch
import transformers
from transformers import DynamicCache, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
from provenance import capture
from qwen_decoder import kernel_census, stats
from sparkfuse.measure import device_time, host_time
from sparkfuse.partition import partition_export, ordered_collectives

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("hidden", "probabilities", "key", "value")


class Step(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = Qwen2DecoderLayer(cfg, 0)

    def forward(self, x, c, s, m, k, v):
        cache = DynamicCache.from_legacy_cache(((k, v),))
        y, p = self.layer(
            x,
            attention_mask=m,
            position_embeddings=(c, s),
            past_key_value=cache,
            use_cache=True,
            output_attentions=True,
        )
        return y, p, *cache.to_legacy_cache()[0]


def suite(offset):
    torch.manual_seed(60313 + offset)
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
    model = Step(cfg).eval()
    for n in (model.layer.input_layernorm, model.layer.post_attention_layernorm):
        n.weight.normal_(1, 0.18)
    return model.to("cuda", torch.bfloat16), Qwen2RotaryEmbedding(cfg, device="cuda")


def trajectory(model, rope, past, seed):
    """Only the reference owns a persistent DynamicCache. Its prefix is populated by real projections."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = torch.bfloat16
    dev = "cuda"
    pos0 = torch.tensor([[0], [13]], device=dev)

    def hidden(n):
        return torch.randn((2, n, 1536), dtype=dt, device=dev, generator=g)

    x = hidden(past)
    tables = rope(x, pos0 + torch.arange(past, device=dev)[None, :])
    mask = torch.full((2, 1, past, past), float("-inf"), device=dev, dtype=dt).triu(1)
    mask[1, :, 1:, 0] = float("-inf")
    state = DynamicCache()
    model.layer(
        x,
        attention_mask=mask,
        position_embeddings=tables,
        past_key_value=state,
        use_cache=True,
        output_attentions=False,
    )
    stages = []
    for n in (past, past + 1):
        k, v = state.to_legacy_cache()[0]
        assert k.shape == v.shape == (2, 2, n, 128)
        x = hidden(1)
        c, s = rope(x, pos0 + n)
        mask = torch.zeros((2, 1, 1, n + 1), device=dev, dtype=dt)
        mask[1, :, :, 0] = float("-inf")
        mask[0, :, :, 1] = float("-inf")
        y, p = model.layer(
            x,
            attention_mask=mask,
            position_embeddings=(c, s),
            past_key_value=state,
            use_cache=True,
            output_attentions=True,
        )
        ko, vo = state.to_legacy_cache()[0]
        stages.append(((x, c, s, mask, k, v), (y, p, ko, vo)))
    return stages


def bits(x):
    return x.contiguous().view(torch.int16)


def same(a, b):
    return bool(torch.equal(bits(a), bits(b)))


def check(out, ref, args, before, strict):
    metrics = {key: stats(a, b) for key, a, b in zip(LABELS, out, ref, strict=True)}
    k, v = args[-2:]
    n = k.shape[2]
    ko, vo = out[-2:]
    st = [list(x.stride()) for x in out]
    expected = [list(x.stride()) for x in ref]

    def ptr(x):
        return x.untyped_storage().data_ptr()

    distinct = len({ptr(ko), ptr(vo)}) == 2 and not ({ptr(ko), ptr(vo)} & {ptr(x) for x in args})
    unchanged = all(same(x, y) for x, y in zip((k, v), before, strict=True))
    prefix = all(same(x[:, :, :n], y) for x, y in zip((ko, vo), before, strict=True))
    missing = int(out[1][args[3].isneginf().expand_as(out[1])].count_nonzero())
    rowerr = float((out[1].float().sum(-1) - 1).abs().max())
    ok = (
        distinct
        and unchanged
        and prefix
        and missing == 0
        and rowerr <= 0.004
        and all(
            v["tolerance_pass"] and (not strict or v["unequal_bits"] == 0) for v in metrics.values()
        )
        and (not strict or st == expected)
    )
    return dict(
        outputs=metrics,
        output_strides=st,
        expected_strides=expected,
        cache_input_strides=[list(x.stride()) for x in (k, v)],
        new_storage=distinct,
        input_unchanged=unchanged,
        prefix_exact=prefix,
        masked_nonzero=missing,
        max_row_sum_error=rowerr,
        passed=ok,
    )


@torch.inference_mode()
def run(
    model,
    rope,
    length,
    offset,
    verify_only,
    profile,
    emit_dir,
    head_repeats=False,
    framework_softmax=False,
):
    export = trajectory(model, rope, length, 60401 + offset)
    graphs = []
    paths = []
    for i, (args, ref) in enumerate(export):
        ep = torch.export.export(model, args)
        stage = {"eager": model, "exported_aten": ep.module()}
        details = {}
        variants = [("rotary", False), ("append", True)]
        if head_repeats:
            variants.append(("repeat", True))
        for name, enable in variants:
            gm, records = partition_export(
                ep,
                host_backend="native",
                rotary_pairs=True,
                paired_appends=enable,
                paired_head_repeats=name == "repeat",
                framework_softmax=framework_softmax,
                schedule_policy=ordered_collectives,
            )
            gm.graph.lint()
            assert sum(r.get("reason") == "framework_softmax" for r in records) == int(
                framework_softmax
            )
            stage[name] = gm
            hits = [r for r in records if r.get("kind") == "paired_append"]
            assert len(hits) == int(enable) and any(r.get("kind") == "rotary_pair" for r in records)
            reps = [r for r in records if r.get("kind") == "paired_head_repeat"]
            assert len(reps) == int(name == "repeat")
            if enable:
                if reps:
                    hits = reps
                src = gm.get_submodule(hits[0]["module"]).program.source
                assert sha256(src.encode()).hexdigest() == hits[0]["cuda_sha256"]
                if emit_dir and (not head_repeats or reps):
                    emit_dir.mkdir(parents=True, exist_ok=True)
                    (emit_dir / f"p{length}-step{i + 1}.cu").write_text(src)
            details[name] = dict(fx=str(gm.graph), records=records)
        if not verify_only:
            stage["inductor"] = torch.compile(
                model, backend="inductor", fullgraph=True, dynamic=False
            )
        paths.append(stage)
        graphs.append(dict(aten=str(ep.graph), partitions=details))
    seeds = [s + offset for s in (60527, 60637, 60733, 60869, 60943)]
    results = {k: dict(trajectories=[]) for k in paths[0]}
    control = []
    for seed in seeds:
        steps = trajectory(model, rope, length, seed)
        # Negative controls: touching all visible past V changes hidden but not attention scores;
        # changing just a masked V has no effect on hidden or probabilities.
        args, ref = steps[0]
        x, c, s, m, k, v = args
        target = v.clone()
        target[0, :, 1].fill_(27)
        target[1, :, 0].fill_(27)
        masked = model(x, c, s, m, k, target)
        erased = model(x, c, s, m, k, torch.zeros_like(v))
        ctrl = dict(
            seed=seed,
            masked_hidden_exact=same(masked[0], ref[0]),
            masked_prob_exact=same(masked[1], ref[1]),
            erased_hidden_bits=int((bits(erased[0]) != bits(ref[0])).sum()),
            erased_prob_exact=same(erased[1], ref[1]),
        )
        control.append(ctrl)
        for name, rr in results.items():
            cached = tuple(x.clone() for x in args[-2:])
            traj = []
            for i, (src, ref) in enumerate(steps):
                mixed = (*src[:4], *cached)
                before = tuple(x.clone() for x in cached)
                try:
                    out = paths[i][name](*mixed)
                    row = check(out, ref, mixed, before, name != "inductor")
                    cached = out[-2:]
                except Exception as exc:
                    if name != "inductor":
                        raise
                    row = dict(passed=False, error=type(exc).__name__)
                    traj.append(row)
                    break
                traj.append(row)
            rr["trajectories"].append(
                dict(
                    seed=seed, steps=traj, passed=len(traj) == 2 and all(x["passed"] for x in traj)
                )
            )
    allowed = []
    for name, r in results.items():
        r["passed"] = all(a["passed"] for a in r["trajectories"])
        if r["passed"]:
            allowed.append(name)
    orders = [[], []]
    if not verify_only:
        rng = random.Random(72011 + length)
        for name in allowed:
            results[name]["timing"] = [dict(windows=[]), dict(windows=[])]
        for i, (args, ref) in enumerate(steps):
            for _ in range(7):
                order = allowed.copy()
                rng.shuffle(order)
                orders[i].append(order)
                for name in order:

                    def fn():
                        return paths[i][name](*args)

                    results[name]["timing"][i]["windows"].append(
                        dict(
                            device=device_time(fn, groups=7, unroll=4, replay=3),
                            host=host_time(fn, samples=15),
                        )
                    )
            for name in allowed:
                t = results[name]["timing"][i]
                t.update(
                    {
                        m + "_us": median(w[m]["median_us"] for w in t["windows"])
                        for m in ("device", "host")
                    }
                )
                if profile:
                    t["profile"] = kernel_census(lambda: paths[i][name](*args))
    return dict(
        initial_prefix=length,
        graphs=graphs,
        controls=control,
        orderings=orders,
        backends=results,
        passed=all(
            r["masked_hidden_exact"]
            and r["masked_prob_exact"]
            and r["erased_hidden_bits"] > 0
            and r["erased_prob_exact"]
            for r in control
        )
        and all(results[k]["passed"] for k in results if k != "inductor"),
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--past", type=int, nargs="+", default=[31, 255, 1023])
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument(
        "--head-repeats",
        action="store_true",
        help="also evaluate the two-output materializing head-repeat pass, leaving both GEMMs unchanged",
    )
    p.add_argument(
        "--framework-softmax",
        action="store_true",
        help="keep the original softmax as a PyTorch boundary for every variant",
    )
    p.add_argument("--profile", action="store_true")
    p.add_argument("--emit-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.past) < 2 or len(set(a.past)) != len(a.past):
        p.error("distinct prefix lengths >=2 required")
    if a.profile and a.verify_only:
        p.error("use --profile for measurement only")
    model, rope = suite(a.seed_offset)
    doc = dict(
        scope="Two successive one-token forwards of one actual random-weight BF16 Qwen2DecoderLayer, H1536/I8960, B2, Q12/KV2. Each backend carries its own K/V into step 2; inputs to the layer are independent hidden vectors, not sampled tokens. Separate static export for each length and input layout. No paged/in-place cache or model generation throughput.",
        protocol="Stateful HF DynamicCache reference prefills real K/V. Two masked positions across a batch; phase tables outside timing. Correctness on five trajectories (last supplies the immutable common timing inputs), separate export seed. Seven shuffled windows per step. Admission requires both steps of all five trajectories. Returned cache prefix and input immutability are bit-exact for all; SparkFuse also requires bitwise agreement and matching strides for four outputs. Inductor uses the existing elementwise tolerance. Device graph and synchronized fresh-output host timings separately; profiler outside timing.",
        weight_seed=60313 + a.seed_offset,
        export_seed=60401 + a.seed_offset,
        seeds=[s + a.seed_offset for s in (60527, 60637, 60733, 60869, 60943)],
        timing_seed=60943 + a.seed_offset,
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        verify_only=a.verify_only,
        head_repeats=a.head_repeats,
        framework_softmax=a.framework_softmax,
        profile=a.profile,
        provenance=capture(ROOT / ".runtime/decode-audit/sources"),
        cases=[],
    )
    for length in a.past:
        c = run(
            model,
            rope,
            length,
            a.seed_offset,
            a.verify_only,
            a.profile,
            a.emit_dir,
            a.head_repeats,
            a.framework_softmax,
        )
        doc["cases"].append(c)
        print(
            length,
            c["passed"],
            {k: [x["device_us"] for x in v.get("timing", [])] for k, v in c["backends"].items()},
            flush=True,
        )
    doc["passed"] = all(c["passed"] for c in doc["cases"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    tmp.replace(a.output)
    return 0 if doc["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
