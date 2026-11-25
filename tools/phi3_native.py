"""Unmodified HF Phi3DecoderLayer with its native packed QKV and full/partial RoPE.

Compare the same exported graph before/after sliced_rotary, and carry each path's
own prefill K/V into decode and its first update into the next update.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn
import transformers
from transformers import DynamicCache, Phi3Config
from transformers.models.phi3.modeling_phi3 import (
    Phi3DecoderLayer,
    Phi3RotaryEmbedding,
    apply_rotary_pos_emb,
)
from kernel_trace import kernel_census
from mlp_projection_audit import compare, digest, measure
from provenance import capture
from qwen_decoder import stats
from sparkfuse.partition import partition_export, ordered_collectives

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("prefill", "decode1", "decode2")
LABELS = ("hidden", "probabilities", "key", "value")


class Step(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, c, s, mask, *kv):
        state = DynamicCache.from_legacy_cache((tuple(kv),)) if kv else DynamicCache()
        y, p = self.layer(
            x,
            attention_mask=mask,
            position_embeddings=(c, s),
            past_key_value=state,
            use_cache=True,
            output_attentions=True,
        )
        return y, p, *state.to_legacy_cache()[0]


def suite(offset, rd):
    torch.manual_seed(60313 + offset)
    cfg = Phi3Config(
        hidden_size=1536,
        intermediate_size=4096,
        num_attention_heads=12,
        num_key_value_heads=2,
        num_hidden_layers=1,
        max_position_embeddings=4096,
        original_max_position_embeddings=4096,
        partial_rotary_factor=rd / 128,
        attention_dropout=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    cfg._attn_implementation = "eager"
    m = Step(Phi3DecoderLayer(cfg, 0)).eval()
    for n in (m.layer.input_layernorm, m.layer.post_attention_layernorm):
        n.weight.normal_(1, 0.18)
    m = m.to("cuda", torch.bfloat16)
    return m, Phi3RotaryEmbedding(cfg, device="cuda")


def reference(m, rope, n, seed):
    """One actual eager layer, one persistent HF cache and one projection hook for all three calls."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dt = torch.bfloat16
    pos = torch.tensor([[0], [13]], device="cuda")
    out = []
    state = DynamicCache()
    kv = ()
    wide = []
    hook = m.layer.self_attn.qkv_proj.register_forward_hook(lambda mod, args, y: wide.append(y))
    try:
        for i, t in enumerate((n, 1, 1)):
            x = torch.randn(2, t, 1536, device="cuda", dtype=dt, generator=g)
            c, s = rope(x, pos + (torch.arange(t, device="cuda")[None, :] if not i else n + i - 1))
            if i:
                mask = torch.zeros(2, 1, 1, n + i, device="cuda", dtype=dt)
                mask[1, :, :, 0] = float("-inf")
                mask[0, :, :, 1] = float("-inf")
            else:
                mask = torch.full((2, 1, n, n), float("-inf"), device="cuda", dtype=dt).triu(1)
                mask[1, :, 1:, 0] = float("-inf")
            args = (x, c, s, mask, *kv)
            y, p = m.layer(
                x,
                attention_mask=mask,
                position_embeddings=(c, s),
                past_key_value=state,
                use_cache=True,
                output_attentions=True,
            )
            kv = state.to_legacy_cache()[0]
            out.append((args, (y, p, *kv)))
    finally:
        hook.remove()
    assert len(wide) == 3
    return out, wide


class Epilogue(nn.Module):
    def forward(self, x, c, s):
        q, k, v = (
            u.view(*x.shape[:2], -1, 128).transpose(1, 2)
            for u in (x[..., :1536], x[..., 1536:1792], x[..., 1792:])
        )
        q, k = apply_rotary_pos_emb(q, k, c, s)
        return q, k, v


def paths(mod, args, verify, emit, layer):
    ep = torch.export.export(mod, args)
    paths = {"eager": mod}
    graphs = {}
    kw = (
        dict(
            ordered_scores="slots",
            schedule_policy=ordered_collectives,
            packed_gates=True,
            rotary_pairs=True,
            paired_appends=True,
            paired_head_repeats=True,
        )
        if layer
        else {}
    )
    for n in ("base", "sliced"):
        gm, rs = partition_export(ep, sliced_rotary=n == "sliced", host_backend="native", **kw)
        gm.graph.lint()
        paths[n] = gm
        hits = [z for z in rs if z.get("kind") == "sliced_rotary_pair"]
        assert len(hits) == int(n == "sliced")
        assert not layer or sum(z.get("kind") == "packed_swiglu" for z in rs) == 1
        if hits:
            (hit,) = hits
            src = gm.get_submodule(hit["module"]).program.source
            key = sha256(src.encode()).hexdigest()
            assert key == hit["cuda_sha256"]
            if emit:
                emit.mkdir(parents=True, exist_ok=True)
                (emit / (key + ".cu")).write_text(src)
        graphs[n] = dict(fx=str(gm.graph), records=rs)
    if not verify:
        paths["inductor"] = torch.compile(
            ep.module(), backend="inductor", fullgraph=True, dynamic=False
        )
    return paths, dict(aten=str(ep.graph), variants=graphs)


def ptr(z):
    return z.untyped_storage().data_ptr()


def bits(z):
    return z.contiguous().view(torch.int16)


def check_layer(y, ref, args, saved, strict):
    ds = {n: stats(z, r) for n, z, r in zip(LABELS, y, ref, strict=True)}
    fresh = len({ptr(z) for z in y}) == len(y) and not {ptr(z) for z in y} & {ptr(z) for z in args}

    # Even when V from the first step is a noncompact view, its offset and storage
    # span are observable via the returned cache. Only benchmark exact backends if retained.
    def meta(out):
        return [
            dict(
                stride=list(z.stride()),
                offset=z.storage_offset(),
                storage_bytes=z.untyped_storage().nbytes(),
            )
            for z in out
        ]

    actual = meta(y)
    expected = meta(ref)
    unchanged = all(torch.equal(bits(z), bits(b)) for z, b in zip(args[4:], saved, strict=True))
    prefix = (
        all(torch.equal(bits(z[:, :, : b.shape[2]]), bits(b)) for z, b in zip(y[-2:], saved))
        if saved
        else True
    )
    masked = int(y[1][args[3].isneginf().expand_as(y[1])].count_nonzero())
    rowerr = float((y[1].float().sum(-1) - 1).abs().max())
    passed = (
        fresh
        and unchanged
        and prefix
        and masked == 0
        and rowerr <= 0.004
        and (not strict or actual == expected)
        and all(d["tolerance_pass"] and (not strict or d["unequal_bits"] == 0) for d in ds.values())
    )
    return dict(
        outputs=ds,
        metadata=actual,
        expected_metadata=expected,
        distinct_outputs=fresh,
        input_cache_unchanged=unchanged,
        prefix_exact=prefix,
        masked_nonzero=masked,
        max_row_sum_error=rowerr,
        passed=passed,
    )


def check_epi(y, ref, args):
    wide = args[0]
    ds = [compare(z, r) for z, r in zip(y, ref, strict=True)]
    v = y[-1]
    alias = ptr(v) == ptr(wide) and v.storage_offset() == ref[-1].storage_offset()
    fresh = len({ptr(z) for z in (*args, *y[:2])}) == len({ptr(z) for z in args}) + 2
    return dict(
        tensors=ds,
        visible_v_alias=alias,
        fresh_qk=fresh,
        passed=alias
        and fresh
        and all(
            z["stride_match"]
            and z["unequal_bits"] == 0
            and z["layout"]["storage_offset"] == r.storage_offset()
            for z, r in zip(ds, ref, strict=True)
        ),
    )


def perform(m, rope, rd, n, offset, verify, emit):
    orig, wide = reference(m, rope, n, 60401 + offset)
    stages = []
    epis = []
    audit = []
    epi = Epilogue()
    for (a, _), w in zip(orig, wide, strict=True):
        fs, gr = paths(m, a, verify, emit, True)
        ei, ge = paths(epi, (w, *a[1:3]), verify, emit, False)
        stages.append(fs)
        epis.append(ei)
        audit.append(dict(layer=gr, isolated=ge))
    seeds = [offset + s for s in (60527, 60637, 60733, 60869, 60943)]
    panel = {name: [] for name in fs}
    isos = [[] for _ in STAGES]
    for seed in seeds:
        real, ww = reference(m, rope, n, seed)
        for name, trajectories in panel.items():
            checks = []
            kv = ()
            for (a, ref), fns in zip(real, stages, strict=True):
                args = (*a[:4], *kv)
                saved = tuple(z.clone() for z in kv)
                try:
                    y = fns[name](*args)
                    checks.append(check_layer(y, ref, args, saved, name != "inductor"))
                    kv = y[-2:]
                except Exception as exc:
                    if name != "inductor":
                        raise
                    checks.append(dict(passed=False, error=type(exc).__name__))
                    break
            trajectories.append(
                dict(
                    seed=seed,
                    checks=checks,
                    passed=len(checks) == 3 and all(x["passed"] for x in checks),
                )
            )
        if seed in (seeds[0], seeds[2], seeds[4]):
            for i, ((a, result), w, ef) in enumerate(zip(real, ww, epis, strict=True)):
                inp = (w, *a[1:3])
                before = [digest(z) for z in inp]
                ref = epi(*inp)
                cnt = w.shape[1]
                cache = [
                    compare(z, h[:, :, -cnt:]) for z, h in zip(ref[1:], result[-2:], strict=True)
                ]
                assert all(v["unequal_bits"] == 0 for v in cache)
                methods = {}
                for name, fn in ef.items():
                    try:
                        d = check_epi(fn(*inp), ref, inp)
                        d["input_unchanged"] = [digest(z) for z in inp] == before
                        d["passed"] &= d["input_unchanged"]
                        methods[name] = d
                    except Exception as exc:
                        if name != "inductor":
                            raise
                        methods[name] = dict(passed=False, error=type(exc).__name__)
                isos[i].append(
                    dict(seed=seed, input_sha256=before, cache_suffix=cache, methods=methods)
                )
    allowed = [n for n, v in panel.items() if all(z["passed"] for z in v)]
    details = []
    for i, ((a, _), w, fs, ef) in enumerate(zip(real, ww, stages, epis, strict=True)):
        ok = [n for n in ef if all(z["methods"][n]["passed"] for z in isos[i])]
        einputs = (w, *a[1:3])
        layer_time = {}
        epi_time = {}
        layer_prof = {}
        if not verify:
            funcs = {n: (lambda n=n: fs[n](*a)) for n in allowed}
            if i != 0 or w.shape[1] <= 127:
                layer_time = measure(funcs, offset + rd + n + i)
            else:
                layer_prof = {n: kernel_census(f) for n, f in funcs.items()}
            if i < 2:
                epi_time = measure(
                    {n: (lambda n=n: ef[n](*einputs)) for n in ok}, offset + rd + n + i + 1
                )
        details.append(
            dict(
                stage=STAGES[i],
                graphs=audit[i],
                common_inputs_sha256=[digest(z) for z in a],
                layer=layer_time,
                profile_only=layer_prof,
                isolated=dict(rows=isos[i], **epi_time),
            )
        )
    passed = all(n in allowed for n in ("eager", "base", "sliced")) and all(
        v["methods"][n]["passed"] for ss in isos for v in ss for n in ("eager", "base", "sliced")
    )
    print(
        rd,
        n,
        passed,
        allowed,
        [h["layer"].get("median_us", {}).get("device") for h in details],
        flush=True,
    )
    return dict(
        rotary_dim=rd,
        prefix=n,
        seeds=seeds,
        trajectories=panel,
        admitted_layer=allowed,
        stages=details,
        passed=passed,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-offset", type=int, default=124093)
    p.add_argument("--past", nargs="+", type=int, default=[127, 1024])
    p.add_argument("--rotary-dim", nargs="+", type=int, default=[64, 128])
    p.add_argument("--emit-dir", type=Path)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if any(x < 2 for x in a.past) or any(x < 2 or x > 128 or x % 2 for x in a.rotary_dim):
        p.error("prefix >=2 and even rotary width in [2,128] required")
    result = dict(
        protocol="Native random-weight Phi3DecoderLayer in Transformers eager-attention mode. No parameter conversions or model rewriting. Both hybrids share the same input export, norm/softmax, packed MLP and other passes. Five unseen prefill->decode->decode trajectories; each path receives its own earlier cache, compared against a persistent HF DynamicCache. Four outputs and their metadata, old cache, masked probabilities, plus a separate check of all native packed epilogue bits, V alias and real cache suffixes. Layer Inductor uses existing tolerance; others exact. Only passing backends timed on common last-seed inputs, 7 shuffled windows for both device replay and synchronized host; full prefill >127 is profiled but not timed; no model throughput claim.",
        provenance=capture(ROOT / ".runtime/phi3-native/sources"),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        weight_seed=60313 + a.seed_offset,
        export_seed=60401 + a.seed_offset,
        verify_only=a.verify_only,
        weights={},
        cases=[],
    )
    for rd in a.rotary_dim:
        m, rope = suite(a.seed_offset, rd)
        result["weights"][rd] = {k: digest(v) for k, v in m.named_parameters()}
        for n in a.past:
            result["cases"].append(
                perform(m, rope, rd, n, a.seed_offset, a.verify_only, a.emit_dir)
            )
    result["passed"] = all(c["passed"] for c in result["cases"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
    tmp.replace(a.output)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
