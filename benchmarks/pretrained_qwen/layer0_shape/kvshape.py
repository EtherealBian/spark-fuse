"""Isolate the first-block K/V projection with frozen BF16 weights; no attention or language head."""

import hashlib
import json
import torch
from torch.nn import functional as fn
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2RotaryEmbedding, rotate_half

ARMS = ("whole", "split", "padded")
STAGES = ("embedding", "normalized", "cos", "sin", "key_linear", "value", "key_rotated")


def need(c, msg):
    if not c:
        raise ValueError(msg)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def witness(x):
    t = x.detach().contiguous().cpu()
    need(t.dtype == torch.bfloat16 and bool(t.isfinite().all()), "BF16 finite required")
    return t.view(torch.uint8).numpy().tobytes()


def plan(n, p, arm):
    need(0 < p < n and arm in ARMS, "invalid split")
    # The padded arm performs independent, token-local projections only. No dummy token enters a cache.
    return (
        [(0, n, n)]
        if arm == "whole"
        else [(0, p, p if arm == "split" else n), (p, n, n - p if arm == "split" else n)]
    )


def probe(x, p, arm, norm, rope, w, head_dim):
    need(x.ndim == 3 and x.shape[0] == 1 and x.dtype == torch.bfloat16, "batch-one BF16 rows")
    n = x.shape[1]
    selected = (p - 1, p, n - 1)
    need(len(set(selected)) == 3, "three unique positions required")
    observed = {k: {} for k in STAGES}
    counts = []
    for lo, hi, rows in plan(n, p, arm):
        # Every arm owns contiguous storage. Padding goes after real rows, not in front of them.
        z = x.new_zeros(1, rows, x.shape[-1])
        z[:, : hi - lo].copy_(x[:, lo:hi])
        pos = torch.arange(lo, lo + rows, device=z.device).reshape(1, -1)
        g = norm(z)
        co, si = rope(z, pos)
        key = fn.linear(g, w["k_proj.weight"], w["k_proj.bias"])
        value = fn.linear(g, w["v_proj.weight"], w["v_proj.bias"])
        kg = key.view(1, rows, -1, head_dim)
        cc = co.unsqueeze(2)
        ss = si.unsqueeze(2)
        rot = kg * cc + rotate_half(kg) * ss
        stages = dict(
            embedding=z, normalized=g, cos=co, sin=si, key_linear=key, value=value, key_rotated=rot
        )
        for j in selected:
            if lo <= j < hi:
                for k, v in stages.items():
                    observed[k][j] = v[0, j - lo].reshape(-1)
        counts.append(rows)
    return {
        k: torch.stack([obs[i] for i in selected]).contiguous() for k, obs in observed.items()
    }, counts


def official(wdir, ids):
    from checkpoint32b import SPEC, REV, MODEL
    from safetensors import safe_open
    from transformers import Qwen2Config

    names = ["model.embed_tokens.weight", "model.layers.0.input_layernorm.weight"] + [
        "model.layers.0.self_attn." + s
        for s in ("k_proj.weight", "k_proj.bias", "v_proj.weight", "v_proj.bias")
    ]
    locks = SPEC["files"]
    receipts = {}

    def check(k):
        m = locks[k]
        p = wdir / k
        need(p.stat().st_size == m["size"], "weight file size")
        if "sha256" in m:
            with p.open("rb") as f:
                out = hashlib.file_digest(f, "sha256").hexdigest()
            need(out == m["sha256"], "shard hash")
            receipts[k] = dict(sha256=out, bytes=m["size"])
        else:
            b = p.read_bytes()
            g = hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()
            need(g == m["git_blob"], "config/index hash")
            receipts[k] = dict(git_blob=g, bytes=m["size"])

    for k in ["model.safetensors.index.json", "config.json"]:
        check(k)
    index = json.loads((wdir / "model.safetensors.index.json").read_bytes())["weight_map"]
    shards = {index[k] for k in names}
    need(shards == {"model-00001-of-00017.safetensors"}, "unexpected touched shards")
    shard = next(iter(shards))
    check(shard)
    cfg = Qwen2Config.from_json_file(wdir / "config.json")
    need(
        (cfg.hidden_size, cfg.num_key_value_heads, cfg.num_attention_heads, cfg.rms_norm_eps)
        == (5120, 8, 40, 1e-6),
        "geometry",
    )
    norm = Qwen2RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(
        device="cuda", dtype=torch.bfloat16
    )
    rope = Qwen2RotaryEmbedding(cfg, device="cuda")
    with safe_open(wdir / shard, framework="pt", device="cpu") as f:
        e = f.get_slice(names[0])
        need(e.get_shape() == [152064, 5120], "embedding extent")
        unique = sorted(set(ids))
        need(all(type(i) is int and 0 <= i < 152064 for i in unique), "token out of bounds")
        table = torch.cat([e[i : i + 1] for i in unique], dim=0).cuda()
        norm.weight.copy_(f.get_tensor(names[1]))
        w = {k.rsplit("self_attn.", 1)[1]: f.get_tensor(k).cuda() for k in names[2:]}
    need(
        all(t.dtype == torch.bfloat16 for t in (table, norm.weight, *w.values()))
        and all(
            tuple(w[k].shape) == ((1024, 5120) if k.endswith(".weight") else (1024,)) for k in w
        ),
        "wrong K/V params",
    )
    return (
        dict(
            model=MODEL,
            revision=REV,
            verified_files=receipts,
            selected_embedding_rows=len(unique),
            inv_freq_dtype=str(rope.inv_freq.dtype),
        ),
        {x: i for i, x in enumerate(unique)},
        table,
        norm,
        rope,
        w,
    )
