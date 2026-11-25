"""A real Qwen2Attention eager-attention module, with QK, PV and projections left in PyTorch."""

import argparse, json
from pathlib import Path
import torch, transformers
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RotaryEmbedding
from sparkfuse.partition import partition_export
from sparkfuse.measure import compare, device_time, host_time


@torch.inference_mode()
def fixture(batch, seq):
    cfg = Qwen2Config(
        hidden_size=1536,
        intermediate_size=8960,
        num_attention_heads=12,
        num_key_value_heads=2,
        num_hidden_layers=1,
    )
    cfg._attn_implementation = "eager"
    torch.manual_seed(641)
    dt = torch.bfloat16
    mod = Qwen2Attention(cfg, layer_idx=0).eval().to("cuda", dt)
    rotary = Qwen2RotaryEmbedding(cfg, device="cuda").eval()

    def inputs(seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(batch, seq, 1536, device="cuda", dtype=dt, generator=g)
        pos = torch.arange(seq, device="cuda")[None, :].expand(batch, -1)
        cos, sin = rotary(x, pos)
        # A materialized mask per batch, shared by heads. Hide the first key for
        # alternating batches (except row 0) so wrong batch coordinates change outputs.
        mask = torch.full((batch, 1, seq, seq), float("-inf"), device="cuda", dtype=dt).triu(1)
        mask[1::2, :, 1:, 0] = float("-inf")
        return x, (cos, sin), mask

    return mod, inputs


@torch.inference_mode()
def run(batch, seq, include_inductor):
    dt = torch.bfloat16
    mod, inputs = fixture(batch, seq)
    a = inputs(64)
    ep = torch.export.export(mod, a)
    gm, d = partition_export(ep, host_backend="native")
    assert any("module" in r and any("softmax" in n for n in r["ops"]) for r in d), d
    paths = {"eager": mod, "sparkfuse_mixed": gm}
    if include_inductor:
        paths["inductor"] = torch.compile(mod, dynamic=False, fullgraph=True)
    results = {}
    for label, fn in paths.items():
        rec = results[label] = {"checks": []}
        for seed in [64, 939]:
            xx = inputs(seed)
            ref = mod(*xx)
            out = fn(*xx)
            try:
                rec["checks"].append({"seed": seed, "parity": "pass", "errors": compare(out, ref)})
            except AssertionError as exc:
                rec["checks"].append({"seed": seed, "parity": "fail", "diagnostic": str(exc)})
        if all(x["parity"] == "pass" for x in rec["checks"]):
            rec.update(device=device_time(lambda: fn(*a)), host=host_time(lambda: fn(*a)))
    return dict(
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        dtype=str(dt),
        batch=batch,
        sequence=seq,
        heads=12,
        kv_heads=2,
        random_weights_seed=641,
        attention="eager implementation, explicit additive causal mask, no KV cache/checkpoint/decoder benchmark",
        graph=str(ep.graph),
        mixed_graph=str(gm.graph),
        regions=d,
        backends=results,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=97)
    p.add_argument("--inductor", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rec = run(a.batch, a.seq, a.inductor)
    a.output.write_text(json.dumps(rec, indent=2) + "\n")
    if any(c["parity"] != "pass" for b in rec["backends"].values() for c in b["checks"]):
        raise SystemExit(2)
