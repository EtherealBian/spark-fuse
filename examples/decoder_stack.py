"""Real two-/three-layer decoder stacks sharing one functional, externally visible KV state."""

import torch
from transformers import DynamicCache, LlamaConfig, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding


class DecodeStack(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def execute(self, x, c, s, cache, mask):
        for layer in self.layers:
            x = layer(
                x,
                attention_mask=mask,
                position_embeddings=(c, s),
                past_key_value=cache,
                use_cache=True,
                output_attentions=False,
            )[0]
        return x

    def forward(self, x, c, s, *args):
        layers = len(self.layers)
        past = tuple((args[i * 2], args[i * 2 + 1]) for i in range(layers))
        mask = args[-1] if len(args) == 2 * layers + 1 else None
        cache = DynamicCache.from_legacy_cache(past)
        y = self.execute(x, c, s, cache, mask)
        return (y, *(t for kv in cache.to_legacy_cache() for t in kv))


def suite(family, depth, seed):
    torch.manual_seed(seed)
    if family == "qwen2":
        cfg_cls, cls, rope = Qwen2Config, Qwen2DecoderLayer, Qwen2RotaryEmbedding
        extra = dict(use_sliding_window=False, sliding_window=None)
        g = 2
    elif family == "llama":
        cfg_cls, cls, rope = LlamaConfig, LlamaDecoderLayer, LlamaRotaryEmbedding
        extra = {}
        g = 4
    else:
        raise ValueError("unknown family")
    cfg = cfg_cls(
        hidden_size=512,
        intermediate_size=1024,
        num_attention_heads=8,
        num_key_value_heads=g,
        num_hidden_layers=depth,
        attention_dropout=0.0,
        **extra,
    )
    cfg._attn_implementation = "sdpa"
    return DecodeStack([cls(cfg, i) for i in range(depth)]).to(
        device="cuda", dtype=torch.bfloat16
    ).eval(), rope(cfg, device="cuda").eval()


def trajectory(m, rope, past, masked, seed, steps=4):
    rng = torch.Generator(device="cuda").manual_seed(seed)
    offset = torch.tensor([[0], [19]], device="cuda")
    state = DynamicCache()
    out = []

    def hidden(n):
        return torch.randn(2, n, 512, device="cuda", dtype=torch.bfloat16, generator=rng)

    x = hidden(past)
    c, s = rope(x, offset + torch.arange(past, device="cuda"))
    m.execute(x, c, s, state, None)
    for t in range(past, past + steps):
        x = hidden(1)
        c, s = rope(x, offset + t)
        mask = None
        if masked:
            mask = torch.zeros(2, 1, 1, t + 1, dtype=x.dtype, device="cuda")
            mask[0, ..., ::7] = -torch.inf
            mask[1, ..., ::5] = -torch.inf
            if seed % 3 == 0:
                mask[1].fill_(-torch.inf)
        args = (x, c, s, *(v for pair in state.to_legacy_cache() for v in pair)) + (
            (mask,) if masked else ()
        )
        y = m.execute(x, c, s, state, mask)
        out.append((args, (y, *(v for pair in state.to_legacy_cache() for v in pair))))
    return out


def export(m, ex, masked, upper=2048):
    p = torch.export.Dim(
        "past", min=2, max=upper
    )  # `*args` is one tuple in the input tree of dynamic_shapes.
    ds = ({}, {}, {}, tuple([{2: p}] * (2 * len(m.layers)) + ([{3: p + 1}] if masked else [])))
    return torch.export.export(m, ex, dynamic_shapes=ds, strict=True)
