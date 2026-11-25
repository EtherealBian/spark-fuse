"""A real Qwen2DecoderLayer's functional single-token boundary; no pretrained weights."""

import torch
from transformers import DynamicCache, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding


class QwenSDPAStep(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = Qwen2DecoderLayer(config, 0)

    def forward(self, x, c, s, k, v, mask=None):
        cache = DynamicCache.from_legacy_cache(((k, v),))
        y = self.layer(
            x,
            attention_mask=mask,
            position_embeddings=(c, s),
            past_key_value=cache,
            use_cache=True,
            output_attentions=False,
        )[0]
        return y, *cache.to_legacy_cache()[0]


def suite(seed):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        hidden_size=512,
        intermediate_size=1152,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=1,
        use_sliding_window=False,
        sliding_window=None,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "sdpa"
    m = QwenSDPAStep(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
    return m, Qwen2RotaryEmbedding(cfg, device="cuda").eval()


def trajectory(m, rope, past, masked, seed, *, steps=2):
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError("expected 1..64 decode steps")
    rng = torch.Generator(device="cuda").manual_seed(seed)
    offset = torch.tensor([[0], [19]], device="cuda")
    state = DynamicCache()
    stages = []

    width = m.layer.input_layernorm.weight.numel()

    def hidden(n):
        return torch.randn(2, n, width, device="cuda", dtype=torch.bfloat16, generator=rng)

    x = hidden(past)
    pos = offset + torch.arange(past, device="cuda")
    cs = rope(x, pos)
    m.layer(
        x, position_embeddings=cs, past_key_value=state, use_cache=True, output_attentions=False
    )
    for s in range(past, past + steps):
        k, v = state.to_legacy_cache()[0]
        x = hidden(1)
        c, si = rope(x, offset + s)
        mask = torch.zeros((2, 1, 1, s + 1), device="cuda", dtype=x.dtype) if masked else None
        if masked:
            mask[0, ..., ::7] = -torch.inf
            mask[1, ..., ::5] = -torch.inf
            if seed % 3 == 0:
                mask[1].fill_(-torch.inf)
        args = (x, c, si, k, v) + ((mask,) if masked else ())
        y = m.layer(
            x,
            attention_mask=mask,
            position_embeddings=(c, si),
            past_key_value=state,
            use_cache=True,
            output_attentions=False,
        )[0]
        stages.append((args, (y, *state.to_legacy_cache()[0])))
    return stages
