"""Random-weight, offline HF decoder families using their own projections, RoPE and cache."""

import torch
from transformers import LlamaConfig, MistralConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRotaryEmbedding
from examples.qwen_sdpa_step import QwenSDPAStep, suite as qwen


class FunctionalDecoder(QwenSDPAStep):
    # Share only the functional public cache boundary, not the Qwen2 attention or weights.
    def __init__(self, layer):
        torch.nn.Module.__init__(self)
        self.layer = layer


def suite(family, seed):
    if family == "qwen2":
        return qwen(seed)
    torch.manual_seed(seed)
    if family == "llama":
        cfg = LlamaConfig(
            hidden_size=512,
            intermediate_size=1024,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=1,
            attention_dropout=0.0,
            attention_bias=False,
            mlp_bias=False,
        )
        layer, rope = LlamaDecoderLayer, LlamaRotaryEmbedding
    elif family == "mistral":
        cfg = MistralConfig(
            hidden_size=768,
            intermediate_size=1536,
            num_attention_heads=6,
            num_key_value_heads=2,
            head_dim=128,
            num_hidden_layers=1,
            sliding_window=None,
            attention_dropout=0.0,
        )
        layer, rope = MistralDecoderLayer, MistralRotaryEmbedding
    else:
        raise ValueError("unknown decoder family")
    cfg._attn_implementation = "sdpa"
    return FunctionalDecoder(layer(cfg, 0)).to(device="cuda", dtype=torch.bfloat16).eval(), rope(
        cfg, device="cuda"
    ).eval()
