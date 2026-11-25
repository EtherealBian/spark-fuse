"""One genuine ATen SDPA export: [B,H,1,D] queries, compact [B,G,S,D] KV."""

import torch
from torch.nn.functional import scaled_dot_product_attention as sdpa


class SingleDecode(torch.nn.Module):
    def __init__(self, gqa=True, scale=None, causal=False, dropout=0.0):
        super().__init__()
        self.gqa = gqa
        self.scale = scale
        self.causal = causal
        self.dropout = dropout

    def forward(self, q, k, v, mask=None):
        return sdpa(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout,
            is_causal=self.causal,
            scale=self.scale,
            enable_gqa=self.gqa,
        )
