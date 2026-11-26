"""Inference-only Qwen gate/up: one owning tensor, also used for separate prefill GEMMs."""

from hashlib import sha256
import torch
from torch import nn
from torch.nn import functional as F
from projection import join


class SharedBankMLP(nn.Module):
    def __init__(self, mlp, fused):
        super().__init__()
        self.register_buffer("bank", join(mlp))
        self.down_proj, self.act_fn, self.fused = mlp.down_proj, mlp.act_fn, fused

    def forward(self, x):
        if x.ndim == 3 and x.shape[:2] == (1, 1):
            gu = F.linear(x, self.bank)
            g, u = gu.chunk(2, -1)
            w = g.shape[-1]
            out = self.fused(g.reshape(1, w), u.reshape(1, w)).reshape(1, 1, w)
        else:
            g, u = self.bank.chunk(2, 0)
            out = self.act_fn(F.linear(x, g)) * F.linear(x, u)
        return self.down_proj(out)


def resident(model):
    # Replace one layer at a time. Never keep a list of original MLPs or banks.
    from examples.graphs import Gate
    from sparkfuse import compile_export
    from sparkfuse.native import native_dispatch

    if model.training or model.config.model_type != "qwen2" or model.config.hidden_act != "silu":
        raise ValueError("requires a frozen eval Qwen2 with SiLU")
    p = next(model.parameters())
    a = p.new_zeros((1, model.config.intermediate_size))
    k = compile_export(torch.export.export(Gate(), (a, torch.empty_like(a))), device=a.device)
    fuse = native_dispatch(k)
    for layer in model.model.layers:
        layer.mlp = SharedBankMLP(layer.mlp, fuse)
    return sha256(k.source.encode()).hexdigest()
