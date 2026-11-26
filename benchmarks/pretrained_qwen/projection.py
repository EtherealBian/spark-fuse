"""One-token, batch-one BF16/float projection ablation with immutable weights."""

from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F


def join(mlp):
    a, b = mlp.gate_proj, mlp.up_proj
    if (
        not isinstance(a, nn.Linear)
        or not isinstance(b, nn.Linear)
        or a.bias is not None
        or b.bias is not None
        or a.weight.requires_grad
        or b.weight.requires_grad
        or a.weight.shape != b.weight.shape
        or a.weight.dtype != b.weight.dtype
        or a.weight.device != b.weight.device
    ):
        raise ValueError("requires identical-size frozen biasless linear weights on one device")
    weight = torch.cat((a.weight, b.weight), dim=0)
    left, right = weight.chunk(2, dim=0)
    if not torch.equal(left, a.weight) or not torch.equal(right, b.weight):
        raise ValueError("packing changed weights")
    return weight


class PackedDecodeMLP(nn.Module):
    def __init__(self, mlp, bank, fused=None):
        super().__init__()
        self.fallback = mlp
        self.register_buffer("bank", bank, persistent=False)
        self.fused = fused

    def forward(self, x):
        if x.ndim != 3 or x.shape[:2] != (1, 1):
            return self.fallback(x)
        gu = F.linear(x, self.bank)
        g, u = gu.chunk(2, -1)
        if self.fused is None:
            joined = self.fallback.act_fn(g) * u
        else:
            w = g.shape[-1]
            joined = self.fused(g.reshape(1, w), u.reshape(1, w)).reshape(1, 1, w)
        return self.fallback.down_proj(joined)


def packed(model, banks, fused=None):
    copy = deepcopy(model, {id(t): t for t in model.parameters()})
    for layer, bank in zip(copy.model.layers, banks, strict=True):
        layer.mlp = PackedDecodeMLP(layer.mlp, bank, fused)
    return copy
