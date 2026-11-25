"""Signed per-token activation quantizer and a framework W8A8 projection consumer."""

import torch
from torch import nn


def quantize(z):
    f = z.float()
    scale = f.abs().amax(-1, keepdim=True).clamp_min(1.0e-5) / 127.0
    q = (f / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


class ResidualQuant(nn.Module):
    def forward(self, x, r):
        z = x + r
        q, scale = quantize(z)
        return z, q, scale


class QuantProjection(nn.Module):
    """Expose the intermediate tensors so boundary and accumulator errors stay observable."""

    def __init__(self, weight, channel_scale):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("channel_scale", channel_scale)

    def forward(self, x, r):
        z = x + r
        q, scale = quantize(z)
        acc = torch._int_mm(q.reshape(-1, q.shape[-1]), self.weight)
        y = (acc.float() * scale.reshape(-1, 1)) * self.channel_scale
        return z, q, scale, acc, y.reshape(*z.shape[:-1], self.weight.shape[-1])
