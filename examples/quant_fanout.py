"""Static signed-int8 query/key/value projection example, not a complete attention layer."""

import torch
from torch import nn
from .token_quant import quantize


class QuantFanout(nn.Module):
    """Expose residual, codes, scales, and an independent accumulator/output per projection."""

    def __init__(self, weights, channels):
        super().__init__()
        self.count = len(weights)
        for i, (w, c) in enumerate(zip(weights, channels, strict=True)):
            self.register_buffer(f"weight_{i}", w)
            self.register_buffer(f"channel_{i}", c)

    def forward(self, x, residual):
        z = x + residual
        q, s = quantize(z)
        lhs = q.reshape(-1, q.shape[-1])
        row = s.reshape(-1, 1)
        out = [z, q, s]
        for i in range(self.count):
            acc = torch._int_mm(lhs, getattr(self, f"weight_{i}"))
            y = (acc.float() * row) * getattr(self, f"channel_{i}")
            out.extend((acc, y))
        return tuple(out)
