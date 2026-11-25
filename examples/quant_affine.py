"""Biased signed Q/K/V projections with explicit ordered FP32 dequantization and narrow outputs."""

import torch
from .quant_fanout import QuantFanout
from .token_quant import quantize


class QuantAffineFanout(QuantFanout):
    def __init__(self, weights, channels, biases, output_dtypes, *, debug_outputs=False):
        super().__init__(weights, channels)
        if len(weights) != len(biases) or len(weights) != len(output_dtypes):
            raise ValueError("one bias and narrow dtype per projection")
        self.output_dtypes = tuple(output_dtypes)
        self.debug_outputs = debug_outputs
        for i, b in enumerate(biases):
            self.register_buffer(f"bias_{i}", b)

    def forward(self, x, r):
        z = x + r
        codes, scale = quantize(z)
        a = codes.reshape(-1, codes.shape[-1])
        row = scale.reshape(-1, 1)
        outputs = [z, codes, scale] if self.debug_outputs else []
        for i in range(self.count):
            acc = torch._int_mm(a, getattr(self, f"weight_{i}"))
            f = acc.float()
            t = f * row
            v = t * getattr(self, f"channel_{i}")
            b = v + getattr(self, f"bias_{i}")
            y = b.to(self.output_dtypes[i])
            outputs.extend((acc, f, t, v, b, y) if self.debug_outputs else (y,))
        return tuple(outputs)
