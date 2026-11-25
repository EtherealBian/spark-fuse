"""Swap only the three Q/K/V linears of a real Transformers BertLayer for signed W8A8 linears."""

import torch
from torch import nn
from .token_quant import quantize


class TokenLinear(nn.Module):
    """Offline per-output-channel RHS quantization; each public linear independently quantizes x."""

    def __init__(self, linear):
        super().__init__()
        w = linear.weight.detach().float()
        channel = w.abs().amax(1, keepdim=True).clamp_min(1.0e-5) / 127.0
        self.register_buffer("rhs", (w / channel).round().clamp(-127, 127).to(torch.int8).t())
        self.register_buffer("channel", channel.reshape(-1).contiguous())
        self.register_buffer("bias", linear.bias.detach().float().clone())

    def forward(self, x):
        q, s = quantize(x)
        a = torch._int_mm(q.reshape(-1, q.shape[-1]), self.rhs)
        y = ((a.float() * s.reshape(-1, 1)) * self.channel + self.bias).to(x.dtype)
        return y.reshape(*x.shape[:-1], self.bias.shape[0])


class EncoderBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x, residual=None):
        hidden = x if residual is None else x + residual
        return self.block(hidden, output_attentions=True)


def replace_qkv(block):
    attention = block.attention.self
    for key in ("query", "key", "value"):
        setattr(attention, key, TokenLinear(getattr(attention, key)))
    return block


class ObservableStack(nn.Module):
    """Each distinct EncoderBlock keeps its hidden state and attention probabilities observable."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, residual=None):
        hidden = x
        outputs = []
        for i, layer in enumerate(self.layers):
            hidden, probs = (
                layer(hidden, residual) if i == 0 and residual is not None else layer(hidden)
            )
            outputs.extend((hidden, probs))
        return tuple(outputs)
