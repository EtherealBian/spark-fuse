import torch
from torch import nn
from torch.nn import functional as F


class ResidualNorm(nn.Module):
    def __init__(self, expose=False):
        super().__init__()
        self.expose = expose

    def forward(self, x, skip, w):
        r = x + skip
        y = F.rms_norm(r, (x.shape[-1],), w, 1e-6)
        return (y, r) if self.expose else y


class ResidualLayerNorm(nn.Module):
    def forward(self, x, skip, weight, bias):
        r = x + skip
        return F.layer_norm(r, (x.shape[-1],), weight, bias, 1e-5), r


class Gate(nn.Module):
    def forward(self, gate, up):
        return F.silu(gate) * up


class DecomposedNorm(nn.Module):
    def forward(self, x, skip, w):
        h = x + skip
        f = h.float()
        return w * (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


class ScoreSoftmax(nn.Module):
    """Attention-score scaling + additive bias + stable softmax; no QK/PV GEMM."""

    def __init__(self, expose=False):
        super().__init__()
        self.expose = expose

    def forward(self, scores, bias):
        logits = scores * 0.125 + bias
        prob = torch.softmax(logits, dim=-1)
        return (prob, logits) if self.expose else prob


class Mean(nn.Module):
    def forward(self, x):
        return x.square().mean(-1, keepdim=True)


def case(name, rows, width, dtype, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*sz):
        return torch.randn(sz, dtype=dtype, device="cuda", generator=g)

    if name == "layernorm2":
        return ResidualLayerNorm(), (rand(rows, width), rand(rows, width), rand(width), rand(width))
    if name == "gate":
        return Gate(), (rand(rows, width), rand(rows, width))
    if name in ("scores", "scores2"):
        return ScoreSoftmax(name == "scores2"), (rand(rows, width), rand(width))
    if name == "mean":
        return Mean(), (rand(rows, width),)
    mod = {
        "resnorm": lambda: ResidualNorm(False),
        "resnorm2": lambda: ResidualNorm(True),
        "decomp": DecomposedNorm,
    }[name]()
    return mod, (rand(rows, width), rand(rows, width), rand(width))
