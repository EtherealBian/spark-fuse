"""A bounded Qwen2 MLP integration. Attention, projections, weights and cache stay in HF."""

from copy import deepcopy
from hashlib import sha1, sha256
from pathlib import Path

import torch
from torch import nn
from examples.graphs import Gate
from sparkfuse import compile_export
from sparkfuse.native import native_dispatch

MODEL = "Qwen/Qwen2-0.5B-Instruct"
REV = "c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
WEIGHTS = "130282af0dfa9fe5840737cc49a0d339d06075f83c5a315c3372c9a0740d0b96"
BLOBS = {
    "config.json": "463b055262b6c66c4629a74a4b300bfe2ed31d3c",
    "tokenizer.json": "33ea6c72ebb92a237fa2bdf26c5ff16592efcdae",
    "tokenizer_config.json": "ff55d7b9eb1384e5d4d7e75dc0f564c1a8833d6e",
}


def verify_assets(directory):
    import hashlib

    d = Path(directory)
    for name, expected in BLOBS.items():
        data = (d / name).read_bytes()
        if sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != expected:
            raise ValueError("not the pinned upstream blob: " + name)
    with (d / "model.safetensors").open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != WEIGHTS:
            raise ValueError("checkpoint checksum mismatch")


class GatedMLP(nn.Module):
    def __init__(self, old, paths):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = old.gate_proj, old.up_proj, old.down_proj
        self.act_fn, self.paths = old.act_fn, paths

    def forward(self, x):
        g, u = self.gate_proj(x), self.up_proj(x)
        f = self.paths.get(g.shape[1]) if g.ndim == 3 and g.shape[0] == 1 else None
        if f is None:
            return self.down_proj(self.act_fn(g) * u)
        # Both operands are fresh dense linear outputs. The native launcher validates layout/dtype.
        return self.down_proj(f(g[0], u[0]).unsqueeze(0))


def specialize(model, lengths):
    cfg = model.config
    if cfg.model_type != "qwen2" or cfg.hidden_act != "silu":
        raise ValueError("requires Qwen2 with SiLU")
    clone = deepcopy(model, {id(t): t for t in model.parameters()})
    paths, sources = {}, {}
    exemplar = next(model.parameters())
    width = cfg.intermediate_size
    for n in sorted(set(lengths)):
        a = exemplar.new_zeros((n, width))
        b = torch.empty_like(a)
        k = compile_export(torch.export.export(Gate(), (a, b)), device=a.device)
        paths[n] = native_dispatch(k)
        sources[str(n)] = sha256(k.source.encode()).hexdigest()
    for layer in clone.model.layers:
        layer.mlp = GatedMLP(layer.mlp, paths)
    assert all(x is y for x, y in zip(model.parameters(), clone.parameters(), strict=True))
    return clone, sources
