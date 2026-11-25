"""ALiBi-style causal positional prior only (no projected QK scores, no value product)."""

import torch
from torch import nn


class CausalPositionPrior(nn.Module):
    def forward(self, slopes, query_pos, key_pos, mask):
        bias = (key_pos - query_pos) * slopes + mask
        return bias.softmax(dim=-1), bias


def inputs(heads, queries, keys, device, seed, offset=0):
    if keys < queries:
        raise ValueError("causal example expects keys >= queries")
    # Small, randomly perturbed positive slopes provide independent calibration requests.
    g = torch.Generator(device=device).manual_seed(seed)
    slopes = torch.exp2(-torch.linspace(1, 8, heads, device=device))
    slopes = slopes * (1 + 0.1 * torch.rand(heads, generator=g, device=device))
    q = torch.arange(keys - queries, keys, device=device, dtype=torch.float32).view(1, queries, 1)
    k = torch.arange(keys, device=device, dtype=torch.float32).view(1, 1, keys)
    mask = torch.zeros(1, queries, keys, device=device).masked_fill_(k > q, -torch.inf)
    args = (slopes.view(heads, 1, 1), q, k, mask)
    if offset:
        args = tuple(
            torch.cat((v.new_empty(offset), v.flatten()))[offset:].view_as(v) for v in args
        )
    return args
