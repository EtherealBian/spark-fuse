"""One fixed-capacity export; each batch row supplies an independent live KV length on CUDA."""

import torch
from torch.nn.functional import scaled_dot_product_attention


class RaggedGatherDecode(torch.nn.Module):
    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def forward(self, q, k, v, table, lengths, bias):
        p, g, t, d = k.shape
        b, c = table.shape
        ids = table.clamp(0, p - 1).reshape(-1)

        def bank(x):
            y = (
                x.index_select(0, ids)
                .reshape(b, c, g, t, d)
                .permute(0, 2, 1, 3, 4)
                .reshape(b, g, c * t, d)
            )
            return torch.narrow(y, 2, 0, self.capacity)

        keys, values = bank(k), bank(v)
        pos = torch.arange(self.capacity, device=q.device, dtype=torch.int32)
        inactive = pos >= lengths.reshape(b, 1, 1, 1)
        mask = bias.masked_fill(inactive, -torch.inf)
        return scaled_dot_product_attention(q, keys, values, attn_mask=mask, enable_gqa=True).clone(
            memory_format=torch.contiguous_format
        )


PROFILES = (
    "full",
    "mixed",
    "tiny",
    "empty",
    "tile_edges",
    "extremes",
    "descending",
    "single_active",
)


def row_lengths(b, t, s, profile):
    options = {
        "full": [s],
        "mixed": [s, 1, s // 3, 0],
        "tiny": [0, 1, 9, 17],
        "empty": [0],
        "tile_edges": [t - 1, t, t + 1, 2 * t + 1],
        "extremes": [-(2**31), 2**31 - 1, 1, -11],
        "descending": [s - i * s // b for i in range(b)],
        "single_active": [s] + [0] * (b - 1),
    }
    values = options[profile]
    return [values[i % len(values)] for i in range(b)]


def ragged_request(cfg, seed, profile, table_pattern):
    from .paged_decode_example import request

    q, k, v, table, bias = request(cfg, seed, table_pattern)
    lens = torch.tensor(
        row_lengths(cfg[0], cfg[4], cfg[5], profile), device=q.device, dtype=torch.int32
    )
    return q, k, v, table, lens, bias


def explicit_mask(xs):
    q, k, v, tab, lengths, bias = xs
    s = bias.size(-1)
    m = bias.masked_fill(
        torch.arange(s, device=q.device, dtype=torch.int32) >= lengths.reshape(q.size(0), 1, 1, 1),
        -torch.inf,
    )
    return q, k, v, tab, m
