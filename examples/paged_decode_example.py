"""Functional, read-only paged KV attention with an explicit saturating physical-page map."""

import torch
from torch.nn.functional import scaled_dot_product_attention


class PageGatherDecode(torch.nn.Module):
    def __init__(self, sequence):
        super().__init__()
        self.sequence = sequence

    def forward(self, q, keys, values, table, mask=None):
        # Index policy is part of this graph, not inserted by the compiler.
        physical, g, page, d = keys.shape
        b, c = table.shape
        ids = table.clamp(0, physical - 1).reshape(-1)

        def gather(x):
            y = (
                x.index_select(0, ids)
                .reshape(b, c, g, page, d)
                .permute(0, 2, 1, 3, 4)
                .reshape(b, g, c * page, d)
            )
            return torch.narrow(y, 2, 0, self.sequence)

        return scaled_dot_product_attention(
            q, gather(keys), gather(values), attn_mask=mask, enable_gqa=True
        ).clone(memory_format=torch.contiguous_format)


def request(cfg, seed, pattern):
    """Independent physical banks and logical maps, including shared pages and saturating endpoints."""
    b, h, g, d, t, s, mask, kind = cfg
    c = (s + t - 1) // t
    p = b * c + 5
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[kind]
    rng = torch.Generator(device="cuda").manual_seed(seed)

    def rand(sh):
        return torch.randn(sh, device="cuda", dtype=dtype, generator=rng)

    q, k, v = [rand(sh) for sh in ((b, h, 1, d), (p, g, t, d), (p, g, t, d))]
    ids = (
        torch.randperm(p, device="cuda", generator=rng, dtype=torch.int32)[: b * c]
        .reshape(b, c)
        .clone()
    )
    if pattern == "repeated":
        ids.copy_(ids[0:1, :1].clone().expand_as(ids))
    elif pattern == "shared":
        ids[:, : max(1, c // 2)] = ids[0, : max(1, c // 2)].clone()
    elif pattern == "endpoints":
        ids[:, ::2] = torch.iinfo(torch.int32).min
        ids[:, 1::2] = torch.iinfo(torch.int32).max
    elif pattern != "shuffled":
        raise ValueError("unknown page workload")
    if not mask:
        return q, k, v, ids
    bias = rand((mask, 1, 1, s))
    bias[0, ..., ::7] = -torch.inf
    if pattern == "endpoints":
        bias[-1].fill_(-torch.inf)
    return q, k, v, ids, bias


def materialize(xs, sequence):
    q, k, v, table, *mask = xs
    b, c = table.shape
    p, g, t, d = k.shape
    ids = table.clamp(0, p - 1).reshape(-1)
    return (
        q,
        *(
            bank.index_select(0, ids)
            .reshape(b, c, g, t, d)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, g, c * t, d)[:, :, :sequence]
            .contiguous()
            for bank in (k, v)
        ),
        *mask,
    )
