"""Functional cache write/read on private batch banks. Both positions are per-row CUDA data."""

import torch


class CacheSlot(torch.nn.Module):
    def __init__(self, visible=(False, False)):
        super().__init__()
        self.visible = visible

    def forward(self, k, v, kt, vt, write, read):
        b, g, cap, d = k.shape

        def indices(x):
            return x.clamp(0, cap - 1).long().reshape(b, 1, 1, 1).expand(b, g, 1, d)

        wi, ri = indices(write), indices(read)
        ko = k.scatter(2, wi, kt)
        vo = v.scatter(2, wi, vt)
        return (
            ko.gather(2, ri),
            vo.gather(2, ri),
            *((ko,) if self.visible[0] else ()),
            *((vo,) if self.visible[1] else ()),
        )


def request(cfg, seed, pattern):
    b, g, s, d, kind = cfg
    r = torch.Generator(device="cuda").manual_seed(seed)
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[kind]

    def bits(shape):
        return torch.randint(
            -32768, 32768, shape, device="cuda", dtype=torch.int16, generator=r
        ).view(dt)

    # Include NaN payloads, infinities and signed zeros, not just finite training-like values.
    k, v, kt, vt = [bits(x) for x in ((b, g, s, d), (b, g, s, d), (b, g, 1, d), (b, g, 1, d))]
    w = torch.randint(0, s, (b,), device="cuda", dtype=torch.int32, generator=r)
    q = torch.randint(0, s, (b,), device="cuda", dtype=torch.int32, generator=r)
    if pattern == "equal":
        q.copy_(w)
    elif pattern == "different":
        q.copy_((w + 1) % s)
    elif pattern == "mixed":
        q[::2] = w[::2]
    elif pattern == "endpoints":
        w[::2] = -2147483648
        w[1::2] = 2147483647
        q[:] = 2147483647
    else:
        raise ValueError("unknown index pattern")
    return k, v, kt, vt, w, q
