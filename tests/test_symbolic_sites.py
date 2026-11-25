import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention
from sparkfuse.symbolic_sites import partition_symbolic_decode
from sparkfuse.ir import Unsupported


class Step(torch.nn.Module):
    def __init__(self, *, public_repeat=False, expose=False, dropout=0.0):
        super().__init__()
        self.linear = torch.nn.Linear(512, 128, bias=False)
        self.public_repeat = public_repeat
        self.expose = expose
        self.dropout = dropout

    def forward(self, q, old_k, old_v, new_k, new_v, mask):
        k = torch.cat((old_k, new_k), 2)
        v = torch.cat((old_v, new_v), 2)
        s = k.shape[2]

        def rep(x):
            return x.unsqueeze(2).expand(2, 2, 4, s, 64).reshape(2, 8, s, 64)

        a, b = rep(k), rep(v)
        y = scaled_dot_product_attention(q, a, b, attn_mask=mask, dropout_p=self.dropout)
        h = self.linear(y.transpose(1, 2).reshape(2, 1, -1))
        return (h, k, v) + ((a,) if self.public_repeat else ()) + ((y,) if self.expose else ())


def req(s):
    return tuple(
        torch.randn(sh, dtype=torch.bfloat16, device="cuda")
        for sh in (
            (2, 8, 1, 64),
            (2, 2, s, 64),
            (2, 2, s, 64),
            (2, 2, 1, 64),
            (2, 2, 1, 64),
            (2, 1, 1, s + 1),
        )
    )


def export(m, xs):
    p = torch.export.Dim("past", min=2, max=129)
    return torch.export.export(
        m, xs, dynamic_shapes=({}, {2: p}, {2: p}, {}, {}, {3: p + 1}), strict=True
    )


@pytest.mark.parametrize("public,domain", [(False, "checked"), (True, "bounded")])
@torch.inference_mode()
def test_one_layer_region_preserves_caches_and_live_repeat(public, domain):
    m = Step(public_repeat=public).to(device="cuda", dtype=torch.bfloat16).eval()
    ex, au = req(7), req(51)
    ep = export(m, ex)
    graph = str(ep.graph)
    part = partition_symbolic_decode(ep, ex, au, domain=domain)
    assert len(part.sites) == 1, part.skipped
    rec = part.sites[0]
    assert (
        rec["plan"]["lower"] == 3
        and rec["plan"]["upper"] == 130
        and rec["offset_from_symbol"] == 1
        and len(rec["erased_repeats"]) == 2 - int(public)
    )
    outputs = []
    storage = set()
    for s in (2, 31, 32, 61, 128, 129):
        xs = req(s)
        xs[-1][0].fill_(-torch.inf)
        ref = m(*xs)
        ys = part.module(*xs)
        for j, (val, truth) in enumerate(zip(ys, ref, strict=True)):
            torch.testing.assert_close(
                val,
                truth,
                rtol=0.02 if j == 0 else 0,
                atol=0.012 if j == 0 else 0,
                check_stride=True,
            )
            assert val.data_ptr() not in storage
            storage.add(val.data_ptr())
            outputs.append(val)
        assert torch.equal(ys[1][:, :, :s], xs[1]) and torch.equal(ys[2][:, :, :s], xs[2])
    assert graph == str(ep.graph)
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        part.module(*req(130))


@torch.inference_mode()
def test_observable_attention_and_rng_fail_closed():
    ex, au = req(7), req(45)
    m = Step(expose=True).to(device="cuda", dtype=torch.bfloat16).eval()
    p = partition_symbolic_decode(export(m, ex), ex, au)
    assert not p.sites and len(p.skipped) == 1
    assert all(torch.equal(a, b) for a, b in zip(p.module(*au), m(*au)))
    m = Step(dropout=0.1).to(device="cuda", dtype=torch.bfloat16).eval()
    ep = export(m, ex)
    before = torch.cuda.get_rng_state()
    with pytest.raises(Unsupported):
        partition_symbolic_decode(ep, ex, au)
    assert torch.equal(before, torch.cuda.get_rng_state())


@torch.inference_mode()
def test_numerical_guard_and_exported_relation():
    ex, au = req(7), req(45)
    m = Step().to(device="cuda", dtype=torch.bfloat16).eval()
    p = partition_symbolic_decode(export(m, ex), ex, au)
    assert p.sites
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        p.module(*ex[:-1], au[-1])
    ex[0][0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        p.module(*ex)
