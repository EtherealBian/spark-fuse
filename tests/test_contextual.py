"""Timing fixtures verify decisions (not speed); CUDA modules, wires and requests are real."""

import json
from pathlib import Path
import pytest
import torch
from sparkfuse import Schedule, Unsupported
from sparkfuse import contextual as ctx
from sparkfuse.partition import partition_export


def window(a, b):
    return {
        "order": ["challenger", "incumbent"],
        "samples_us": {"incumbent": [a] * 7, "challenger": [b] * 7},
    }


@pytest.mark.parametrize(
    "challenger,succeeds",
    [
        ([99] * 7, False),
        ([94] * 5 + [102] * 2, False),
        ([94] * 6 + [1000], True),
        ([200] * 7, False),
    ],
)
def test_preregistered_paired_rule(challenger, succeeds):
    d = ctx._decide([window(100, b) for b in challenger], "incumbent", "challenger", 1.02)
    assert (d["selected"] == "challenger") == succeeds


def test_fail_closed_samples_and_ties():
    assert ctx._nominate([window(7, 7)] * 5, "incumbent") == "incumbent"
    for x in [0, -1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            ctx._nominate([window(1, x)] * 5, "incumbent")
    with pytest.raises(ValueError):
        ctx._decide([], "x", "y", 1.02)


def test_exact_is_not_numeric_equality():
    minus = torch.tensor([-0.0])
    plus = torch.tensor([0.0])
    same = torch.tensor([0x7FC01234], dtype=torch.int32).view(torch.float32)
    assert ctx._exact(same, same.clone(), "nan") == 1
    for a, b in [
        (minus, plus),
        (same, torch.tensor([0x7FC01235], dtype=torch.int32).view(torch.float32)),
        (same, (same,)),
        (same.expand(3), same.repeat(3)),
    ]:
        with pytest.raises(ctx.SiteMismatch):
            ctx._exact(a, b, "test")
    offsets = torch.arange(8)
    assert ctx._alias_pattern(offsets[:3], offsets[2:5]) != ctx._alias_pattern(
        offsets[:3], offsets[1:4]
    )
    assert ctx._alias_pattern(offsets[:3], offsets[2:5]) != ctx._alias_pattern(
        offsets[:3], offsets[2:5].clone()
    )


def test_no_clobber_receipt(tmp_path):
    dest = tmp_path / "record.json"
    ctx._write_once(dest, {"n": 1})
    original = dest.read_bytes()
    with pytest.raises(FileExistsError):
        ctx._write_once(dest, {"n": 2})
    assert dest.read_bytes() == original and list(tmp_path.iterdir()) == [dest]


class TwoSites(torch.nn.Module):
    def __init__(self, neg=False):
        super().__init__()
        self.neg = neg
        self.register_buffer(
            "weights", torch.linspace(-1, 1, 128, device="cuda", dtype=torch.bfloat16).repeat(3, 1)
        )

    def forward(self, x, y):
        # Two independent regions with identical IR, stable identity, separate timing decisions.
        a = torch.nn.functional.silu(x) * self.weights
        b = torch.nn.functional.silu(y) * self.weights
        return {"first": a, "second": -b if self.neg else b, "input_view": x.view(-1)}


def requests():
    def make(seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        return tuple(
            torch.randn(3, 128, dtype=torch.bfloat16, device="cuda", generator=gen)
            for _ in range(2)
        )

    return [make(i) for i in (319, 773, 1193, 1927)]


def fake_windows(gm, name, mods, args, seed, n):
    return [
        dict(
            order=list(mods),
            samples_us={
                k: [100 if mod.kernel.schedule.vec == 1 else 90] * 7 for k, mod in mods.items()
            },
        )
        for _ in range(n)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("host", ["native", "python"])
def test_one_site_only_with_shared_ir_and_json_roundtrip(host, monkeypatch, tmp_path):
    req = requests()
    model = TwoSites()
    ep = torch.export.export(model, req[0])
    opts = {"host_backend": host}
    base, recs = partition_export(ep, **opts)
    hits = [r["module"] for r in recs if "module" in r]
    assert len(hits) == 2
    assert ctx.signature(base.get_submodule(hits[0]).kernel.plan, "cuda") == ctx.signature(
        base.get_submodule(hits[1]).kernel.plan, "cuda"
    )
    monkeypatch.setattr(ctx, "_windows", fake_windows)
    s = Schedule(128, "flat", False, 4)
    bad = Schedule(12, "flat", False, 1)
    p = tmp_path / "receipt.json"
    gm, r = ctx.select_site(ep, hits[0], [s, bad], *req[:3], partition_options=opts, path=p)
    assert (
        json.loads(p.read_text()) == r
        and r["selected"] == vars(s)
        and r["rejected"][ctx._key(bad)]["phase"] == "legality"
    )
    assert (
        gm.get_submodule(hits[0]).kernel.schedule == s
        and gm.get_submodule(hits[1]).kernel.schedule == base.get_submodule(hits[1]).kernel.schedule
    )
    with torch.cuda.stream(torch.cuda.Stream()):
        torch.cuda.current_stream().wait_stream(torch.cuda.default_stream())
        replay, verified = ctx.replay_site(ep, r, req[-1], partition_options=opts)
        y = replay(*req[-1])
        torch.cuda.current_stream().synchronize()
    assert (
        verified == {"live_ins": 2, "live_outs": 1, "whole": 3}
        and ctx._exact(y, model(*req[-1]), "model") == 3
    )
    assert ctx._alias_pattern(req[-1], y) == ctx._alias_pattern(req[-1], model(*req[-1]))
    with pytest.raises(ValueError, match="fresh"):
        ctx.replay_site(ep, r, req[1], partition_options=opts)
    with pytest.raises(ValueError, match="invalid context receipt"):
        ctx.replay_site(ep, dict(r, seed=-9), req[-1], partition_options=opts)
    with pytest.raises(ctx.SiteMismatch, match="fingerprint"):
        ctx.replay_site(
            torch.export.export(TwoSites(neg=True), req[0]), r, req[-1], partition_options=opts
        )
    shifted = tuple(
        torch.empty(x.numel() + 1, dtype=x.dtype, device=x.device)[1:].reshape(x.shape).copy_(x)
        for x in req[-1]
    )
    with pytest.raises(ctx.SiteMismatch, match="fingerprint"):
        ctx.replay_site(ep, r, shifted, partition_options=opts)
    with torch.no_grad():
        model.weights.add_(1)
    with pytest.raises(ctx.SiteMismatch, match="fingerprint"):
        ctx.replay_site(ep, r, req[-1], partition_options=opts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "phase,exc,accepted",
    [
        ("discovery", ctx.SiteMismatch, True),
        ("confirm", ctx.SiteMismatch, True),
        ("audit", ctx.SiteMismatch, False),
        ("discovery", RuntimeError, False),
    ],
)
def test_failures_do_not_publish_bad_state(phase, exc, accepted, monkeypatch, tmp_path):
    req = requests()
    ep = torch.export.export(TwoSites(), req[0])
    base, rr = partition_export(ep)
    hit = next(r["module"] for r in rr if "module" in r)
    anchor = base.get_submodule(hit).kernel.schedule
    s = Schedule(128, "flat", False, 4)
    dest = tmp_path / "receipt.json"
    check = ctx._check
    target = {"discovery": req[0], "confirm": req[1], "audit": req[2]}[phase]

    def fail(gm, name, mod, args, oracle):
        if mod.kernel.schedule == s and args is target:
            raise exc("sentinel")
        return check(gm, name, mod, args, oracle)

    monkeypatch.setattr(ctx, "_check", fail)
    monkeypatch.setattr(ctx, "_windows", fake_windows)
    if accepted:
        _, r = ctx.select_site(ep, hit, [s], *req[:3], path=dest)
        assert (
            dest.exists()
            and r["selected"] == vars(anchor)
            and r["rejected"][ctx._key(s)]["reason"] == "sentinel"
        )
    else:
        with pytest.raises(exc, match="sentinel"):
            ctx.select_site(ep, hit, [s], *req[:3], path=dest)
        assert not dest.exists() and not list(Path(tmp_path).iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_requests_and_seeded_operators():
    req = requests()
    ep = torch.export.export(TwoSites(), req[0])
    s = [Schedule(128, "flat", False, 4)]
    with pytest.raises(ValueError, match="distinct"):
        ctx.select_site(ep, "invalid", s, *([req[0]] * 3))
    with pytest.raises(Unsupported, match="ordinary"):
        ctx.select_site(ep, "invalid", s, *req[:3])

    class Random(torch.nn.Module):
        def forward(self, x, y):
            return x * torch.rand_like(y) + y

    with pytest.raises(Unsupported, match="random"):
        ctx.select_site(torch.export.export(Random(), req[0]), "invalid", s, *req[:3])
