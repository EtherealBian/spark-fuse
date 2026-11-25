from copy import deepcopy
import json

import pytest
import torch
from torch import nn
from examples.bert_quant import TokenLinear
import sparkfuse.serving_policy as p


class Fanout(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.layers = nn.ModuleList(
            TokenLinear(nn.Linear(128, 32 + 8 * i, device="cuda", dtype=torch.bfloat16))
            for i in range(n)
        )

    def forward(self, x):
        return tuple(linear(x) for linear in self.layers)


def setup(seed=13, n=3):
    torch.manual_seed(seed)
    m = Fanout(n).eval()
    g = torch.Generator(device="cuda").manual_seed(seed + 227)
    r = [
        (torch.randn(1, 17, 128, device="cuda", dtype=torch.bfloat16, generator=g),)
        for _ in range(30)
    ]
    return torch.export.export(m, r[0]), r


def samples(names, values, n):
    return [
        dict(order=list(names), samples_us={k: [v] * 8 for k, v in zip(names, values, strict=True)})
        for _ in range(n)
    ]


@pytest.mark.parametrize(
    "wins,ratio,arm",
    [(7, 1.02, "independent"), (5, 1.2, "independent"), (6, 1.2, "packed"), (7, 1.03, "packed")],
)
def test_paired_gate(wins, ratio, arm):
    rows = samples(["independent", "packed"], [103, 100], 7)
    for i, w in enumerate(rows):
        w["samples_us"]["independent"] = [ratio * 100 if i < wins else 99] * 8
    assert p._decision(rows, "packed")["selected"] == arm


def test_tie_and_bad_samples():
    rows = samples(p.ARMS, [1, 1, 1], 5)
    assert p._nominate(rows, p.ARMS) == p.ANCHOR
    for bad in [0, float("nan"), -1, False, float("inf")]:
        rows[1]["samples_us"]["shared"][3] = bad
        with pytest.raises(ValueError, match="invalid request timing"):
            p._nominate(rows, p.ARMS)


def clock(handles, rows, objective, seed, n):
    # Synthetic, deterministic clocks only for control-flow/error-injection tests.
    prices = {"independent": 100.0, "shared": 97.0, "packed": 85.0}
    return samples(handles, [prices[k] for k in handles], n)


def record_handles(mp):
    held = []
    old = p._capture

    def capture(*a):
        exe = old(*a)
        held.append((a[-1], exe))
        return exe

    mp.setattr(p, "_capture", capture)
    return held


def select(ep, r, **opts):
    return p.select_request_policy(
        ep, *r[:2], r[2:10], r[10:18], r[18:26], objective="latency", **opts
    )


@torch.inference_mode()
def test_real_winner_replay_and_no_overwrite(monkeypatch, tmp_path):
    ep, r = setup()
    monkeypatch.setattr(p, "_windows", clock)
    held = record_handles(monkeypatch)
    f = tmp_path / "receipt.json"
    h, rec = select(ep, r, receipt_path=f)
    with h:
        assert rec["body"]["decision"]["selected"] == "packed"
        assert [n for n, e in held if e._graph is not None] == ["packed"]
        p._exact(h(*r[-1]), ep.module()(*r[-1]), "winner")
    held.clear()
    with p.replay_request_policy(ep, *r[26:28], f, objective="latency") as run:
        assert [n for n, _ in held] == ["packed"]
        p._exact(run(*r[-1]), ep.module()(*r[-1]), "replay")
    with pytest.raises(FileExistsError):
        select(ep, r, receipt_path=f)
    assert all(e._graph is None for n, e in held) and json.loads(f.read_bytes()) == rec
    with pytest.raises(ValueError, match="context mismatch"):
        p.replay_request_policy(ep, *r[26:28], rec, objective="queue8")
    with pytest.raises(ValueError, match="repeated request content"):
        p.replay_request_policy(ep, *r[:2], rec, objective="latency")
    held.clear()
    # Change one byte without changing the object, shape or layout.
    weight = next(iter(ep.state_dict.values()))
    idx = (0,) * weight.ndim
    weight[idx] += 1
    with pytest.raises(ValueError, match="context mismatch"):
        p.replay_request_policy(ep, *r[26:28], rec, objective="latency")
    assert not held
    bad = deepcopy(rec)
    bad["body"]["decision"]["selected"] = "shared"
    with pytest.raises(ValueError, match="checksum"):
        p._validate(bad)
    bad["sha256"] = p._hash(bad["body"])
    with pytest.raises(ValueError, match="inconsistent selection"):
        p._validate(bad)


@pytest.mark.parametrize("mode", ["same-storage", "same-content", "bad-shape", "short"])
@torch.inference_mode()
def test_distinct_inputs_before_any_capture(mode, monkeypatch):
    ep, r = setup()
    monkeypatch.setattr(p, "_capture", lambda *a: pytest.fail("unexpected capture"))
    if mode == "same-storage":
        r[19] = r[3]
    if mode == "same-content":
        r[19] = (r[3][0].clone(),)
    if mode == "bad-shape":
        r[19] = (r[19][0].reshape(17, 1, 128),)
    if mode == "short":
        r = r[:24]
    with pytest.raises((ValueError, p.Unsupported)):
        select(ep, r)


@pytest.mark.parametrize(
    "exception", [RuntimeError, torch.OutOfMemoryError, p.SiteMismatch, p.Unsupported]
)
@torch.inference_mode()
def test_semantic_rejection_not_operational_failure(monkeypatch, exception):
    ep, r = setup()
    monkeypatch.setattr(p, "_windows", clock)
    held = record_handles(monkeypatch)
    orig = p._kinds

    def verify(h, name):
        if name == "shared":
            raise exception("injected")
        return orig(h, name)

    monkeypatch.setattr(p, "_kinds", verify)
    if exception in (p.SiteMismatch, p.Unsupported):
        exe, rec = select(ep, r)
        with exe:
            assert (
                set(rec["body"]["rejections"]) == {"shared"}
                and rec["body"]["decision"]["selected"] == "packed"
            )
    else:
        with pytest.raises(exception, match="injected"):
            select(ep, r)
        assert [n for n, _ in held] == ["independent", "shared"]
    assert all(e._graph is None for _, e in held)


@torch.inference_mode()
def test_unseen_audit_fails_closed_after_selection(monkeypatch, tmp_path):
    ep, r = setup()
    held = record_handles(monkeypatch)
    phase = []
    orig = p._gate

    def clocks(handles, rows, *args):
        phase.append(id(rows[0][0]))
        return clock(handles, rows, *args)

    def audit(h, rows, refs, state):
        if rows[0][0] is r[18][0]:
            raise p.SiteMismatch("audit mismatch")
        return orig(h, rows, refs, state)

    monkeypatch.setattr(p, "_windows", clocks)
    monkeypatch.setattr(p, "_gate", audit)
    f = tmp_path / "unused.json"
    with pytest.raises(p.SiteMismatch, match="audit mismatch"):
        select(ep, r, receipt_path=f)
    assert (
        phase == [id(r[2][0]), id(r[10][0])]
        and all(e._graph is None for _, e in held)
        and not f.exists()
    )


@torch.inference_mode()
def test_single_branch_cannot_claim_shared_or_packed(monkeypatch):
    ep, r = setup(n=1)
    monkeypatch.setattr(p, "_windows", clock)
    exe, rec = select(ep, r)
    with exe:
        assert rec["body"]["admitted"] == ["independent"] and set(rec["body"]["rejections"]) == {
            "shared",
            "packed",
        }
        p._exact(exe(*r[-1]), ep.module()(*r[-1]), "anchor")


def test_residual_and_plain_are_both_actual_quantizers():
    from types import SimpleNamespace

    for op in ("residual_token_quant", "activation_token_quant"):
        h = SimpleNamespace(
            metadata={
                "regions": [
                    {"kind": k}
                    for k in [op, "shared_hidden_token_quantizers", "packed_int32_projection"]
                ]
            }
        )
        p._kinds(h, "packed")
