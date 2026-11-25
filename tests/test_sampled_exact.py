import sqlite3
from contextlib import closing

import pytest
import torch
from torch import nn

from sparkfuse import calibration as cal
from sparkfuse.measure import compare
from sparkfuse.partition import partition_export
from sparkfuse.sampled_exact import tune_shared_exact


class RepeatedScore(nn.Module):
    def forward(self, x, proj):
        a = (x * 0.125).softmax(-1)
        b = a @ proj
        return a, (b * 0.125).softmax(-1)


def fixture():
    g = torch.Generator(device="cuda").manual_seed(366404)
    args = [
        tuple(
            torch.randn(s, dtype=torch.float32, device="cuda", generator=g)
            for s in [(11, 97), (97, 97)]
        )
        for _ in range(4)
    ]
    m = RepeatedScore()
    return m, torch.export.export(m, args[0]), args


def test_reference_capture_and_joint_dispatch_roundtrip(monkeypatch, tmp_path):
    m, ep, args = fixture()
    path = tmp_path / "dispatch.sqlite"
    orig = cal.partition_export

    def forbid_tainted_capture(*a, **kw):
        gm, rows = orig(*a, **kw)
        if "database" not in kw:

            def wrong(*unused):
                raise RuntimeError("discovery must propagate the unfused oracle")

            for row in rows:
                if "module" in row:
                    gm.get_submodule(row["module"]).dispatch = wrong
        return gm, rows

    monkeypatch.setattr(cal, "partition_export", forbid_tainted_capture)
    gm, result = cal.calibrate_export(ep, *args[:3], path, budget=4, sampled_exact=True)
    assert result["sampled_exact"] and result["captured_from"] == "unfused_aten"
    assert len(result["records"]) == 2
    with closing(sqlite3.connect(path)) as db:
        assert db.execute("SELECT count(*) FROM schedules").fetchone()[0] == 1
    r0, r1 = result["records"].values()
    assert r0 == r1 and set(r0["sites"]) == set(result["records"]) == set(r0["third_input_errors"])
    for row in result["mixed_request_errors"].values():
        assert len(row) == 2 and all(v["rtol"] == v["atol"] == v["unequal"] == 0 for v in row)
    reload, _ = partition_export(ep, database=path, host_backend="native")
    for f in [gm, reload]:
        compare(f(*args[-1]), m(*args[-1]), exact=(0, 1))


@pytest.mark.parametrize("phase", ["holdout", "audit"])
def test_no_equal_schedule_or_independent_audit_failure(monkeypatch, phase, tmp_path):
    _, ep, args = fixture()
    path = tmp_path / "dispatch.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE sentinel(v)")
        db.execute("INSERT INTO sentinel VALUES (42)")
    before = path.read_bytes()
    visited = []

    def inject(ep, sites, **kwargs):
        name = next(reversed(sites))
        idx = 1 if phase == "holdout" else 2
        a, ref = sites[name][idx]
        sites[name][idx] = (a, tuple(v + 1 for v in ref))
        visited.append(name)
        return tune_shared_exact(ep, sites, **kwargs)

    monkeypatch.setattr(cal, "tune_shared_exact", inject)
    with pytest.raises(AssertionError) as exc:
        cal.calibrate_export(ep, *args[:3], path, budget=2, sampled_exact=True)
    assert visited and path.read_bytes() == before
    if phase == "holdout":
        assert (
            "no candidate passed" in str(exc.value)
            and "holdout" in str(exc.value)
            and visited[0] in str(exc.value)
        )
    else:
        assert any("independent audit" in n and visited[0] in n for n in exc.value.__notes__)


@pytest.mark.parametrize(
    "ratios,admitted", [([1.1] * 5, True), ([1.1, 1.1, 1.1, 0.9, 0.9], False), ([1.02] * 5, False)]
)
def test_speed_claim_needs_paired_holdout_evidence(monkeypatch, ratios, admitted):
    import sparkfuse.sampled_exact as joint
    from sparkfuse import Schedule

    _, ep, args = fixture()
    gm, parts = partition_export(ep)
    n = next(d["module"] for d in parts if "module" in d)
    sub = gm.get_submodule(n).kernel.ep
    oracle = torch.fx.GraphModule({}, sub.graph)
    sites = {"first": [((a[0],), oracle(a[0])) for a in args[:3]]}
    anchor, other = Schedule(128, "warp_xor", True), Schedule(256, "warp_xor", False)
    monkeypatch.setattr(joint, "_shortlist", lambda *_: [anchor, other])
    timings = [5.0, 4.0]
    for i, ratio in enumerate(ratios):
        pair = [5.0, 5.0 / ratio]
        timings.extend(reversed(pair) if i % 2 else pair)
    stream = iter(timings)
    monkeypatch.setattr(joint, "device_time", lambda *a, **kw: {"median_us": next(stream)})
    rec = joint.tune_shared_exact(sub, sites, budget=2, seed=915)
    assert rec["admitted"] is admitted and rec["selected"] == vars(other if admitted else anchor)
    assert len(rec["paired_holdout_us"]) == 5 and next(stream, None) is None


def test_conflicting_shared_profiles_never_overwrite(tmp_path):
    path = tmp_path / "db.sqlite"
    with pytest.raises(AssertionError, match="shared dispatch key"):
        cal._save(
            path,
            [
                {"key": "collision", "selected": {"vec": 1}},
                {"key": "collision", "selected": {"vec": 4}},
            ],
        )
    assert not path.exists()
