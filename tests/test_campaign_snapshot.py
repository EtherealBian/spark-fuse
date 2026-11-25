"""CPU-only checks for portable evidence. Synthetic times deliberately differ by clock."""

import gzip
import json
from pathlib import Path
import pytest
from tools import campaign_snapshot as cs
from tools.report import render


def entry(n=1, compiler="a" * 64):
    name = f"resnorm2-{n}x8-float16"
    phase = "broad-search"
    base = {"threads": 32}
    win = {"threads": 64}
    err = [{"unequal": 1}, {"rtol": 0, "atol": 0, "unequal": 0, "max_abs": 0}]

    def clock(val):
        return {"samples_us": [val + 0.5, val, val - 0.5], "median_us": val}

    q = dict(
        seed=5,
        compiler=compiler,
        candidate=win,
        default=base,
        selected=win,
        heldout_speedup_vs_default=2.0,
        paired_holdout_us=[{"default": 4, "candidate": 2} for _ in range(5)],
        admitted=True,
        trials=[{"schedule": base, "device": clock(4)}, {"schedule": win, "device": clock(2)}],
        invalid=[],
        heldout_errors=err,
    )
    c = dict(
        case="resnorm2",
        shape=[n, 8],
        dtype="torch.float16",
        seed=2005,
        schedule=win,
        backends={
            k: dict(parity="pass", errors=err, device=clock(i + 1), host_sync=clock(3 - i))
            for i, k in enumerate(cs.BACKENDS)
        },
    )
    return {
        "worker": dict(
            job=dict(
                id=name, case="resnorm2", phase=phase, rows=n, width=8, dtype="float16", seed=5
            ),
            comparison=c,
            tuning=q,
            status="ok",
        ),
        "index": dict(
            id=name,
            phase=phase,
            status="ok",
            us={k: v["device"]["median_us"] for k, v in c["backends"].items()},
            invalid=0,
            trials=2,
            admitted=True,
            holdout=2.0,
            compiler=compiler,
        ),
    }


def make_run(tmp_path, entries, tail=b""):
    run = tmp_path / "example"
    ev = run / "evidence"
    ev.mkdir(parents=True)
    with (run / "index.jsonl").open("wb") as f:
        for val in entries:
            r, w = val["index"], val["worker"]
            f.write(json.dumps(r).encode() + b"\n")
            suffix = ".json" if r["status"] == "ok" else ".rejected.json"
            if w is not None:
                (ev / (r["id"] + suffix)).write_text(json.dumps(w))
        f.write(tail)
    return run


def test_seal_and_regenerate_without_gpu(tmp_path):
    missing = {
        "index": dict(
            id="old-1", phase="broad-search", status="error", stderr=str(Path.home()) + "/old.txt"
        ),
        "worker": None,
    }
    run = make_run(tmp_path, [entry(), entry(2, "b" * 64), missing], b'{"unfinished":')
    (run / "evidence" / "never-indexed.json").write_text("invalid")
    out = tmp_path / "public" / "mid.md"
    render(run, out)
    p = out.with_suffix(".json.gz")
    d = json.loads(gzip.decompress(p.read_bytes()))
    v = cs.verify_file(p)
    assert v["indexed"] == 3 and v["passed"] == 2 and v["distinct_sources_verified"] == 0
    assert d["records"][-1]["index"]["stderr"] == "$HOME/old.txt"
    assert len(d["summary"]["by_compiler"]) == 3
    assert d["summary"]["outcomes"]["failure-without-worker-evidence"] == 1
    g = d["summary"]["by_compiler"]["a" * 64]
    assert g["host_sync"]["sparkfuse_faster"] == 1 and g["device"]["sparkfuse_faster"] == 0
    assert d["ignored_unterminated_index_bytes"] > 0 and "stopped:" not in cs.completeness(d)
    second = p.parent / "offline.md"
    render(p, second)
    assert out.read_bytes() == second.read_bytes()
    d["observed_supervisor"].update(completed=2, failed=1, last_id="old-1", stop_reason="deadline")
    assert "partial append" in cs.completeness(d)
    d["ignored_unterminated_index_bytes"] = 0
    assert cs.completeness(d) == "stopped: deadline"
    d["observed_supervisor"]["last_id"] = "other"
    assert "do not match" in cs.completeness(d)


@pytest.mark.parametrize(
    "fault",
    [
        "median",
        "sourcepath",
        "repeat",
        "identity",
        "selection",
        "index-clock",
        "residual",
        "seed",
        "missing",
        "backend",
        "panel",
        "trials",
    ],
)
def test_reject_conflicting_evidence(fault):
    e = entry()
    r = e["index"]
    w = e["worker"]
    b = w["comparison"]["backends"]
    q = w["tuning"]
    a = [e]
    if fault == "median":
        b["eager"]["device"]["median_us"] = 15
    if fault == "sourcepath":
        w["provenance"] = {"source_tree_sha256": "0" * 64, "archive": "../other.tar.gz"}
    if fault == "repeat":
        a.append(e)
    if fault == "identity":
        w["job"]["id"] = "another"
    if fault == "selection":
        q["paired_holdout_us"][0] = dict(default=1, candidate=18)
        q["paired_holdout_us"][1] = dict(default=1, candidate=19)
    if fault == "index-clock":
        r["us"]["eager"] = 99
    if fault == "residual":
        b["sparkfuse"]["errors"][1]["unequal"] = 1
    if fault == "seed":
        w["comparison"]["seed"] = 5
    if fault == "missing":
        e["worker"] = None
    if fault == "backend":
        b.pop("inductor")
    if fault == "panel":
        r["phase"] = "score-prefill-like"
    if fault == "trials":
        q["trials"][0]["device"]["samples_us"] = [0.01] * 3
    with pytest.raises(ValueError):
        cs.audit(a, "general")


def test_rejected_comparison_and_unattributed_discovery():
    e = entry()
    e["index"]["status"] = "error"
    w = e["worker"]
    w["status"] = "parity-failure"
    w["comparison"]["backends"]["inductor"] = {"parity": "fail", "diagnostic": "unequal"}
    f = entry(2)
    f["index"].pop("compiler")
    f["index"]["status"] = "error"
    f["worker"] = {
        "job": f["worker"]["job"],
        "status": "discovery-error",
        "diagnostic": "no trials",
    }
    s, r = cs.audit([e, f], "general")
    assert (
        s["passed"] == 0 and r[0]["backend_failures"] == ["inductor"] and r[1]["compiler"] is None
    )
    assert s["by_compiler"]["unrecorded"]["device"]["median_inductor_over_sparkfuse"] is None


def test_integrity_and_legacy_parity(tmp_path):
    e = entry()
    del e["worker"]["comparison"]["backends"]["inductor"]["parity"]
    run = make_run(tmp_path, [e])
    d = cs.snapshot(run)
    g = d["summary"]["by_compiler"]["a" * 64]
    assert g["implicit_parity_in_passed"] == 1
    g["admitted"] = 0
    with pytest.raises(ValueError, match="summary differs"):
        cs.verify_doc(d, tmp_path)
    d["records"][0]["index"]["status"] = "error"
    with pytest.raises(ValueError, match="digest mismatch"):
        cs.verify_doc(d, tmp_path)


def test_no_path_traversal_in_index(tmp_path):
    run = tmp_path / "example"
    run.mkdir()
    (run / "index.jsonl").write_text('{"id":"../unrelated", "status":"error"}\n')
    with pytest.raises(ValueError, match="case ID"):
        cs.snapshot(run)
