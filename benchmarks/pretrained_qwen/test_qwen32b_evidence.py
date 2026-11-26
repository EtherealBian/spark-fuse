"""Fast offline falsification tests for the full-model 32B evidence contract."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit32b as a
import checkpoint32b as cp


def row(n):
    def c(i):
        return dict(
            prompt=i,
            input_ids=[i] * n,
            reference=[i] * 32,
            free=[i] * 32,
            max_abs_by_step=[0.0] * 32,
            reference_logits_sha256="a" * 64,
            logits_sha256="a" * 64,
            logits_bitwise_equal=True,
            score_dtype="torch.bfloat16",
        )

    return dict(
        length=n,
        capacity=n + 32,
        checks=[c(1), c(2)],
        setup_peak_allocated=cp.PARAMETERS * 2 + 4096,
        warm_peak_allocated=cp.PARAMETERS * 2 + 2048,
        warm_peak_reserved=cp.PARAMETERS * 2 + 8192,
        windows=[
            dict(prompt=i, ttft_ms=5.0, decode_ms=[2.0] * 31, tokens=[i] * 32) for i in (1, 2, 2, 1)
        ],
    )


def receipt(k):
    return dict(
        format="sparkfuse.qwen25-32b.v1",
        backend=k,
        model=cp.MODEL,
        revision=cp.REV,
        parameters=cp.PARAMETERS,
        layers=64,
        batch=1,
        attention="sdpa",
        generated=32,
        dtype="bfloat16",
        torch="fixture",
        transformers="fixture",
        accelerate="fixture",
        cuda="fixture",
        device="NVIDIA GB10",
        source_sha256={
            name: hashlib.sha256(Path(a.__file__).with_name(name).read_bytes()).hexdigest()
            for name in a.FILES
        },
        model_storage_before=cp.PARAMETERS * 2,
        model_storage_after=cp.PARAMETERS * 2,
        gate_source_sha256="e" * 64,
        cases=[row(n) for n in (128, 2048)],
    )


@pytest.fixture
def reports(tmp_path):
    for k in ("eager", "resident"):
        (tmp_path / f"qwen32b-{k}.json").write_text(json.dumps(receipt(k)))
    return tmp_path


def edit(d, fn):
    p = d / "qwen32b-resident.json"
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))


def test_rebuild(reports):
    o = a.build(reports)
    assert o["strict_pilot_passed"]
    r = o["cases"]["128"]["resident"]
    assert r["decode_samples"] == 124 and r["decode_p95_ms"] == 2


@pytest.mark.parametrize(
    "fn",
    [
        lambda r: r.update(revision="0" * 40),
        lambda r: r["source_sha256"].update({"scale32b.py": "0" * 64}),
        lambda r: r.update(accelerate="different"),
        lambda r: r["cases"][0]["checks"][0].update(logits_sha256="0" * 64),
        lambda r: r["cases"][0]["checks"][0].update(max_abs_by_step=[0.5] * 32),
        lambda r: r["cases"][0]["checks"][0].update(reference=[5] * 32),
        lambda r: r["cases"][0]["checks"][1].update(input_ids=[1] * 128),
        lambda r: r["cases"][1]["windows"][0].update(decode_ms=[float("nan")] * 31),
        lambda r: r["cases"][1]["windows"].pop(),
        lambda r: r["cases"][1].update(warm_peak_reserved=1),
    ],
)
def test_corruption_rejected(reports, fn):
    edit(reports, fn)
    with pytest.raises(ValueError):
        a.build(reports)


@pytest.mark.parametrize(
    "fn",
    [
        lambda r: r["cases"][0]["checks"][0].update(free=[9] * 32),
        lambda r: r["cases"][0]["windows"][0].update(tokens=[9] * 32),
        lambda r: r["cases"][0]["checks"][0].update(
            max_abs_by_step=[0.5] * 32, logits_sha256="0" * 64, logits_bitwise_equal=False
        ),
        lambda r: r.update(model_storage_after=cp.PARAMETERS * 2 + 256),
    ],
)
def test_honest_negative_preserved(reports, fn):
    edit(reports, fn)
    assert not a.build(reports)["strict_pilot_passed"]


def test_checkpoint_pin():
    entries = cp.SPEC["files"]
    assert len([s for s in entries.values() if "sha256" in s]) == 17
    assert cp.SPEC["revision"] == cp.REV and cp.SPEC["bf16_parameters"] == cp.PARAMETERS
    assert sum(v["size"] for v in entries.values() if "sha256" in v) == 65_527_841_856


def test_checkpoint_refuses_unknown_sidecars(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected or missing"):
        cp.verify(tmp_path)


def test_checkpoint_rehashes_content(tmp_path, monkeypatch):
    m = copy.deepcopy(cp.SPEC)
    name = "x.safetensors"
    m["files"] = {name: dict(size=1, sha256="0" * 64)}
    monkeypatch.setattr(cp, "SPEC", m)
    (tmp_path / name).write_bytes(b"a")
    with pytest.raises(ValueError, match="digest mismatch"):
        cp.verify(tmp_path)
