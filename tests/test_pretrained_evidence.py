"""Public pretrained evidence can be audited offline; no weight download on CI."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "pretrained_qwen"
REPORTS = ROOT / "reports"


@pytest.mark.parametrize(
    "script,inputs,manifest",
    [
        ("audit.py", ("pretrained-qwen-gb10.json",), "pretrained-qwen-audit.json"),
        (
            "audit_projections.py",
            ("pretrained-projections-gb10.json", "pretrained-qwen-gb10.json"),
            "pretrained-projections-audit.json",
        ),
    ],
)
def test_manifests_rebuild_without_site_packages(script, inputs, manifest):
    data = subprocess.check_output(
        [sys.executable, "-I", "-S", str(BENCH / script), *[str(REPORTS / p) for p in inputs]],
        cwd=BENCH,
    )
    assert data == (REPORTS / manifest).read_bytes()


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("weights", "packed weight budget"),
        ("prompt", "independent held-out prompts"),
        ("tokens", "eligibility gate"),
        ("speed", "paired decision mismatch"),
    ],
)
def test_projection_auditor_rejects_tampering(tmp_path, kind, reason):
    p = REPORTS / "pretrained-qwen-gb10.json"
    j = json.loads((REPORTS / "pretrained-projections-gb10.json").read_bytes())
    c = j["cases"]["128"]
    if kind == "weights":
        j["weight_pack_bytes"] -= 1
    elif kind == "prompt":
        c["inputs"][0] = json.loads(p.read_bytes())["cases"]["128"]["inputs"][0]
    elif kind == "tokens":
        c["validation"]["packed_fused"][0]["generated"][0] = -1
    elif kind == "speed":
        c["comparisons"][0]["target_met"] = not c["comparisons"][0]["target_met"]
    raw = tmp_path / "panel.json"
    raw.write_text(json.dumps(j))
    r = subprocess.run(
        [sys.executable, "-I", "-S", str(BENCH / "audit_projections.py"), str(raw), str(p)],
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and reason in r.stderr
