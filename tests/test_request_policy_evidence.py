"""Recompute the published decision and holdout medians without CUDA or Transformers."""

from pathlib import Path
import json
from statistics import median
from sparkfuse.contextual import _hash
from sparkfuse.serving_policy import _validate, _scores


def test_whole_request_policy_evidence():
    report = Path(__file__).resolve().parents[1] / "reports/request-policy-gb10.json"
    d = json.loads(report.read_bytes())
    cs = d["cases"]
    assert not d["smoke"] and len(cs) == 12
    key = d["provenance"]["source_tree_sha256"]
    assert (report.parent / "sources" / (key + ".tar.gz")).is_file()
    pairs = {}
    for c in cs:
        b = _validate(c["receipt"])
        assert c["objective"] == b["context"]["objective"]
        assert set(b["admitted"]) == {"independent", "shared", "packed"} and not b["rejections"]
        k = b["builds"]["packed"]["kinds"]
        assert (
            k["packed_int32_projection"] == 1
            and k["shared_hidden_token_quantizers"] == 1
            and k["residual_token_quant" if c["residual"] else "activation_token_quant"] == 1
        )
        refhashes = c["new_ref_sha256"]
        assert len(refhashes) == 4 and all(len(row) == 2 for row in refhashes)
        assert len(c["new_cpu_qkv_oracle"]) == 4 and all(
            len(row["branches"]) == 3 for row in c["new_cpu_qkv_oracle"]
        )
        fresh = {_hash(row) for row in c["new_inputs_sha256"]}
        assert len(fresh) == 4 and fresh.isdisjoint(x for xs in b["requests"].values() for x in xs)
        assert all(v == [2] * 8 for arm in b["gates"].values() for v in arm.values())
        m = {k: median(v) for k, v in _scores(b["audit"], b["admitted"], 7).items()}
        assert m == c["audit_us"]
        saved = m["independent"] - m[b["decision"]["selected"]]
        assert saved == c["saved_us_vs_independent"]
        assert c["amortize_vs_independent_requests"] == (
            1000 * (c["selection_wall_ms"] - b["builds"]["independent"]["ms"]) / saved
            if saved > 0
            else None
        )
        key = (c["seed"], c["m"])
        p = pairs.setdefault(key, {})
        assert c["objective"] not in p
        p[c["objective"]] = (b["requests"], b["context"]["state"], c["new_ref_sha256"])
    assert (
        {m for seed, m in pairs} == {17, 33, 129}
        and len(pairs) == 6
        and all(
            sorted(p) == ["latency", "queue8"] and p["latency"] == p["queue8"]
            for p in pairs.values()
        )
    )
