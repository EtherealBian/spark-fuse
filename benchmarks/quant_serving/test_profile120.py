import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from moe_blocks import measure
from profile120 import ROOT, choose, from_evidence, measured_cohorts, strict_mix, verify_target
from audit_bundle import attest
from runtime_receipt import PIN


def fixture():
    def tok(port, path, data):
        return {"tokens": list(data["content"].encode())}

    def response(port, ids, reuse, n):
        return dict(
            ids=[3] * n,
            stop="limit",
            processed=len(ids),
            reused=0,
            predicted=n,
            prompt_ms=10.0,
            decode_ms=20.0,
            wall_ms=40.0,
        )

    with patch("moe_blocks.post", tok), patch("moe_blocks.complete", response):
        bs = [measure(0, i) for i in range(4)]
    for i in (1, 2):
        for row in bs[i]["rows"]:
            row["wall_ms"] = 35.0
    return bs


class Profile(unittest.TestCase):
    def test_rejects_order_reversal_regression_even_if_aggregate_wins(self):
        b = fixture()
        plan = choose(b)
        self.assertEqual(plan["selected_ubatch"], 512)
        b[2]["rows"][0]["wall_ms"] = 43.0
        plan = choose(b)
        self.assertEqual(plan["selected_ubatch"], 128)
        self.assertTrue(plan["gates"]["positive_gain_and_margin_in_both_orders"])
        self.assertFalse(
            plan["gates"]["each_selected_prompt_within_regression_budget_in_both_orders"]
        )
        b[2]["rows"][0]["wall_ms"] = 35.0
        for r in b[2]["rows"]:
            r["wall_ms"] = 39.5
        self.assertFalse(choose(b)["gates"]["positive_gain_and_margin_in_both_orders"])

    def test_cohorts_only_permit_observed_fingerprints(self):
        b = fixture()
        b[3]["rows"][0]["ids"][-1] = 7
        plans = measured_cohorts(b)
        self.assertEqual(plans["256"]["selected_ubatch"], 128)
        self.assertEqual(plans["2048"]["selected_ubatch"], 512)
        for n, plan in plans.items():
            self.assertEqual(
                plan["request_counts"],
                {r["input_hash"]: 1 for r in b[0]["rows"] if str(r["size"]) == n},
            )
        unseen = "0" * 64
        self.assertEqual(choose(b, {unseen: 100})["selected_ubatch"], 128)

    def test_token_drift_and_unknown_inputs_fail_closed(self):
        b = fixture()
        k = b[0]["rows"][0]["input_hash"]
        other = b[0]["rows"][1]["input_hash"]
        b[2]["rows"][0]["ids"][10] = 77
        self.assertEqual(choose(b, {k: 600, other: 1})["different_token_sha256"], [k])
        self.assertEqual(choose(b, {other: 2})["selected_ubatch"], 512)
        result = choose(b, {"f" * 64: 100})
        self.assertEqual(result["selected_ubatch"], 128)
        self.assertEqual(result["orders"], {})

    def test_rejects_invalid_counts_unlike_inputs_and_mutated_receipts(self):
        b = fixture()
        k = b[0]["rows"][0]["input_hash"]
        for mix in ({}, {k: True}, {k: 0}, {k: 2.5}, {"abc": 2}):
            with self.assertRaises(ValueError):
                choose(b, mix)
        with self.assertRaises(ValueError):
            strict_mix([(k, 1), (k, 2)])
        bad = copy.deepcopy(b)
        bad[0]["rows"][1] = bad[0]["rows"][0]
        with self.assertRaises(ValueError):
            choose(bad)
        with self.assertRaises(ValueError):
            choose(b, min_gain=float("nan"))
        with (
            tempfile.TemporaryDirectory() as td,
            patch(
                "profile120.bundle", return_value={"evidence_sha256": {"block-0.json": "f" * 64}}
            ),
        ):
            d = Path(td)
            (d / "block-0.json").write_text(json.dumps(b[0]))
            with self.assertRaisesRegex(ValueError, "changed"):
                from_evidence(d)

    def test_fresh_target_requires_matched_weights_geometry_and_loaded_binary(self):
        lock = ROOT / "gpt-oss-120b-mxfp4.lock.json"
        cfg = json.loads(lock.read_bytes())
        cp = dict(
            cfg,
            schema=1,
            total_bytes=sum(e["bytes"] for e in cfg["files"]),
            lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        )
        rt = dict(
            context=dict(n_seq_max=1, n_ctx=4096, n_ctx_seq=4096, n_batch=512, n_ubatch=128),
            llama_cpp_revision=PIN,
            device="NVIDIA GB10",
            gpu_layers_including_output=37,
            server_sha256="a" * 64,
            driver="580.10",
            runtime_lib_sha256={
                k: "b" * 64 for k in ("libggml-base.so", "libggml-cuda.so", "libllama.so")
            },
            eog_token_ids=[199999],
            backend_buffers=[dict(device="CUDA0", kind="model", reported_mib=1.0)],
        )
        base = attest(lock, cp, rt, 37, 1, 128)
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)

            def put(name, data):
                (d / name).write_text(json.dumps(data))

            put("checkpoint.json", cp)
            put("runtime.json", rt)
            self.assertEqual(verify_target(base, d)["verified_current_ubatch"], 128)
            for bad in [
                dict(rt, server_sha256="c" * 64),
                dict(rt, driver="580.11"),
                dict(rt, context=dict(rt["context"], n_ctx=8192)),
                dict(rt, context=dict(rt["context"], n_ubatch=256)),
                dict(
                    rt,
                    runtime_lib_sha256=dict(rt["runtime_lib_sha256"], **{"libllama.so": "c" * 64}),
                ),
            ]:
                put("runtime.json", bad)
                with self.assertRaises(ValueError):
                    verify_target(base, d)
            put("runtime.json", rt)
            put("checkpoint.json", dict(cp, revision="a" * 40))
            with self.assertRaises(ValueError):
                verify_target(base, d)
