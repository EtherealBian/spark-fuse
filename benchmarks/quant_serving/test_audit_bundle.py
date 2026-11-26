import copy
import hashlib
import json
import unittest
from pathlib import Path
from audit_bundle import PIN, attest, excluded_pilot


class CrossPanelEvidence(unittest.TestCase):
    def test_rejects_runtime_or_weight_identity_changes(self):
        lock = Path(__file__).resolve().with_name("qwen72b-q4km.lock.json")
        m = json.loads(lock.read_bytes())
        cp = dict(
            m,
            schema=1,
            total_bytes=sum(v["bytes"] for v in m["files"]),
            lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        )
        r = dict(
            context=dict(n_seq_max=1, n_ctx=4096, n_ctx_seq=4096, n_batch=512, n_ubatch=128),
            llama_cpp_revision=PIN,
            device="NVIDIA GB10",
            gpu_layers_including_output=81,
            driver="1.0",
            eog_token_ids=[128247, 151643, 151645, 151662, 151663, 151664],
            server_sha256="a" * 64,
            runtime_lib_sha256={
                k: "f" * 64 for k in ("libggml-base.so", "libggml-cuda.so", "libllama.so")
            },
            backend_buffers=[dict(device="CUDA0", kind="model", reported_mib=1024.0)],
        )
        self.assertEqual(attest(lock, cp, r, 81, 1, 128)["driver"], "1.0")
        for k, v in [
            ("runtime_lib_sha256", {}),
            ("gpu_layers_including_output", 80),
            ("server_sha256", "x"),
            ("eog_token_ids", [151645]),
            ("eog_token_ids", [True]),
            ("device", "CPU"),
            ("backend_buffers", []),
            ("context", dict(r["context"], n_seq_max=2)),
        ]:
            with self.assertRaises(ValueError):
                attest(lock, cp, dict(r, **{k: v}), 81, 1, 128)
        bad = copy.deepcopy(cp)
        bad["files"][0]["sha256"] = "9" * 64
        with self.assertRaises(ValueError):
            attest(lock, bad, r, 81, 1, 128)

    def test_early_stop_is_evidence_not_an_equal_length_pair(self):
        ids = list(range(256))
        meta = dict(
            case="probe",
            length=256,
            shared=240,
            target="T",
            primer="P",
            target_ids=ids,
            primer_ids=ids[:240] + [90],
            order=["reuse", "cold"],
        )
        cold = dict(ids=[4] * 24, predicted=24, stop="limit", processed=256, reused=0)
        warm = dict(ids=[4] * 14 + [151643], predicted=15, stop="eos", processed=16, reused=240)
        row = dict(meta, cold=cold, reuse=warm)
        p = dict(
            schema=1,
            attempted_pairs=8,
            request_ignore_eos=True,
            explicit_eog_mask=False,
            rejected_pairs=[row],
        )
        result = excluded_pilot(p, [meta], {151643})
        self.assertFalse(result["included_in_timings"])
        self.assertEqual(result["reuse_tokens"], 15)
        for k, v in [("stop", "limit"), ("ids", [4] * 15), ("processed", 15), ("predicted", 24)]:
            modified = copy.deepcopy(p)
            modified["rejected_pairs"][0]["reuse"][k] = v
            with self.assertRaises(ValueError):
                excluded_pilot(modified, [meta], {151643})
