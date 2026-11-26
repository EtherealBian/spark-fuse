import copy
import unittest
from unittest.mock import patch
from moe_blocks import audit, measure, paired_changes
from runtime_receipt import read_log


class Blocks(unittest.TestCase):
    def test_abba_and_gate(self):
        def tok(port, path, body):
            return {"tokens": list(body["content"].encode())}

        def done(port, ids, reuse, n):
            return dict(
                stop="limit",
                ids=[4] * n,
                processed=len(ids),
                reused=0,
                predicted=n,
                prompt_ms=30.0,
                decode_ms=10.0,
                wall_ms=45.0,
            )

        with patch("moe_blocks.post", tok), patch("moe_blocks.complete", done):
            bs = [measure(0, b) for b in range(4)]
        self.assertTrue(audit(bs)["all_tokens_exact"])
        for k, v in [
            ("size", 256.0),
            ("case", False),
            ("reused", False),
            ("processed", 256.0),
            ("predicted", 24.0),
        ]:
            corrupted = copy.deepcopy(bs)
            corrupted[0]["rows"][0][k] = v
            with self.subTest(k=k), self.assertRaises(ValueError):
                audit(corrupted)
        duplicated = copy.deepcopy(bs)
        for b in duplicated:
            for key in ("input_ids", "input_hash"):
                b["rows"][1][key] = b["rows"][0][key]
        with self.assertRaises(ValueError):
            audit(duplicated)
        bs[1]["rows"][0]["ids"][0] = 10
        self.assertFalse(audit(bs)["all_tokens_exact"])
        bs[1]["rows"][0]["processed"] = 1
        with self.assertRaises(ValueError):
            audit(bs)

    def test_pair_sign_survives_order_reversal_and_exposes_first_divergence(self):
        rows = [
            dict(input_hash="abc", ids=[3, 4, 5], prompt_ms=t - 3, decode_ms=2, wall_ms=t)
            for t in (12, 8, 11, 10)
        ]
        rows[2]["ids"] = [3, 44, 5]
        outcome = paired_changes(rows)
        self.assertEqual([p["saved_ms"]["wall_ms"] for p in outcome["comparisons"]], [4, -1])
        self.assertEqual([p["first_divergent_token"] for p in outcome["comparisons"]], [None, 2])
        self.assertFalse(outcome["wall_faster_in_both_orders"])
        self.assertFalse(outcome["same_tokens_in_both_orders"])
        # The comparison array is intentionally per-prompt; do not pool unlike prompts.
        self.assertEqual(outcome["input_sha256"], "abc")

    def test_runtime_receipt(self):
        s = "NVIDIA GB10\noffloaded 81/81 layers to GPU\nn_seq_max = 2\nn_ctx = 8192\nn_ctx_seq = 4096\nn_batch = 512\nn_ubatch = 128\nCUDA0 model buffer size = 34399.70 MiB"
        self.assertEqual(read_log(s, 81, 2, 128)["backend_buffers"][0]["device"], "CUDA0")
        # The two-slot server logs a second, slot-local n_ctx. It is not the global context.
        self.assertEqual(
            read_log(s + "\nslot init_slot: new slot, n_ctx = 4096", 81, 2, 128)["context"][
                "n_ctx"
            ],
            8192,
        )
        for bad in (
            s.replace("81/81", "79/81"),
            s.replace("NVIDIA GB10", "GPU"),
            s.replace("128", "256"),
            s.replace("CUDA0", "CPU"),
        ):
            with self.assertRaises(ValueError):
                read_log(bad, 81, 2, 128)
