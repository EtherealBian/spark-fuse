import unittest
from unittest.mock import patch
from moe_blocks import audit, measure
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
        bs[1]["rows"][0]["ids"][0] = 10
        self.assertFalse(audit(bs)["all_tokens_exact"])
        bs[1]["rows"][0]["processed"] = 1
        with self.assertRaises(ValueError):
            audit(bs)

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
