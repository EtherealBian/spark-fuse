import hashlib
import tempfile
from pathlib import Path
from unittest import TestCase
import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from run import Runner
from attribute import capture, store
from attribution_review import ARMS, compare, read


class Probe(TestCase):
    @torch.inference_mode()
    def test_real_static_cache_three_arms_same_fixed_teacher_and_dirty_restore(self):
        torch.manual_seed(639)
        old = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old)
        torch.set_num_threads(1)
        cfg = Qwen2Config(
            vocab_size=73,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            max_position_embeddings=32,
            tie_word_embeddings=False,
            use_sliding_window=False,
        )
        cfg._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(cfg).eval()
        r = Runner(model, cfg, 25, torch.device("cpu"), torch.float32)
        ids = (3, 6, 4, 5, 10, 11, 22, 33, 42, 15)
        forced = [45, 3, 72, 67]
        q = {a: capture(r, ids, 6, forced, a, 73) for a in ARMS}
        self.assertEqual(tuple(q["whole"][0].shape), (4, 73))
        self.assertEqual(tuple(q["whole"][1].shape), (4, 3, 2, 8))
        for j in (0, 1):
            self.assertTrue(torch.equal(q["inplace"][j], q["restored"][j]))
            torch.testing.assert_close(q["whole"][j], q["inplace"][j], atol=2e-6, rtol=1e-5)
        alt = capture(r, ids, 6, [34, 1, 3, 8], "whole", 73)
        self.assertTrue(torch.equal(alt[0][0], q["whole"][0][0]))
        self.assertFalse(torch.equal(alt[0][1], q["whole"][0][1]))
        self.assertTrue(torch.equal(alt[1], q["whole"][1]))

    def test_complete_bit_payload_shape_digest_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            t = torch.tensor([[0.0, -0.0, 1.0], [-1.0, 256.0, 0.5]], dtype=torch.bfloat16)
            m = store(p, "x", t)
            a = read(p, "x.bf16", m, (2, 3))
            self.assertEqual(a[0, 1], 0x8000)
            self.assertEqual(a[1, 0], 0xBF80)
            with self.assertRaises(ValueError):
                read(p, "x.bf16", m, (3, 2))
            f = p / "x.bf16"
            b = f.read_bytes()
            f.write_bytes(bytes([b[0] ^ 2]) + b[1:])
            with self.assertRaises(ValueError):
                read(p, "x.bf16", m, (2, 3))
            z = bytes([0xC0, 0x7F]) + b[2:]
            f.write_bytes(z)
            m["sha256"] = hashlib.sha256(z).hexdigest()
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                read(p, "x.bf16", m, (2, 3))

    def test_competing_argmax_tie_and_sampled_physical_kv(self):
        a = np.array([[0x3F80, 0x3F80, 0], [0, 0x4000, 0x3F80]], dtype="<u2")
        b = a.copy()
        b[1, 2] = 0x4040
        ka = np.zeros((6, 3, 2, 8), dtype="<u2")
        kb = ka.copy()
        kb[3, 2, 1, 5] = 0x8000
        s = compare(a, b, ka, kb)
        self.assertEqual(s["first_different_greedy_step"], 2)
        self.assertEqual(
            s["at_first_changed_argmax"],
            dict(
                reference_id=1,
                candidate_id=2,
                reference_margin_between_two=1.0,
                candidate_margin_between_two=1.0,
            ),
        )
        self.assertEqual(s["different_logit_values_by_step"], [0, 1])
        self.assertEqual(
            s["first_changed_sampled_kv_bank"],
            dict(
                layer_0_based=1,
                bank="value",
                counts_at_last_prefix_first_suffix_last_input=[0, 0, 1],
            ),
        )
        self.assertEqual(compare(a, a, kb, kb)["changed_sampled_kv_scalars"], 0)
