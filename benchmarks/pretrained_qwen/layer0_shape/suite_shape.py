import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
)
import kvshape as q
import review_shape as v


class Shape(unittest.TestCase):
    def test_compact_schedules_do_not_change_real_positions(self):
        self.assertEqual(q.plan(128, 96, "split"), [(0, 96, 96), (96, 128, 32)])
        self.assertEqual(q.plan(128, 96, "padded"), [(0, 96, 128), (96, 128, 128)])
        self.assertEqual(q.plan(128, 96, "whole"), [(0, 128, 128)])
        for p, a in [(128, "split"), (0, "whole"), (96, "other")]:
            with self.assertRaises(ValueError):
                q.plan(128, p, a)

    @torch.inference_mode()
    def test_native_rotary_and_padding_with_same_genuine_rows(self):
        torch.manual_seed(19)
        cfg = Qwen2Config(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64,
            num_hidden_layers=1,
            max_position_embeddings=64,
            rope_theta=1000000.0,
        )
        norm = Qwen2RMSNorm(32, eps=1e-6).bfloat16()
        rope = Qwen2RotaryEmbedding(cfg)
        x = torch.randn(1, 12, 32).bfloat16()
        w = {
            k + ".weight": torch.eye(32, dtype=torch.bfloat16)[:16].clone()
            for k in ("k_proj", "v_proj")
        }
        w.update({k + ".bias": torch.zeros(16, dtype=torch.bfloat16) for k in ("k_proj", "v_proj")})
        idx = [6, 7, 11]
        g = norm(x)
        co, si = rope(x, torch.arange(12)[None])
        native = g[:, :, :16].reshape(1, 12, 2, 8).transpose(1, 2)
        _, oracle = apply_rotary_pos_emb(native, native, co, si)
        o = oracle.transpose(1, 2)[0, idx].reshape(3, 16)
        for arm in q.ARMS:
            s, counts = q.probe(x, 7, arm, norm, rope, w, 8)
            self.assertTrue(torch.equal(s["key_rotated"], o))
            self.assertTrue(torch.equal(s["value"], g[0, idx, :16]))
            self.assertTrue(torch.equal(s["embedding"], x[0, idx]))
            self.assertEqual(sum(counts), 24 if arm == "padded" else 12)
        self.assertEqual(
            q.witness(torch.tensor([-0.0, 0.0], dtype=torch.bfloat16)), bytes.fromhex("00800000")
        )

    def test_wire_offsets_shape_digests_and_finite(self):
        b = bytearray()
        m = {}
        for k, n in v.STAGES.items():
            a = b"\x80\x3f" * (n * 3)
            m[k] = dict(offset=len(b), bytes=len(a), sha256=v.digest(a))
            b.extend(a)
        pos, obj = v.segments(bytes(b), 0, m)
        self.assertEqual(pos, len(b))
        self.assertTrue(all(x.shape == (3, v.STAGES[k]) for k, x in obj.items()))
        for what in ("bytes", "offset", "sha256"):
            bad = copy.deepcopy(m)
            bad["sin"][what] = 1
            with self.subTest(what=what), self.assertRaises(ValueError):
                v.segments(bytes(b), 0, bad)
        b[:2] = b"\xc0\x7f"
        m["embedding"]["sha256"] = v.digest(b[: m["embedding"]["bytes"]])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            v.segments(bytes(b), 0, m)

    def test_real_oracle_reader_checks_whole_file_and_physical_bank_order(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            p = root / "128-1-0-0-inplace-kv.bf16"
            buf = np.zeros((128, 3, 8, 128), dtype="<u2")
            buf[1, 2, 7, 87] = 45
            raw = buf.tobytes()
            p.write_bytes(raw)
            r = {"kv": dict(bytes=len(raw), sha256=v.digest(raw))}
            out = v.oracle(root, (128, 1, 0), 0, "inplace", r)
            self.assertEqual(np.count_nonzero(out["key_rotated"]), 0)
            self.assertEqual(out["value"][2, 7 * 128 + 87], 45)
            buf[-1, -1, -1, -1] = 1
            p.write_bytes(buf.tobytes())
            with self.assertRaisesRegex(ValueError, "parent KV"):
                v.oracle(root, (128, 1, 0), 0, "inplace", r)

    def test_counts_include_signed_zero_not_just_numeric_disagreement(self):
        a = {k: np.zeros((3, n), dtype="<u2") for k, n in v.STAGES.items()}
        b = {k: x.copy() for k, x in a.items()}
        b["key_linear"][1, 0] = 0x8000
        d = v.comparison(a, b)
        self.assertEqual(d["key_linear"], [0, 1, 0])
        self.assertTrue(all(not any(d[k]) for k in d if k != "key_linear"))
