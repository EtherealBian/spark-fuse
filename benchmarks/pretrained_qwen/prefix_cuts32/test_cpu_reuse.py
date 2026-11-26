"""Fixed CPU targets; tiny FP32 model is a harness check, never 32B evidence."""

import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from prefix_snapshot import PrefixFork
from run import Runner


class Tiny(unittest.TestCase):
    @torch.inference_mode()
    def test_same_id_sequences_at_neighboring_cuts(self):
        torch.set_num_threads(1)
        torch.manual_seed(617)
        cfg = Qwen2Config(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            use_sliding_window=False,
        )
        cfg._attn_implementation = "sdpa"
        m = Qwen2ForCausalLM(cfg).eval()
        r = Runner(m, cfg, 32, torch.device("cpu"), torch.float32)
        fixed = (tuple(range(1, 25)), tuple(range(1, 19)) + (31, 32, 33, 34, 35, 36))
        ref = []
        for target in fixed:
            r.cache.reset()
            ref.append(r.forward(torch.tensor([target]), r.pos[:24]).clone())
        for p in (5, 6, 11, 12, 17, 18):
            with self.subTest(p=p):
                f = PrefixFork(r, "one_owner", fixed[0][:p], 97)
                self.assertEqual(f.storage_bytes, 2 * 2 * 2 * p * 8 * 4)
                for i in (1, 0, 1):
                    target = fixed[i]
                    j = f.restore(r, "one_owner", target)
                    actual = r.forward(torch.tensor([target[j:]]), r.pos[j:24])
                    torch.testing.assert_close(actual, ref[i], rtol=1e-5, atol=2e-6)
                    self.assertEqual(actual.argmax().item(), ref[i].argmax().item())
