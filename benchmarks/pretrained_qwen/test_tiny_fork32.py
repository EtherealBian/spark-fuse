from unittest import TestCase
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from prefix_snapshot import PrefixFork
from run import Runner


class Tiny(TestCase):
    @torch.inference_mode()
    def test_real_cpu_static_cache_matches_recomputed_branches(self):
        torch.manual_seed(983)
        old = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old)
        torch.set_num_threads(1)
        cfg = Qwen2Config(
            vocab_size=89,
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
        r = Runner(m, cfg, 24, torch.device("cpu"), torch.float32)
        p = (1, 2, 3, 4, 5, 6, 7)
        f = PrefixFork(r, "unit", p, 89)
        self.assertEqual(f.storage_bytes, 2 * 2 * 2 * len(p) * 8 * 4)
        for rest in [(28, 9, 10), (11, 7), (28, 9, 10)]:
            target = p + rest
            r.cache.reset()
            full = r.forward(torch.tensor([target]), r.pos[: len(target)]).clone()
            j = f.restore(r, "unit", target)
            part = r.forward(torch.tensor([rest]), r.pos[j : len(target)])
            torch.testing.assert_close(full, part, rtol=1e-5, atol=2e-6)
            self.assertEqual(full.argmax().item(), part.argmax().item())
