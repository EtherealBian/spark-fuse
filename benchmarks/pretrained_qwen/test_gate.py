"""Guard the actual integration separately from end-to-end timing/teacher-forced logits."""

from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GatedMLP
from projection import join, PackedDecodeMLP


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(5, 11)
        self.up_proj = nn.Linear(5, 11)
        self.down_proj = nn.Linear(11, 5)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class GateTests(unittest.TestCase):
    def test_shape_guards_and_shared_weights(self):
        torch.manual_seed(703)
        old = Tiny().eval()
        calls = []

        def fused(g, u):
            self.assertEqual(tuple(g.shape), (3, 11))
            calls.append(True)
            return torch.nn.functional.silu(g) * u

        new = GatedMLP(old, {3: fused})
        for shape in [(2, 3, 5), (1, 4, 5), (3, 5), (1, 3, 5)]:
            x = torch.randn(*shape)
            self.assertTrue(torch.equal(old(x), new(x)))
        self.assertEqual(calls, [True])
        self.assertIs(new.gate_proj.weight, old.gate_proj.weight)

    def test_packing_order_and_fallback(self):
        old = Tiny().eval()
        old.gate_proj = nn.Linear(5, 11, bias=False)
        old.up_proj = nn.Linear(5, 11, bias=False)
        with self.assertRaises(ValueError):
            join(old)
        old.requires_grad_(False)
        w = join(old)
        self.assertTrue(torch.equal(w[:11], old.gate_proj.weight))
        self.assertTrue(torch.equal(w[11:], old.up_proj.weight))
        calls = []

        def gate(g, u):
            calls.append((g.shape, u.shape))
            return torch.nn.functional.silu(g) * u

        packed = PackedDecodeMLP(old, w, gate)
        for shape in [(1, 1, 5), (1, 2, 5), (2, 1, 5)]:
            x = torch.randn(*shape)
            torch.testing.assert_close(packed(x), old(x), rtol=1e-5, atol=1e-6)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(ValueError):
            join(Tiny().requires_grad_(False))


if __name__ == "__main__":
    unittest.main()
