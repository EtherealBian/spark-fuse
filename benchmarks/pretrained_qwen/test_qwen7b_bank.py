"""An owned bank must free original weights, preserve prefill and reject trainable input."""

from pathlib import Path
import gc
import sys
import unittest
import weakref
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resident import SharedBankMLP
from test_gate import Tiny


class ResidentTests(unittest.TestCase):
    def test_bank_and_outputs(self):
        torch.manual_seed(701)
        old = Tiny().eval()
        old.gate_proj = nn.Linear(5, 11, bias=False)
        old.up_proj = nn.Linear(5, 11, bias=False)
        with self.assertRaises(ValueError):
            SharedBankMLP(old, lambda a, b: F.silu(a) * b)
        old.requires_grad_(False)
        calls = []

        def fuse(a, b):
            calls.append(a.shape)
            return F.silu(a) * b

        m = SharedBankMLP(old, fuse)
        self.assertEqual(
            m.bank.untyped_storage().nbytes(), 2 * old.up_proj.weight.untyped_storage().nbytes()
        )
        for view in m.bank.chunk(2, 0):
            self.assertEqual(view.untyped_storage().data_ptr(), m.bank.untyped_storage().data_ptr())
        for shape in ((1, 1, 5), (1, 19, 5), (2, 1, 5), (3, 5)):
            x = torch.randn(*shape)
            torch.testing.assert_close(m(x), old(x))
        self.assertEqual(len(calls), 1)
        a, b = weakref.ref(old.gate_proj.weight), weakref.ref(old.up_proj.weight)
        del old
        gc.collect()
        self.assertIsNone(a())
        self.assertIsNone(b())


if __name__ == "__main__":
    unittest.main()
