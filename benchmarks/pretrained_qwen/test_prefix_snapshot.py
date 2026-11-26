import unittest
from types import SimpleNamespace
import torch
from prefix_snapshot import PrefixFork, banks


class Fake:
    def __init__(self):
        self.pos = torch.arange(16)
        self.cache = SimpleNamespace(
            key_cache=[torch.zeros(1, 2, 16, 4) for _ in range(2)],
            value_cache=[torch.zeros(1, 2, 16, 4) for _ in range(2)],
        )
        self.cache.reset = lambda: [t.zero_() for t in banks(self)]

    def forward(self, ids, pos):
        for i, t in enumerate(banks(self)):
            t[:, :, pos, :] = ids.float().reshape(1, 1, -1, 1) + i


class Snapshot(unittest.TestCase):
    def test_restores_independent_compact_rows_and_zeroes_tail(self):
        r = Fake()
        f = PrefixFork(r, "a", (4, 7, 3), 16)
        self.assertEqual(f.storage_bytes, 4 * 2 * 3 * 4 * 4)
        wanted = [x.clone() for x in banks(r)]
        ptrs = [x.data_ptr() for x in banks(r)]
        for last in (1, 8):
            for x in banks(r):
                x.fill_(55)
            self.assertEqual(f.restore(r, "a", (4, 7, 3, last)), 3)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(banks(r), wanted)))
            self.assertEqual([x.data_ptr() for x in banks(r)], ptrs)

    def test_rejections_before_write(self):
        r = Fake()
        q = Fake()
        f = PrefixFork(r, "a", (4, 7), 16)
        for runner, key, target in [
            (q, "a", (4, 7, 1)),
            (r, "b", (4, 7, 1)),
            (r, "a", (4, 7)),
            (r, "a", (4, 8, 1)),
            (r, "a", (4, 7, True)),
            (r, "a", (4, 7, 16)),
            (r, "a", tuple(range(16))),
        ]:
            before = [t.clone() for t in banks(runner)]
            with self.subTest(key=key, target=target), self.assertRaises(ValueError):
                f.restore(runner, key, target)
            self.assertTrue(all(torch.equal(t, b) for t, b in zip(banks(runner), before)))
        r.cache.key_cache[0] = r.cache.key_cache[0].clone() + 9
        with self.assertRaises(ValueError):
            f.restore(r, "a", (4, 7, 1))
        self.assertTrue((r.cache.key_cache[0] >= 9).all().item())
