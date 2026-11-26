import itertools
import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from slot_policy import minimum_assignment, SlotRouter


class Joint(unittest.TestCase):
    def test_matches_exhaustive_rectangular_oracle(self):
        rng = random.Random(1781)
        for rows in range(1, 5):
            for cols in range(rows, 7):
                for _ in range(8):
                    c = [[rng.randrange(-5, 7) for _ in range(cols)] for _ in range(rows)]
                    out = minimum_assignment(c)
                    self.assertEqual(len(set(out)), rows)
                    self.assertEqual(
                        sum(c[i][k] for i, k in enumerate(out)),
                        min(
                            sum(c[i][k] for i, k in enumerate(p))
                            for p in itertools.permutations(range(cols), rows)
                        ),
                    )
        c = [[0] * 64 for _ in range(64)]
        self.assertEqual(minimum_assignment(c), list(range(64)))

    def test_tradeoff_foreign_scope_and_failed_completion(self):
        def seeded():
            r = SlotRouter(2, 20, 1)
            ls = r.acquire_batch([("a", [1, 1, 1, 1, 2]), ("a", [1, 2, 3])])
            for lease in ls:
                r.finish(lease, success=True)
            return r

        for policy, expected in [("greedy", [3, 1]), ("joint", [1, 4])]:
            r = seeded()
            ls = r.acquire_batch([("a", [1, 1, 1, 3]), ("a", [1, 1, 1, 1, 3])], strategy=policy)
            self.assertEqual([lease.prefix for lease in ls], expected)
            self.assertIsNone(r.acquire_batch([("a", [1])]))
            for lease in ls:
                r.finish(lease, success=False)
            self.assertEqual(r.acquire("a", [1]).prefix, 0)
        r = seeded()
        ls = r.acquire_batch([("b", [1, 1, 1, 1, 2]), ("a", [1, 1, 1, 1, 3])])
        self.assertEqual([lease.prefix for lease in ls], [0, 4])
        r.reset()
        with self.assertRaises(ValueError):
            r.finish(ls[0], success=True)

    def test_atomic_all_or_none_and_no_partial_reservation(self):
        r = SlotRouter(2, 10, 1)
        gate = Barrier(2)

        def reserve(_):
            gate.wait()
            return r.acquire_batch([("tenant", [3]), ("tenant", [4])])

        with ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(reserve, [0, 1]))
        self.assertEqual(sum(out is None for out in results), 1)
        self.assertIsNone(r.acquire("any", [5]))
        r.reset()
        with self.assertRaises(ValueError):
            r.acquire_batch([("a", [1]), ("a", [-1])])
        self.assertEqual(len(r.acquire_batch([("a", [1]), ("a", [2])])), 2)
