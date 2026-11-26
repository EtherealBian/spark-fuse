import unittest
from concurrent.futures import ThreadPoolExecutor
from slot_policy import Lease, SlotRouter


class Slots(unittest.TestCase):
    def test_hot_prefix_and_scope_isolation(self):
        r = SlotRouter(2, 64, 3)
        a = [1, 2, 3, 4]
        b = [7, 8, 9, 4]
        for ids in (a, b):
            lease = r.acquire("s", ids)
            self.assertFalse(lease.reuse)
            r.finish(lease, success=True)
        lease = r.acquire("s", [1, 2, 3, 8])
        self.assertEqual((lease.slot, lease.prefix, lease.reuse), (0, 3, True))
        r.finish(lease, success=True)
        lease = r.acquire("other", b)
        self.assertFalse(lease.reuse)
        self.assertEqual(lease.prefix, 0)

    def test_mutation_failure_restarts_and_foreign_leases(self):
        r = SlotRouter(1, 64, 2)
        v = [1, 2, 3]
        lease = r.acquire("s", v)
        v[0] = 9
        r.finish(lease, success=True)
        a = r.acquire("s", [1, 2, 4])
        self.assertEqual(a.prefix, 2)
        with self.assertRaises(ValueError):
            r.finish(Lease(a.slot, a.prefix, a.reuse), success=True)
        r.finish(a, success=False)
        a = r.acquire("s", [1, 2, 4])
        self.assertFalse(a.reuse)
        r.reset()
        with self.assertRaises(ValueError):
            r.finish(a, success=True)
        b = r.acquire("s", [1, 2, 4])
        self.assertFalse(b.reuse)
        r.finish(b, success=True)
        with self.assertRaises(ValueError):
            r.finish(b, success=False)

    def test_parallel_reservations_are_disjoint(self):
        r = SlotRouter(4, 64, 1)
        with ThreadPoolExecutor(16) as ex:
            ls = list(ex.map(lambda i: r.acquire("s", [i]), range(32)))
        self.assertEqual(sorted(lease.slot for lease in ls if lease is not None), [0, 1, 2, 3])

    def test_round_robin_and_threshold(self):
        r = SlotRouter(2, 64, 3, "round_robin")
        slots = []
        for a in ([1, 2, 3], [5, 6, 7], [1, 2, 4], [5, 6, 7]):
            lease = r.acquire("s", a)
            slots.append((lease.slot, lease.prefix))
            r.finish(lease, success=True)
        self.assertEqual(slots, [(0, 0), (1, 0), (0, 0), (1, 3)])

    def test_invalid_and_busy(self):
        r = SlotRouter(1, 4, 1)
        for scope, ids in [
            ("", [1]),
            ("a", []),
            ("a", [True]),
            ("a", [-1]),
            ("a", [1, 2, 3, 4, 5]),
        ]:
            with self.assertRaises(ValueError):
                r.acquire(scope, ids)
        a = r.acquire("a", [1])
        self.assertIsNone(r.acquire("a", [3]))
        r.finish(a, success=False)
