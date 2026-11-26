import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from slot_admission import BatchAdmission, AdmissionBusy, AdmissionClosed

A = [("tenant-a", [4, 5])]
B = [("tenant-b", [4, 5])]


def make(n=2, **kw):
    return BatchAdmission(
        n,
        16,
        min_prefix=1,
        max_waiters=kw.get("waiters", 8),
        max_waiting_tokens=kw.get("tokens", 64),
    )


def wait(b, n):
    end = time.monotonic() + 2
    while b.snapshot()["waiting_batches"] != n:
        if time.monotonic() > end:
            raise AssertionError("waiter did not enter/leave")
        time.sleep(0.001)


class Admission(unittest.TestCase):
    def test_zero_queue_exact_scope_and_failed_completion(self):
        b = make(1, waiters=0, tokens=0)
        (lease,) = b.reserve(A)
        self.assertFalse(lease.reuse)
        with self.assertRaises(AdmissionBusy):
            b.reserve(A)
        with self.assertRaises(TimeoutError):
            b.reserve(A, timeout_s=0)
        b.finish(lease, success=True)
        (lease,) = b.reserve(A, timeout_s=0)
        self.assertEqual(lease.prefix, 2)
        b.finish(lease, success=True)
        (lease,) = b.reserve(B)
        self.assertFalse(lease.reuse)
        b.finish(lease, success=False)
        (lease,) = b.reserve(B)
        self.assertFalse(lease.reuse)
        b.finish(lease, success=True)
        self.assertEqual(b.snapshot()["router_calls"], 4)

    def test_head_of_line_all_or_none(self):
        b = make()
        (x,) = b.reserve(A)
        with ThreadPoolExecutor(2) as ex:
            head = ex.submit(b.reserve, A + B, timeout_s=3)
            wait(b, 1)
            tail = ex.submit(b.reserve, B, timeout_s=3)
            wait(b, 2)
            self.assertEqual(b.snapshot()["active_slots"], 1)
            b.finish(x, success=True)
            joint = head.result(2)
            self.assertEqual(len(joint), 2)
            self.assertFalse(tail.done())
            b.finish(joint[0], success=True)
            (last,) = tail.result(2)
            b.finish(joint[1], success=False)
            b.finish(last, success=True)
        self.assertEqual(b.snapshot()["active_slots"], 0)

    def test_deadline_removes_head_and_unblocks_successor(self):
        b = make()
        (x,) = b.reserve(A)
        with ThreadPoolExecutor(2) as ex:
            head = ex.submit(b.reserve, A + B, timeout_s=0.35)
            wait(b, 1)
            tail = ex.submit(b.reserve, B, timeout_s=3)
            wait(b, 2)
            with self.assertRaises(TimeoutError):
                head.result(2)
            (lease,) = tail.result(2)
            self.assertNotEqual(lease.slot, x.slot)
            b.finish(x, success=False)
            b.finish(lease, success=True)
        s = b.snapshot()
        self.assertEqual((s["timeouts"], s["waiting_tokens"], s["router_calls"]), (1, 0, 2))

    def test_mutable_input_is_frozen_before_wait_and_budget_is_refunded(self):
        b = make(1, tokens=2)
        (x,) = b.reserve(A)
        row = [["tenant-a", [4, 5]]]
        with ThreadPoolExecutor(1) as ex:
            f = ex.submit(b.reserve, row, timeout_s=3)
            wait(b, 1)
            row[0][0] = "tenant-b"
            row[0][1].clear()
            with self.assertRaises(AdmissionBusy):
                b.reserve(B, timeout_s=1)
            b.finish(x, success=True)
            (lease,) = f.result(2)
            self.assertEqual(lease.prefix, 2)
            b.finish(lease, success=False)
        self.assertEqual(b.snapshot()["waiting_tokens"], 0)

    def test_close_wakes_all_without_releasing_inflight(self):
        b = make(1)
        (x,) = b.reserve(A)
        with ThreadPoolExecutor(3) as ex:
            fs = [ex.submit(b.reserve, A, timeout_s=3) for _ in range(3)]
            wait(b, 3)
            b.close()
            b.close()
            for f in fs:
                with self.assertRaises(AdmissionClosed):
                    f.result(2)
        with self.assertRaises(AdmissionClosed):
            b.reserve(A)
        self.assertEqual(b.snapshot()["active_slots"], 1)
        b.finish(x, success=False)
        self.assertEqual(b.snapshot()["closed_waiters"], 3)

    def test_rejects_double_foreign_or_invalid_finish(self):
        b, c = make(), make()
        (x,) = b.reserve(A)
        (y,) = c.reserve(B)
        with self.assertRaises(ValueError):
            b.finish(y, success=True)
        with self.assertRaises(ValueError):
            b.finish(x, success=1)
        b.finish(x, success=False)
        with self.assertRaises(ValueError):
            b.finish(x, success=False)
        c.finish(y, success=True)
        self.assertEqual(b.snapshot()["active_slots"], 0)

    def test_bad_args_never_enter_queue(self):
        b = make()
        for bad in ([], [("", [1])], [("a", [True])], [("a", [-1])], [("a", [0] * 17)], A + B + A):
            with self.assertRaises(ValueError):
                b.reserve(bad)
        for t in (-1, True, float("nan"), float("inf"), 10**10000):
            with self.assertRaises(ValueError):
                b.reserve(A, timeout_s=t)
        self.assertTrue(all(v == 0 for v in b.snapshot().values()))
