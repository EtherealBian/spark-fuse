from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from audit_feedback_stack import GIB, decoded, qualify, guard, verified_summary


class Acceptance(unittest.TestCase):
    def test_fast_operator_does_not_promote_whole_model(self):
        for loop in (False, True):
            for native in (None, False, True):
                r = qualify(True, loop, native)
                self.assertFalse(r["recommend_default_change"])
                self.assertEqual(r["native_vs_queued"], "not_directly_measured")
                self.assertEqual(
                    r["native_vs_torch_graph"],
                    {None: "not_evaluated_here", True: "passed", False: "did_not_meet_gate"}[
                        native
                    ],
                )
        with self.assertRaises(ValueError):
            qualify(True, False, 1)

    def test_duplicate_keys_nonfinite_and_byte_identity(self):
        for bad in (b'{"a":1,"a":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                decoded(bad)
        with TemporaryDirectory() as td:
            p = Path(td)
            (p / "summary.json").write_text('{"count": 1}')
            with self.assertRaises(ValueError):
                verified_summary(p, {"count": 1})

    def test_admission_not_just_a_boolean(self):
        a = dict(
            available_bytes=90 * GIB,
            required_bytes=84 * GIB,
            free_disk_bytes=8 * GIB,
            required_disk_bytes=2 * GIB,
            admitted=True,
        )
        m = dict(
            schema=2,
            floor_bytes=6 * GIB,
            reason="workload-finished",
            samples=4,
            below=0,
            min_available_bytes=8 * GIB,
            phases={
                k: dict(samples=2, min_available_bytes=8 * GIB) for k in ("loading", "requests")
            },
        )
        guard(a, m)
        for k, value in [
            ("required_bytes", 0),
            ("free_disk_bytes", True),
            ("available_bytes", 0),
            ("admitted", 1),
        ]:
            with self.assertRaises(ValueError):
                guard({**a, k: value}, m)
        with self.assertRaises(ValueError):
            guard(a, {**m, "reason": "monitoring"})
