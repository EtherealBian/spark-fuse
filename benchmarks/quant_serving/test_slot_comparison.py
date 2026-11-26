from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from audit_slot_matchers import read, replay


class BothPanels(TestCase):
    def test_keeps_unsuccessful_design(self):
        data = replay(Path(__file__).resolve().parents[2] / "reports" / "slot-matchers")
        old, new = (data["panels"][k] for k in ("ordered64", "scoped16"))
        self.assertEqual((old["pairs"], new["pairs"]), (48, 80))
        self.assertEqual(old["outside_regression_bounds"], ["same-tenant-miss"])
        self.assertFalse(old["passes_opt_in_gate"])
        self.assertTrue(new["passes_opt_in_gate"])
        self.assertEqual((new["ten_percent_wins_on_64"], data["default"]), (9, "scan"))

    def test_reject_links_and_oversized_inputs(self):
        with TemporaryDirectory() as td:
            a = Path(td) / "a"
            a.write_bytes(b"x")
            b = Path(td) / "b"
            b.symlink_to(a)
            with self.assertRaises(ValueError):
                read(b)
            with a.open("wb") as f:
                f.truncate(4_000_001)
            with self.assertRaises(ValueError):
                read(a)
