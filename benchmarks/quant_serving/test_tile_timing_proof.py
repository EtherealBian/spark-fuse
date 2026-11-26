import json
from pathlib import Path
import shutil
import tempfile
import unittest
from tile_timing_proof import observations, reconcile_block, excluded


class Streams(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = Path(tmp.name)
        for n in ("runtime.json", "rows.json"):
            shutil.copyfile(Path("reports/qwen235b") / n, self.d / n)
        self.rows, _, self.clean, _ = observations(self.d)
        (self.d / "timing.json").write_bytes(self.clean)

    def test_live_bytes_not_just_json_equivalence(self):
        _, h = reconcile_block(self.d, 0)
        self.assertEqual(set(h), {"timing.json", "rows.json", "runtime.json"})
        (self.d / "timing.json").write_bytes(self.clean + b"\n")
        with self.assertRaisesRegex(ValueError, "stdout"):
            reconcile_block(self.d, 0)

    def test_same_rows_wrong_tile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "block"):
            reconcile_block(self.d, 1)

    def test_exclusion_requires_original_stdout_and_rows(self):
        pref = "".join(
            "case {rep} {case} {processed} {predicted}\n".format(**r) for r in self.rows["rows"]
        ).encode()
        f = self.d / "stdout.txt"
        f.write_bytes(pref + self.clean)
        self.assertIs(excluded(self.d, self.rows["inputs"])["included_in_comparison"], False)
        for bad in (
            self.clean,
            pref[1:] + self.clean,
            pref + json.dumps(json.loads(self.clean)).encode(),
        ):
            f.write_bytes(bad)
            with self.assertRaisesRegex(ValueError, "cause"):
                excluded(self.d, self.rows["inputs"])
