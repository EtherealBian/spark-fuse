import copy
import json
from pathlib import Path
import tempfile
import unittest
from admit235 import assess
from audit235 import GIB, admission, verify_inventory


class AdmissionContract(unittest.TestCase):
    def setUp(self):
        here = Path(__file__).resolve().parent
        self.lock = here / "qwen235b-q2k.lock.json"
        self.budget = json.loads((here / "qwen235b-headers.json").read_bytes())

    def test_boundary_exactly_and_one_byte_short(self):
        b = self.budget
        ram = b["source_object_bytes"] + b["uniform_dense_fp16_kv_bytes"] + 22 * GIB
        for phase in ("launch", "transfer"):
            disk = 16 * GIB + (b["source_object_bytes"] if phase == "transfer" else 0)
            receipt = assess(b, ram, disk, phase)
            admission(receipt, b, phase)
            for mem, space in ((ram - 1, disk), (ram, disk - 1)):
                r = assess(b, mem, space, phase)
                self.assertFalse(r["approved"])
                with self.assertRaises(ValueError):
                    admission(r, b, phase)
        for values in ((True, 5, "launch"), (0, -1, "launch"), (1, 1, "other")):
            with self.assertRaises(ValueError):
                assess(b, *values)

    def test_digests_cannot_be_substituted_consistently(self):
        b = copy.deepcopy(self.budget)
        b["shards"][1]["directory_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            verify_inventory(self.lock, b)
        with tempfile.TemporaryDirectory() as tmp:
            changed = json.loads(self.lock.read_bytes())
            changed["files"][0]["sha256"] = "1" * 64
            fake = Path(tmp) / "lock.json"
            fake.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                verify_inventory(fake, self.budget)
