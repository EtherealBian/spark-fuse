import copy
import json
from pathlib import Path
import unittest
from audit235 import GIB, admission, verify_inventory


class Audit(unittest.TestCase):
    def test_reject_unfunded_or_wrong_phase(self):
        b = dict(source_object_bytes=80 * GIB, uniform_dense_fp16_kv_bytes=GIB)
        r = dict(
            phase="transfer",
            available_ram=104 * GIB,
            required_ram=103 * GIB,
            free_disk=110 * GIB,
            required_disk=96 * GIB,
            reserved_ram=22 * GIB,
            headroom_after_object_and_kv=23 * GIB,
            approved=True,
        )
        admission(r, b, "transfer")
        for k, v in [
            ("available_ram", 102 * GIB),
            ("headroom_after_object_and_kv", 21 * GIB),
            ("required_ram", 102 * GIB),
            ("free_disk", 95 * GIB),
            ("approved", 1),
            ("phase", "launch"),
        ]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                admission(dict(r, **{k: v}), b, "transfer")

    def test_actual_header_budget_tampering(self):
        here = Path(__file__).resolve().parent
        lock = here / "qwen235b-q2k.lock.json"
        b = json.loads((here / "qwen235b-headers.json").read_bytes())
        verify_inventory(lock, b)
        for key in (
            "logical_gguf_elements",
            "packed_tensor_bytes",
            "aligned_directory_bytes",
            "uniform_dense_fp16_kv_bytes",
            "tensor_padding_or_trailer_bytes",
        ):
            other = copy.deepcopy(b)
            other[key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_inventory(lock, other)
