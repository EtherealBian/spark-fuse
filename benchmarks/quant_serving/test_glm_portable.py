from pathlib import Path
import unittest
from audit_glm355 import GIB, assess, gate, read_inputs
from glm_directory import BLOCKS, packed_size

D = Path(__file__).resolve().parent


class Portable(unittest.TestCase):
    def pinned(self):
        return read_inputs(D / "glm45-udiq1s.lock.json", D / "glm45-budget.json")

    def test_pinned_geometry(self):
        m, b = self.pinned()
        self.assertEqual(sum(x["bytes"] for x in m["files"]), b["source_object_bytes"])
        self.assertEqual(b["layers"], 93)
        self.assertEqual((b["active_experts"], b["total_experts"]), (8, 160))
        self.assertEqual(set(map(int, b["tensor_types"])), set(BLOCKS))
        self.assertEqual(b["uniform_dense_fp16_kv_bytes"], 2 * 4096 * 93 * 8 * (128 + 128))
        self.assertEqual(
            sum(b["packed_tensor_bytes_by_type"].values())
            + b["aligned_directory_bytes"]
            + b["tensor_padding_or_trailer_bytes"],
            b["source_object_bytes"],
        )

    def test_both_admissions_at_their_boundary(self):
        _, b = self.pinned()
        ram = 125451373440
        for phase, disk in [
            ("transfer", 16 * GIB + b["source_object_bytes"]),
            ("launch", 16 * GIB),
        ]:
            x = assess(phase, b, ram, disk)
            gate(phase, b, x)
            for a, d in ((ram - 1, disk), (ram, disk - 1)):
                with self.subTest(phase=phase, a=a, d=d), self.assertRaises(ValueError):
                    gate(phase, b, assess(phase, b, a, d) | {"approved": True})

    def test_no_favorable_type_coercions(self):
        _, b = self.pinned()
        x = assess("launch", b, 126000000000, 20 * GIB)
        for patch in [
            {"approved": 1},
            {"available_ram": True},
            {"reserved_ram": 0},
            {"phase": "transfer"},
        ]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                gate("launch", b, x | patch)

    def test_block_axis_packing(self):
        for kind, (_, stride, nbytes) in BLOCKS.items():
            self.assertEqual(packed_size((2 * stride, 7, 3), kind), 2 * nbytes * 7 * 3)
            if stride > 1:
                with self.assertRaises(ValueError):
                    packed_size((stride - 1, 7), kind)
