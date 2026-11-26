"""Alter measured evidence, including a re-signed ledger, and assert the accounting fails closed."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from audit_expert import MODELS, audit


class MeasuredLedgerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The portable checkout stores ledgers in reports/expert-ledger; never invent one if missing.
        cls.root = Path.cwd()
        cls.reports = cls.root / "reports"
        cls.folder = cls.reports / "expert-ledger"
        cls.sources = cls.root / "benchmarks/quant_serving"

    def test_measured_receipts_and_dense_control(self):
        r = audit(self.folder, self.sources, self.reports)
        self.assertEqual(r["72"]["one_token_expert_bytes"], 0)
        self.assertEqual(
            (r["120"]["one_token_expert_bytes"], r["235"]["one_token_expert_bytes"]),
            (1908541440, 5137760256),
        )

    def test_rejects_coherent_digest_with_invalid_accounting(self):
        for label in (
            "extra_expert",
            "drop_tensor",
            "move_shard",
            "imaginary_saving",
            "invented_dense",
            "changed_abi",
            "miss_ledger",
        ):
            with self.subTest(case=label), TemporaryDirectory() as td:
                dest = Path(td)
                for f in ["verified.json", *(k + ".json" for k in MODELS)]:
                    (dest / f).write_bytes((self.folder / f).read_bytes())
                key = "72" if label == "invented_dense" else "235"
                p = dest / (key + ".json")
                x = json.loads(p.read_bytes())
                v = dest / "verified.json"
                proof = json.loads(v.read_bytes())
                ref = proof["models"][key]
                if label == "extra_expert":
                    x["packed_bytes_by_group"]["routed_expert"] += 128
                elif label == "drop_tensor":
                    x["shards"][0]["tensors"] -= 1
                elif label == "move_shard":
                    x["shards"].reverse()
                elif label == "imaginary_saving":
                    x["routed_expert_union_bounds"]["8"]["min_bytes"] -= 1
                elif label == "invented_dense":
                    x["routed_expert_union_bounds"] = {"1": dict(min_bytes=0, max_bytes=0)}
                elif label == "changed_abi":
                    ref["abi_sha256"] = "0" * 64
                else:
                    del proof["models"]["120"]
                b = json.dumps(x, indent=2).encode()
                p.write_bytes(b)
                ref["ledger_sha256"] = hashlib.sha256(b).hexdigest()
                v.write_text(json.dumps(proof))
                with self.assertRaises(ValueError):
                    audit(dest, self.sources, self.reports)


if __name__ == "__main__":
    unittest.main()
