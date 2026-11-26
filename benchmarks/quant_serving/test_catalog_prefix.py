import copy
import unittest
from catalog_prefix import arms, audit, digest, order


def fixture():
    src = []
    raw = []
    for ix in range(8):
        n = 256 if ix < 4 else 2048
        shared = n - 17
        ids = [ix + 1] * n
        t = digest(ids)
        base = [2] * 24
        src.append(
            dict(
                length=n, target_ids=ids, target=t, case=str(ix), shared=shared, cold={"ids": base}
            )
        )
        opts = arms(n, shared)
        aa = {}
        for k in order(ix, opts):
            cut, reuse = opts[k]
            prev = cut if reuse else 0

            def rec(length, p, h):
                return dict(
                    ids=[2] * length,
                    predicted=length,
                    stop="limit",
                    processed=p,
                    reused=h,
                    prompt_ms=4,
                    decode_ms=1 if length > 1 else 0,
                    wall_ms=6,
                )

            aa[k] = dict(cut=cut, prime=rec(1, cut, 0), request=rec(24, n - prev, prev))
        raw.append(dict(size=n, case=str(ix), target=t, shared=shared, arms=aa))
    return src, raw


class PreparedPrefixes(unittest.TestCase):
    def test_counterfactuals_have_matched_inputs_and_exact_boundaries(self):
        src, raw = fixture()
        report = audit(raw, src)
        self.assertEqual(report["cases"]["2048"]["arms"]["block-512"]["exact"], 4)
        raw[0]["arms"]["full"]["request"]["ids"][3] = 4
        report = audit(raw, src)
        self.assertEqual(report["cases"]["256"]["arms"]["full"]["first_divergent_steps"], [4])
        self.assertEqual(report["cases"]["256"]["arms"]["full"]["exact"], 3)
        for target, k, v in [
            ("request", "reused", 0),
            ("prime", "reused", 10),
            ("request", "predicted", 15),
            ("request", "stop", "eos"),
        ]:
            changed = copy.deepcopy(raw)
            changed[0]["arms"]["full"][target][k] = v
            with self.assertRaises(ValueError):
                audit(changed, src)
        changed = copy.deepcopy(raw)
        changed[0]["arms"] = dict(reversed(changed[0]["arms"].items()))
        with self.assertRaises(ValueError):
            audit(changed, src)

    def test_break_even_is_case_paired_and_never_implies_exactness(self):
        src, raw = fixture()
        for arm in raw[0]["arms"].values():
            arm["prime"]["wall_ms"] = 10
        raw[0]["arms"]["cold-a"]["request"]["wall_ms"] = 12
        target = raw[0]["arms"]["full"]["request"]
        target["wall_ms"] = 8
        target["ids"][0] = 5
        values = audit(raw, src)["cases"]["256"]["arms"]["full"]["paired_estimates"]
        self.assertEqual(
            values[0],
            {"case": "0", "equal_to_cold": False, "saved_per_hit_ms": 4, "break_even_hits": 3},
        )
        self.assertIsNone(values[1]["break_even_hits"])
        target["wall_ms"] = 1
        with self.assertRaisesRegex(ValueError, "latency"):
            audit(raw, src)

    def test_attestation_binds_both_sessions_and_all_outputs(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import json
        from audit_catalog import load_inputs, source_files

        base = Path(__file__).resolve().parents[2] / "reports/qwen72b"
        old = json.loads((base / "prefix-rows.json").read_bytes())
        _, raw = fixture()
        for ix, (row, source) in enumerate(zip(raw, old, strict=True)):
            size, shared = source["length"], source["shared"]
            row.update(case=source["case"], target=source["target"], shared=shared)
            for name, (cut, reuse) in arms(size, shared).items():
                a = row["arms"][name]
                a["cut"] = cut
                a["prime"]["processed"] = cut
                a["request"].update(
                    processed=size - cut if reuse else size, reused=cut if reuse else 0
                )
        with TemporaryDirectory() as td:
            d = Path(td)
            for key, path in source_files(base).items():
                if key != "rows":
                    (d / f"{key}.json").write_bytes(path.read_bytes())

            def dump(name, value):
                (d / name).write_text(json.dumps(value))

            dump("rows.json", raw)
            report = load_inputs(d, base, include_rows=True)
            self.assertEqual(len(report["evidence_sha256"]), 6)
            self.assertEqual(load_inputs(d, base, include_rows=False), base / "prefix-rows.json")
            for stage in ("prime", "request"):
                a = raw[0]["arms"]["block-128"][stage]["ids"]
                a[0] = 151645
                dump("rows.json", raw)
                with self.assertRaisesRegex(ValueError, "suppressed EOG"):
                    load_inputs(d, base, include_rows=True)
                a[0] = 2
            dump("rows.json", raw)
            rt = json.loads((d / "runtime.json").read_bytes())
            rt["server_sha256"] = "1" * 64
            dump("runtime.json", rt)
            with self.assertRaisesRegex(ValueError, "different binaries"):
                load_inputs(d, base, include_rows=True)
