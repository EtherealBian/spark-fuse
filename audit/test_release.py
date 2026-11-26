"""Run with python3 -I -S audit/test_release.py (no dependencies and no GPU)."""

from pathlib import Path
import hashlib, json, sqlite3, sys, unittest
from contextlib import closing
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release import acceptance, deadline_gate, logical_catalog, sanitizer_log, source_digest, STAGES, CUTOFF_STAGE, separate_windows


class EvidenceTests(unittest.TestCase):
    def test_sanitizers_require_each_tool_its_tests_and_clean_exit(self):
        def block(t):
            s = (
                "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                if t == "racecheck"
                else "ERROR SUMMARY: 0 errors"
            )
            return f"demo: {t}\n5 passed, 1 warning in 2.4s\n========= {s}\n"

        clean = "".join(map(block, ["memcheck", "initcheck", "racecheck", "synccheck"]))
        self.assertEqual(len(sanitizer_log(clean, expected_init=True)), 4)
        for bad in (
            clean.replace("initcheck", "memcheck"),
            clean.replace("0 warnings", "1 warning"),
            clean.replace("5 passed", "0 passed"),
            clean.replace(block("synccheck"), ""),
        ):
            with self.subTest(), self.assertRaises(ValueError):
                sanitizer_log(bad, expected_init=True)

    def test_deadline_needs_independent_complete_matching_smoke_tree(self):
        def write(p, x):
            p.write_text(json.dumps(x))

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = {
                "checks": [
                    {
                        "check": "tests",
                        "status": "pass",
                        "returncode": 0,
                        "tail": "5 passed in 0.65s",
                    }
                ],
                "passed": True,
                "completed_epoch": 120.0,
            }
            original = CUTOFF_STAGE + "-stage-acceptance"
            rerun = "gb10-overnight-acceptance"
            key = "a" * 64
            smoke = {
                "provenance": {"source_tree_sha256": key, "archive": "sources/" + key + ".tar.gz"}
            }
            deadline = "1970-01-01T00:05:00+00:00"
            f = lambda n: root / (n + ".json")
            for tag in (original, rerun):
                write(f(tag), release)
                write(f(tag + "-probe"), smoke)
            self.assertEqual(deadline_gate(root, key, deadline)["source_stamped"], 1)
            for edit in ("late", "suite", "source", "missing"):
                write(f(rerun), release)
                write(f(rerun + "-probe"), smoke)
                if edit == "late":
                    write(f(rerun), dict(release, completed_epoch=300.1))
                elif edit == "suite":
                    write(
                        f(rerun),
                        dict(release, checks=[dict(release["checks"][0], tail="2 passed")]),
                    )
                elif edit == "source":
                    write(
                        f(rerun + "-probe"),
                        {"provenance": dict(smoke["provenance"], source_tree_sha256="b" * 64)},
                    )
                else:
                    f(rerun + "-probe").unlink()
                with self.subTest(edit=edit), self.assertRaises(ValueError):
                    deadline_gate(root, key, deadline)
        with self.assertRaises(ValueError):
            acceptance(dict(release, checks=release["checks"] * 2))

    def test_supplement_is_later_and_extends_the_cutoff_suite(self):
        def record(ts, n, gates):
            tests = {"check": "tests", "status": "pass", "returncode": 0, "tail": f"{n} passed"}
            return {"passed": True, "completed_epoch": ts, "checks": [tests] + [dict(tests, check=g) for g in gates]}
        end = "1970-01-01T00:05:00+00:00"
        early = record(250, 5, ["dispatch"])
        late = record(360, 6, ["dispatch", "rows"])
        self.assertEqual(separate_windows(early, late, end)["added_gates"], ["rows"])
        for first, last in [(early, dict(late, completed_epoch=299)),
                            (dict(early, completed_epoch=301), late),
                            (early, record(360, 5, ["dispatch", "rows"])),
                            (early, record(360, 6, ["replacement", "rows"]))]:
            with self.subTest(), self.assertRaises(ValueError):
                separate_windows(first, last, end)

    def test_catalog_digest_is_order_independent_but_row_sensitive(self):
        with TemporaryDirectory() as tmp:
            results = []
            for i, rows in enumerate((("a", "b"), ("b", "a"), ("a", "c"))):
                db = Path(tmp) / str(i)
                with closing(sqlite3.connect(db)) as conn, conn:
                    conn.execute("create table schedules(key text,record text)")
                    conn.executemany(
                        "insert into schedules values (?,?)",
                        [(k, json.dumps({"key": k})) for k in rows],
                    )
                results.append(logical_catalog(db))
            self.assertEqual(results[0], results[1])
            self.assertNotEqual(results[0], results[2])
            self.assertEqual(results[0][0], 2)

    def test_source_identity_ignores_reports_but_includes_cuda_host_code(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "sparkfuse/csrc/host.cpp"
            p.parent.mkdir(parents=True)
            p.write_bytes(b"hello")
            code = hashlib.sha256(
                b"sparkfuse/csrc/host.cpp\0" + hashlib.sha256(b"hello").digest()
            ).hexdigest()
            self.assertEqual(source_digest(root), code)
            (root / "notes.txt").write_text("hello")
            self.assertEqual(source_digest(root), code)
            p.write_bytes(b"bye")
            self.assertNotEqual(source_digest(root), code)


if __name__ == "__main__":
    unittest.main()
