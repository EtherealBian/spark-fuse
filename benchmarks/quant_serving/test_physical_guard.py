import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from physical_guard import accepted, observe, empty_phases, update, FLOOR


class Guard(unittest.TestCase):
    def test_negative_gate_and_phase_reconciliation(self):
        s = dict(samples=0, min_available_bytes=None, below=0)
        phases = empty_phases()
        for phase, n in [
            ("loading", FLOOR - 2),
            ("loading", FLOOR - 1),
            ("requests", FLOOR),
            ("requests", FLOOR - 1),
            ("requests", FLOOR),
            ("requests", FLOOR + 1),
        ]:
            s = observe(s, phases, phase, n)
        report = dict(s, phases=phases, schema=2, floor_bytes=FLOOR, reason="workload-finished")
        accepted(report)
        self.assertEqual(s, dict(samples=6, min_available_bytes=FLOOR - 2, below=0))
        for key, value in [
            ("below", 3),
            ("reason", "monitoring"),
            ("reason", "low-memory-stop"),
            ("reason", "monitor-stopped"),
            ("phases", {}),
            ("samples", 8),
            ("min_available_bytes", FLOOR),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                accepted(dict(report, **{key: value}))
        bad = copy.deepcopy(report)
        bad["phases"]["requests"]["samples"] = 0
        with self.assertRaises(ValueError):
            accepted(bad)
        with self.assertRaises(ValueError):
            update(s, True)

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open"), "Linux pidfd only")
    def test_real_child_lives_through_phase_handshake(self):
        with (
            tempfile.TemporaryDirectory() as td,
            subprocess.Popen(["/usr/bin/sleep", "15"]) as child,
        ):
            p = Path(td)
            out = p / "guard.json"
            try:
                with subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        str(Path(__file__).with_name("physical_guard.py")),
                        "watch",
                        str(child.pid),
                        "/usr/bin/sleep",
                        str(out),
                    ]
                ) as watcher:
                    try:
                        for _ in range(60):
                            if out.is_file():
                                break
                            time.sleep(0.05)
                        self.assertEqual(json.loads(out.read_bytes())["reason"], "monitoring")
                        (p / "requests.started").touch()
                        time.sleep(1.2)
                        (p / "requests.finished").touch()
                        self.assertEqual(watcher.wait(6), 0)
                        accepted(json.loads(out.read_bytes()))
                        self.assertIsNone(child.poll())
                    finally:
                        if watcher.poll() is None:
                            watcher.terminate()
            finally:
                if child.poll() is None:
                    child.terminate()
