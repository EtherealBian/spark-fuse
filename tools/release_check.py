"""Final acceptance runs; does not replace slower results or clear failed cases."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(until, report):
    results = []
    commands = [("tests", [sys.executable, "-m", "pytest", "-q", "tests"])]
    commands.append(
        (
            "installed_wheel",
            [
                sys.executable,
                str(ROOT / "tools/wheel_smoke.py"),
                "--output",
                str(report.with_name(report.stem + "-wheel.json")),
            ],
        )
    )
    opt = os.environ.get("SPARKFUSE_QWEN_PY")
    qreport = report.with_name(report.stem + "-qwen.json")
    if opt:
        commands.append(
            (
                "qwen_modules",
                [opt, str(ROOT / "tools/qwen_integration.py"), "--output", str(qreport)],
            )
        )
        commands.append(
            (
                "qwen_modules_native",
                [
                    opt,
                    str(ROOT / "tools/qwen_integration.py"),
                    "--host-backend",
                    "native",
                    "--output",
                    str(report.with_name(report.stem + "-qwen-native.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_attention_native",
                [
                    opt,
                    str(ROOT / "tools/qwen_scores.py"),
                    "--output",
                    str(report.with_name(report.stem + "-attention.json")),
                ],
            )
        )
    for name, cmd in commands:
        left = until - time.time() - 3
        if left < 1:
            results.append({"check": name, "status": "no time"})
            break
        t = time.monotonic()
        try:
            r = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=min(180, left),
            )
            results.append(
                {
                    "check": name,
                    "status": "pass" if r.returncode == 0 else "fail",
                    "returncode": r.returncode,
                    "tail": r.stdout[-3500:],
                    "elapsed_s": time.monotonic() - t,
                }
            )
        except subprocess.TimeoutExpired:
            results.append({"check": name, "status": "timeout"})
    report.write_text(
        json.dumps(
            {
                "checks": results,
                "passed": all(x["status"] == "pass" for x in results),
                "completed_epoch": time.time(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("until", type=float)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    check(a.until, a.report)
