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
        commands.append(
            (
                "bert_self_output_native",
                [
                    opt,
                    str(ROOT / "tools/bert_output.py"),
                    "--verify-only",
                    "--output",
                    str(report.with_name(report.stem + "-bert.json")),
                ],
            )
        )
        commands.append(
            (
                "bert_encoder_layer_fixed_order",
                [
                    opt,
                    str(ROOT / "tools/bert_layer.py"),
                    "--verify-only",
                    "--output",
                    str(report.with_name(report.stem + "-bert-layer.json")),
                ],
            )
        )
        commands.append(
            (
                "bert_host_call_equivalence",
                [
                    opt,
                    str(ROOT / "tools/host_call_panel.py"),
                    "--case",
                    "bert",
                    "--seq",
                    "33",
                    "--verify-only",
                    "--output",
                    str(report.with_name(report.stem + "-host-calls.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_decoder_layer_residual_barrier",
                [
                    opt,
                    str(ROOT / "tools/qwen_decoder.py"),
                    "--verify-only",
                    "--output",
                    str(report.with_name(report.stem + "-qwen-decoder.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_decoder_paired_rotary",
                [
                    opt,
                    str(ROOT / "tools/qwen_decoder.py"),
                    "--verify-only",
                    "--rotary",
                    "--seed-offset",
                    "11039",
                    "--output",
                    str(report.with_name(report.stem + "-rotary.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_decode_cache_carry",
                [
                    opt,
                    str(ROOT / "tools/qwen_cached_decode.py"),
                    "--past",
                    "63",
                    "--verify-only",
                    "--seed-offset",
                    "12049",
                    "--output",
                    str(report.with_name(report.stem + "-cache.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_decode_head_repeat",
                [
                    opt,
                    str(ROOT / "tools/qwen_cached_decode.py"),
                    "--past",
                    "255",
                    "1025",
                    "--verify-only",
                    "--head-repeats",
                    "--framework-softmax",
                    "--seed-offset",
                    "49481",
                    "--output",
                    str(report.with_name(report.stem + "-head-repeat.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_decode_ordered_scores",
                [
                    opt,
                    str(ROOT / "tools/qwen_cached_decode.py"),
                    "--past",
                    "1023",
                    "2047",
                    "--verify-only",
                    "--score-panel",
                    "--seed-offset",
                    "59879",
                    "--output",
                    str(report.with_name(report.stem + "-ordered-scores.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_decode_guarded_bf16_scores",
                [
                    opt,
                    str(ROOT / "tools/qwen_cached_decode.py"),
                    "--past",
                    "1535",
                    "2047",
                    "--verify-only",
                    "--score-panel",
                    "--guard-panel",
                    "--seed-offset",
                    "76031",
                    "--output",
                    str(report.with_name(report.stem + "-guarded-scores.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_decode_cache_fanout",
                [
                    opt,
                    str(ROOT / "tools/qwen_cached_decode.py"),
                    "--past",
                    "63",
                    "1024",
                    "2047",
                    "--verify-only",
                    "--fanout-panel",
                    "--seed-offset",
                    "86783",
                    "--output",
                    str(report.with_name(report.stem + "-fanout.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_decode_packed_swiglu",
                [
                    opt,
                    str(ROOT / "tools/qwen_packed_decode.py"),
                    "--past",
                    "63",
                    "2047",
                    "--verify-only",
                    "--seed-offset",
                    "91733",
                    "--output",
                    str(report.with_name(report.stem + "-packed-gate.json")),
                ],
            )
        )
        for proj in ("gated-gelu", "gated-gelu_pytorch_tanh"):
            commands.append(
                (
                    "t5_" + proj,
                    [
                        opt,
                        str(ROOT / "tools/t5_gate.py"),
                        "--proj",
                        proj,
                        "--verify-only",
                        "--output",
                        str(report.with_name(report.stem + "-t5-" + proj + ".json")),
                    ],
                )
            )
    commands.append(
        (
            "all_low_precision_activation_encodings",
            [
                sys.executable,
                str(ROOT / "tools/activation_audit.py"),
                "--output",
                str(report.with_name(report.stem + "-activations.json")),
            ],
        )
    )
    commands.append(
        (
            "wide_cached_score_schedules",
            [
                sys.executable,
                str(ROOT / "tools/wide_score_audit.py"),
                "--output",
                str(report.with_name(report.stem + "-wide-scores.json")),
            ],
        )
    )
    commands.append(
        (
            "dynamic_native_shape_matrix",
            [
                sys.executable,
                str(ROOT / "tools/dynamic_native_panel.py"),
                "--verify-only",
                "--output",
                str(report.with_name(report.stem + "-dynamic-native.json")),
            ],
        )
    )
    sealed = sorted((ROOT / "reports").glob("campaign-*.json.gz"))
    if sealed:
        commands.append(
            (
                "offline_campaign_evidence",
                [
                    sys.executable,
                    str(ROOT / "tools/campaign_snapshot.py"),
                    "verify",
                    *map(str, sealed),
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
    passed = len(results) == len(commands) and all(x["status"] == "pass" for x in results)
    tmp = report.with_suffix(report.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            {
                "checks": results,
                "passed": passed,
                "completed_epoch": time.time(),
            },
            indent=2,
        )
        + "\n"
    )

    tmp.replace(report)
    return passed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("until", type=float)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    raise SystemExit(0 if check(a.until, a.report) else 1)
