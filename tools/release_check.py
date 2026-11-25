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
    commands.append(
        (
            "residual_token_quant_projection",
            [
                sys.executable,
                str(ROOT / "tools/token_quant_panel.py"),
                "--smoke",
                "--output",
                str(report.with_name(report.stem + "-token-quant.json")),
            ],
        )
    )
    commands.append(
        (
            "int32_dequantization_ablation",
            [
                sys.executable,
                str(ROOT / "tools/projection_epilogue_panel.py"),
                "--smoke",
                "--output",
                str(report.with_name(report.stem + "-projection-epilogue.json")),
            ],
        )
    )
    commands.append(
        (
            "frozen_w8a8_rhs_layout",
            [
                sys.executable,
                str(ROOT / "tools/quant_state_panel.py"),
                "--smoke",
                "--output",
                str(report.with_name(report.stem + "-quant-state.json")),
            ],
        )
    )
    commands.append(
        (
            "grouped_projection_epilogues",
            [
                sys.executable,
                str(ROOT / "tools/group_epilogue_panel.py"),
                "--smoke",
                "--output",
                str(report.with_name(report.stem + "-grouped-epilogue.json")),
            ],
        )
    )
    commands.append(
        (
            "packed_frozen_projection_fanout",
            [
                sys.executable,
                str(ROOT / "tools/packed_projection_panel.py"),
                "--smoke",
                "--output",
                str(report.with_name(report.stem + "-packed-projection.json")),
            ],
        )
    )
    commands.append(
        (
            "native_resident_request_io",
            [
                sys.executable,
                str(ROOT / "tools/resident_io_panel.py"),
                "--smoke",
                "--output",
                str(report.with_name(report.stem + "-resident-io.json")),
            ],
        )
    )
    commands.append(
        (
            "packed_int8_shape_family",
            [
                sys.executable,
                str(ROOT / "tools/packed_family_panel.py"),
                "--smoke",
                "--output",
                str(report.with_name(report.stem + "-packed-family.json")),
            ],
        )
    )
    opt = os.environ.get("SPARKFUSE_QWEN_PY")
    qreport = report.with_name(report.stem + "-qwen.json")
    if opt:
        commands.append(
            (
                "shared_state_shape_family",
                [
                    opt,
                    str(ROOT / "tools/shape_family_panel.py"),
                    "--smoke",
                    "--output",
                    str(report.with_name(report.stem + "-shape-family.json")),
                ],
            )
        )
        commands.append(
            (
                "resident_qwen_requests",
                [
                    opt,
                    str(ROOT / "tools/resident_qwen.py"),
                    "--smoke",
                    "--output",
                    str(report.with_name(report.stem + "-resident.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_context_site_replay",
                [
                    opt,
                    str(ROOT / "tools/qwen_context.py"),
                    "--smoke",
                    "--output",
                    str(report.with_name(report.stem + "-qwen-context.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_rank3_packed_boundary",
                [
                    opt,
                    str(ROOT / "tools/qwen_packed_mean.py"),
                    "--verify-only",
                    "--seq",
                    "33",
                    "--seed-offset",
                    "80141",
                    "--output",
                    str(report.with_name(report.stem + "-qwen-packed-mean.json")),
                ],
            )
        )
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
        commands.append(
            (
                "qwen_qkv_checkpoint_layout",
                [
                    opt,
                    str(ROOT / "tools/qwen_qkv_layout.py"),
                    "--past",
                    "127",
                    "--verify-only",
                    "--seed-offset",
                    "91733",
                    "--output",
                    str(report.with_name(report.stem + "-qkv-layout.json")),
                ],
            )
        )
        commands.append(
            (
                "qwen_joint_packed_rotary",
                [
                    opt,
                    str(ROOT / "tools/qwen_packed_rotary.py"),
                    "--past",
                    "127",
                    "--verify-only",
                    "--seed-offset",
                    "91733",
                    "--output",
                    str(report.with_name(report.stem + "-packed-rotary.json")),
                ],
            )
        )
        commands.append(
            (
                "phi3_native_prefill_cached",
                [
                    opt,
                    str(ROOT / "tools/phi3_native.py"),
                    "--past",
                    "127",
                    "--rotary-dim",
                    "64",
                    "128",
                    "--seed-offset",
                    "137413",
                    "--verify-only",
                    "--output",
                    str(report.with_name(report.stem + "-phi3.json")),
                ],
            )
        )
        commands.append(
            (
                "phi3_chunked_cache_updates",
                [
                    opt,
                    str(ROOT / "tools/phi3_cache.py"),
                    "--verify-only",
                    "--past",
                    "127",
                    "--rotary-dim",
                    "64",
                    "128",
                    "--seed-offset",
                    "166909",
                    "--output",
                    str(report.with_name(report.stem + "-phi3-cache.json")),
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
    commands.append(
        (
            "decomp_precision_replay",
            [
                sys.executable,
                str(ROOT / "tools/decomp_casts.py"),
                "--case",
                "decomp-31x1024-bfloat16",
                "--verify-only",
                "--seed-offset",
                "59419",
                "--output",
                str(report.with_name(report.stem + "-decomp.json")),
            ],
        )
    )
    commands.append(
        (
            "packed_mean_bitwise",
            [
                sys.executable,
                str(ROOT / "tools/packed_mean_panel.py"),
                "--verify-only",
                "--index",
                "1",
                "--index",
                "5",
                "--seed-offset",
                "60317",
                "--output",
                str(report.with_name(report.stem + "-packed-mean.json")),
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
