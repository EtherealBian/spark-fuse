"""Rebuild a portable audit from the case index, including failures and slower cases."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median


def render(run, output):
    rows = [json.loads(x) for x in (run / "index.jsonl").read_text().splitlines()]
    ok = [r for r in rows if r["status"] == "ok"]
    bad = [r for r in rows if r["status"] != "ok"]
    text = [
        "# DGX Spark / GB10 tuning report",
        "",
        f"{len(ok)} cases passed all three backends; {len(bad)} interrupted or failed. "
        f"{sum(r['trials'] for r in ok)} discovery timings; "
        f"{sum(r['invalid'] for r in ok)} rejected schedules.",
        "",
        "Times are warm CUDA Graph device-replay medians in microseconds. "
        "The benchmark reuses addresses and inputs; small tensors may fit in L2. "
        "It does not measure token throughput or cold DRAM bandwidth. "
        "Full synchronized Python-call latency and all sampling rounds are in each evidence JSON.",
        "",
        "| case / dtype | eager | Inductor | SparkFuse | vs Inductor |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in ok:
        u = r["us"]
        text.append(
            f"| {r['id']} | {u['eager']:.3f} | {u['inductor']:.3f} | {u['sparkfuse']:.3f} | {u['inductor'] / u['sparkfuse']:.2f}x |"
        )
    text += ["", "## Coverage", ""]
    for phase, n in Counter(r["phase"] for r in ok).items():
        text.append(f"- {phase}: {n}")
    if ok:
        ratios = [r["us"]["inductor"] / r["us"]["sparkfuse"] for r in ok]
        text += [
            "",
            f"Median of per-case Inductor/SparkFuse ratios: {median(ratios):.3f}x. "
            "Each tested shape is counted once; this is not a model-weighted speedup.",
        ]
    if bad:
        text += ["", "## Failed / interrupted cases", ""] + [
            f"- `{r['id']}`: {r['status']}. See index for full diagnostics." for r in bad
        ]
    output.write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    render(a.run, a.output)
