"""Derive paid timings for the two requests actually observed. No extrapolated fanout."""

from pathlib import Path
import hashlib
import json
import math
import sys
from statistics import median


def check(x, msg):
    if not x:
        raise ValueError(msg)


def positive(x):
    return type(x) in (int, float) and math.isfinite(x) and x > 0


def analyze(raw):
    cells = []
    for cell in raw["cases"]:
        rows = []
        for j, pair in enumerate(cell["pairs"]):
            a, b = pair["waves"]
            ordered = (a["arm"], b["arm"])
            check(
                pair["pair"] == j
                and ordered == (("cold", "snapshot") if j in (0, 3) else ("snapshot", "cold")),
                "arm order",
            )
            waves = {a["arm"]: a, b["arm"]: b}
            c, s = waves["cold"], waves["snapshot"]
            ct, st = c["targets"], s["targets"]
            br = [t["branch"] for t in ct]
            check(br == [t["branch"] for t in st] == ([0, 1] if j < 2 else [1, 0]), "branch order")
            check(c["prepare_ms"] == 0 and positive(s["prepare_ms"]), "preparation")
            check(
                all(positive(t["wall_ms"]) for t in ct + st)
                and all(positive(w["wave_ms"]) for w in (c, s)),
                "timer",
            )
            cw, sw = c["wave_ms"], s["wave_ms"]
            prep = s["prepare_ms"]
            branch_gains = [x["wall_ms"] - y["wall_ms"] for x, y in zip(ct, st, strict=True)]
            rc = cw - sum(t["wall_ms"] for t in ct)
            rs = sw - prep - sum(t["wall_ms"] for t in st)
            check(min(rc, rs) >= 0, "negative residual")
            gain = cw - sw
            check(abs(gain - (sum(branch_gains) - prep + rc - rs)) < 1e-7, "unbalanced ledger")
            rows.append(
                dict(
                    pair=j,
                    first_arm=ordered[0],
                    first_branch=br[0],
                    prepare_ms=prep,
                    first_target_saved_ms=branch_gains[0],
                    second_target_saved_ms=branch_gains[1],
                    first_net_before_unassigned_overhead_ms=branch_gains[0] - prep,
                    residual_difference_ms=rc - rs,
                    cold_residual_ms=rc,
                    snapshot_residual_ms=rs,
                    net_wave_saved_ms=gain,
                    paired_fraction_saved=gain / cw,
                )
            )
        check(len(rows) == 4, "not four crossed pairs")
        keys = (
            "prepare_ms",
            "first_target_saved_ms",
            "second_target_saved_ms",
            "first_net_before_unassigned_overhead_ms",
            "net_wave_saved_ms",
            "paired_fraction_saved",
        )
        summary = {
            k: dict(low=min(v), median=median(v), high=max(v))
            for k in keys
            for v in [[r[k] for r in rows]]
        }
        cells.append(
            dict(
                length=cell["length"],
                prefix=cell["prefix"],
                family=cell["family"],
                n_pairs=len(rows),
                wins=sum(r["net_wave_saved_ms"] > 0 for r in rows),
                summary=summary,
                pairs=rows,
            )
        )
    return cells


def report(f):
    b = f.read_bytes()
    raw = json.loads(b)
    return dict(
        schema="sparkfuse.prefix_cost.v1",
        source_sha256=hashlib.sha256(b).hexdigest(),
        units="milliseconds; positive savings favor snapshot; ratios unitless",
        scope="Sequential, two measured requests per wave. First subtotal excludes unassigned outer-loop overhead and is not TTFT. Medians are computed per statistic, not added across columns. No extrapolation to other branch counts.",
        cells=analyze(raw),
    )


if __name__ == "__main__":
    print(json.dumps(report(Path(sys.argv[1])), indent=2, allow_nan=False))
