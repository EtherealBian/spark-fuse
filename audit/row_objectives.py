"""Select a request path on one seed; evaluate the frozen choice on the other.

Offline diagnostic for the exact bounded-row workload in catalog_rows_panel.py. Does not
perform new GPU timings, retune a schedule, or change the compiler's default behavior.
"""

from collections import defaultdict
from hashlib import sha256
import argparse
import json
import math
from pathlib import Path
from statistics import median

CALIBRATE, VALIDATE = 4670017, 4687319
GENERIC, TABLE = "generic_native", "table_native"
CLOCKS = ("rotating_sync_us", "queued8_us_per_forward")
WINS, RATIO = 5, 1.03


def require(cond, message):
    if not cond:
        raise ValueError(message)


def sample_medians(case, clock):
    rot = case["rotating"]
    windows = rot["windows"]
    require(len(windows) == 7, "expected seven paired windows")
    out = {}
    for path in (GENERIC, TABLE):
        require(case["gates"][path]["admitted"] is True, "path not admitted")
        values = []
        for window in windows:
            order = window["order"]
            require(order.count(path) == 1, "invalid backend order")
            vv = window["samples"][path][clock]
            require(
                len(vv) == (16 if clock == CLOCKS[0] else 5)
                and all(type(v) in (float, int) and math.isfinite(v) and v > 0 for v in vv),
                "invalid samples",
            )
            values.append(median(vv))
        require(
            median(values) == rot["median_us"][path][clock], "stored median differs from samples"
        )
        out[path] = values
    return out


def score(values):
    a, b = values[GENERIC], values[TABLE]
    return dict(
        paired_wins=sum(x > y for x, y in zip(a, b, strict=True)),
        median_paired_ratio=median(x / y for x, y in zip(a, b, strict=True)),
        generic_us=median(a),
        table_us=median(b),
    )


def decide(cal, has_hit):
    s = score(cal)
    return (
        TABLE
        if has_hit and s["paired_wins"] >= WINS and s["median_paired_ratio"] >= RATIO
        else GENERIC
    )


def evaluate(a, b, clock):
    # Deliberately freeze the choice before reading any validation timing.
    cal = sample_medians(a, clock)
    path = decide(cal, any(v["selected"] is not None for v in a["variants"]))
    valid = sample_medians(b, clock)
    score_b = score(valid)
    score_b["selected_us"] = median(valid[path])
    score_b["selected_over_generic_us"] = score_b["selected_us"] / score_b["generic_us"]
    return dict(clock=clock, selected=path, calibration=score(cal), heldout=score_b)


def analyze(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    require(data["smoke"] is False and bool(data["cases"]), "requires a full timing panel")
    rows = defaultdict(dict)
    for c in data["cases"]:
        key = tuple(c["config"])
        seed = c["seed"]
        require(
            seed in (CALIBRATE, VALIDATE) and seed not in rows[key], "duplicate or unexpected seed"
        )
        rows[key][seed] = c
    cases = []
    for key, pair in sorted(rows.items()):
        require(set(pair) == {CALIBRATE, VALIDATE}, "missing held-out seed")
        a, b = pair[CALIBRATE], pair[VALIDATE]
        require(
            all(
                a[k] == b[k]
                for k in ("config", "rows", "bounds", "variants", "hints", "generic_source_sha256")
            ),
            "regimes or kernels differ across seeds",
        )
        require(
            len(a["inputs"]) == len(b["inputs"]) == len(a["rows"])
            and not set(x[0] for x in a["inputs"]).intersection(x[0] for x in b["inputs"]),
            "calibration and holdout activations overlap",
        )
        cases.append(
            dict(
                config=list(key),
                rows=a["rows"],
                exact_rows=[v["rows"] for v in a["variants"] if v["selected"] is not None],
                objectives=[evaluate(a, b, k) for k in CLOCKS],
            )
        )
    totals = {}
    for j, k in enumerate(CLOCKS):
        entries = [c["objectives"][j] for c in cases]
        admitted = [x for x in entries if x["selected"] == TABLE]
        ratios = [x["heldout"]["selected_over_generic_us"] for x in admitted]
        totals[k] = dict(
            configurations=len(entries),
            selected_table=len(admitted),
            heldout_faster=sum(v < 1 for v in ratios),
            heldout_slower=sum(v > 1 for v in ratios),
            heldout_slower_over_3pct=sum(v > 1.03 for v in ratios),
            heldout_selected_generic=len(entries) - len(admitted),
        )
    return dict(
        format="sparkfuse.heldout-request-objectives.v1",
        input_sha256=sha256(raw).hexdigest(),
        source_tree_sha256=data["provenance"]["source_tree_sha256"],
        device=data["device"],
        torch=data["torch"],
        cuda=data["cuda"],
        calibration_seed=CALIBRATE,
        heldout_seed=VALIDATE,
        rule=dict(
            min_wins_of_7=WINS, min_median_paired_generic_over_table=RATIO, exact_hit_required=True
        ),
        cases=cases,
        summary=totals,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("panel", type=Path)
    a = p.parse_args()
    print(json.dumps(analyze(a.panel), sort_keys=True, indent=2, allow_nan=False))
