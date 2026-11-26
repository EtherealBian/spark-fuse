"""Reconstruct both clients' timelines and bound dual-slot native decode calls, without a GPU."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from struct import pack, unpack


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size < 2**20,
        "invalid evidence file",
    )
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(v):
    return unpack("=f", pack("=f", v))[0]


def native(snapshot):
    """Enumerate all integer numerators using the pinned float divide and default stream format."""
    v = snapshot["values"]

    def integer(k):
        t = v[k]
        require(
            isinstance(t, str) and t.isascii() and t.isdecimal() and str(int(t)) == t,
            "noncanonical counter",
        )
        return int(t)

    require(
        integer("requests_processing") == integer("requests_deferred") == 0, "boundary not idle"
    )
    d = integer("n_decode_total")
    require(0 < d <= 4096, "outside bounded panel")
    possible = [
        b for b in range(2 * d + 1) if format(f32(b / d), ".6g") == v["n_busy_slots_per_decode"]
    ]
    require(len(possible) == 1, "not a unique native numerator")
    return integer("prompt_tokens_total"), integer("tokens_predicted_total"), d, possible[0]


def joint_bounds(d, b):
    require(
        type(d) is int and type(b) is int and d > 0 and 0 <= b <= 2 * d, "invalid slot accounting"
    )
    # For zero/one/two processing slots: b-d = n2-n0. Do not silently assume n0=0.
    return max(0, b - d), b // 2


def divergence(ids, ref):
    require(len(ids) == len(ref) == 16, "not the fixed-length panel")
    return next((i + 1 for i, (x, y) in enumerate(zip(ids, ref)) if x != y), None)


def timeline(w, base):
    a, b = native(w["before"]), native(w["after"])
    n = w["length"]
    lead = w["lead"]
    mode = w["mode"]
    require(
        w["before"]["epoch"] == w["after"]["epoch"]
        and mode in ("serial", "together")
        and lead in (0, 1)
        and n in (128, 1024),
        "bad wave",
    )
    pd, td, dc, slots = (y - x for x, y in zip(a, b))
    require((pd, td) == (2 * n, 32) and dc <= slots <= 2 * dc, "work differs")
    low, high = joint_bounds(dc, slots)
    rr = w["rows"]
    require(len(rr) == 2, "need two responses")
    clients = []
    for j, r in enumerate(rr):
        case = f"{n}:{j}"
        ids = r["ids"]
        start = r["start_ns"]
        end = r["end_ns"]
        require(
            r["case"] == case
            and r["slot"] == (j ^ lead)
            and type(start) is type(end) is int
            and 0 <= start < end,
            "wrong client or clock",
        )
        require(
            r["processed"] == n
            and r["predicted"] == 16
            and r["reused"] == 0
            and r["stop"] == "limit"
            and r["truncated"] is False
            and r["input_sha256"] == base[case]["input_sha256"]
            and all(type(i) is int and i >= 0 for i in ids),
            "wrong request work",
        )
        clients.append(
            dict(
                case=case,
                slot=r["slot"],
                dispatch_from_eligibility_ns=start,
                service_ns=end - start,
                completion_from_eligibility_ns=end,
                first_different_id_vs_one_slot=divergence(ids, base[case]["ids"]),
            )
        )
    first, last = sorted(r["end_ns"] for r in rr)
    overlap = first - max(r["start_ns"] for r in rr)
    if mode == "serial":
        require(
            rr[lead]["end_ns"] <= rr[1 - lead]["start_ns"] and slots == dc, "serial arm overlaps"
        )
    else:
        require(overlap > 0, "together arm did not overlap on client")
    return dict(
        rep=w["rep"],
        length=n,
        mode=mode,
        lead=lead,
        clients=clients,
        first_completion_ns=first,
        last_completion_ns=last,
        client_overlap_ns=max(0, overlap),
        native_decode_calls=dc,
        native_busy_slot_samples=slots,
        dual_slot_calls_lower_bound=low,
        dual_slot_calls_upper_bound=high,
    )


def rebuild(out, ref):
    sm = load(out / "summary.json")
    br = load(ref / "rows.json")["rows"]
    base = {r["case"]: r for r in br if r["rep"] == 0}
    require(
        len(br) == 8 and len(base) == 4 and all(r["ids"] == base[r["case"]]["ids"] for r in br),
        "reference is not repeatable",
    )
    require(sm["baseline_sha256"] == sha(ref / "summary.json"), "reference changed")
    epochs = set()
    blocks = []
    comparisons = []
    for block in range(2):
        p = out / f"block-{block}" / "rows.json"
        raw = load(p)
        ws = raw["waves"]
        require(
            raw["block"] == block
            and len(ws) == 8
            and sha(p) == sm["evidence_sha256"][f"block-{block}"]["rows.json"],
            "uncommitted rows",
        )
        epoch = ws[0]["before"]["epoch"]
        require(
            epoch not in epochs and native(ws[0]["before"])[:2] == (256, 6),
            "not fresh after the two warmups",
        )
        epochs.add(epoch)
        chron = []
        keys = set()
        prev = ws[0]["before"]
        for w in ws:
            key = (w["rep"], w["length"], w["mode"])
            require(
                key not in keys and w["before"] == prev and w["before"]["epoch"] == epoch,
                "broken counter chronology",
            )
            keys.add(key)
            prev = w["after"]
            chron.append(timeline(w, base))
        require(
            keys
            == {(r, n, m) for r in (0, 1) for n in (128, 1024) for m in ("serial", "together")},
            "incomplete crossover",
        )
        blocks.append(dict(block=block, waves=chron))
        for rep in (0, 1):
            for n in (128, 1024):
                arm = {w["mode"]: w for w in chron if (w["rep"], w["length"]) == (rep, n)}
                s, t = arm["serial"], arm["together"]
                k = next(i for i, w in enumerate(chron) if (w["rep"], w["length"]) == (rep, n))
                rows = {w["mode"]: w["rows"] for w in ws if (w["rep"], w["length"]) == (rep, n)}
                require(s["lead"] == t["lead"] == ((rep + block) % 2), "wrong lead")
                first_ratio = t["first_completion_ns"] / s["first_completion_ns"]
                tail_ratio = t["last_completion_ns"] / s["last_completion_ns"]
                delta = [
                    divergence(rows["together"][j]["ids"], rows["serial"][j]["ids"]) for j in (0, 1)
                ]
                baseline = all(
                    c["first_different_id_vs_one_slot"] is None
                    for a in arm.values()
                    for c in a["clients"]
                )
                pair = sm["lengths"][str(n)]["pairs"][2 * block + rep]

                def near(a, b):
                    return abs(a - b) < 1e-10

                require(
                    (pair["block"], pair["rep"], pair["lead"], pair["first_mode"])
                    == (block, rep, s["lead"], chron[k]["mode"])
                    and near(pair["first_completion_change"], first_ratio - 1)
                    and near(pair["pair_saving"], 1 - tail_ratio),
                    "published ratios differ",
                )
                require(
                    pair["serial_pair_ms"] == s["last_completion_ns"] / 1e6
                    and pair["together_pair_ms"] == t["last_completion_ns"] / 1e6
                    and pair["serial_decode_calls"] == s["native_decode_calls"]
                    and pair["together_decode_calls"] == t["native_decode_calls"]
                    and pair["together_excess_busy_slots"] == t["dual_slot_calls_lower_bound"]
                    and pair["equal_outputs"] == [v is None for v in delta]
                    and pair["all_match_one_slot"] == baseline,
                    "published paired evidence differs",
                )
                comparisons.append(
                    dict(
                        block=block,
                        rep=rep,
                        length=n,
                        first_response_change=first_ratio - 1,
                        last_response_saving=1 - tail_ratio,
                        client_completion_change=[
                            t["clients"][j]["completion_from_eligibility_ns"]
                            / s["clients"][j]["completion_from_eligibility_ns"]
                            - 1
                            for j in (0, 1)
                        ],
                        first_different_id_between_arms=delta,
                        match_one_slot=baseline,
                        proven_dual_slot_calls=t["dual_slot_calls_lower_bound"],
                    )
                )
    overall = True
    for n in (128, 1024):
        cases = [c for c in comparisons if c["length"] == n]
        result = sm["lengths"][str(n)]
        checks = dict(
            all_four_pair_makespans_improve_at_least_5pct=all(
                c["last_response_saving"] >= 0.05 for c in cases
            ),
            first_completion_within_10pct_all_pairs=all(
                c["first_response_change"] <= 0.10 for c in cases
            ),
            exact_outputs=all(
                c["first_different_id_between_arms"] == [None, None] and c["match_one_slot"]
                for c in cases
            ),
            joint_backend_occupancy_all_pairs=all(c["proven_dual_slot_calls"] > 0 for c in cases),
        )
        require(all(result[k] is v for k, v in checks.items()), "wrong gates")
        overall &= all(checks.values())
        for mode in ("serial", "together"):
            durations = [
                w["last_completion_ns"] / 1e6
                for b in blocks
                for w in b["waves"]
                if w["length"] == n and w["mode"] == mode
            ]
            require(result[f"median_{mode}_pair_ms"] == median(durations), "incorrect median")
    require(
        sm["qualified_for_tested_workload"] is overall
        and (sm["blocks"], sm["measured_responses"], sm["warmups_excluded"]) == (2, 32, 4),
        "wrong overall decision",
    )
    return dict(
        schema="glm355.observed-timelines.v1",
        summary_sha256=sha(out / "summary.json"),
        blocks=blocks,
        pairs=comparisons,
        qualified_for_tested_workload=overall,
        scope="Client intervals are measured from common eligibility, not first-token latency. Bounds use the sum of native processing slots sampled at decode calls, not per-kernel batch composition.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("one_slot", type=Path)
    a = p.parse_args()
    print(json.dumps(rebuild(a.run, a.one_slot), indent=2, allow_nan=False))
