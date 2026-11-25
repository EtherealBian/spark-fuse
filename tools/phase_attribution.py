"""Strict single-stream FX-to-CUDA attribution and replay summaries (Kineto Chrome JSON)."""

from statistics import median


def require(condition, message):
    if not condition:
        raise ValueError(message)


def kernels(events):
    require(
        not any(e.get("cat") in ("gpu_memcpy", "gpu_memset") for e in events),
        "unattributed GPU copy or memset",
    )
    ks = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    require(
        ks and len({tuple(e["args"][k] for k in ("device", "context", "stream")) for e in ks}) == 1,
        "expected one GPU stream in one context",
    )
    require(
        all(e["dur"] > 0 for e in ks)
        and all(
            a["ts"] < b["ts"] and a["ts"] + a["dur"] <= b["ts"] + b["dur"]
            for a, b in zip(ks, ks[1:])
        ),
        "invalid or reordered GPU intervals",
    )
    return ks


def signature(event):
    a = event["args"]
    return {
        **{k: a[k] for k in ("grid", "block", "shared memory", "registers per thread")},
        "kernel": event["name"],
    }


def launches(events):
    table = {}
    for e in events:
        if (
            e.get("cat") in ("cuda_runtime", "cuda_driver")
            and "Launch" in e["name"]
            and "correlation" in e.get("args", {})
        ):
            c = e["args"]["correlation"]
            require(c not in table, "ambiguous launch correlation")
            table[c] = e
    return table


def map_ordinary(events, node_meta):
    annotations = [
        e
        for e in events
        if e.get("cat") == "user_annotation"
        and e.get("ph") == "X"
        and e["name"].startswith("phase/")
    ]
    table = launches(events)
    order = []
    seen = set()
    for k in kernels(events):
        c = k["args"]["correlation"]
        require(c in table and c not in seen, "missing or reused ordinary launch")
        seen.add(c)
        launch = table[c]
        owners = [
            e["name"][6:]
            for e in annotations
            if (e["pid"], e["tid"]) == (launch["pid"], launch["tid"])
            and e["ts"] <= launch["ts"]
            and launch["ts"] + launch["dur"] <= e["ts"] + e["dur"]
        ]
        require(len(owners) == 1 and owners[0] in node_meta, "kernel does not have one FX owner")
        node = owners[0]
        order.append(dict(ordinal=len(order), node=node, **node_meta[node], signature=signature(k)))
    require(seen == set(table), "ordinary trace contains a launch without a matching kernel")
    return order


def replay_intervals(events, mapping, repeats):
    ks = kernels(events)
    table = launches(events)
    require(
        len(table) == repeats and all("GraphLaunch" in launch["name"] for launch in table.values()),
        "unexpected graph launch count or unrelated launch",
    )
    expected = [x["signature"] for x in mapping]
    ids = None
    cursor = 0
    result = []
    require(
        len({k["args"]["graph id"] for k in ks}) == 1 and ks[0]["args"]["graph id"] != 0,
        "mixed or missing graph ID",
    )
    for c in sorted(table, key=lambda c: table[c]["ts"]):
        chunk = ks[cursor : cursor + len(mapping)]
        cursor += len(chunk)
        require(
            [signature(e) for e in chunk] == expected
            and all(e["args"]["correlation"] == c for e in chunk),
            "captured replay and labeled launch signature/order differ",
        )
        nodes = [e["args"]["graph node id"] for e in chunk]
        require(
            len(set(nodes)) == len(nodes) and (ids is None or ids == nodes),
            "ambiguous or unstable graph-node order",
        )
        ids = nodes
        anchor = chunk[0]["ts"]
        result.append([[round((e["ts"] - anchor) * 1000), round(e["dur"] * 1000)] for e in chunk])
    require(cursor == len(ks), "unexpected trailing kernels")
    return ids, result


def phase_totals(mapping, replay):
    require(len(mapping) == len(replay) > 0 and replay[0][0] == 0, "bad interval count/origin")
    totals = {}
    end = busy = gaps = overlap = 0
    last_start = -1
    for row, (offset, duration) in zip(mapping, replay, strict=True):
        # Kineto can report small overlaps even on a single stream. Keep the
        # overlap explicitly; subtracting summed durations is not idle time.
        require(
            offset > last_start and duration > 0 and offset + duration >= end,
            "invalid interval order",
        )
        gaps += max(0, offset - end)
        overlap += max(0, end - offset)
        last_start, end = offset, offset + duration
        busy += duration
        group = row["group"]
        totals[group] = totals.get(group, 0) + duration
    return {
        **totals,
        "sum_kernel_durations": busy,
        "interval_union": busy - overlap,
        "reported_overlap": overlap,
        "inter_kernel_gap": gaps,
        "first_to_last": end,
    }


def summarize(mapping, windows):
    groups = []
    for w in windows:
        totals = [phase_totals(mapping, r) for r in w["intervals_ns"]]
        groups.append({k: median(t[k] for t in totals) / 1000 for k in totals[0]})
    return {
        **{k: median(g[k] for g in groups) for k in groups[0]},
        "unprofiled_graph": median(median(w["event_samples_us"]) for w in windows),
    }
