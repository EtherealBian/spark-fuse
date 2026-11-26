"""Retrospective answer-field diagnostic. Keep the original whole-output classification unchanged."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from glm_manifest import audit, need


def boundary(s):
    text = s.strip()
    opening, closing = "<think>", "</think>"
    if opening not in text and closing not in text:
        return "plain", text
    if text.startswith(opening) and text.count(opening) == 1:
        if closing not in text:
            return "unclosed", None
        if text.count(closing) == 1:
            return "closed", text.split(closing, 1)[1].strip()
    return "malformed", None


def score(value, current, stale):
    return (
        "current"
        if value == current
        else "stale"
        if value == stale
        else "unavailable"
        if value is None
        else "other"
    )


def inspect(raw, rt, timing):
    n = audit(raw, rt["eog_token_ids"])
    need(n == timing, "original whole-output verdicts no longer reproduce")
    result = {}
    pairs = []
    for arm in ("cold", "reuse"):
        p = Counter()
        z = Counter()
        strict = Counter()
        other = Counter()
        for r in raw["rows"]:
            if r["arm"] != arm:
                continue
            k = r["case"]
            case = n["cases"][k]
            reply = r["target"]
            mode, field = boundary(reply["decoded"])
            sf = score(field, case["expected"], case["stale"])
            a = case["arms"][arm]
            w = score(reply["decoded"].strip(), case["expected"], case["stale"])
            need(
                w == {"correct": "current", "stale": "stale", "other": "other"}[a["verdict"]],
                "strict verdict drift",
            )
            p[mode] += 1
            z[sf] += 1
            strict[w] += 1
            other[mode + ":" + reply["stop"]] += 1
            pairs.append(
                dict(
                    case=k,
                    arm=arm,
                    strict_whole=w,
                    envelope=mode,
                    answer_field=sf,
                    stop=reply["stop"],
                )
            )
        result[arm] = dict(
            strict_whole=dict(sorted(strict.items())),
            envelope=dict(sorted(p.items())),
            answer_field=dict(sorted(z.items())),
            envelope_stop=dict(sorted(other.items())),
        )
    return dict(
        schema="sparkfuse.glm-manifest-boundary.v1",
        result=result,
        cases=pairs,
        scope="Retrospective formatting diagnostic on the original eight messages; does not replace strict scoring. A leading balanced literal think block may be removed once; incomplete or malformed envelopes are not searched for an answer. Output contents are not reproduced in this derived report.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    d = p.parse_args().run
    b = {n: (d / n).read_bytes() for n in ("rows.json", "runtime.json", "timing.json")}
    r = inspect(*(json.loads(x) for x in b.values()))
    r["source_sha256"] = {k: hashlib.sha256(v).hexdigest() for k, v in b.items()}
    print(json.dumps(r, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
