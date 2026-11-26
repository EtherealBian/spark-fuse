"""Recount delivered IDs from the already-audited native four-launch natural-EOS panel.
The source rows preserve the endpoint fields, not the original HTTP byte stream. The
adapter is replayed offline; the costs below measure native requests, NOT this adapter.
"""

import argparse
import hashlib
import json
from pathlib import Path
from completion_meter import Counts, native
from audit_lookup235 import review
from lookup235 import GEN, V


def replay(lock, root, base):
    validated = review(lock, root, base)
    original = (root / "summary.json").read_bytes()
    if original != (json.dumps(validated, indent=2, allow_nan=False) + "\n").encode():
        raise ValueError("source panel does not reproduce")
    modes = {}
    orders = []
    checksum = hashlib.sha256(original).hexdigest()
    for b in range(4):
        d = root / f"block-{b}"

        def get(n):
            return json.loads((d / n).read_bytes())

        raw = get("lookup-rows.json")
        setting = get("lookup-setting.json")
        eog = get("runtime.json")["eog_token_ids"]
        mode = setting["mode"]
        groups = {}
        for rec in raw["rows"]:
            fam = rec["key"].split(":")[0]
            r = rec["response"]
            timings = {"predicted_n": r["predicted"]}
            if r["counters_present"]:
                timings.update(draft_n=r["drafted"], draft_n_accepted=r["accepted"])
            # The request driver recorded integer perf_counter_ns differences divided by 1e6.
            # Convert back only if this is an unambiguous round trip for the observed sample.
            ns = round(r["wall_ms"] * 1_000_000)
            if ns / 1_000_000 != r["wall_ms"]:
                raise ValueError("not an integer ns wall sample")
            count = native(
                dict(stop_type=r["stop"], tokens=r["ids"], timings=timings),
                wall_ns=ns,
                draft_max=setting["draft_max"] if mode != "none" else 0,
                n_predict=GEN,
                vocab_size=V,
                eog=eog,
            )
            groups[fam] = groups.get(fam, Counts()) + count
        block = sum(groups.values(), Counts())
        modes[mode] = modes.get(mode, Counts()) + block
        orders.append(
            dict(
                block=b,
                mode=mode,
                by_family={k: groups[k].report() for k in sorted(groups)},
                total=block.report(),
            )
        )
    return dict(
        schema="sparkfuse.native-delivery.v1",
        source_summary_sha256=checksum,
        blocks=orders,
        by_mode={k: v.report() for k, v in modes.items()},
        scope="Offline reader of archived requests. Includes returned terminal EOG, reported separately; never count speculative lookahead as delivered output. Rates use the sum of serial HTTP request durations. No streaming, multi-client throughput or new latency measurement.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for x in ("lock", "root", "base"):
        p.add_argument(x, type=Path)
    a = p.parse_args()
    print(json.dumps(replay(a.lock, a.root, a.base), indent=2, allow_nan=False))
