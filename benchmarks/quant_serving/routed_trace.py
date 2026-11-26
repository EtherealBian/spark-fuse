"""Compare a two-slot prefix-aware dispatcher against round-robin, using native slot IDs."""

import argparse
import json
from pathlib import Path
from prefix_replay import complete, digest, post
from slot_policy import SlotRouter

PATTERNS = {"alternating": "ABABAB", "switchback": "ABBAAB"}
POLICIES = ("round_robin", "affinity")
QUESTIONS = ("retention", "ordering", "versioning", "delivery", "rollback", "durability")
LENGTH = 1024
NGEN = 16


def workload(tokenize, pattern, rep):
    prefixes = {
        k: tokenize((f"{kind} ledger {rep}. {topic}. " * 270))
        for k, kind, topic in [
            ("A", "Amber", "Events settle after a write-ahead log entry"),
            ("B", "Cobalt", "Reports are sorted by the reconciliation timestamp"),
        ]
    }
    tasks = []
    for i, k in enumerate(PATTERNS[pattern]):
        suffix = tokenize(f"\nCase {i}. Explain {QUESTIONS[i]} in one sentence:")
        size = LENGTH - len(suffix)
        if size < 64 or size >= len(prefixes[k]):
            raise ValueError("prompt construction failed")
        tasks.append(prefixes[k][:size] + suffix)
    return tasks


def run(port):
    def tokenize(s):
        return post(port, "tokenize", dict(content=s, add_special=False, parse_special=False))[
            "tokens"
        ]

    results = []
    inputs = {}
    for pi, pattern in enumerate(PATTERNS):
        for rep in range(2):
            tasks = workload(tokenize, pattern, rep)
            inputs[f"{pattern}:{rep}"] = tasks
            arms = POLICIES if (pi + rep) % 2 == 0 else POLICIES[::-1]
            for arm in arms:
                router = SlotRouter(2, 4096, 32, policy=arm)
                rows = []
                for ids in tasks:
                    lease = router.acquire("frozen-checkpoint", ids)
                    assert lease is not None
                    try:
                        w = complete(port, ids, lease.reuse, NGEN, slot=lease.slot)
                    except BaseException:
                        router.finish(lease, success=False)
                        raise
                    router.finish(lease, success=True)
                    rows.append(dict(target=digest(ids), slot=lease.slot, prefix=lease.prefix, **w))
                results.append(dict(pattern=pattern, rep=rep, policy=arm, rows=rows))
                print("trace", pattern, rep, arm, sum(w["reused"] for w in rows), flush=True)
    return results, inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows, inputs = run(a.port)
    a.output.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    a.output.with_name("trace-inputs.json").write_text(
        json.dumps(inputs, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    main()
