"""Account for delivered tokens and backend lookahead separately in audited ABBA runs.

The candidate's workload gate comes from the original experiment, not an acceptance-rate
cutoff selected after measurement. A null ratio means no tokens were proposed.
"""

import json
from pathlib import Path
from audit_lookup235 import review
from lookup235 import ORDER, need


def account(summary):
    comparison = summary["comparison"]
    cases = comparison["cases"]
    need(
        set(cases) == set(ORDER) and summary["target_requests"] == 24 and summary["launches"] == 4,
        "missing native ABBA data",
    )
    families = {}
    for family in ("copy", "edit", "new"):
        selected = [cases[k] for k in ORDER if k.startswith(family + ":")]
        splits = []
        for j, label in enumerate(("off_then_on", "on_then_off")):
            pairs = [x["pairs"][j] for x in selected]
            off = sum(x["baseline_wall_ms"] for x in pairs)
            on = sum(x["lookup_wall_ms"] for x in pairs)
            draft = sum(x["drafted"] for x in pairs)
            hit = sum(x["accepted"] for x in pairs)
            served = sum(x["lookup_delivered"] for x in pairs)
            pred = sum(x["lookup_predicted"] for x in pairs)
            splits.append(
                dict(
                    comparison_order=label,
                    cases=len(pairs),
                    baseline_ms=off,
                    lookup_ms=on,
                    observed_wall_saving_fraction=1 - on / off,
                    identical_outputs=sum(x["identical"] for x in pairs),
                    baseline_delivered=sum(x["baseline_delivered"] for x in pairs),
                    lookup_delivered=served,
                    lookup_backend_predicted=pred,
                    backend_minus_delivered=pred - served,
                    drafted=draft,
                    accepted=hit,
                    proposal_acceptance=(hit / draft if draft else None),
                )
            )
        gate = comparison["workloads"][family]
        families[family] = dict(
            orders=splits,
            preregistered_case_passes=gate["cases_meeting_gate"],
            both_cases_pass=gate["both_cases_pass"],
        )
    return dict(
        schema="sparkfuse.lookup-account.v1",
        families=families,
        scope="Each order sums the two measured prompts in that family, not independently concurrent requests. Ratios use sums. Accepted drafts are not evidence of faster delivery or identical output; read both comparison orders and the original per-case acceptance gate.",
    )


if __name__ == "__main__":
    import sys

    lock, d, base = map(Path, sys.argv[1:])
    s = review(lock, d, base)
    need(
        (json.dumps(s, indent=2, allow_nan=False) + "\n").encode()
        == (d / "summary.json").read_bytes(),
        "audited summary does not match recorded report",
    )
    print(json.dumps(account(s), indent=2, allow_nan=False))
