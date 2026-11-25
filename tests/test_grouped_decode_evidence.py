"""Offline checks for logical parity, live outputs and physically distinct KV request rings."""

import json
import math
from pathlib import Path
from statistics import median


def test_grouped_decode_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "grouped-decode-gb10.json").read_bytes())
    assert not d["smoke"] and (root / d["provenance"]["archive"]).is_file()
    cases = d["cases"]
    assert (
        len(cases) == 10
        and len({c["seed"] for c in cases}) == 2
        and len({(c["seed"], tuple(c["spec"])) for c in cases}) == 10
        and len({tuple(c["spec"]) for c in cases}) == 5
    )
    for c in cases:
        b, h, g, s, width, mask, dt, _ = c["spec"]
        assert (
            len(c["input_sha256"]) == 10
            and len({a[0] for a in c["input_sha256"]}) == 10
            and len(set(c["cpu_fp64_sha256"])) == 8
        )
        expected = {f"group{r}_tile{t}" for r in (1, 2, 4) for t in (32, 128)}
        assert set(c["plan"]) == expected and set(c["bitwise_parity"]) == expected
        for name, plan in c["plan"].items():
            p = plan["plan"]
            r, t = p["head_group"], p["tile"]
            assert name == f"group{r}_tile{t}" and h % (r * g) == 0
            chunks = math.ceil(s / t)
            assert p["workspace_bytes"] == (
                4 * b * h * chunks * (width + 2) if chunks > 1 else 0
            ) and len(plan["binary"]) == 1 + (chunks > 1)
            same = c["bitwise_parity"][f"group1_tile{t}"]
            assert (
                c["bitwise_parity"][name] == same
                and len(same["partials"]) == len(same["narrow"]) == 8
            )
            if chunks > 1:
                assert (
                    plan["binary"][1]["source_sha256"]
                    == c["plan"][f"group1_tile{t}"]["binary"][1]["source_sha256"]
                )
            assert c["hot"]["profiles"][name]["count"] == len(plan["binary"])
        paths = {n for n, v in c["gates"].items() if v["admitted"]}
        assert expected | {"eager"} <= paths and (
            "inductor" in c["gates"] or "inductor" in c["build_failures"]
        )
        assert paths == set(c["requests"]["median_us"]) == set(c["hot"]["median_us"]["device"])
        for name, gate in c["gates"].items():
            assert len(gate["requests"]) == 8 and gate["admitted"] == (
                gate["distinct"]
                and all(m["structure"] and m["outside_tolerance"] == 0 for m in gate["requests"])
            )
        for n in paths:
            for mode in ("fixed_sync_us", "rotating_sync_us", "queued8_us_per_forward"):
                ws = c["requests"]["windows"]
                assert len(ws) == 7 and c["requests"]["median_us"][n][mode] == median(
                    median(w["samples"][n][mode]) for w in ws
                )
        ring = c["locality"]
        if s == 257:
            assert ring is None
            continue
        slots = ring["slots"]
        fp = ring["footprint"]
        assert (
            4 <= slots <= 64
            and ring["seed"] != c["seed"]
            and len(ring["input_sha256"]) == slots
            and not {r[0] for r in ring["input_sha256"]} & {r[0] for r in c["input_sha256"]}
        )
        assert (
            fp["hot"]["kv"] == fp["shared_kv"]["kv"]
            and fp["hot"]["kv"]["storages"] == 2
            and fp["independent_kv"]["kv"]["storages"] == 2 * slots
        )
        assert (
            fp["hot"]["query"]["storages"] == 1
            and fp["shared_kv"]["query"]["storages"]
            == fp["independent_kv"]["query"]["storages"]
            == slots
        )
        assert (
            fp["independent_kv"]["kv"]["bytes"]
            == slots * fp["hot"]["kv"]["bytes"]
            >= 3 * ring["l2_bytes"]
        )
        assert len(ring["orders"]) == 9
        assert set(ring["gates"]) == {f"{n}:{m}" for n in paths for m in fp}
        accepted = {k for k, v in ring["gates"].items() if v["admitted"]}
        assert all(set(row) == accepted and len(row) == len(accepted) for row in ring["orders"])
        for tag, gate in ring["gates"].items():
            assert len(gate["requests"]) == slots and gate["admitted"] == (
                gate["live_outputs"]["storages"] == slots
                and all(m["structure"] and m["outside_tolerance"] == 0 for m in gate["requests"])
            )
            if tag.split(":")[0] != "inductor":
                assert gate["admitted"]
        for tag in accepted:
            assert len(ring["device_us"][tag]) == 9 and ring["median_us"][tag] == median(
                ring["device_us"][tag]
            )
