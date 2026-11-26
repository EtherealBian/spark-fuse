import json
from pathlib import Path
from statistics import median


def test_paged_attention_direct_read_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    r = json.loads((root / "paged-decode-gb10.json").read_bytes())
    assert not r["smoke"] and (root / r["provenance"]["archive"]).is_file()
    cases = r["cases"]
    assert (
        len(cases) == 12
        and len({tuple(c["config"]) for c in cases}) == 6
        and {c["seed"] for c in cases} == {4093111, 4117903}
        and len({(tuple(c["config"]), c["seed"]) for c in cases}) == 12
    )
    for c in cases:
        b, h, g, d, t, s, mask, dt = c["config"]
        pages = (s + t - 1) // t
        assert (
            c["patterns"] == ["shuffled", "shared", "repeated", "endpoints"] * 2
            and c["physical_pages"] == b * pages + 5
            and c["dense_kv_logical_bytes"] == 4 * b * g * s * d
            and len(set(c["fp64"])) >= 4
            and len({xs[0] for xs in c["inputs"]}) == 8
            and c["inductor_backend_calls"] == 1
        )
        for n in ("paged_w1", "paged_w4"):
            v = c["variants"][n]
            assert (
                v["partial_workspace_bytes"] == (4 * b * h * pages * (d + 2) if pages > 1 else 0)
                and len(v["binaries"]) == 1 + int(pages > 1)
                and v["setup"]["wall_ms"] > 0
            )
            for z in v["binaries"]:
                assert c["binaries"][z]["binary_sha256"] == z
        gate = c["gates"]
        assert set(gate) == {"eager", "gather_then_warp", "paged_w1", "paged_w4", "inductor_static"}
        ok = set()
        base = [x["sha256"] for x in gate["gather_then_warp"]["rows"]]
        for n, v in gate.items():
            assert len(v["rows"]) == 8 and v["admitted"] == (
                v["all_live"]
                and all(x["structure"] and not x["outside_tolerance"] for x in v["rows"])
            )
            assert c["matched_materialized_bits"][n] == sum(
                z["sha256"] == h for z, h in zip(v["rows"], base, strict=True)
            )
            if v["admitted"]:
                ok.add(n)
        assert {"eager", "gather_then_warp", "paged_w1", "paged_w4"} <= ok
        hot = c["hot"]
        mix = c["rotating"]
        assert (
            len(hot["windows"]) == len(mix["windows"]) == 7
            and set(hot["profiles"]) == set(c["allocation"]) == set(mix["median_us"]) == ok
        )
        for n in ok:
            assert c["allocation"][n]["live_output_bytes"] == b * h * d * 2
            if n.startswith("paged_w"):
                assert hot["profiles"][n]["count"] == 1 + int(pages > 1)
            for m in ("device", "host"):
                assert hot["median_us"][m][n] == median(
                    median(w[m][n]["samples_us"]) for w in hot["windows"]
                )
            for m, z in mix["median_us"][n].items():
                assert z == median(median(w["samples"][n][m]) for w in mix["windows"])
