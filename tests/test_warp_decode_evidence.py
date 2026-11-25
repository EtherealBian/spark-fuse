import json
from pathlib import Path
from statistics import median


def test_warp_schedule_and_reference_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    r = json.loads((root / "warp-decode-gb10.json").read_bytes())
    assert not r["smoke"] and (root / r["provenance"]["archive"]).is_file()
    cs = r["cases"]
    assert (
        len(cs) == 8
        and len({tuple(c["config"]) for c in cs}) == 4
        and {c["seed"] for c in cs} == {3921719, 3932693}
    )
    assert len({(tuple(c["config"]), c["seed"]) for c in cs}) == 8
    keys = {"cta32", "cta64", "warp1_t32", "warp4_t32", "warp1_t64", "warp4_t64"}
    for c in cs:
        b, h, g, d, m, dt = c["config"]
        assert (
            c["sizes"] == [2, 31, 32, 33, 127, 513, 2049, 8193]
            and c["qualification"] == [17, 65]
            and len({xs[0] for xs in c["inputs"]}) == 24
            and len(c["fp64"]) == 24
            and c["inductor_backend_calls"] >= 1
        )
        vs = c["variants"]
        assert set(vs) == keys and set(c["gates"]) == keys | {"eager", "inductor_dynamic"}
        for tile in (32, 64):
            vv = [v for v in vs.values() if v["tile"] == tile]
            assert (
                {v["warps"] for v in vv} == {0, 1, 4}
                and len({v["binaries"][0] for v in vv}) == 3
                and len({v["binaries"][1] for v in vv}) == 1
            )
            for v in vv:
                assert (
                    all(c["binaries"][k]["binary_sha256"] == k for k in v["binaries"])
                    and v["setup"]["wall_ms"] > 0
                    and v["peak_workspace_bytes"]
                    == 4 * b * h * ((8193 + tile - 1) // tile) * (d + 2)
                )
        ok = set()
        for n, z in c["gates"].items():
            assert len(z["requests"]) == 24 and z["admitted"] == (
                z["all_live"]
                and all(v["structure"] and not v["outside_tolerance"] for v in z["requests"])
            )
            if z["admitted"]:
                ok.add(n)
        assert keys | {"eager"} <= ok and {int(s) for s in c["hot"]} == set(c["sizes"])
        for s, t in c["hot"].items():
            assert len(t["windows"]) == 7 and set(t["profiles"]) == ok
            for n in ok:
                if n in keys:
                    assert t["profiles"][n]["count"] == 1 + int(int(s) > vs[n]["tile"])
                for clock in ("device", "host"):
                    assert t["median_us"][clock][n] == median(
                        median(w[clock][n]["samples_us"]) for w in t["windows"]
                    )
        mix = c["mixed"]
        assert set(mix["median_us"]) == ok and len(mix["windows"]) == 7
        for k, m in mix["median_us"].items():
            for n, v in m.items():
                assert v == median(median(w["samples"][k][n]) for w in mix["windows"])
