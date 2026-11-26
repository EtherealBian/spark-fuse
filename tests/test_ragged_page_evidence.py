import json
from statistics import median
from pathlib import Path


def test_row_lengths_observables_and_per_profile_costs():
    root = Path(__file__).resolve().parents[1] / "reports"
    r = json.loads((root / "ragged-page-gb10.json").read_bytes())
    assert not r["smoke"] and (root / r["provenance"]["archive"]).is_file()
    cs = r["cases"]
    assert (
        len(cs) == 8
        and len({tuple(c["config"]) for c in cs}) == 4
        and len({(tuple(c["config"]), c["seed"]) for c in cs}) == 8
        and {c["seed"] for c in cs} == {4211831, 4227139}
    )
    for c in cs:
        b, h, g, d, t, s, mask, dt = c["config"]
        cap = (s + t - 1) // t
        assert (
            c["inductor_backend_calls"] == 1
            and len(c["fp64"]) == 8
            and len({x[0] for x in c["inputs"]}) == 8
        )
        aa = c["activity"]
        assert len(aa) == 8 and {x["profile"] for x in aa} == {
            "full",
            "mixed",
            "tiny",
            "empty",
            "tile_edges",
            "extremes",
            "descending",
            "single_active",
        }
        for row in aa:
            ns = [max(0, min(n, s)) for n in row["raw_lengths"]]
            assert (
                row["live_tokens"] == ns
                and row["live_pages"] == sum((n + t - 1) // t for n in ns)
                and row["capacity_pages"] == b * cap
            )
        for v in c["variants"].values():
            assert (
                v["workspace_bytes"] == (0 if cap == 1 else 4 * b * h * cap * (d + 2))
                and len(v["binaries"]) == 1 + int(cap > 1)
                and all(c["binaries"][k]["binary_sha256"] == k for k in v["binaries"])
            )
        gates = c["gates"]
        assert set(gates) == {"eager", "ragged", "explicit_mask_paged", "inductor_static"} and [
            z["sha256"] for z in gates["ragged"]["rows"]
        ] == [z["sha256"] for z in gates["explicit_mask_paged"]["rows"]]
        ok = set()
        for n, g in gates.items():
            assert len(g["rows"]) == len(g["nonzero_empty_row_bits"]) == 8 and g["admitted"] == (
                g["all_live"]
                and not any(g["nonzero_empty_row_bits"])
                and all(z["structure"] and not z["outside_tolerance"] for z in g["rows"])
            )
            if g["admitted"]:
                ok.add(n)
        assert {"eager", "ragged", "explicit_mask_paged"} <= ok and set(c["hot"]) == {
            "full",
            "mixed",
            "tiny",
            "empty",
        }
        for name, hot in c["hot"].items():
            assert (
                set(hot["profiles"]) == ok
                and len(hot["windows"]) == 7
                and hot["profiles"]["ragged"]["count"] == 1 + int(cap > 1)
            )
            for n in ok:
                assert c["allocation"][name][n]["output_storage_bytes"] == b * h * d * 2
                for m in ("device", "host"):
                    assert hot["median_us"][m][n] == median(
                        median(w[m][n]["samples_us"]) for w in hot["windows"]
                    )
        rot = c["rotating"]
        assert len(rot["windows"]) == 7 and set(rot["median_us"]) == ok
        for n, v in rot["median_us"].items():
            for m, z in v.items():
                assert z == median(median(w["samples"][n][m]) for w in rot["windows"])
