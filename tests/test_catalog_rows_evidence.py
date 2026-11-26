from pathlib import Path
import json
from statistics import median


def test_exact_keys_unhinted_fallback_and_shared_clock_populations():
    root = Path(__file__).resolve().parents[1] / "reports"
    r = json.loads((root / "catalog-rows-gb10.json").read_bytes())
    cs = r["cases"]
    assert (
        not r["smoke"]
        and (root / r["provenance"]["archive"]).is_file()
        and len(cs) == 10
        and len({(tuple(c["config"]), c["seed"]) for c in cs}) == 10
        and len({tuple(c["config"]) for c in cs}) == 5
    )
    for c in cs:
        cfg = c["config"]
        rows = c["rows"]
        assert (
            len(rows) == len(c["inputs"]) == len({z[0] for z in c["inputs"]}) == 8
            and set(rows) == {1, 2, 7, 32, 48, 65, 97, 129}
            and c["bounds"] == [1, 129]
            and 1 <= c["inductor_backend_calls"] <= 8
        )
        hits = {v["rows"] for v in c["variants"] if v["selected"] is not None}
        want = {
            ("decomp", 768): {1, 32},
            ("resnorm2", 768): {32},
            ("gate", 768): {1},
            ("mean", 768): {32},
            ("gate", 1024): set(),
        }
        assert (
            hits == want[tuple(cfg[:2])]
            and len(c["generic_source_sha256"]) == 64
            and all((v["source_sha256"] is not None) == (v["rows"] in hits) for v in c["variants"])
        )
        assert [v["rows"] for v in c["variants"]] == sorted(c["hints"])
        g = c["gates"]
        allowed = set()
        assert set(g) == {
            "eager",
            "table_native",
            "generic_native",
            "split_python",
            "inductor_dynamic",
        }
        for n, v in g.items():
            assert len(v["rows"]) == 8 and v["admitted"] == (
                v["ownership"]
                and all(
                    q["structure"] and q["exact_bits"] and q["within_tolerance"] for q in v["rows"]
                )
            )
            if v["admitted"]:
                allowed.add(n)
        for j, rows_n in enumerate(rows):
            ref = g["table_native"]["rows"][j]["outputs"]
            assert ref == g["split_python"]["rows"][j]["outputs"]
            if rows_n not in hits:
                assert ref == g["generic_native"]["rows"][j]["outputs"]
            if cfg[0] == "resnorm2":
                assert ref[1] == g["eager"]["rows"][j]["outputs"][1]
        assert {"table_native", "generic_native", "split_python", "eager"} <= allowed and set(
            c["hot"]
        ) == {str(min(hits, default=48)), "97"}
        for h in c["hot"].values():
            assert (
                set(h["profiles"]) == allowed
                and len(h["windows"]) == 7
                and h["profiles"]["table_native"]["count"] == 1
            )
            for n in allowed:
                for k in ("device", "host"):
                    assert h["median_us"][k][n] == median(
                        median(w[k][n]["samples_us"]) for w in h["windows"]
                    )
        rot = c["rotating"]
        assert set(rot["median_us"]) == allowed and len(rot["windows"]) == 7
        for n in allowed:
            for k, us in rot["median_us"][n].items():
                assert us == median(median(w["samples"][n][k]) for w in rot["windows"])
