import json
from pathlib import Path
from statistics import median


def test_multilayer_observable_cache_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    r = json.loads((root / "protected-stack-gb10.json").read_bytes())
    assert not r["smoke"] and (root / r["provenance"]["archive"]).is_file()
    cs = r["cases"]
    assert len(cs) == 8 and {(c["family"], c["depth"], c["seed"], c["masked"]) for c in cs} == {
        (f, d, s, m)
        for f, d in [("qwen2", 2), ("llama", 3)]
        for s in (3998173, 4011227)
        for m in (True, False)
    }
    observed_drift = set()
    for c in cs:
        d = c["depth"]
        assert (
            c["past"] == [2, 15, 30, 31, 125, 255, 513, 2045]
            and c["qualification"] == [17, 65]
            and c["steps"] == 4
            and len({xs[0] for tr in c["inputs"] for xs in tr}) == 32
            and c["inductor_backend_calls"] >= 1
        )
        v = c["variants"]["protected"]
        pr = v["protection"]
        assert (
            pr["flat_indices"] == list(range(1, 2 * d + 1))
            and len(pr["outputs"]) == 2 * d
            and pr["unchanged_dependencies"]
            and len(pr["skipped_attention"]) == d - 1
            and len(v["attention"]) == 1
            and not v["appends"]
            and len(pr["protected_nodes"]) > d * 20
        )
        for ids in v["attention_binaries"]:
            assert len(ids) == 2 and all(c["binaries"][k]["binary_sha256"] == k for k in ids)
        assert (
            all(
                len(z) == 4
                and all(
                    q["attention"]["structure"] and not q["attention"]["outside_tolerance"]
                    for q in z
                )
                for z in c["fp64"].values()
            )
            and len(c["fp64"]) == 2
        )
        ok = set()
        exact = set()
        for name, g in c["gates"].items():
            steps = g["steps"]
            assert len(steps) == 8 and all(len(tr) == 4 for tr in steps)
            rows = [z for tr in steps for z in tr]
            assert all(len(z["outputs"]) == 2 * d + 1 for z in rows)
            admit = g["all_live"] and all(
                z["fresh"]
                and z["geometry"]
                and z["own_prefix"]
                and all(x["outside"] == 0 for x in z["outputs"])
                for z in rows
            )
            diff = [
                sum(z["outputs"][i]["different_bytes"] != 0 for z in rows)
                for i in range(1, 1 + 2 * d)
            ]
            assert (
                admit == g["admitted"]
                and diff == g["differing_cache_tensors_by_output"]
                and g["exact_caches"] == (admit and not any(diff))
            )
            assert all(
                (x["sha256"] == x["reference_sha256"]) == (x["different_bytes"] == 0)
                for z in rows
                for x in z["outputs"]
            )
            if admit:
                ok.add(name)
            if g["exact_caches"]:
                exact.add(name)
            if name == "unprotected" and any(diff[2:]):
                observed_drift.add(c["family"])
        assert (
            exact == set(c["byte_exact_cache_paths"])
            and {"protected", "hf_eager", "exported_aten"} <= exact
            and ok == set(c["numerically_timed"])
        )
        assert set(c["gates"]) | set(c["build_failures"]) == {
            "unprotected",
            "protected",
            "hf_eager",
            "exported_aten",
            "inductor_dynamic",
        }
        for t in c["hot"].values():
            assert len(t["windows"]) == 7 and set(t["profiles"]) == ok
            for n in ok:
                for clock in ("device", "host"):
                    assert t["median_us"][clock][n] == median(
                        median(w[clock][n]["samples_us"]) for w in t["windows"]
                    )
        mix = c["rotating"]
        assert set(mix["median_us"]) == ok and len(mix["windows"]) == 7
        for n, ms in mix["median_us"].items():
            for m, us in ms.items():
                assert us == median(median(w["samples"][n][m]) for w in mix["windows"])
    assert observed_drift == {"qwen2", "llama"}
