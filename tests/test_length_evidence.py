"""Recheck dynamic length coverage, numerical/ABI admissions and timing medians offline."""

import json
from pathlib import Path
from statistics import median


def test_symbolic_kv_length_report():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "symbolic-kv-gb10.json").read_bytes())
    assert not d["smoke"] and (root / d["provenance"]["archive"]).is_file()
    assert len(d["cases"]) == len({(tuple(c["config"]), c["seed"]) for c in d["cases"]}) == 4
    for c in d["cases"]:
        b, h, _, w, mask, _ = c["config"]
        ls = c["held_lengths"]
        n = c["requests_per_held_length"]
        assert (
            n == 3
            and ls == [2, 31, 32, 33, 129, 513, 1025, 2049]
            and not set(ls) & set(c["qualifying_lengths"])
        )
        assert (
            len(c["binary"])
            == len({v["source_sha256"] for v in c["binary"]})
            == len({v["binary_sha256"] for v in c["binary"]})
            == 2
            and len(c["static_kernel_cache_keys"]) == 13
        )
        assert c["workspace_bytes"] == [
            0 if s <= 32 else 4 * b * h * ((s + 31) // 32) * (w + 2) for s in ls
        ]
        assert (
            len({x[0] for x in c["input_sha256"]}) == len(set(c["cpu_fp64_sha256"])) == len(ls) * n
        )
        admitted = set()
        for key, g in c["gates"].items():
            os = g["outputs"]
            assert len(os) == n * len(ls) and g["admitted"] == (
                g["all_live"] and all(x["structure"] and x["outside_tolerance"] == 0 for x in os)
            )
            if g["admitted"]:
                admitted.add(key)
            if key == "length_cuda":
                assert [x["sha256"] for x in os] == c["static_cuda_sha256"]
        assert {"eager", "length_cuda"} <= admitted
        for key, count in c["inductor_backend_calls_after_timing"].items():
            trace = c["first_seen_requests"][key]
            assert (
                trace
                and [r["length"] for r in trace] == ls
                and all(r["first_request_ms"] > 0 for r in trace)
                and [r["backend_calls"] for r in trace] == sorted(r["backend_calls"] for r in trace)
                and count >= trace[-1]["backend_calls"]
            )
        hot = c["hot"]
        assert set(hot) == {"2", "33", "1025", "2049"}
        for res in hot.values():
            assert set(res["profiles"]) == admitted and len(res["windows"]) == 7
            for mode in ("device", "host"):
                for key, v in res["median_us"][mode].items():
                    assert v == median(median(a[mode][key]["samples_us"]) for a in res["windows"])
        x = c["mixed"]
        assert len(x["windows"]) == 7 and set(x["median_us"]) == admitted
        for key, clocks in x["median_us"].items():
            for clock, value in clocks.items():
                assert value == median(median(a["samples"][key][clock]) for a in x["windows"])
