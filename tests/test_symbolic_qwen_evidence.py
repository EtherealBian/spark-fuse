"""Audit the independently carried four-token requests and raw clocks without a GPU."""

import json
from pathlib import Path
from statistics import median


def test_real_symbolic_layer_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "symbolic-qwen-gb10.json").read_bytes())
    assert not d["smoke"] and (root / d["provenance"]["archive"]).is_file()
    assert {(c["masked"], c["seed"]) for c in d["cases"]} == {
        (m, s) for m in (True, False) for s in (3511217, 3522353)
    }
    for c in d["cases"]:
        assert (
            c["steps"] == 4
            and c["initial_past"] == [2, 29, 30, 31, 125, 253, 509, 2045]
            and c["qualification_past"] == [17, 65]
        )
        assert (
            len(c["input_sha256"]) == 8
            and len({xs[0] for t in c["input_sha256"] for xs in t}) == 32
            and len(c["weight_sha256"]) == 12
        )
        assert (
            c["binary"]["native"] == c["binary"]["ctypes"]
            and len({b["binary_sha256"] for b in c["binary"]["native"]}) == 2
        )
        for host, rec in c["sites"].items():
            assert (
                rec["host_backend"] == host
                and len(rec["erased_repeats"]) == 2
                and rec["offset_from_symbol"] == 1
                and (rec["plan"]["lower"], rec["plan"]["upper"]) == (3, 2049)
            )
        for start, row, ws in zip(
            c["initial_past"], c["site_fp64"], c["workspace_bytes"], strict=True
        ):
            assert len(row) == 4 and [z["sequence"] for z in row] == list(
                range(start + 1, start + 5)
            )
            for i, (z, w) in enumerate(zip(row, ws, strict=True)):
                s = z["sequence"]
                assert (
                    len(z["reference_sha256"]) == 64
                    and z["cpu_error"]["structure"]
                    and z["cpu_error"]["outside_tolerance"] == 0
                )
                assert z["kv_pitch"] == [128 * s, 64 * s, 64, 1] and z["incoming_cache_pitch"] == (
                    [128 * (s - 1), 64, 128, 1] if i == 0 else [128 * (s - 1), 64 * (s - 1), 64, 1]
                )
                assert w == (0 if s <= 32 else 4 * 2 * 8 * ((s + 31) // 32) * 66)
        admitted = set()
        assert c["gates"]["native"] == c["gates"]["ctypes"]
        for name, g in c["gates"].items():
            flat = [z for row in g["steps"] for z in row]
            assert len(g["steps"]) == 8 and all(len(row) == 4 for row in g["steps"])
            for z in flat:
                o = z["outputs"]
                assert len(o) == 3 and z["within_tolerance"] == all(x["outside"] == 0 for x in o)
                assert z["exact_cache"] == all(x["different_bytes"] == 0 for x in o[1:])
                assert all(
                    (x["ref_sha256"] == x["sha256"]) == (x["different_bytes"] == 0) for x in o
                )
            assert g["admitted"] == (
                g["all_live"]
                and all(
                    z["within_tolerance"] and z["geometry"] and z["fresh"] and z["prefix_bitwise"]
                    for z in flat
                )
            )
            assert g["exact_caches"] == (g["admitted"] and all(z["exact_cache"] for z in flat))
            if name != "inductor_dynamic":
                assert g["exact_caches"]
            if g["admitted"]:
                admitted.add(name)
        assert ("inductor_dynamic" in c["build_failures"]) != ("inductor_dynamic" in c["gates"])
        r = c["rotating"]
        assert len(r["windows"]) == 7 and set(r["median_us"]) == admitted
        for k, modes in r["median_us"].items():
            for mode, us in modes.items():
                assert us == median(median(w["samples"][k][mode]) for w in r["windows"])
        assert set(c["hot"]) == {"2", "125", "2045"}
        for h in c["hot"].values():
            assert len(h["windows"]) == 7 and set(h["profiles"]) == admitted
            for mode, times in h["median_us"].items():
                for k, us in times.items():
                    assert us == median(median(w[mode][k]["samples_us"]) for w in h["windows"])
