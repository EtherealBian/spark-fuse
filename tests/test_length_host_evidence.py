"""The host-only ablation never times a failed ABI or numerical gate."""

import json
from pathlib import Path
from statistics import median


def test_native_length_host_data():
    root = Path(__file__).resolve().parents[1] / "reports"
    d = json.loads((root / "length-host-gb10.json").read_bytes())
    assert not d["smoke"] and (root / d["provenance"]["archive"]).is_file()
    assert len(d["cases"]) == len({(tuple(c["config"]), c["seed"]) for c in d["cases"]}) == 4
    for c in d["cases"]:
        ls = c["lengths"]
        assert (
            ls == [2, 31, 32, 33, 129, 513, 1025, 2049]
            and len(c["same_binary"]) == 2
            and len({b["binary_sha256"] for b in c["same_binary"]}) == 2
        )
        assert len({v[0] for v in c["input_sha256"]}) == len(set(c["cpu_fp64_sha256"])) == 24
        assert (
            c["gates"]["native"] == c["gates"]["ctypes"]
            and c["inductor_graphs_when_warm"] == c["inductor_graphs_after_timing"]
        )
        admitted = set()
        for k, g in c["gates"].items():
            assert len(g["requests"]) == 24 and g["admitted"] == (
                g["all_live"]
                and all(z["structure"] and not z["outside_tolerance"] for z in g["requests"])
            )
            if g["admitted"]:
                admitted.add(k)
        assert {"eager", "ctypes", "native"} <= admitted
        x = c["mixed"]
        assert len(x["windows"]) == 7 and set(x["median_us"]) == admitted
        for k, sub in x["median_us"].items():
            for mode, t in sub.items():
                assert t == median(median(w["samples"][k][mode]) for w in x["windows"])
        for x in c["hot"].values():
            assert set(x["profiles"]) == admitted and len(x["windows"]) == 7
            for mode in ("host", "device"):
                for k, t in x["median_us"][mode].items():
                    assert t == median(median(w[mode][k]["samples_us"]) for w in x["windows"])
        for x in c["checked"].values():
            assert len(x["windows"]) == 7 and set(x["median_us"]) == {
                a + "_" + b for a in ("native", "ctypes") for b in ("checked", "bounded")
            }
            for k, t in x["median_us"].items():
                assert t == median(median(w["samples"][k]) for w in x["windows"])
