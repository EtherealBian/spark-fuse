import json
from pathlib import Path
from statistics import median


def test_visibility_ownership_and_all_bit_patterns_in_measured_panel():
    root = Path(__file__).resolve().parents[1] / "reports"
    r = json.loads((root / "slot-forward-gb10.json").read_bytes())
    assert not r["smoke"] and (root / r["provenance"]["archive"]).is_file()
    cs = r["cases"]
    assert len(cs) == 16 and {(c["config"][2], tuple(c["visible"]), c["seed"]) for c in cs} == {
        (s, (k, v), z)
        for s in (31, 4093)
        for k in (True, False)
        for v in (True, False)
        for z in (4374611, 4381177)
    }
    for c in cs:
        b, g, s, d, dt = c["config"]
        m = sum(c["visible"])
        assert (
            c["inductor_backend_calls"] == 1
            and len(c["oracle"]) == len(c["patterns"]) == len(c["inputs"]) == 8
            and len({r[0] for r in c["inputs"]}) == 8
            and len(c["binary"]["binary_sha256"]) == 64
        )
        gates = c["gates"]
        assert set(gates) == {"eager", "inductor_static", "slot_forward"}
        ok = set()
        for n, z in gates.items():
            assert len(z["rows"]) == 8
            admit = z["all_live"]
            for row, ref in zip(z["rows"], c["oracle"], strict=True):
                assert len(ref) == 2 + m and row["matches"] == [
                    a == t for a, t in zip(row["hashes"], ref, strict=True)
                ]
                admit = admit and row["structure"] and all(row["matches"])
            assert z["admitted"] == admit
            if admit:
                ok.add(n)
        hot = c["hot"]
        rot = c["rotating"]
        assert (
            {"eager", "slot_forward"} <= ok
            and set(hot["profiles"]) == set(rot["median_us"]) == ok
            and len(hot["windows"]) == len(rot["windows"]) == 7
            and hot["profiles"]["slot_forward"]["count"] == 1
        )
        for name in ok:
            assert c["allocation"][name]["output_bytes"] == b * g * d * 2 * (2 + s * m)
            for k in ("device", "host"):
                assert hot["median_us"][k][name] == median(
                    median(w[k][name]["samples_us"]) for w in hot["windows"]
                )
            for k, us in rot["median_us"][name].items():
                assert us == median(median(w["samples"][name][k]) for w in rot["windows"])
