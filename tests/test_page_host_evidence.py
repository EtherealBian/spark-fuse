from pathlib import Path
from statistics import median
import json


def test_identical_cuda_host_call_isolation():
    root = Path(__file__).resolve().parents[1] / "reports"
    r = json.loads((root / "page-host-gb10.json").read_bytes())
    assert not r["smoke"] and (root / r["provenance"]["archive"]).is_file()
    cs = r["cases"]
    assert (
        len(cs) == 8
        and len({(c["ragged"], tuple(c["config"])) for c in cs}) == 4
        and len({(c["ragged"], tuple(c["config"]), c["seed"]) for c in cs}) == 8
        and {c["seed"] for c in cs} == {4309823, 4316429}
    )
    for c in cs:
        b, h, g, d, t, s, mask, dt = c["config"]
        cap = (s + t - 1) // t
        v = c["variants"]
        names = {"split", "native"}
        assert (
            set(v) == names
            and v["split"]["sources"] == v["native"]["sources"]
            and v["split"]["binaries"] == v["native"]["binaries"]
            and len(v["native"]["binaries"]) == 1 + int(cap > 1)
        )
        assert len(c["fp64"]) == len(c["inputs"]) == 8 and len({z[0] for z in c["inputs"]}) == 8
        gs = c["gates"]
        assert len(gs["native"]["rows"]) == 8 and [q["sha256"] for q in gs["native"]["rows"]] == [
            q["sha256"] for q in gs["split"]["rows"]
        ]
        hot = c["hot"]
        rot = c["rotating"]
        assert (
            len(hot["windows"]) == len(rot["windows"]) == 7
            and set(rot["median_us"]) == set(hot["profiles"]) == names
        )
        for name in names:
            vv = v[name]
            assert all(
                c["binaries"][k]["source_sha256"] == sha and c["binaries"][k]["binary_sha256"] == k
                for k, sha in zip(vv["binaries"], vv["sources"], strict=True)
            )
            assert vv["scratch_bytes"] == (0 if cap == 1 else b * h * cap * (d + 2) * 4)
            assert gs[name]["all_live"] and all(
                q["structure"] and not q["outside_tolerance"] for q in gs[name]["rows"]
            )
            assert (
                hot["profiles"][name]["count"] == 1 + int(cap > 1)
                and c["allocation"][name]["live_output_bytes"] == b * h * d * 2
            )
            for m in ("device", "host"):
                assert hot["median_us"][m][name] == median(
                    median(w[m][name]["samples_us"]) for w in hot["windows"]
                )
            for m, us in rot["median_us"][name].items():
                assert us == median(median(w["samples"][name][m]) for w in rot["windows"])
