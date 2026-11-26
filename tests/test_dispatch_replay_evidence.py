from pathlib import Path
import gzip, hashlib, json
from statistics import median
from tools.import_campaign import classify


def test_catalog_gates_and_recomputed_latency_windows():
    root = Path(__file__).resolve().parents[1] / "reports"
    r = json.loads((root / "dispatch-replay-gb10.json").read_bytes())
    bundle = root / "campaign-dispatch-sample.json.gz"
    doc = json.load(gzip.open(bundle))
    catalog = r["catalog"]
    entries, counts = classify(doc["records"], catalog["compiler"])
    assert not r["smoke"] and (root / r["provenance"]["archive"]).is_file()
    assert (
        len(entries) == catalog["unique_keys"] == 10
        and counts == catalog["counts"]
        and hashlib.sha256(bundle.read_bytes()).hexdigest() == catalog["bundle_sha256"]
        and doc["records_sha256"] == catalog["records_sha256"]
    )
    assert (
        catalog["verified"]["distinct_sources_verified"] == 9
        and catalog["verified"]["indexed"] == 12
    )
    cs = r["cases"]
    assert (
        len(cs) == len({tuple(c["config"]) for c in cs}) == 12
        and len({c["signature"] for c in cs}) == 12
        and {c["signature"] for c in cs if c["entry"]} == set(entries)
    )
    for c in cs:
        assert c["entry"] == entries.get(c["signature"])
        assert c["selected"] == (c["entry"]["selected"] if c["entry"] else c["default"])
        assert (c["program_sha256"]["catalog"] == c["program_sha256"]["default"]) == (
            not c["entry"]
        )
        assert (
            len(c["inputs"]) == len({tuple(x) for x in c["inputs"]}) == 8
            and c["inductor_backend_calls"] == 1
        )
        allowed = set()
        g = c["gates"]
        assert set(g) == {"eager", "catalog", "default", "inductor_static"}
        for n, v in g.items():
            checks = v["checks"]
            assert len(checks) == 8 and v["admitted"] == all(
                x["within_tolerance"] and x["exact_bits"] and x["shape"] for x in checks
            )
            for x in checks:
                assert x["within_tolerance"] == (x["errors"] is not None)
                if c["config"][0] == "resnorm2":
                    assert x["exact_bits"] == (x["output_sha256"][1] == x["reference_sha256"][1])
            if v["admitted"]:
                allowed.add(n)
        hot = c["hot"]
        rot = c["rotating"]
        assert (
            {"catalog", "eager"} <= allowed
            and set(hot["profiles"]) == set(rot["median_us"]) == allowed
            and len(hot["windows"]) == len(rot["windows"]) == 7
        )
        for n in allowed:
            for k in ("host", "device"):
                assert hot["median_us"][k][n] == median(
                    median(w[k][n]["samples_us"]) for w in hot["windows"]
                )
            for k, us in rot["median_us"][n].items():
                assert us == median(median(w["samples"][n][k]) for w in rot["windows"])
