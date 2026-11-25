"""Recompute the evidence gates and paired medians from the persisted report, off GPU."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median


def test_qualified_decode_evidence():
    root = Path(__file__).resolve().parents[1] / "reports"
    report = json.loads((root / "decode-sdpa-gb10.json").read_bytes())
    assert not report["smoke"] and (root / report["provenance"]["archive"]).is_file()
    cases = report["cases"]
    seeds = {c["seed"] for c in cases}
    assert (
        len(cases) == 14
        and len(seeds) == 2
        and len({(c["seed"], tuple(c["spec"])) for c in cases}) == 14
        and len({tuple(c["spec"]) for c in cases}) == 7
    )
    for c in cases:
        b, h, g, s, d, mask, dt, sc = c["spec"]
        assert (
            len(c["input_sha256"]) == 5
            and len({r[0] for r in c["input_sha256"]}) == 5
            and len(set(c["cpu_fp64_sha256"])) == 3
        )
        assert "aten.scaled_dot_product_attention.default" in c["export_graph"]
        paths = {"eager", "tile32", "tile64", "tile128"}
        assert paths <= set(c["gates"]) and "inductor" in (c["gates"] | c["build_failures"])
        for backend, gate in c["gates"].items():
            assert len(gate["requests"]) == 3 and gate["admitted"] == (
                gate["independent_storage"]
                and all(r["outside_tolerance"] == 0 and r["structure"] for r in gate["requests"])
            )
            if backend in paths:
                assert gate["admitted"]
        t = c["timing"]
        names = {k for k, v in c["gates"].items() if v["admitted"]}
        assert len(t["windows"]) == 7
        for w in t["windows"]:
            assert set(w["order"]) == names and len(w["order"]) == len(names)
        for clock in ("device", "host"):
            for name in names:
                samples = [w[clock][name]["samples_us"] for w in t["windows"]]
                assert all(
                    len(x) == (7 if clock == "device" else 15) and all(v > 0 for v in x)
                    for x in samples
                )
                assert t["median_us"][clock][name] == median(median(row) for row in samples)
        for name, meta in c["generated"].items():
            tile = int(name.removeprefix("tile"))
            p = meta["plan"]
            chunks = (s + tile - 1) // tile
            assert (
                p["tile"] == tile
                and p["chunks"] == chunks
                and p["workspace_bytes"] == (0 if chunks == 1 else b * h * chunks * (d + 2) * 4)
            )
            assert len(meta["kernels"]) == 1 + int(chunks > 1) == t["profiles"][name]["count"]
            assert all(
                len(k["source_sha256"]) == sha256().digest_size * 2
                and k["resources"]["registers"] > 0
                for k in meta["kernels"]
            )
