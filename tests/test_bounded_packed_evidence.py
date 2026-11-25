"""Recheck eligibility, order, timing and source identity without CUDA or a model download."""

from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from tools.verify_sources import verify_reports

ROOT = Path(__file__).resolve().parents[1]


def test_bounded_panel_artifacts():
    files = [ROOT / "reports" / f"bounded-packed-rows{s}-gb10.json" for s in ("", "-heldout")]
    assert verify_reports(files)["verified_reports"] == 2
    docs = [json.loads(p.read_text()) for p in files]
    assert len({d["weight_seed"] for d in docs}) == len({d["packed_sha256"] for d in docs}) == 2
    assert docs[0]["schedules"] == docs[1]["schedules"]
    for d, suffix in zip(docs, ("", "-heldout"), strict=True):
        deployed = json.loads(
            (ROOT / "reports" / f"qwen-decode-packed-swiglu{suffix}-gb10.json").read_bytes()
        )
        assert d["packed_sha256"] == deployed["checkpoint"]["packed_sha256"]
        assert (
            d["export_seed"] == deployed["export_seed"]
            and d["timing_seed"] == deployed["seeds"][-1]
        )

        def code(record):
            key = record["cuda_sha256"]
            data = (ROOT / "examples/generated/packed-rows" / (key + ".cu")).read_bytes()
            assert sha256(data).hexdigest() == key
            return data

        for n, v in d["schedules"].items():
            assert (
                v["bounds"] == [1, 256]
                and v["width"] == 8960
                and v["dtype"] == "bf16"
                and v["schedule"] == n
                and v["max_packet_elements"] == (1 if n == "scalar" else 8)
            )
            assert b"int rows" in code(v)
        assert [c["rows"] for c in d["cases"]] == [2, 1, 4, 8, 16, 32, 62, 128, 254]
        all_n = {
            "eager",
            "export_dynamic",
            "static_fx",
            "static_native",
            "bounded_scalar",
            "bounded_adaptive",
            "inductor_dynamic",
        }
        for c in d["cases"]:
            code(c["static_record"])
            assert c["stage"] == ("decode1" if c["rows"] == 2 else "prefill_slice") and c["input"][
                "shape"
            ] == [c["rows"], 17920]
            seeds = c["samples"]
            assert [s["seed"] for s in seeds] == deployed["seeds"] and d["export_seed"] not in [
                s["seed"] for s in seeds
            ]
            assert len({s["input_sha256"] for s in seeds}) == 5
            assert all(set(s["outputs"]) == all_n for s in seeds)
            accepted = {
                n
                for n in all_n
                if all(
                    s["outputs"][n].get("unequal_bits") == 0
                    and all(
                        s["outputs"][n][k] for k in ("stride_match", "fresh", "input_unchanged")
                    )
                    for s in seeds
                )
            }
            assert (
                all_n - {"inductor_dynamic"} <= accepted
                and set(c["profiles"]) == accepted
                and len(c["windows"]) == 7
            )
            assert c["profiles"]["static_native"] == c["profiles"]["static_fx"]
            for n, k in [("bounded_scalar", 1), ("bounded_adaptive", 8)]:
                p = c["profiles"][n]
                assert (
                    p["count"] == 1
                    and sum(v for s, v in p["kernel_names"].items() if f"packed_swiglu<{k}>" in s)
                    == 1
                )
            for w in c["windows"]:
                assert len(w["order"]) == len(accepted) and set(w["order"]) == accepted
                for metric, size in [("host", 15), ("device", 7)]:
                    assert set(w[metric]) == accepted and all(
                        len(w[metric][n]["samples_us"]) == size for n in accepted
                    )
            for metric in ("host", "device"):
                assert c["median_us"][metric] == {
                    n: median(median(w[metric][n]["samples_us"]) for w in c["windows"])
                    for n in accepted
                }
