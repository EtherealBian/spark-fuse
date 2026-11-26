"""Reconstruct a paired-serving study from two real launches and its one-slot capacity reference."""

import hashlib, json, math, sys
from pathlib import Path
from statistics import median
from audit_glm355 import gate, read_inputs, rebuild as capacity
from physical_guard import accepted
from pair355 import need, reduce_block
from glm_token_contract import compare


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    need(
        p.is_file() and not p.is_symlink() and p.stat().st_size <= 16 * 1024**2, "bad evidence file"
    )
    return json.loads(p.read_bytes())


def summarize(blocks):
    cases = {}
    proof = []
    for n in (128, 1024):
        pairs = []
        for b, rows in enumerate(blocks):
            for rep in range(2):
                cells = [r for r in rows if r["length"] == n and r["rep"] == rep]
                need(
                    len(cells) == 2 and set(x["mode"] for x in cells) == {"serial", "together"},
                    "unmatched arms",
                )
                a = next(r for r in cells if r["mode"] == "serial")
                t = next(r for r in cells if r["mode"] == "together")
                pairs.append(
                    dict(
                        block=b,
                        rep=rep,
                        first_mode=cells[0]["mode"],
                        lead=a["lead"],
                        serial_pair_ms=a["makespan_ms"],
                        together_pair_ms=t["makespan_ms"],
                        pair_saving=1 - t["makespan_ms"] / a["makespan_ms"],
                        first_completion_change=t["first_ms"] / a["first_ms"] - 1,
                        serial_decode_calls=a["decode_calls"],
                        together_decode_calls=t["decode_calls"],
                        together_excess_busy_slots=t["excess_busy_slots"],
                        equal_outputs=[u == v for u, v in zip(a["ids"], t["ids"], strict=True)],
                        all_match_one_slot=all(a["baseline_exact"] + t["baseline_exact"]),
                    )
                )
        need(
            [p["block"] for p in pairs] == [0, 0, 1, 1]
            and all(
                {p["first_mode"] for p in pairs if p["block"] == b} == {"serial", "together"}
                for b in (0, 1)
            ),
            "unbalanced block/order",
        )
        pfast = all(p["pair_saving"] >= 0.05 for p in pairs)
        pfirst = all(p["first_completion_change"] <= 0.10 for p in pairs)
        pexact = all(all(p["equal_outputs"]) and p["all_match_one_slot"] for p in pairs)
        pjoint = all(p["together_excess_busy_slots"] > 0 for p in pairs)
        cases[str(n)] = dict(
            pairs=pairs,
            median_serial_pair_ms=median([p["serial_pair_ms"] for p in pairs]),
            median_together_pair_ms=median([p["together_pair_ms"] for p in pairs]),
            all_four_pair_makespans_improve_at_least_5pct=pfast,
            first_completion_within_10pct_all_pairs=pfirst,
            exact_outputs=pexact,
            joint_backend_occupancy_all_pairs=pjoint,
        )
        proof.append(pfast and pfirst and pexact and pjoint)
    return dict(
        lengths=cases,
        qualified_for_tested_workload=all(proof),
        scope="Paired requests only; burst makespan and first-completion latency are both reported. Native busy-slot counts prove simultaneous processing states at decode calls, not that every CUDA kernel served both tokens. A throughput win can still delay the first response. No long-context, natural-EOS, quality, or production SLA claim.",
    )


def audit(lock, out, ref, contract):
    base = Path("reports/qwen235b/runtime.json")
    refsum = capacity(lock, ref, base)
    need(
        refsum == read(ref / "summary.json") and refsum["tested_repeatability_gate"] is True,
        "single-slot reference not repeatable",
    )
    need(
        sha(contract) == "bb20002a4c22c93b48575fe0292ed4aca0345e5d64d9ba2a72f859bcc0e86015",
        "changed independent contract",
    )
    prior = read(ref / "runtime.json")
    raw = read(ref / "rows.json")
    compare(read(contract), raw, prior)
    mask = prior["eog_token_ids"]
    _, bud = read_inputs(lock, ref / "header-budget.json")
    dig = {}
    blocks = []
    epochs = []
    observations = []
    for i in range(2):
        d = out / f"block-{i}"
        names = (
            "runtime.json",
            "checkpoint.json",
            "admission-transfer.json",
            "admission.json",
            "header-budget.json",
            "memory-guard.json",
            "rows.json",
        )
        r = {k: read(d / k) for k in names}
        dig[f"block-{i}"] = {k: sha(d / k) for k in names}
        runtime = r["runtime.json"]
        need(
            all(
                sha(d / k) == sha(ref / k)
                for k in ("checkpoint.json", "header-budget.json", "admission-transfer.json")
            ),
            "different checkpoint, header budget or transfer approval",
        )
        gate("transfer", bud, r["admission-transfer.json"])
        gate("launch", bud, r["admission.json"])
        accepted(r["memory-guard.json"])
        need(
            runtime["context"]
            == dict(n_seq_max=2, n_ctx=4096, n_ctx_seq=2048, n_batch=512, n_ubatch=128),
            "wrong per-slot/total KV budget",
        )
        need(
            all(
                runtime[k] == prior[k]
                for k in (
                    "device",
                    "llama_cpp_revision",
                    "server_sha256",
                    "runtime_lib_sha256",
                    "driver",
                    "eog_token_ids",
                    "gpu_layers_including_output",
                )
            ),
            "runtime/placement differs from capacity",
        )
        need(
            runtime["gpu_layers_including_output"] == 94 and runtime["device"] == "NVIDIA GB10",
            "wrong model/device",
        )
        buf = runtime["backend_buffers"]
        need(
            all(
                type(x["reported_mib"]) in (float, int)
                and math.isfinite(x["reported_mib"])
                and x["reported_mib"] > 0
                for x in buf
            ),
            "invalid buffer sizes",
        )

        def grouped(b, k):
            return sorted((x["device"], x["reported_mib"]) for x in b if x["kind"] == k)

        need(
            grouped(buf, "model") == grouped(prior["backend_buffers"], "model")
            and any(x["kind"] == "KV" and x["device"] == "CUDA0" for x in buf),
            "different model placement or missing KV allocation",
        )
        blocks.append(reduce_block(r["rows.json"], raw, mask, i))
        epochs.append(r["rows.json"]["waves"][0]["before"]["epoch"])
        observations.append(
            dict(
                context=runtime["context"],
                backend_buffers=buf,
                minimum_sampled_mem_available=r["memory-guard.json"]["min_available_bytes"],
            )
        )
    need(epochs[0] != epochs[1], "not two fresh backend processes")
    return dict(
        schema="glm355.paired-serving.v1",
        blocks=2,
        measured_responses=32,
        warmups_excluded=4,
        baseline_sha256=sha(ref / "summary.json"),
        token_contract_sha256=sha(contract),
        evidence_sha256=dig,
        observations=observations,
        **summarize(blocks),
    )


if __name__ == "__main__":
    print(json.dumps(audit(*map(Path, sys.argv[1:])), indent=2, allow_nan=False))
