"""Cold-prompt 235B prefill tiling: identical already published inputs and fixed output budget."""

from contextlib import redirect_stdout
import sys
import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import median
from measure235 import run, audit, require

TILES = (128, 256, 256, 128)


def bind(pid, binary, block, tile):
    p = Path("/proc") / str(pid)
    require(p.joinpath("exe").samefile(binary), "wrong backend process")
    argv = p.joinpath("cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    for name, value in {
        "--ubatch-size": tile,
        "--batch-size": 512,
        "--parallel": 1,
        "--ctx-size": 4096,
        "--gpu-layers": "all",
        "--flash-attn": "on",
        "--cache-ram": 0,
        "--host": "127.0.0.1",
    }.items():
        require(
            argv.count(name) == 1 and argv[argv.index(name) + 1] == str(value),
            "backend flag drift: " + name,
        )
    require(
        not any(x.startswith(("--spec", "--draft")) for x in argv) and "--no-context-shift" in argv,
        "unexpected extra decode mode",
    )
    start = p.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
    seed = ":".join((Path("/proc/sys/kernel/random/boot_id").read_text().strip(), str(pid), start))
    return dict(
        block=block, tile=tile, process_fingerprint=hashlib.sha256(seed.encode()).hexdigest()
    )


def compare(blocks, base, eog):
    audit(base, eog)
    require(len(blocks) == 4, "four processes required")
    seen = set()
    data = []
    keys = [f"{n}:{j}" for n in (128, 1024) for j in (0, 1)]
    for idx, ((setting, raw), ub) in enumerate(zip(blocks, TILES, strict=True)):
        fp = setting["process_fingerprint"]
        require(
            type(setting["block"]) is int
            and (setting["block"], setting["tile"]) == (idx, ub)
            and re.fullmatch("[0-9a-f]{64}", fp)
            and fp not in seen,
            "process order, tile or identity invalid",
        )
        seen.add(fp)
        audit(raw, eog)
        require(raw["inputs"] == base["inputs"], "workload differs from baseline")
        data.append({(r["case"], r["rep"]): r for r in raw["rows"]})
    gold = {(r["case"], r["rep"]): r["ids"] for r in base["rows"]}
    cases = {}
    for key in keys:
        rows = []
        exact = True
        repeatable = (
            all(d[(key, 0)]["ids"] == d[(key, 1)]["ids"] for d in data)
            and gold[(key, 0)] == gold[(key, 1)]
        )
        base_stable = all(
            data[i][(key, r)]["ids"] == gold[(key, r)] for r in (0, 1) for i in (0, 3)
        )
        for a, b in ((0, 1), (3, 2)):
            pair = []
            for rep in (0, 1):
                x = data[a][(key, rep)]
                y = data[b][(key, rep)]
                div = next(
                    (
                        j
                        for j, (left, right) in enumerate(zip(x["ids"], y["ids"], strict=True), 1)
                        if left != right
                    ),
                    None,
                )
                exact &= div is None
                pair.append(
                    dict(
                        rep=rep,
                        first_divergence_1based=div,
                        wall_saving=1 - y["wall_ms"] / x["wall_ms"],
                        prefill_saving=1 - y["prompt_ms"] / x["prompt_ms"],
                        decode_saving=1 - y["decode_ms"] / x["decode_ms"],
                    )
                )
            rows.append(
                dict(
                    control_block=a,
                    candidate_block=b,
                    median_wall_saving=median(r["wall_saving"] for r in pair),
                    median_prefill_saving=median(r["prefill_saving"] for r in pair),
                    paired_requests=pair,
                )
            )
        cases[key] = dict(
            within_block_reproducible=repeatable,
            control_matches_public=base_stable,
            cross_tile_exact=exact,
            orders=rows,
        )
    long = [v for k, v in cases.items() if k.startswith("1024:")]
    short = [v for k, v in cases.items() if k.startswith("128:")]
    stable = all(
        c["within_block_reproducible"] and c["control_matches_public"] and c["cross_tile_exact"]
        for c in cases.values()
    )
    timing = all(
        min(o["median_wall_saving"], o["median_prefill_saving"]) >= 0.05
        for c in long
        for o in c["orders"]
    ) and all(o["median_wall_saving"] >= -0.03 for c in short for o in c["orders"])
    return dict(
        cases=cases,
        meets_timing_gate=timing,
        meets_exact_output_gate=stable,
        recommend_for_this_workload=stable and timing,
        scope="One Spark, fixed 16-token cold-prompt requests. Same checkpoint and server; four independent launches (128/256/256/128). Long prompts must save 5% in wall and prefill for both request orders and individual prompts, short prompts can regress at most 3% in wall time. No claim on unconstrained text quality, new lengths, or peak memory.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("run", "audit", "bind"))
    p.add_argument("--block", type=int, choices=range(4), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int)
    p.add_argument("--pid", type=int)
    p.add_argument("--server", type=Path)
    a = p.parse_args()
    tile = TILES[a.block]
    rt = json.loads((a.out / "runtime.json").read_bytes())
    require(
        rt["context"] == dict(n_seq_max=1, n_ctx=4096, n_ctx_seq=4096, n_batch=512, n_ubatch=tile)
        and rt["gpu_layers_including_output"] == 95
        and rt["device"] == "NVIDIA GB10",
        "wrong context/device",
    )
    if a.mode == "bind":
        with (a.out / "tile-setting.json").open("x") as f:
            json.dump(bind(a.pid, a.server, a.block, tile), f, indent=2)
            f.write("\n")
        return
    dest = a.out / "rows.json"
    partial = a.out / "rows.json.partial"
    tmp = a.out / "rows.json.tmp"
    if a.mode == "run":
        require(
            not any(q.exists() for q in (dest, partial, tmp)), "fresh output directory required"
        )

        def save(obj):
            tmp.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
            tmp.replace(partial)

        # Keep progress on stderr; stdout is compared byte-for-byte with offline replay.
        with redirect_stdout(sys.stderr):
            r = run(a.port, save)
        audit(r, rt["eog_token_ids"])
        partial.replace(dest)
    print(
        json.dumps(
            audit(json.loads(dest.read_bytes()), rt["eog_token_ids"]), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
