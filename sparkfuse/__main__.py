"""Offline frontend for trusted torch.export archives; emission does not build or launch code."""

import argparse
import hashlib
import json
from pathlib import Path
import torch
from .codegen import Schedule, generate
from .ir import Unsupported, lower
from .tuning import backend_id


def hash_file(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def frontend(path, schedule):
    # torch.export archives are deserialized by PyTorch. This is not a sandbox.
    ep = torch.export.load(path)
    if any(s.kind.name != "USER_OUTPUT" for s in ep.graph_signature.output_specs):
        raise Unsupported("external effects are not part of the launch ABI")
    if any(
        s.kind.name not in ("USER_INPUT", "PARAMETER", "BUFFER", "CONSTANT_TENSOR")
        for s in ep.graph_signature.input_specs
    ):
        raise Unsupported("unsupported graph input binding")
    p = lower(ep)
    source = generate(p, schedule)
    manifest = {
        "format": "sparkfuse.cuda-source.v1",
        "input_archive_sha256": hash_file(path),
        "compiler_sha256": backend_id(),
        "torch": str(torch.__version__),
        "schedule": vars(schedule),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "plan": p.explain(),
        "bindings": [
            {
                "value_id": i,
                "kind": x.kind.name,
                "target": str(x.target) if x.target is not None else None,
            }
            for i, x in zip(p.inputs, ep.graph_signature.input_specs)
        ],
        "abi": "launch(void **ptr, int rows, void *cuda_stream): input pointers in binding order then outputs in plan order. Source only; no weights or GPU binary included.",
    }
    return source, manifest, str(ep.graph) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m sparkfuse",
        description="Inspect or emit CUDA from a trusted static torch.export archive; no GPU timing or binary compilation.",
    )
    p.add_argument("action", choices=("inspect", "emit"))
    p.add_argument("archive", type=Path, help="trusted .pt2 file only")
    p.add_argument(
        "--out", type=Path, help="emit into a NEW directory; never overwrite an existing directory"
    )
    p.add_argument("--threads", type=int, default=128)
    p.add_argument(
        "--layout",
        choices=(
            "block",
            "warp",
            "flat",
            "warp_xor",
            "warp_slots",
            "warp_bf16_guard",
            "block_welford",
            "compact_mean",
            "mean_tree",
            "mean_cohort",
            "mean_packed_tile",
        ),
        default="block",
    )
    p.add_argument("--vec", type=int, default=1)
    p.add_argument("--cache", action="store_true")
    a = p.parse_args(argv)
    if (a.action == "emit") != (a.out is not None):
        p.error("provide --out exactly when emitting")
    if a.out is not None and (a.out.exists() or a.out.is_symlink()):
        p.error("output destination already exists")
    try:
        cu, m, fx = frontend(a.archive, Schedule(a.threads, a.layout, a.cache, a.vec))
    except (Unsupported, FileNotFoundError) as e:
        p.exit(2, f"SparkFuse: {e}\n")
    text = json.dumps(m, indent=2, allow_nan=False) + "\n"
    if a.action == "inspect":
        print(text, end="")
        return 0
    a.out.mkdir(exist_ok=False)
    (a.out / "kernel.cu").write_bytes(cu.encode("utf-8"))
    (a.out / "graph.txt").write_text(fx)
    (a.out / "manifest.json").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
