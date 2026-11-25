"""Host and device ablation: one exported row range vs per-shape FX and its guarded native region."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import transformers
from mlp_projection_audit import captures, compare, digest, layout, measure
from packed_gate_panel import Epilogue
from provenance import capture
from qwen_cached_decode import suite
from sparkfuse.packed_rows import PackedRows
from sparkfuse.packed_swiglu import fuse_packed_swiglu

ROOT = Path(__file__).resolve().parents[1]


def save_source(source, key, folder):
    assert sha256(source.encode()).hexdigest() == key
    if folder:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / (key + ".cu")).write_bytes(source.encode())


@torch.inference_mode()
def run(offset, folder):
    model, rope = suite(offset)
    pack = torch.cat([model.layer.mlp.gate_proj.weight, model.layer.mlp.up_proj.weight])
    m = Epilogue()
    seeds = [offset + i for i in (60527, 60637, 60733, 60869, 60943)]
    ex = offset + 60401
    # Rows are independent. Take the first N real token rows, NOT a Qwen2 layer with N-token decode.
    acts = {s: captures(model, rope, 127, s) for s in [ex, *seeds]}

    def tensor(seed, rows, stage):
        x, ref = acts[seed][stage]
        return F.linear(x, pack).reshape(-1, pack.shape[0])[:rows], ref[2].reshape(
            -1, pack.shape[0] // 2
        )[:rows]

    x, _ = tensor(ex, 7, 0)
    sym = torch.export.Dim("tokens", min=1, max=256)
    ep = torch.export.export(m, (x,), dynamic_shapes=({0: sym},))
    exmod = ep.module()
    runners = {k: PackedRows(ep, schedule=k) for k in ("adaptive", "scalar")}
    for r in runners.values():
        save_source(r.source, r.record["cuda_sha256"], folder)
    dyn_ind = torch.compile(exmod, backend="inductor", dynamic=True, fullgraph=True)
    cases = []
    for rows, stage in (
        (2, 1),
        (1, 0),
        (4, 0),
        (8, 0),
        (16, 0),
        (32, 0),
        (62, 0),
        (128, 0),
        (254, 0),
    ):
        x, _ = tensor(ex, rows, stage)
        se = torch.export.export(m, (x,))
        gm = se.module()
        (rec,) = fuse_packed_swiglu(gm, "cuda", "native", vectorize=True)
        region = gm.get_submodule(rec["module"])
        save_source(region.program.source, rec["cuda_sha256"], folder)
        # Diagnostic control only: positional ABI verified for this single-region graph. Same CUDA bytes.
        dispatch = region.dispatch
        paths = {
            "eager": m,
            "export_dynamic": exmod,
            "static_fx": gm,
            "static_native": lambda a: dispatch(a)[0],
            "bounded_scalar": runners["scalar"],
            "bounded_adaptive": runners["adaptive"],
            "inductor_dynamic": dyn_ind,
        }
        samples = []
        for s in seeds:
            x, ref = tensor(s, rows, stage)
            before = digest(x)
            metrics = {}
            for n, fn in paths.items():
                try:
                    y = fn(x)
                    metrics[n] = compare(y, ref)
                    metrics[n]["fresh"] = (
                        y.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
                    )
                    metrics[n]["input_unchanged"] = digest(x) == before
                except Exception as e:
                    if n != "inductor_dynamic":
                        raise
                    metrics[n] = dict(error=type(e).__name__)
            samples.append(
                dict(seed=s, input_sha256=before, reference_sha256=digest(ref), outputs=metrics)
            )
        admitted = {
            n
            for n in paths
            if all(
                s["outputs"][n].get("unequal_bits") == 0
                and all(s["outputs"][n][q] for q in ("stride_match", "fresh", "input_unchanged"))
                for s in samples
            )
        }
        assert set(paths) - {"inductor_dynamic"} <= admitted
        timing = measure(
            {n: (lambda f=f: f(x)) for n, f in paths.items() if n in admitted},
            offset + 73111 + rows,
        )
        print(
            rows,
            {m: {k: round(v, 3) for k, v in d.items()} for m, d in timing["median_us"].items()},
            flush=True,
        )
        cases.append(
            dict(
                rows=rows,
                stage="decode1" if stage == 1 else "prefill_slice",
                input=layout(x),
                static_record=rec,
                samples=samples,
                **timing,
            )
        )
    return dict(
        protocol="Standalone gate, BF16, on dense rows from a real random-weight Qwen2 gate/up projection. Only the two-row case is true decode; other cases are prefixes of the flattened prefill. Five held-out seeds, last one timed; seven shuffled windows, synchronized host calls and separate CUDA Graph hot device events. Static native removes FX dispatch but uses the exact guarded submodule entry and allocates fresh outputs. Bounded paths reuse one guarded native executable per schedule across all cases. Not a decoder-layer or service measurement. Inductor gated by exact product bits/stride like all others.",
        weight_seed=60313 + offset,
        packed_sha256=digest(pack),
        export_seed=ex,
        timing_seed=seeds[-1],
        export_shape=[7, 17920],
        dynamic_fx=str(ep.graph),
        schedules={k: r.record for k, r in runners.items()},
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        provenance=capture(ROOT / ".runtime/packed-rows/sources"),
        cases=cases,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--emit-dir", type=Path)
    p.add_argument("--seed-offset", type=int, default=80831)
    a = p.parse_args()
    d = run(a.seed_offset, a.emit_dir)
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, allow_nan=False, separators=(",", ":")) + "\n")
    tmp.replace(a.output)
