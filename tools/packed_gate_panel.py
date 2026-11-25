"""Isolate the generated SwiGLU epilogue on tensors from actual Qwen2 projections."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import transformers
from mlp_projection_audit import captures, compare, digest, layout, measure
from provenance import capture
from qwen_cached_decode import suite
from sparkfuse.packed_swiglu import fuse_packed_swiglu

ROOT = Path(__file__).resolve().parents[1]


class Epilogue(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, -1)
        return F.silu(a) * b


def run(model, rope, packed, past, stage, offset, emit_dir):
    m = Epilogue()
    exp = 60401 + offset
    seeds = [x + offset for x in (60527, 60637, 60733, 60869, 60943)]
    x0, _ = captures(model, rope, past, exp)[stage]
    ep = torch.export.export(m, (F.linear(x0, packed),))
    fns = {
        "eager": m,
        "inductor": torch.compile(ep.module(), backend="inductor", fullgraph=True, dynamic=False),
    }
    records = {}
    for k, vec in [("scalar", False), ("adaptive", True)]:
        g = ep.module()
        (rec,) = fuse_packed_swiglu(g, "cuda", "native", vectorize=vec)
        fns[k] = g
        records[k] = rec
        src = g.get_submodule(rec["module"]).program.source
        assert rec["cuda_sha256"] == sha256(src.encode()).hexdigest()
        if emit_dir:
            emit_dir.mkdir(parents=True, exist_ok=True)
            (emit_dir / (rec["cuda_sha256"] + ".cu")).write_text(src)
    seen = []
    for seed in seeds:
        x, ref = captures(model, rope, past, seed)[stage]
        projected = F.linear(x, packed)
        a, b = projected.chunk(2, -1)
        pre = [compare(t, y) for t, y in zip((a, b), ref[:2], strict=True)]
        assert all(not q["unequal_bits"] for q in pre)
        metrics = {}
        for n, f in fns.items():
            try:
                metrics[n] = compare(f(projected), ref[2])
            except Exception as exc:
                if n != "inductor":
                    raise
                metrics[n] = dict(error=type(exc).__name__)
        seen.append(
            dict(
                seed=seed,
                packed_tensor=layout(projected),
                input_sha256=digest(projected),
                reference_sha256=digest(ref[2]),
                projections=pre,
                outputs=metrics,
            )
        )
    allowed = [
        n
        for n in fns
        if all(
            s["outputs"][n].get("unequal_bits") == 0 and s["outputs"][n]["stride_match"]
            for s in seen
        )
    ]
    assert {"eager", "scalar", "adaptive"} <= set(allowed)
    timing = measure(
        {n: (lambda n=n: fns[n](projected)) for n in allowed}, 71917 + offset + stage + past
    )
    return dict(
        prefix=past,
        step=("prefill", "decode1", "decode2")[stage],
        graph=str(ep.graph),
        schedules=records,
        samples=seen,
        **timing,
    )


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--emit-dir", type=Path)
    ap.add_argument("--seed-offset", type=int, default=80831)
    a = ap.parse_args()
    m, r = suite(a.seed_offset)
    pack = torch.cat([m.layer.mlp.gate_proj.weight, m.layer.mlp.up_proj.weight])
    doc = dict(
        protocol="Only the epilogue on a packed dense BF16 projection output. Inputs originate from real HF Qwen2 prefill/decode activations; split projections and gated product checked against actual HF intermediates. No GEMM, checkpoint conversion or allocator setup in device timing. Five held-out seeds, separate export seed, same last input for all warm replay timings. Inductor must return exact visible product bits and stride. Host timings include allocation, dispatch and synchronization.",
        weight_seed=60313 + a.seed_offset,
        export_seed=60401 + a.seed_offset,
        timing_seed=60943 + a.seed_offset,
        packed_sha256=digest(pack),
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        provenance=capture(ROOT / ".runtime/packed-swiglu/sources"),
        cases=[],
    )
    for past, stage in ((31, 1), (31, 0), (127, 0)):
        c = run(m, r, pack, past, stage, a.seed_offset, a.emit_dir)
        doc["cases"].append(c)
        print(past, c["step"], c["median_us"], flush=True)
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
