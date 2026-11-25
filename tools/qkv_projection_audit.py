"""Audit biased asymmetric Q/K/V packing on live HF Qwen2 inputs before permitting a rewrite."""

import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
import transformers
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from kernel_trace import kernel_census
from mlp_projection_audit import compare, digest, layout, measure
from provenance import capture
from qwen_cached_decode import suite, trajectory

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("three", "packed_views", "packed_copies", "q_plus_kv", "late_bias", "inductor")


def capture_values(model, rope, prefix, seed):
    att = model.layer.self_attn
    xs = [[] for _ in range(3)]
    ys = [[] for _ in range(3)]
    phases = []
    handles = []

    def tap(i):
        def hook(module, args, out):
            xs[i].append(args[0])
            ys[i].append(out)

        return hook

    def tables(module, args, kwargs):
        phases.append(kwargs["position_embeddings"])

    try:
        handles.append(att.register_forward_pre_hook(tables, with_kwargs=True))
        for i, name in enumerate("qkv"):
            handles.append(getattr(att, name + "_proj").register_forward_hook(tap(i)))
        steps = trajectory(model, rope, prefix, seed)
    finally:
        for h in handles:
            h.remove()
    assert len(phases) == 3 and all(len(y) == 3 for y in ys)
    stores = [steps[0][0][-2:], steps[0][1][-2:], steps[1][1][-2:]]
    out = []
    for i in range(3):
        x = xs[0][i]
        qkv = tuple(v[i] for v in ys)
        assert all(v[i] is x for v in xs)
        last = tuple(t[:, :, -x.shape[1] :, :] for t in stores[i])
        out.append((x, qkv, phases[i], last))
    return out


def run(offset, prefixes):
    model, rope = suite(offset)
    att = model.layer.self_attn
    ps = [getattr(att, n + "_proj") for n in "qkv"]
    sizes = [p.out_features for p in ps]
    w, b = (torch.cat([getattr(p, k) for p in ps]) for k in ("weight", "bias"))
    k0 = sizes[0]

    def original(x):
        return tuple(p(x) for p in ps)

    inductor = torch.compile(original, backend="inductor", fullgraph=True, dynamic=False)

    def project(n, x):
        if n == "inductor":
            return inductor(x)
        if n == "three":
            return tuple(p(x) for p in ps)
        if n == "q_plus_kv":
            return (ps[0](x), *F.linear(x, w[k0:], b[k0:]).split(sizes[1:], -1))
        y = F.linear(x, w) + b if n == "late_bias" else F.linear(x, w, b)
        s = y.split(sizes, -1)
        return (
            tuple(t.clone(memory_format=torch.contiguous_format) for t in s)
            if n == "packed_copies"
            else s
        )

    def rotate(t, phase):
        q, k, v = (z.view(*z.shape[:-1], -1, att.head_dim).transpose(1, 2) for z in t)
        rq, rk = apply_rotary_pos_emb(q, k, *phase)
        return rq, rk, v

    seeds = [offset + i for i in (60527, 60733, 60943)]
    panel = dict(
        weight_seed=60313 + offset,
        weight_sha256=[digest(p.weight) for p in ps],
        bias_sha256=[digest(p.bias) for p in ps],
        nonzero_bias=[int(p.bias.count_nonzero()) for p in ps],
        packed_weight_sha256=digest(w),
        packed_bias_sha256=digest(b),
        sizes=sizes,
        seeds=seeds,
        cases=[],
    )
    for prefix in prefixes:
        live = [capture_values(model, rope, prefix, s) for s in seeds]
        for stage in range(3):
            samples = []
            for seed, stages in zip(seeds, live, strict=True):
                x, ref, phase, cache = stages[stage]
                refrot = rotate(ref, phase)
                cache_eq = [compare(z, r) for z, r in zip(refrot[1:], cache, strict=True)]
                assert all(c["unequal_bits"] == 0 for c in cache_eq)
                methods = {}
                for n in NAMES:
                    try:
                        t = project(n, x)
                    except Exception as exc:
                        if n != "inductor":
                            raise
                        methods[n] = dict(
                            error=type(exc).__name__, private_exact=False, public_exact=False
                        )
                        continue
                    ds = [compare(z, r) for z, r in zip(t, ref, strict=True)]
                    ro = [compare(z, r) for z, r in zip(rotate(t, phase), refrot, strict=True)]
                    own = len({z.untyped_storage().data_ptr() for z in t}) == 3 and all(
                        z.untyped_storage().data_ptr() != x.untyped_storage().data_ptr() for z in t
                    )
                    exact = all(not z["unequal_bits"] for z in ds + ro)
                    public = exact and own and all(z["stride_match"] for z in ds)
                    methods[n] = dict(
                        projections=ds,
                        rotated_qkv=ro,
                        independent_storage=own,
                        private_exact=exact,
                        public_exact=public,
                    )
                assert methods["three"]["public_exact"]
                samples.append(
                    dict(
                        seed=seed,
                        input_sha256=digest(x),
                        reference_sha256=[digest(z) for z in ref],
                        input=layout(x),
                        actual_cache=cache_eq,
                        methods=methods,
                    )
                )
            admitted = [n for n in NAMES if all(s["methods"][n]["private_exact"] for s in samples)]
            # Bias-order control and GEMM family observations can expose a numerical boundary; not timings of the rejected path.
            bare = [F.linear(x, p.weight) for p in ps]
            pieces = F.linear(x, w).split(sizes, -1)
            ctrl = dict(
                wide_unbiased_vs_separate=[
                    compare(z, r) for z, r in zip(pieces, bare, strict=True)
                ],
                rounded_then_bias_vs_hf=[
                    compare(z + p.bias, r) for z, p, r in zip(bare, ps, ref, strict=True)
                ],
            )
            signatures = {
                n: kernel_census(lambda n=n: project(n, x))
                for n in NAMES
                if all("error" not in s["methods"][n] for s in samples)
            }
            timing = measure(
                {n: (lambda n=n: project(n, x)) for n in admitted}, offset + prefix * 17 + stage
            )
            c = dict(
                prefix=prefix,
                stage=("prefill", "decode1", "decode2")[stage],
                timing_seed=seeds[-1],
                samples=samples,
                controls=ctrl,
                all_kernel_signatures=signatures,
                **timing,
            )
            panel["cases"].append(c)
            print(
                offset,
                prefix,
                c["stage"],
                {
                    n: [v["unequal_bits"] for v in samples[0]["methods"][n]["projections"]]
                    for n in ("packed_views", "q_plus_kv")
                },
                c["median_us"]["device"],
                flush=True,
            )
    return panel


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prefix", type=int, nargs="+", default=[31, 127, 255])
    p.add_argument("--seed-offset", type=int, nargs="+", default=[80831, 86783])
    a = p.parse_args()
    d = dict(
        protocol="Independent pre-packed checkpoint snapshots, no production rewrite. Real Q,K,V outputs from biased HF modules and the same normalized hidden input. Compare all projection bits, then Q/K after reference RoPE and V; K/V new suffix also cross-checked with observed HF cache. Private gates require value equality; independent visible projections additionally require original strides and separate storage. Timing is projection-only, excludes RoPE and cache, seven shuffled windows on last input. All variants profiled for launch identification even when their latency is withheld. Weight/bias packing not timed. Late bias intentionally rounds the unbiased GEMM before add. No universal baseline claim.",
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        provenance=capture(ROOT / ".runtime/qkv-admission/sources"),
        panels=[run(o, a.prefix) for o in a.seed_offset],
    )
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
