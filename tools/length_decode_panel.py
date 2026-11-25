"""One symbolic KV length family: fresh-output FP64 gates, shape churn and two clocks."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch._dynamo.backends.registry import lookup_backend
from decode_sdpa_panel import inputs, cpu_reference, metrics
from examples.length_decode import CanonicalDecode, export_length_decode
from examples.single_decode import SingleDecode
from ordered_score_panel import binary
from paired_timing import measure
from serving_clocks import clocks
from sparkfuse.decode_attention import compile_decode_attention
from sparkfuse.length_attention import compile_length_decode
from token_quant_panel import digest
from provenance import capture

# None of these eight evaluation lengths is used to qualify SparkFuse.
QUAL = (17, 65)
LENGTHS = (2, 31, 32, 33, 129, 513, 1025, 2049)
CASES = ((2, 8, 2, 64, 2, "bf16"), (1, 16, 4, 128, 0, "fp16"))
SEEDS = (3031751, 3042703)


def counted_inductor(model, dynamic):
    actual = lookup_backend("inductor")
    graphs = []

    def backend(gm, xs):
        graphs.append(len(list(gm.graph.nodes)))
        return actual(gm, xs)

    return torch.compile(model, backend=backend, dynamic=dynamic, fullgraph=True), graphs


def case(cfg, seed, smoke):
    b, h, g, d, mask, dt = cfg

    def request(s, offset):
        xs = inputs((b, h, g, s, d, mask, dt, None), seed + offset)
        return xs[:3] + ((xs[3].to(xs[0].dtype),) if mask else ())

    model = CanonicalDecode().eval()
    bare = SingleDecode().eval()
    ex, au = [request(s, i * 53) for i, s in enumerate(QUAL)]
    ep = export_length_decode(model, ex)
    torch.cuda.synchronize()
    start = time.perf_counter()
    f = compile_length_decode(ep, ex, au)
    torch.cuda.synchronize()
    qual_ms = (time.perf_counter() - start) * 1000
    bins = [binary(SimpleNamespace(path=p, source=p.read_text())) for p in f.paths]
    lengths = (2, 32, 33, 129) if smoke else LENGTHS
    paths = {"eager": model, "length_cuda": f.run_bounded}
    builds = {}
    fail = {}
    counts = {}
    graphs = {}
    # These are Dynamo backend invocations, not a count of the binaries Inductor emits.
    torch._dynamo.reset()
    if not smoke:
        for key, dyn in [("inductor_dynamic", True), ("inductor_static", False)]:
            paths[key], graphs[key] = counted_inductor(model, dyn)
    requests = [
        request(s, 113 * (k + 1) + 4003 * i) for i, s in enumerate(lengths) for k in range(3)
    ]
    req_hashes = [[digest(x) for x in xs] for xs in requests]
    refs = [cpu_reference(xs, None) for xs in requests]
    expected = [model(*xs) for xs in requests]
    for name, fn in list(paths.items()):
        builds[name] = []
        for i, s in enumerate(lengths):
            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                y = fn(*requests[3 * i])
                torch.cuda.synchronize()
            except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as exc:
                assert name.startswith("inductor")
                fail[name] = dict(length=s, type=type(exc).__name__)
                del paths[name]
                break
            builds[name].append(
                dict(
                    length=s,
                    first_request_ms=1000 * (time.perf_counter() - start),
                    backend_calls=len(graphs.get(name, [])) if name in graphs else None,
                )
            )
            del y
    gates = {}
    static_parity = []
    static_bins = set()
    for name, fn in paths.items():
        ys = [fn(*xs) for xs in requests]
        checks = [
            metrics(y, r, e, xs) for y, r, e, xs in zip(ys, refs, expected, requests, strict=True)
        ]
        fresh = len({y.untyped_storage().data_ptr() for y in ys}) == len(ys)
        admitted = fresh and all(z["structure"] and z["outside_tolerance"] == 0 for z in checks)
        gates[name] = dict(admitted=admitted, all_live=fresh, outputs=checks)
        assert admitted or name.startswith("inductor"), (name, checks)
    for i, s in enumerate(lengths):
        # Separate comparator: same tile, static-S CUDA. Canonicalization only changes singleton strides.
        xs = requests[3 * i]
        stat = compile_decode_attention(
            torch.export.export(bare, xs), xs, request(s, 90013 + i), tile=f.plan.tile
        )
        static_bins.update(p.path.parent.name for p in stat.programs)
        for k in range(3):
            row = 3 * i + k
            v = digest(stat.run_bounded(*requests[row]))
            assert v == gates["length_cuda"]["outputs"][row]["sha256"]
            static_parity.append(v)
    accepted = {k: v for k, v in paths.items() if gates[k]["admitted"]}
    hot = {}
    mixed = None
    if not smoke:
        for i in (0, 3, 6, 7):
            xs = requests[3 * i + 1]
            hot[str(lengths[i])] = measure(
                {k: (lambda fn=fn: fn(*xs)) for k, fn in accepted.items()}, seed + lengths[i]
            )
        mixed = clocks(accepted, [requests[3 * i + 2] for i in range(8)], seed + 3881)
    assert req_hashes == [[digest(x) for x in xs] for xs in requests]
    counts = {k: len(v) for k, v in graphs.items()}
    return dict(
        config=cfg,
        seed=seed,
        export_graph=str(ep.graph),
        range_constraints=str(ep.range_constraints),
        qualifying_lengths=QUAL,
        held_lengths=lengths,
        requests_per_held_length=3,
        input_sha256=req_hashes,
        cpu_fp64_sha256=[digest(r) for r in refs],
        qualification_ms=qual_ms,
        binary=bins,
        static_kernel_cache_keys=sorted(static_bins),
        workspace_bytes=[f.plan.workspace_bytes(s) for s in lengths],
        build_failures=fail,
        first_seen_requests=builds,
        inductor_backend_calls_after_timing=counts,
        gates=gates,
        static_cuda_sha256=static_parity,
        hot=hot,
        mixed=mixed,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    data = dict(
        smoke=a.smoke,
        torch=str(torch.__version__),
        device=torch.cuda.get_device_name(),
        cuda=torch.version.cuda,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    with torch._dynamo.config.patch(recompile_limit=32):
        for spec in CASES[:1] if a.smoke else CASES:
            for seed in (3081931,) if a.smoke else SEEDS:
                c = case(spec, seed, a.smoke)
                data["cases"].append(c)
                print(
                    spec,
                    seed,
                    c["inductor_backend_calls_after_timing"],
                    c["mixed"]["median_us"] if c["mixed"] else "verified",
                    flush=True,
                )
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
