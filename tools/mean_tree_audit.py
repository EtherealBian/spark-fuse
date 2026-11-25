"""Trace a rounding-boundary failure, then time the actual untapped exported graph separately."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median
import tempfile
import torch
from torch.profiler import profile, ProfilerActivity
from examples.graphs import case
from provenance import capture
from sparkfuse import Schedule, compile_export
from sparkfuse.measure import compare, device_time, host_time
from sparkfuse.tuning import backend_id


class Boundaries(torch.nn.Module):
    def forward(self, x, skip, weight):
        r = x + skip
        f = r.float()
        mu = f.square().mean(-1, keepdim=True)
        inv = torch.rsqrt(mu + 1e-6)
        normalized = f * inv
        low = normalized.to(x.dtype)
        return mu, inv, normalized, low, weight * low, r


def bits(x):
    return x.view(torch.int32 if x.element_size() == 4 else torch.int16)


def compare_bits(out, ref):
    if len(out) != len(ref):
        raise ValueError("output arity")
    result = []
    for y, x in zip(out, ref):
        if y.dtype != x.dtype or y.shape != x.shape:
            raise ValueError("output metadata")
        result.append(
            dict(
                elements=x.numel(),
                bit_mismatches=int((bits(x) != bits(y)).sum()),
                max_abs=float((x.float() - y.float()).abs().max()),
            )
        )
    return result


@torch.inference_mode()
def audit():
    torch.manual_seed(13759)
    root = Path(__file__).resolve().parents[1]
    result = dict(
        compiler=backend_id(),
        provenance=capture(root / ".runtime/mean-tree-audit/sources"),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        torch_git_revision=torch.version.git_version,
        installed_reduce_header_sha256=sha256(
            (Path(torch.__file__).parent / "include/ATen/native/cuda/Reduce.cuh").read_bytes()
        ).hexdigest(),
        device=torch.cuda.get_device_name(),
        scope="Dense BF16/FP16 sources, one explicit FP32 mean of a materialized squared residual. Boundary observations and timing of the actual untapped graph are separate.",
        boundaries=["mean", "inv", "normalized_fp32", "normalized_low", "weighted", "residual"],
        cases=[],
    )
    base = Schedule(128, "block", True, 1)
    oracle = Boundaries()
    rng = random.Random(19341)
    for rows, width, dt in [
        (48, 8960, torch.bfloat16),
        (17, 8160, torch.float16),
        (17, 8164, torch.float16),
        (32, 1536, torch.bfloat16),
    ]:
        m, a = case("decomp", rows, width, dt, 1752)
        trace_ep = torch.export.export(oracle, a)
        ep = torch.export.export(m, a)
        traced = {
            key: compile_export(trace_ep, schedule=s)
            for key, s in [("old", base), ("ordered", None)]
        }
        native = compile_export(ep)
        ordinary = compile_export(ep, schedule=base)
        block = dict(
            rows=rows,
            width=width,
            dtype=str(dt),
            fx=str(ep.graph),
            schedule=vars(native.schedule),
            kernel_id=native.path.parent.name,
            old_kernel_id=ordinary.path.parent.name,
            requests=[],
        )
        for si, seed in enumerate([1752, 1753, 3752, 59107, 71491, 85221]):
            _, v = case("decomp", rows, width, dt, seed)
            if si >= 3:
                v = tuple(
                    torch.cat([x.new_zeros(i), x.flatten()])[i:].view_as(x)
                    for i, x in enumerate(v, 1)
                )
            ref = oracle(*v)
            assert torch.equal(bits(ref[4]), bits(m(*v)))
            got = {k: f(*v) for k, f in traced.items()}
            stats = {k: compare_bits(out, ref) for k, out in got.items()}
            assert all(x["bit_mismatches"] == 0 for x in stats["ordered"])
            req = dict(seed=seed, storage_offsets=[x.storage_offset() for x in v], boundaries=stats)
            assert torch.equal(bits(native(*v)), bits(ref[4])) and torch.equal(
                bits(ordinary(*v)), bits(got["old"][4])
            )
            if (rows, width, seed) == (48, 8960, 1753):

                def at(values):
                    return [float(x[25, 3533 if x.size(-1) > 1 else 0]) for x in values]

                req["witness"] = dict(
                    index=[25, 3533], eager=at(ref), old=at(got["old"]), ordered=at(got["ordered"])
                )
            block["requests"].append(req)
        # Inspect this build's real eager mean kernel topology without timing under the profiler.
        sq = (v[0] + v[1]).float().square()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            sq.mean(-1, keepdim=True)
            torch.cuda.synchronize()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trace.json"
            prof.export_chrome_trace(str(path))
            block["eager_mean_kernels"] = [
                dict(name=e["name"], grid=e["args"]["grid"], block=e["args"]["block"])
                for e in json.loads(path.read_text())["traceEvents"]
                if e.get("cat") == "kernel"
            ]
        # Seventh request, distinct from all numerical witnesses, normal exported output only.
        _, v = case("decomp", rows, width, dt, 91853)
        ref = m(*v)
        ind = torch.compile(m, fullgraph=True, dynamic=False, backend="inductor")
        funcs = {
            "eager": lambda: m(*v),
            "inductor": lambda: ind(*v),
            "old": lambda: ordinary(*v),
            "ordered": lambda: native(*v),
        }
        prepared = {**funcs, "old": ordinary.prepare(*v), "ordered": native.prepare(*v)}
        checked = {}
        times = {}
        seed = 91853
        for name, f in funcs.items():
            y = f()
            quality = compare_bits([y], [ref])
            try:
                err = compare(y, ref)
                checked[name] = dict(passed=True, errors=err, bitwise=quality)
            except AssertionError as e:
                checked[name] = dict(passed=False, diagnostic=str(e), bitwise=quality)
        assert (
            checked["ordered"]["passed"] and checked["ordered"]["bitwise"][0]["bit_mismatches"] == 0
        )
        for _ in range(7):
            order = [k for k in funcs if checked[k]["passed"]]
            rng.shuffle(order)
            for k in order:
                times.setdefault(k, []).append(
                    dict(
                        device=device_time(prepared[k], groups=7, unroll=8, replay=4),
                        host=host_time(funcs[k], 15),
                    )
                )
            block.setdefault("orderings", []).append(order)
        block.update(
            timing_seed=seed,
            checks=checked,
            times=times,
            median_us={
                k: {
                    clock: median(w[clock]["median_us"] for w in ds) for clock in ("device", "host")
                }
                for k, ds in times.items()
            },
        )
        print(rows, width, block["median_us"], flush=True)
        result["cases"].append(block)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    report = audit()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
