"""Admission study, not a compiler rewrite: one versus two BF16 GEMMs for real Qwen2 MLP inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from paired_timing import measure
from provenance import capture
from qwen_cached_decode import bits, suite, trajectory
from sparkfuse.measure import device_time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("two", "packed_views", "packed_copies", "pack_per_call")
STAGES = ("prefill", "first", "second")


def layout(x):
    return dict(
        shape=list(x.shape),
        stride=list(x.stride()),
        dtype=str(x.dtype),
        storage_offset=x.storage_offset(),
    )


def digest(x):
    return sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def compare(x, y):
    assert x.shape == y.shape and x.dtype == y.dtype == torch.bfloat16
    mismatch = bits(x) != bits(y)
    n = int(mismatch.sum())
    r = dict(unequal_bits=n, layout=layout(x), stride_match=x.stride() == y.stride())
    if n:
        coord = tuple(int(i) for i in mismatch.nonzero()[0])
        r.update(
            first_index=coord,
            actual_bf16=int(bits(x)[coord]) & 65535,
            reference_bf16=int(bits(y)[coord]) & 65535,
        )
    return r


def captures(m, r, prefix, seed):
    data = {k: [] for k in ("x", "up_x", "gate", "up", "product", "down")}
    hooks = []
    mlp = m.layer.mlp

    def listen(value, args, inp, out):
        if inp:
            data[inp].append(args[0])
        if out:
            data[out].append(value)

    def tap(inp=None, out=None):
        return lambda module, args, value: listen(value, args, inp, out)

    try:
        for layer, inp, out in [
            (mlp.gate_proj, "x", "gate"),
            (mlp.up_proj, "up_x", "up"),
            (mlp.down_proj, "product", "down"),
        ]:
            hooks.append(layer.register_forward_hook(tap(inp, out)))
        trajectory(m, r, prefix, seed)
    finally:
        for h in hooks:
            h.remove()
    assert all(len(a) == 3 for a in data.values()) and all(
        a is b for a, b in zip(data["x"], data["up_x"], strict=True)
    )
    return [
        (x, (g, u, a, d))
        for x, g, u, a, d in zip(
            *(data[k] for k in ("x", "gate", "up", "product", "down")), strict=True
        )
    ]


def projection(name, x, weights, packed, down):
    if name == "two":
        a, b = (F.linear(x, w) for w in weights)
    else:
        w = torch.cat(weights) if name == "pack_per_call" else packed
        a, b = F.linear(x, w).chunk(2, -1)
        if name == "packed_copies":
            a, b = a.contiguous(), b.contiguous()
    product = F.silu(a) * b
    return a, b, product, F.linear(product, down)


@torch.inference_mode()
def run(offset, prefixes):
    model, rope = suite(offset)
    mlp = model.layer.mlp
    weights = (mlp.gate_proj.weight, mlp.up_proj.weight)
    down = mlp.down_proj.weight
    packed = torch.cat(weights)
    digest0 = [digest(w) for w in (*weights, down, packed)]
    assert all(torch.equal(v, w) for v, w in zip(packed.chunk(2, 0), weights, strict=True))
    record = dict(
        weight_seed=60313 + offset,
        weight_sha256=digest0,
        weight_layouts=[layout(w) for w in (*weights, down, packed)],
        extra_packed_bytes=packed.numel() * packed.element_size(),
        packing=device_time(lambda: torch.cat(weights), groups=7, unroll=2, replay=3),
        cases=[],
    )
    for prefix in prefixes:
        by_step = [[] for _ in STAGES]
        seeds = [k + offset for k in (60527, 60733, 60943)]
        runs = {s: captures(model, rope, prefix, s) for s in seeds}
        for seed, live in runs.items():
            for i, (x, ref) in enumerate(live):
                row = dict(
                    seed=seed,
                    input=layout(x),
                    input_sha256=digest(x),
                    reference=[layout(v) for v in ref],
                    methods={},
                )
                for name in NAMES:
                    out = projection(name, x, weights, packed, down)
                    diff = [compare(a, b) for a, b in zip(out, ref, strict=True)]
                    row["methods"][name] = dict(
                        tensors=diff,
                        shared_gate_up_storage=out[0].untyped_storage().data_ptr()
                        == out[1].untyped_storage().data_ptr(),
                        private_intermediates_exact=all(t["unequal_bits"] == 0 for t in diff)
                        and all(t["stride_match"] for t in diff[-2:]),
                    )
                assert row["methods"]["two"]["private_intermediates_exact"]
                by_step[i].append(row)
        for i, (x, ref) in enumerate(runs[seeds[-1]]):
            checked = by_step[i]
            allowed = [
                n
                for n in NAMES
                if all(s["methods"][n]["private_intermediates_exact"] for s in checked)
            ]
            paths = {n: (lambda n=n: projection(n, x, weights, packed, down)[-1]) for n in allowed}
            paths["hf"] = lambda: mlp(x)

            def inductor_call():
                return compiled(x)

            # Compile/validate the timed down-only callable, never time a numerical rejection.
            compiled = torch.compile(mlp, backend="inductor", fullgraph=True, dynamic=False)
            ind = []
            for s in seeds:
                v, target = runs[s][i]
                try:
                    ind.append(dict(seed=s, **compare(compiled(v), target[-1])))
                except (RuntimeError, torch._dynamo.exc.TorchDynamoException) as e:
                    ind.append(dict(seed=s, error=type(e).__name__))
            if all(t.get("unequal_bits") == 0 and t["stride_match"] for t in ind):
                paths["inductor"] = inductor_call
            assert all(compare(f(), ref[-1])["unequal_bits"] == 0 for f in paths.values())
            q = measure(paths, offset + prefix + i)
            record["cases"].append(
                dict(
                    prefix=prefix,
                    stage=STAGES[i],
                    samples=checked,
                    timing_seed=seeds[-1],
                    inductor_down=ind,
                    **q,
                )
            )
            print(offset, prefix, STAGES[i], allowed, q["median_us"]["device"], flush=True)
    assert digest0 == [digest(w) for w in (*weights, down, packed)]
    changed_gate = weights[0].clone()
    changed_gate[0].zero_()
    new = (changed_gate, weights[1])
    reference = projection("two", x, new, None, down)
    cold = projection("packed_views", x, new, packed, down)
    fresh = projection("pack_per_call", x, new, None, down)
    record["replacement_control"] = dict(
        tensor="gate row zeroed in independent replacement",
        snapshot_gate=compare(cold[0], reference[0]),
        repacked_gate=compare(fresh[0], reference[0]),
    )
    assert (
        record["replacement_control"]["snapshot_gate"]["unequal_bits"] > 0
        and record["replacement_control"]["repacked_gate"]["unequal_bits"] == 0
    )
    return record


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    d = dict(
        protocol="MLP-only admission experiment, not an installed compiler pass. Hook actual BF16 Qwen2 gate/up inputs and four actual reference tensors (gate, up, SiLU product and down output) from stateful prefill and two decoding steps. A prebuilt packed tensor is an immutable experimental snapshot; per-call pack is the live-weight control. Candidate views need not preserve internal projection layout, but product and down must match both bits and stride. Inductor admission checks visible down, not internal projection bits. Timings return only down, no hooks or reference-trajectory construction during timing. No weight-bandwidth or throughput extrapolation.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        provenance=capture(ROOT / ".runtime/mlp-packing/sources"),
        panels=[],
    )
    for offset in (80831, 86783):
        d["panels"].append(run(offset, (31, 127)))
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
