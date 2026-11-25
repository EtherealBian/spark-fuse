"""Same-export ablation on real Phi-3: Q/K slice rotation and functional cache append.

Both compiled variants use sliced rotation on prefill. The only extra switch on
cached calls is sliced_cache. Each method carries its own cache across two updates.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import nn
import torch._inductor.config as inductor_config
import transformers
from mlp_projection_audit import compare, digest, measure
from phi3_native import Epilogue, bits, check_layer, ptr, reference, suite
from provenance import capture
from sparkfuse.partition import ordered_collectives, partition_export


COMPARATORS = {"inductor", "inductor_casts"}


class ExposedAppend(nn.Module):
    def __init__(self):
        super().__init__()
        self.rope = Epilogue()

    def forward(self, wide, cos, sin, pk, pv):
        q, k, v = self.rope(wide, cos, sin)
        return q, torch.cat((pk, k), 2), torch.cat((pv, v), 2), v


def paths(mod, args, *, cached, layer, verify, emit):
    ep = torch.export.export(mod, args)
    funcs = {"eager": mod}
    graphs = {}
    kw = (
        dict(
            ordered_scores="slots",
            schedule_policy=ordered_collectives,
            paired_appends=True,
            paired_head_repeats=True,
            packed_gates=True,
            rotary_pairs=True,
        )
        if layer
        else dict(paired_appends=True)
    )
    for n in ("sliced", "cache"):
        gm, rs = partition_export(
            ep, host_backend="native", sliced_rotary=True, sliced_cache=n == "cache", **kw
        )
        gm.graph.lint()
        funcs[n] = gm
        is_new = cached and n == "cache"
        assert sum(r.get("kind") == "sliced_rotary_pair" for r in rs) == int(not is_new)
        hits = [r for r in rs if r.get("kind") == "sliced_cache_update"]
        assert len(hits) == int(is_new)
        if hits:
            (hit,) = hits
            src = gm.get_submodule(hit["module"]).program.source.encode()
            key = sha256(src).hexdigest()
            assert key == hit["cuda_sha256"]
            if emit:
                emit.mkdir(parents=True, exist_ok=True)
                (emit / (key + ".cu")).write_bytes(src)
        graphs[n] = dict(fx=str(gm.graph), records=rs)
    if not verify:
        funcs["inductor"] = torch.compile(
            ep.module(), backend="inductor", fullgraph=True, dynamic=False
        )
        funcs["inductor_casts"] = torch.compile(
            ep.module(),
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"emulate_precision_casts": True},
        )
    return funcs, dict(aten=str(ep.graph), variants=graphs)


def iso_check(fn, args, ref, before):
    out = fn(*args)
    metrics = [compare(x, z) for x, z in zip(out, ref, strict=True)]
    meta = [(x.storage_offset(), x.untyped_storage().nbytes()) for x in out]
    expected = [(x.storage_offset(), x.untyped_storage().nbytes()) for x in ref]
    fresh = len({ptr(x) for x in out[:3]}) == 3 and not {ptr(x) for x in out[:3]} & {
        ptr(x) for x in args
    }
    alias = ptr(out[3]) == ptr(args[0])
    unchanged = [digest(x) for x in args] == before
    prefix = all(
        torch.equal(bits(x[:, :, : z.shape[2]]), bits(z))
        for x, z in zip(out[1:3], args[-2:], strict=True)
    )
    good = (
        fresh
        and alias
        and unchanged
        and prefix
        and meta == expected
        and all(x["unequal_bits"] == 0 and x["stride_match"] for x in metrics)
    )
    return dict(
        tensors=metrics,
        offset_and_span=meta,
        expected_offset_and_span=expected,
        fresh=fresh,
        raw_v_alias=alias,
        input_unchanged=unchanged,
        prefix_exact=prefix,
        passed=good,
    )


def run(m, rope, rd, n, chunks, offset, verify, emit):
    # More than one shape is deliberately tested. Reset Dynamo's frame cache between
    # independent cases, outside all correctness and timing windows.
    if not verify:
        torch.compiler.reset()
    orig, wide = reference(m, rope, n, offset + 62131, chunks)
    iso = ExposedAppend()
    stages = []
    graphs = []
    for i, ((a, _), w) in enumerate(zip(orig, wide, strict=True)):
        fs, gr = paths(m, a, cached=i > 0, layer=True, verify=verify, emit=emit)
        ef = {}
        ig = None
        if i:
            ef, ig = paths(
                iso, (w, *a[1:3], *a[4:]), cached=True, layer=False, verify=verify, emit=emit
            )
        stages.append((fs, ef))
        graphs.append(dict(layer=gr, isolated=ig))
    seeds = [offset + v for v in (62201, 62273, 62347, 62459, 62539)]
    panel = {name: [] for name in stages[0][0]}
    iso_rows = [[], []]
    for j, seed in enumerate(seeds):
        real, ww = reference(m, rope, n, seed, chunks)
        for name, trajectories in panel.items():
            checks = []
            carry = ()
            for (a, ref), (fs, _) in zip(real, stages, strict=True):
                args = (*a[:4], *carry)
                saved = tuple(z.clone() for z in carry)
                try:
                    y = fs[name](*args)
                    checks.append(check_layer(y, ref, args, saved, name not in COMPARATORS))
                    carry = y[-2:]
                except Exception as e:
                    if name not in COMPARATORS:
                        raise
                    checks.append(dict(error=type(e).__name__, passed=False))
                    break
            trajectories.append(
                dict(
                    seed=seed,
                    checks=checks,
                    passed=len(checks) == 3 and all(z["passed"] for z in checks),
                )
            )
        if j % 2 == 0:
            for i in (1, 2):
                a, res = real[i]
                args = (ww[i], *a[1:3], *a[4:])
                ref = iso(*args)
                before = [digest(z) for z in args]
                methods = {}
                oracle = [compare(z, h) for z, h in zip(ref[1:3], res[-2:], strict=True)]
                assert all(z["unequal_bits"] == 0 and z["stride_match"] for z in oracle)
                for name, fn in stages[i][1].items():
                    try:
                        methods[name] = iso_check(fn, args, ref, before)
                    except Exception as e:
                        if name not in COMPARATORS:
                            raise
                        methods[name] = dict(error=type(e).__name__, passed=False)
                iso_rows[i - 1].append(
                    dict(
                        seed=seed,
                        inputs_sha256=before,
                        whole_hf_cache_oracle=oracle,
                        methods=methods,
                    )
                )
    allowed = [name for name, v in panel.items() if all(z["passed"] for z in v)]
    updates = []
    for i in (1, 2):
        a, _ = real[i]
        args = (ww[i], *a[1:3], *a[4:])
        rows = iso_rows[i - 1]
        ok = [n for n in stages[i][1] if all(row["methods"][n]["passed"] for row in rows)]
        fs, ef = stages[i]
        old = a[4].shape[2]
        q = a[0].shape[1]
        mask = a[3].isneginf()
        ar = torch.arange(q, device="cuda")[:, None]
        future = torch.arange(old + q, device="cuda")[None, :] >= (old + ar + 1)
        assert torch.all(mask[:, :, future]) and torch.equal(
            mask[0, 0, :, 1], torch.ones(q, device="cuda", dtype=torch.bool)
        )
        mt = dict(layer={}, isolated={})
        if not verify:
            mt["layer"] = measure(
                {k: (lambda k=k: fs[k](*a)) for k in allowed}, offset + rd + n + i + q
            )
            mt["isolated"] = measure(
                {k: (lambda k=k: ef[k](*args)) for k in ok}, offset + rd + n + i + q + 1
            )
        updates.append(
            dict(
                stage=i,
                old_tokens=old,
                new_tokens=q,
                masked_entries=int(mask.sum()),
                future_masked_per_batch=int(future.sum()),
                timing_seed=None if verify else seeds[-1],
                layer_inputs_sha256=[digest(z) for z in a],
                history_layout=[
                    dict(shape=list(z.shape), stride=list(z.stride()), offset=z.storage_offset())
                    for z in a[4:]
                ],
                isolated_rows=rows,
                admitted_isolated=ok,
                timing=mt,
            )
        )
    required = {"eager", "cache", "sliced"}
    passed = required.issubset(allowed) and all(
        required.issubset(z["admitted_isolated"]) for z in updates
    )
    print(
        rd,
        n,
        chunks,
        passed,
        allowed,
        [z["timing"]["layer"].get("median_us", {}).get("device") for z in updates],
        flush=True,
    )
    return dict(
        rotary_dim=rd,
        prefix=n,
        chunks=chunks,
        seeds=seeds,
        trajectories=panel,
        admitted_layer=allowed,
        graphs=graphs,
        updates=updates,
        passed=passed,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed-offset", type=int, default=145091)
    p.add_argument("--past", type=int, nargs="+", default=[127, 1024])
    p.add_argument("--rotary-dim", type=int, nargs="+", default=[64, 128])
    p.add_argument("--chunks", nargs="+", default=["1,1", "4,3"])
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--emit-dir", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        chunks = [tuple(map(int, s.split(","))) for s in a.chunks]
    except ValueError:
        p.error("chunks must be comma-separated positive integers")
    if (
        any(len(z) != 2 or min(z) < 1 for z in chunks)
        or any(n < 2 for n in a.past)
        or any(v < 2 or v > 128 or v % 2 for v in a.rotary_dim)
    ):
        p.error("need two positive chunk sizes, prefix >=2, even rotary width in [2,128]")
    result = dict(
        protocol="Unmodified HF Phi3DecoderLayer. Own prefill and successive caches per path. Exact BF16 and output metadata for both hybrids/eager; existing tolerant gate for both whole-layer Inductor variants, exact outputs/aliases for all isolated paths. Five unseen trajectories, three isolated witnesses per shape. Shared last-seed inputs, seven shuffled CUDA Graph and host timing windows only after admission. Prefill verified but not timed here. No pretrained model throughput claim.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        comparator_options={"inductor": {}, "inductor_casts": {"emulate_precision_casts": True}},
        default_emulate_precision_casts=bool(inductor_config.emulate_precision_casts),
        transformers=transformers.__version__,
        weight_seed=60313 + a.seed_offset,
        export_seed=62131 + a.seed_offset,
        weights={},
        verify_only=a.verify_only,
        provenance=capture(Path(__file__).resolve().parents[1] / ".runtime/phi3-cache/sources"),
        cases=[],
    )
    for rd in a.rotary_dim:
        model, rope = suite(a.seed_offset, rd)
        result["weights"][rd] = {k: digest(v) for k, v in model.named_parameters()}
        for n in a.past:
            for chunk in chunks:
                result["cases"].append(
                    run(model, rope, rd, n, chunk, a.seed_offset, a.verify_only, a.emit_dir)
                )
    result["passed"] = all(c["passed"] for c in result["cases"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n")
    tmp.replace(a.output)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
