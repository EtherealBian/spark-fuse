"""Trace explicit QKV checkpoint layouts through a real Qwen2 decoder; not an automatic GEMM rewrite."""

import argparse
from hashlib import sha256
import json
import operator
from pathlib import Path

import torch
from torch import fx, nn
from torch.nn import functional as F
import transformers
from mlp_projection_audit import compare, digest, measure
from provenance import capture
from qkv_projection_audit import capture_values
from qwen_cached_decode import suite
from qwen_packed_decode import run as layer_panel
from sparkfuse.partition import ordered_collectives, partition_export

ROOT = Path(__file__).resolve().parents[1]
NAMES = [f"layer.self_attn.{k}_proj" for k in "qkv"]


class Copies(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.widths = widths

    def forward(self, x):
        return tuple(
            y.clone(memory_format=torch.contiguous_format) for y in x.split(self.widths, -1)
        )


class Bank(nn.Module):
    """Experimental checkpoint layout. Caller explicitly selects its projection arithmetic."""

    def __init__(self, params, widths, mode):
        super().__init__()
        self.weight, self.bias = params
        self.widths = widths
        assert mode in ("separate", "views", "copies")
        self.mode = mode

    def forward(self, x):
        if self.mode == "separate":
            return tuple(
                F.linear(x, w, b)
                for w, b in zip(
                    self.weight.split(self.widths), self.bias.split(self.widths), strict=True
                )
            )
        wide = F.linear(x, self.weight, self.bias)
        v = wide.split(self.widths, -1)
        return (
            tuple(t.clone(memory_format=torch.contiguous_format) for t in v)
            if self.mode == "copies"
            else v
        )


def bank_ep(ep, bank, args):
    """Fixture-specific graph surgery. Re-export to obtain real changed strides, not stale FX metadata."""
    gm = ep.module()
    original = gm.state_dict()
    gm.add_module("sf_qkv", bank)
    nodes = []
    for n in gm.graph.nodes:
        if n.target is not torch.ops.aten.linear.default or len(n.args) != 3:
            continue
        _, w, b = n.args
        if w.op == "get_attr" and w.target in [k + ".weight" for k in NAMES]:
            assert (
                b.op == "get_attr"
                and w.target[:-6] + "bias" == b.target
                and len(b.users) == len(w.users) == 1
            )
            nodes.append(n)
    assert [n.args[1].target for n in nodes] == [k + ".weight" for k in NAMES] and len(
        {n.args[0] for n in nodes}
    ) == 1
    with gm.graph.inserting_before(nodes[0]):
        tup = gm.graph.call_module("sf_qkv", (nodes[0].args[0],))
        repl = [gm.graph.call_function(operator.getitem, (tup, i)) for i in range(3)]
    for n, t in zip(nodes, repl, strict=True):
        n.replace_all_uses_with(t)
        gm.graph.erase_node(n)
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    new = fx.GraphModule(gm, gm.graph)
    expected = set(original) - {k + "." + p for k in NAMES for p in ("weight", "bias")}
    sd = new.state_dict()
    assert set(sd) == expected | {"sf_qkv.weight", "sf_qkv.bias"} and all(
        sd[k].data_ptr() == original[k].data_ptr() for k in expected
    )
    assert sum(a.numel() for a in sd.values()) == sum(a.numel() for a in original.values())
    for k in ("weight", "bias"):
        assert sd["sf_qkv." + k].data_ptr() == getattr(bank, k).data_ptr()
    return torch.export.export(new, args)


def variants(orig, bank_data, args, verify_only, emit_dir):
    params, widths = bank_data
    ep = torch.export.export(orig, args)
    re = torch.export.export(ep.module(), args)
    paths = {"eager": orig}
    graphs = {}
    kwargs = dict(
        rotary_pairs=True,
        paired_appends=True,
        paired_head_repeats=True,
        ordered_scores="slots",
        host_backend="native",
        schedule_policy=ordered_collectives,
    )
    banked = {
        s: bank_ep(ep, Bank(params, widths, s), args) for s in ("separate", "views", "copies")
    }
    for name, e in [
        ("direct", ep),
        ("reexport", re),
        *banked.items(),
        ("one_copy_kernel", banked["copies"]),
    ]:
        g, rr = partition_export(e, split_copies=name == "one_copy_kernel", **kwargs)
        g.graph.lint()
        paths[name] = g
        hits = [r for r in rr if r.get("kind") == "split_copy3"]
        assert len(hits) == int(name == "one_copy_kernel")
        if hits and emit_dir:
            (hit,) = hits
            src = g.get_submodule(hit["module"]).program.source
            assert sha256(src.encode()).hexdigest() == hit["cuda_sha256"]
            emit_dir.mkdir(exist_ok=True, parents=True)
            (emit_dir / (hit["cuda_sha256"] + ".cu")).write_text(src)
        graphs[name] = dict(aten=str(e.graph), fx=str(g.graph), records=rr)
    if not verify_only:
        paths["inductor"] = torch.compile(
            ep.module(), backend="inductor", fullgraph=True, dynamic=False
        )
    return paths, graphs


def isolated(m, rope, params, widths, prefix, offset, verify_only):
    seeds = [offset + k for k in (60527, 60733, 60943)]
    measurements = []
    traces = [capture_values(m, rope, prefix, s) for s in seeds]
    fallback = Bank(params, widths, "separate")
    copies = Copies(widths)
    selectors = ["prefill", "decode1", "decode2"]
    for stage in range(3):
        rows = []
        paths = {}
        rec = None
        for seed, live in zip(seeds, traces, strict=True):
            x, ref, _, _ = live[stage]
            wide = F.linear(x, *params)
            parts = wide.split(widths, -1)
            sliced = fallback(x)
            copy_ref = copies(wide)
            if not paths:
                e = torch.export.export(copies, (wide,))
                g, rr = partition_export(e, split_copies=True, host_backend="native")
                (rec,) = [r for r in rr if r.get("kind") == "split_copy3"]
                paths = {
                    "three_clones": copies,
                    "fx_copy_kernel": g,
                    "native_copy_kernel": g.get_submodule(rec["module"]).dispatch,
                }
                if not verify_only:
                    paths["inductor"] = torch.compile(
                        e.module(), backend="inductor", fullgraph=True, dynamic=False
                    )
            methods = {}
            for n, fn in paths.items():
                try:
                    y = fn(wide)
                    d = [compare(a, b) for a, b in zip(y, copy_ref, strict=True)]
                    independent = len({z.untyped_storage().data_ptr() for z in (wide, *y)}) == 4
                    methods[n] = dict(
                        tensors=d,
                        independent=independent,
                        passed=independent
                        and all(
                            a["unequal_bits"] == 0
                            and a["stride_match"]
                            and a["layout"]["storage_offset"] == 0
                            for a in d
                        ),
                    )
                except Exception as exc:
                    if n != "inductor":
                        raise
                    methods[n] = dict(error=type(exc).__name__, passed=False)
            same = [compare(v, r) for v, r in zip(sliced, ref, strict=True)]
            assert (
                all(v["unequal_bits"] == 0 and v["stride_match"] for v in same)
                and len({v.untyped_storage().data_ptr() for v in (*sliced, x)}) == 4
            )
            rows.append(
                dict(
                    seed=seed,
                    x_sha256=digest(x),
                    wide_sha256=digest(wide),
                    sliced_checkpoint_against_hf=same,
                    wide_against_hf=[compare(v, r) for v, r in zip(parts, ref, strict=True)],
                    copy_backends=methods,
                )
            )
        allowed = [n for n in paths if all(r["copy_backends"][n]["passed"] for r in rows)]
        assert {"three_clones", "fx_copy_kernel", "native_copy_kernel"} <= set(allowed)
        timing = (
            measure({n: (lambda n=n: paths[n](wide)) for n in allowed}, offset + stage + prefix)
            if not verify_only and stage != 2
            else {}
        )
        measurements.append(
            dict(stage=selectors[stage], shape=list(wide.shape), rows=rows, record=rec, **timing)
        )
    return measurements


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed-offset", type=int, default=80831)
    p.add_argument("--past", type=int, nargs="+", default=[127, 1024])
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--emit-dir", type=Path)
    a = p.parse_args()
    if len(set(a.past)) != len(a.past) or min(a.past) < 2:
        p.error("distinct prefixes >=2 required")
    m, rope = suite(a.seed_offset)
    ps = [getattr(m.layer.self_attn, n + "_proj") for n in "qkv"]
    widths = tuple(p.out_features for p in ps)
    params = tuple(
        nn.Parameter(torch.cat([getattr(p, k) for p in ps]), requires_grad=False)
        for k in ("weight", "bias")
    )
    assert all(
        torch.equal(z, getattr(p, k))
        for k, t in zip(("weight", "bias"), params, strict=True)
        for z, p in zip(t.split(widths), ps, strict=True)
    )
    out = dict(
        protocol="Explicit fixed Qwen2 fixture checkpoint layouts, no global GEMM rewrite. Same-source direct/re-export controls. Five held-out two-step trajectories, own previous output cache. All exact output gates and independent storage except the predeclared tolerant Inductor baseline. Only eligible paths timed on common last-seed inputs. HF prefill supplies initial cache. Separate projection fallback and split-copy ABI also checked on three real prefill/decode inputs; copy-only timings use an already-produced wide tensor even where wide GEMM disagrees with HF.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        transformers=transformers.__version__,
        weight_seed=60313 + a.seed_offset,
        export_seed=60401 + a.seed_offset,
        layer_seeds=[s + a.seed_offset for s in (60527, 60637, 60733, 60869, 60943)],
        verify_only=a.verify_only,
        checkpoint=dict(
            widths=widths,
            packed_sha256=[digest(t) for t in params],
            source_sha256=[[digest(getattr(p, k)) for p in ps] for k in ("weight", "bias")],
            parameter_bytes=sum(t.numel() * t.element_size() for t in params),
        ),
        provenance=capture(ROOT / ".runtime/qkv-layer/sources"),
        cases=[],
    )
    for n in a.past:
        isolated_case = isolated(m, rope, params, widths, n, a.seed_offset, a.verify_only)
        c = layer_panel(
            m,
            (params, widths),
            rope,
            n,
            a.seed_offset,
            a.verify_only,
            a.emit_dir,
            variant_builder=variants,
        )
        c["isolated"] = isolated_case
        out["cases"].append(c)
        print(n, c["passed"], [v["device"] for v in c["median_us"]], flush=True)
    out["passed"] = all(c["passed"] for c in out["cases"])
    tmp = a.output.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(a.output)
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
