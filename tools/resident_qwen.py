"""Fixed-shape, multi-request Qwen2 inference: fresh results, shuffled serving clocks, no outer graph."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path

import torch
import transformers
from packed_mean_panel import digest
from provenance import capture
from serving_clocks import build, clocks
from qwen_packed_mean import Qwen2RotaryEmbedding, compile_modes, make_model, whole_check
from sparkfuse import Schedule
from sparkfuse.contextual import _alias_pattern, _layout
from sparkfuse.packed_mean import match
from sparkfuse.partition import ordered_collectives, partition_export
from sparkfuse.resident import capture_export

ROOT = Path(__file__).resolve().parents[1]


def case(seq, weight, frozen, smoke):
    torch.compiler.reset()
    seed = weight + seq * 499
    model, make = make_model(weight, seq)
    rotary = Qwen2RotaryEmbedding(model.layer.self_attn.config, device="cuda")

    def request(i):
        x, _, _, mask = make(seed + i * 173)
        pos = torch.arange(seq, device="cuda")[None, :] + torch.tensor(
            [[i * 3], [13 + i * 5]], device="cuda"
        )
        c, s = rotary(x, pos)
        col = 1 + i % 13
        mask[:, :, col + 1 :, col] = float("-inf")
        return x, c.contiguous(), s.contiguous(), mask

    ex, audit = request(0), request(1)
    reqs = [request(i) for i in range(2, 10)]
    ep = torch.export.export(model, ex)
    at = ep.module()
    plan = Schedule(**frozen["kernels"]["packed"]["schedule"])
    opts = dict(
        host_backend="native",
        rotary_pairs=True,
        schedule_policy=lambda p: plan if match(p) else ordered_collectives(p),
    )
    (gm, recs), partition_cost = build(lambda: partition_export(ep, **opts))
    assert sum(r.get("schedule") == vars(plan) for r in recs) == 1
    prep = {"mixed_partition": partition_cost}
    paths = {"eager": model, "aten": at, "mixed": gm}
    resident = {}
    with ExitStack() as ctx:
        for kind in ("aten", "mixed"):
            name = "resident_" + kind
            exe, prep[name] = build(
                lambda: capture_export(
                    ep, ex, audit, backend=kind, partition_options=opts if kind == "mixed" else None
                )
            )
            ctx.enter_context(exe)
            paths[name] = exe
            resident[name] = exe.metadata
        assert resident["resident_mixed"]["regions"] == recs
        if not smoke:
            paths.update(compile_modes(model))
        want = [model(*r) for r in reqs]
        inp_hashes = [[digest(x) for x in r] for r in reqs]
        assert all(len({h[i] for h in inp_hashes}) == 8 for i in range(4))
        assert len({x.untyped_storage().data_ptr() for r in reqs for x in r}) == 32
        gates = {}
        for name, f in paths.items():
            # Keep all prior results alive across the complete eight-request burst.
            out = [f(*r) for r in reqs]
            checks = []
            independent = len({x.untyped_storage().data_ptr() for y in out for x in y}) == 16
            for y, ref, r in zip(out, want, reqs, strict=True):
                q = whole_check(y, ref, r[-1])
                q["layout_match"] = [_layout(x) for x in y] == [_layout(x) for x in ref]
                q["alias_match"] = _alias_pattern(r, y) == _alias_pattern(r, ref)
                checks.append(q)
            gates[name] = dict(
                distinct_outputs=independent,
                checks=checks,
                passed=independent
                and all(c["passed"] and c["layout_match"] and c["alias_match"] for c in checks),
            )
            del out
        admitted = {n: f for n, f in paths.items() if gates[n]["passed"]}
        assert {"eager", "aten", "mixed", "resident_aten", "resident_mixed"} <= admitted.keys()
        timing = None if smoke else clocks(admitted, reqs, seed + 1009)
        if timing:
            print(
                seq,
                weight,
                {n: round(v["rotating_sync_us"], 2) for n, v in timing["median_us"].items()},
                flush=True,
            )
        return dict(
            sequence=seq,
            weight_seed=weight,
            request_base=seed,
            export_sha256=sha256(str(ep.graph).encode()).hexdigest(),
            weight_sha256={n: digest(t) for n, t in model.named_parameters()},
            schedule=vars(plan),
            input_sha256=inp_hashes,
            reference_sha256=[[digest(x) for x in y] for y in want],
            input_layouts=[_layout(x) for x in ex],
            setup=prep,
            resident=resident,
            gates=gates,
            timing=timing,
        )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    source = ROOT / "reports/qwen-packed-mean-heldout-gb10.json"
    frozen = {c["sequence"]: c for c in json.loads(source.read_bytes())["cases"]}
    res = dict(
        protocol="Single BF16 Qwen2 decoder layer, batch 2. Random model weights; all four inputs change across eight distinct-address requests; no autoregressive generation. Frozen RMS schedule per length. Capture includes private weight snapshot, whole request staging, fresh outputs, two-request qualification. Only bitwise/layout/alias-correct paths on eight retained outputs are timed. Seven seeded shuffled method orders. Fixed and rotating single requests: 16 wall samples per window, CPU submit and synchronized response. Queued: five bursts of eight separate rotating requests, all outputs held through sync, wall us per forward. No outer CUDA graph, not token throughput. Allocator changes during setup are PyTorch counters, not total process memory. Setup uses warm artifact caches; mixed capture includes another partition.",
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(),
        transformers=transformers.__version__,
        rms_report_sha256=sha256(source.read_bytes()).hexdigest(),
        smoke=a.smoke,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for seq in [33] if a.smoke else [33, 65, 129]:
        for weight in [811639] if a.smoke else [813157, 821641]:
            res["cases"].append(case(seq, weight, frozen[seq], a.smoke))
    tmp = a.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    tmp.replace(a.output)


if __name__ == "__main__":
    main()
