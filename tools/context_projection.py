"""Site selection in a mixed residual RMSNorm + dense projection, with a visible residual."""

import argparse
import json
from pathlib import Path

import torch
from paired_timing import measure
from packed_mean_panel import checks, good
from provenance import capture
from sparkfuse import Schedule
from sparkfuse.contextual import replay_site, select_site, _value
from sparkfuse.partition import ordered_collectives, partition_export


class Project(torch.nn.Module):
    def __init__(self, h, dt, gen):
        super().__init__()
        self.register_buffer("scale", torch.randn(h, device="cuda", dtype=dt, generator=gen))
        self.register_buffer(
            "projection", torch.randn(64, h, device="cuda", dtype=dt, generator=gen) / h**0.5
        )

    def forward(self, x, skip):
        r = x + skip
        f = r.float()
        norm = (f * (f.square().mean(-1, keepdim=True) + 2**-12).rsqrt()).to(x.dtype) * self.scale
        return r, torch.nn.functional.linear(norm, self.projection)


def case(n, h, dt, offset, seed):
    torch.compiler.reset()
    model = Project(h, dt, torch.Generator(device="cuda").manual_seed(seed))
    opts = dict(host_backend="native", schedule_policy=ordered_collectives)

    def make(i):
        gen = torch.Generator(device="cuda").manual_seed(seed + i)
        return tuple(
            torch.randn(n * h + offset, device="cuda", dtype=dt, generator=gen)[offset:].reshape(
                n, h
            )
            for _ in range(2)
        )

    ep = torch.export.export(model, make(0))
    anchor, recs = partition_export(ep, **opts)
    (hit,) = [r for r in recs if r.get("module")]
    assert len(hit["live_outs"]) == 2 and "aten.linear.default" in str(anchor.graph)
    sched = [Schedule(t, "mean_packed_tile", True, v) for t, v in ((128, 4), (256, 4), (512, 8))]
    req = [make(i) for i in (13, 37, 83, 151, 211)]
    _, receipt = select_site(
        ep, hit["module"], sched, *req[:3], seed=seed + 313, partition_options=opts
    )
    paths = {
        "eager": model,
        "incumbent": anchor,
        "inductor": torch.compile(model, fullgraph=True, dynamic=False),
        "inductor_casts": torch.compile(
            model, fullgraph=True, dynamic=False, options={"emulate_precision_casts": True}
        ),
    }
    trials = []
    for a in req[3:]:
        gm, replay = replay_site(ep, receipt, a, partition_options=opts)
        paths["selected"] = gm
        ref = model(*a)
        out = {name: checks(fn(*a), ref) for name, fn in paths.items()}
        assert all(good(out[name]) for name in ("eager", "incumbent", "selected"))
        trials.append(
            dict(
                replay=replay,
                values=out,
                input_hashes=[_value(x) for x in a],
                output_hashes=[_value(x) for x in ref],
            )
        )
    allowed = [name for name in paths if all(good(t["values"][name]) for t in trials)]
    tm = measure({name: (lambda name=name: paths[name](*a)) for name in allowed}, seed + 557)
    print(n, h, str(dt), offset, receipt["decision"], tm["median_us"], flush=True)
    return dict(
        shape=[n, h],
        dtype=str(dt),
        offset=offset,
        seed=seed,
        region=hit,
        receipt=receipt,
        trials=trials,
        allowed=allowed,
        whole=tm,
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    res = dict(
        protocol="Synthetic mixed graph: exposed narrow residual + weighted RMSNorm consumed by a 64-feature ATen linear. Ordinary eager, default Inductor, cast-aware Inductor and native partition. Correctness required on both fresh replay requests before timing. Threshold identical to Qwen2. Not a complete trained decoder or a token-throughput benchmark.",
        device=torch.cuda.get_device_name(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        provenance=capture(a.output.parent / "sources"),
        cases=[],
    )
    for n, h, dt in ((128, 2560, torch.bfloat16), (63, 8160, torch.float16)):
        for off in (0, 1):
            res["cases"].append(case(n, h, dt, off, 613007 + n * 991 + off * 313))
    a.output.write_text(json.dumps(res, allow_nan=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
