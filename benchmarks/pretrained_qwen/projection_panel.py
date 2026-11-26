"""Matched graph replay for a two-by-two ablation on independent prompt content."""

import argparse, json, random
from hashlib import sha256
from pathlib import Path
from statistics import median
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import MODEL, REV, WEIGHTS, specialize, verify_assets
from projection import join, packed
from run import Runner, LENGTHS, STEPS, timed, compare, profile
from metrics import p95, summarize

TEXTS = (
    ("The box is red and the bag is green. ", "What color is the bag?"),
    ("A week has seven days. ", "How many days in two weeks?"),
    ("长江流经中国的多个省市。", "这条河叫什么？"),
    ("The square has four equal sides. ", "How many sides?"),
    ("Python sorts numbers in ascending order by default. ", "Sort: 3,1,2."),
)
PAIRS = (
    ("packed_fused", "two_fused"),
    ("packed_eager", "two_eager"),
    ("packed_fused", "packed_eager"),
    ("packed_fused", "inductor"),
)


def token_input(tok, n, i):
    enc = lambda x: tok.encode(x, add_special_tokens=False)
    a = enc("<|im_start|>user\n")
    b = enc("\n" + TEXTS[i][1] + "<|im_end|>\n<|im_start|>assistant\n")
    c = enc(TEXTS[i][0] * 32)
    assert 0 < n - len(a) - len(b) <= len(c)
    return torch.tensor([a + c[: n - len(a) - len(b)] + b], dtype=torch.long, device="cuda")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    verify_assets(args.weights)
    torch.manual_seed(4517)
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(
        args.weights, local_files_only=True, trust_remote_code=False
    )
    m = (
        AutoModelForCausalLM.from_pretrained(
            args.weights,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
        .requires_grad_(False)
    )
    f, ks = specialize(m, [1])
    banks, pack_ms = timed(lambda: [join(l.mlp) for l in m.model.layers])
    gate = f.model.layers[0].mlp.paths[1]
    methods = {
        "two_eager": m,
        "two_fused": f,
        "packed_eager": packed(m, banks),
        "packed_fused": packed(m, banks, gate),
        "inductor": torch.compile(
            m, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
        ),
    }
    result = dict(
        format="sparkfuse.pretrained-projections.v1",
        model=MODEL,
        revision=REV,
        weights_sha256=WEIGHTS,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        layers=len(banks),
        weight_pack_bytes=sum(x.nbytes for x in banks),
        pack_and_weight_check_ms=pack_ms,
        gate_cuda_sha256=ks,
        driver_sha256={
            f: sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
            for f in ("run.py", "model.py", "metrics.py", "projection_panel.py", "projection.py")
        },
        cases={},
    )
    mk = lambda fn: Runner(fn, m.config, max(LENGTHS) + STEPS, torch.device("cuda"), torch.bfloat16)
    paths = {k: mk(v) for k, v in methods.items()}
    ref = mk(m)
    inputs = {(n, i): token_input(tok, n, i) for n in LENGTHS for i in range(5)}
    for name, r in paths.items():
        print("prepare replay", name, flush=True)
        for n in LENGTHS:
            r.request(inputs[n, 0])
        r.setup_replay(inputs[32, 0])
    from torch._dynamo.utils import counters

    result["inductor_graphs_after_setup"] = int(counters["stats"]["unique_graphs"])
    rng = random.Random(4517)
    for n in LENGTHS:
        refs = {}
        val = {}
        print("matched replay requests", n, flush=True)
        for i in range(1, 5):
            _, ts, ys = ref.request(inputs[n, i], observe=True)
            refs[i] = (ts, ys)
        for k, r in paths.items():
            val[k] = [compare(r, inputs[n, i], refs[i]) for i in range(1, 5)]
        rows = []
        for w in range(12):
            order = list(paths)
            rng.shuffle(order)
            i = 1 + w % 4
            row = dict(order=order, prompt=i)
            for k in order:
                row[k] = paths[k].request(inputs[n, i])[0]
            rows.append(row)
        eligible = lambda k: all(q["passed"] for q in val[k])
        summary = {}
        for k in paths:
            xs = [t for r in rows for t in r[k]["decode_ms"]]
            summary[k] = dict(eligible=eligible(k), p50_ms=median(xs), p95_ms=p95(xs))
        cs = [summarize(rows, a, b, eligible(a) and eligible(b)) for a, b in PAIRS]
        result["cases"][str(n)] = dict(
            inputs=[inputs[n, i].tolist()[0] for i in range(1, 5)],
            validation=val,
            windows=rows,
            summary=summary,
            comparisons=cs,
            uncaptured_profile={
                k: profile(mk(methods[k]), inputs[n, 1]) for k in ("two_fused", "packed_fused")
            },
        )
    result["inductor_graphs_after_windows"] = int(counters["stats"]["unique_graphs"])
    result["deployment_target_met"] = all(
        c["comparisons"][-1]["target_met"] for c in result["cases"].values()
    )
    p = args.output
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    t.replace(p)
    print("deployment target:", result["deployment_target_met"])


if __name__ == "__main__":
    main()
