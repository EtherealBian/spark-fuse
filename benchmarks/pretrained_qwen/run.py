"""Warm, batch-one, full-model autoregressive requests. No weights are published."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from statistics import median
from time import perf_counter

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache
from model import MODEL, REV, WEIGHTS, specialize, verify_assets
from metrics import p95, summarize

LENGTHS = (32, 128)
STEPS = 9  # one first token, eight dependent decode tokens; fixed work even after EOS
TEXTS = (
    ("The orchard has three apple trees and two pear trees. ", "How many trees?"),
    ("Water freezes at zero degrees Celsius at standard pressure. ", "When does water freeze?"),
    ("杭州位于中国浙江省，西湖是杭州的一座湖。", "杭州在哪个省？"),
    (
        "In Python, len returns the number of items in a sequence. ",
        "What does len([1,2,3]) return?",
    ),
)


def prompt(tok, n, i):
    enc = lambda x: tok.encode(x, add_special_tokens=False)
    a = enc("<|im_start|>user\n")
    b = enc("\n" + TEXTS[i][1] + "<|im_end|>\n<|im_start|>assistant\n")
    context = enc(TEXTS[i][0] * 32)
    if not 0 < n - len(a) - len(b) <= len(context):
        raise ValueError("prompt budget")
    return torch.tensor([a + context[: n - len(a) - len(b)] + b], dtype=torch.long, device="cuda")


def sync():
    torch.cuda.synchronize()


def timed(fn):
    sync()
    t = perf_counter()
    x = fn()
    sync()
    return x, (perf_counter() - t) * 1000


class Runner:
    def __init__(self, fn, cfg, capacity, device, dtype):
        self.fn = fn
        self.cache = StaticCache(
            cfg, max_batch_size=1, max_cache_len=capacity, device=device, dtype=dtype
        )
        self.pos = torch.arange(capacity, device=device)
        self.graph = None

    def forward(self, x, p):
        return self.fn(
            input_ids=x,
            cache_position=p,
            position_ids=p.view(1, -1),
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=False,
        )[0][:, -1, :]

    def setup_replay(self, sample):
        self.cache.reset()
        logits = self.forward(sample, self.pos[: sample.shape[1]])
        self.buf_x = logits.argmax(-1, keepdim=True)
        self.buf_p = self.pos[sample.shape[1] : sample.shape[1] + 1].clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.forward(self.buf_x, self.buf_p)
        torch.cuda.current_stream().wait_stream(side)
        sync()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.buf_y = self.forward(self.buf_x, self.buf_p)
        sync()
        self.cache.reset()

    def decode(self, x, p):
        if self.graph is None:
            return self.forward(x, p)
        self.buf_x.copy_(x)
        self.buf_p.copy_(p)
        self.graph.replay()
        return self.buf_y

    def request(self, ids, *, observe=False, teacher=None):
        self.cache.reset()
        sync()
        n = ids.shape[1]
        tokens = []
        logits = []
        dur = []
        x = ids
        for step in range(STEPS):

            def advance():
                y = (
                    self.forward(x, self.pos[:n])
                    if step == 0
                    else self.decode(x, self.pos[n + step - 1 : n + step])
                )
                return y, y.argmax(-1, keepdim=True)

            (y, t), ms = timed(advance)
            dur.append(ms)
            if observe:
                logits.append(y.detach().float().cpu().flatten())
            tokens.append(t)
            x = t if teacher is None else teacher[step]
        return dict(ttft_ms=dur[0], decode_ms=dur[1:]), tokens, logits


def compare(run, ids, reference):
    ref_tokens, ref_logits = reference
    _, free, _ = run.request(ids)
    _, _, out = run.request(ids, observe=True, teacher=ref_tokens)
    actual = [int(t.item()) for t in free]
    wanted = [int(t.item()) for t in ref_tokens]
    by_step = []
    for a, b in zip(out, ref_logits, strict=True):
        if not torch.isfinite(a).all():
            raise ValueError("non-finite model output")
        diff = (a - b).abs()
        by_step.append(
            dict(
                max_abs=diff.max().item(),
                mean_abs=diff.mean().item(),
                cosine=torch.nn.functional.cosine_similarity(a[None], b[None]).item(),
            )
        )
    return dict(
        generated=actual,
        reference=wanted,
        teacher_forced_logits=by_step,
        passed=actual == wanted
        and all(x["cosine"] >= 0.999 and x["max_abs"] <= 1.0 for x in by_step),
    )


def profile(run, ids):
    from torch.profiler import profile as prof, ProfilerActivity

    run.cache.reset()
    n = ids.shape[1]
    y = run.forward(ids, run.pos[:n])
    x = y.argmax(-1, keepdim=True)
    sync()
    with prof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        run.decode(x, run.pos[n : n + 1]).argmax(-1, keepdim=True)
        sync()
    kernels = Counter()
    counts = Counter()
    for ev in p.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            kernels[ev.name] += ev.time_range.elapsed_us()
            counts[ev.name] += 1
    return dict(
        kernel_events=counts.total(),
        sum_event_us=sum(kernels.values()),
        top=[dict(name=k, events=counts[k], sum_us=v) for k, v in kernels.most_common(8)],
    )


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=12)
    args = ap.parse_args()
    if args.windows < 3 or args.windows % 3:
        raise ValueError("--windows must be a positive multiple of three, at least three")
    verify_assets(args.weights)
    torch.manual_seed(307)
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(
        args.weights, local_files_only=True, trust_remote_code=False
    )
    cfgargs = dict(
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model, load = timed(
        lambda: AutoModelForCausalLM.from_pretrained(args.weights, **cfgargs).to("cuda").eval()
    )
    assert model.config.num_hidden_layers == 24
    (sf, src), sf_ms = timed(lambda: specialize(model, (*LENGTHS, 1)))
    ind, wrap = timed(
        lambda: torch.compile(
            model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
    )
    cfg = model.config
    cap = max(LENGTHS) + STEPS
    models = {"eager": model, "sparkfuse": sf, "inductor": ind}
    mk = lambda fn: Runner(fn, cfg, cap, torch.device("cuda"), torch.bfloat16)
    runs = {k: mk(v) for k, v in models.items()}
    ids = {(n, i): prompt(tok, n, i) for n in LENGTHS for i in range(4)}
    setup = {}
    graph_setup = {}
    print("weights checked and model initialized", flush=True)
    for k, r in list(runs.items()):
        setup[k] = {}
        for n in LENGTHS:
            print("warmup", k, n, flush=True)
            times, _, _ = r.request(ids[n, 0])
            setup[k][n] = times
        g = mk(models[k])
        _, graph_setup[k] = timed(lambda: g.setup_replay(ids[LENGTHS[0], 0]))
        runs[k + "_replay"] = g
    result = dict(
        format="sparkfuse.pretrained-qwen.v1",
        model=MODEL,
        revision=REV,
        weights_sha256=WEIGHTS,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
        attention="sdpa",
        dtype="bfloat16",
        batch=1,
        layers=len(model.model.layers),
        max_cache_len=cap,
        generated_tokens=STEPS,
        sparkfuse_gate_source_sha256=src,
        setup_ms=dict(
            model_load_and_h2d=load,
            specialize=sf_ms,
            inductor_wrapper=wrap,
            first_requests=setup,
            decode_graph_capture=graph_setup,
        ),
        cases={},
    )
    result["driver_sha256"] = {
        f: sha256((Path(__file__).parent / f).read_bytes()).hexdigest()
        for f in ["run.py", "model.py", "metrics.py"]
    }
    from torch._dynamo.utils import counters

    result["inductor_graphs_after_setup"] = int(counters["stats"]["unique_graphs"])
    rng = random.Random(91871)
    for n in LENGTHS:
        print("validation and timed windows", n, flush=True)
        refs = {}
        for i in (1, 2, 3):
            _, t, y = runs["eager"].request(ids[n, i], observe=True)
            refs[i] = (t, y)
        validation = {
            k: [compare(r, ids[n, i], refs[i]) for i in (1, 2, 3)] for k, r in runs.items()
        }
        rows = []
        for w in range(args.windows):
            order = list(runs)
            rng.shuffle(order)
            i = 1 + w % 3
            row = dict(order=order, prompt=i)
            for k in order:
                row[k] = runs[k].request(ids[n, i])[0]
            rows.append(row)
        passed = lambda k: all(v["passed"] for v in validation[k])
        summary = {}
        for k in runs:
            t = [r[k]["ttft_ms"] for r in rows]
            d = [ms for r in rows for ms in r[k]["decode_ms"]]
            summary[k] = dict(
                eligible=passed(k),
                ttft_p50_ms=median(t),
                ttft_p95_ms=p95(t),
                decode_p50_ms=median(d),
                decode_p95_ms=p95(d),
            )
        pairs = [
            ("sparkfuse", "inductor"),
            ("sparkfuse_replay", "inductor_replay"),
            ("sparkfuse", "eager"),
            ("sparkfuse_replay", "eager_replay"),
            ("sparkfuse_replay", "sparkfuse"),
            ("eager_replay", "eager"),
            ("inductor_replay", "inductor"),
        ]
        result["cases"][str(n)] = dict(
            inputs=[ids[n, i].tolist()[0] for i in (1, 2, 3)],
            validation=validation,
            windows=rows,
            summary=summary,
            comparisons=[summarize(rows, a, b, passed(a) and passed(b)) for a, b in pairs],
            profile={k: profile(runs[k], ids[n, 1]) for k in models},
        )
    result["inductor_graphs_after_windows"] = int(counters["stats"]["unique_graphs"])
    result["deployment_target_met"] = all(
        all(c["target_met"] for c in result["cases"][str(n)]["comparisons"][:2]) for n in LENGTHS
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.output)
    for n, v in result["cases"].items():
        print(n, v["summary"], flush=True)
    print("target met:", result["deployment_target_met"], flush=True)


if __name__ == "__main__":
    main()
