"""One local greedy response from the pinned checkpoint. This CLI is not a timer."""

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from model import REV, verify_assets, specialize
from projection import join, packed
from run import Runner


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument(
        "--backend", choices=("eager", "sparkfuse", "packed", "inductor"), default="eager"
    )
    p.add_argument("--replay", action="store_true")
    args = p.parse_args()
    if not args.prompt.strip() or args.max_new_tokens < 1:
        p.error("provide a nonempty prompt and positive token budget")
    verify_assets(args.weights)
    tok = AutoTokenizer.from_pretrained(
        args.weights, local_files_only=True, trust_remote_code=False
    )
    model = (
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
    ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to("cuda")
    n = ids.shape[1]
    cap = n + args.max_new_tokens
    if cap > model.config.max_position_embeddings:
        p.error("prompt plus output budget exceeds model context window")
    fn = model
    if args.backend in ("sparkfuse", "packed"):
        sf, _ = specialize(model, [1])
        fn = sf
        if args.backend == "packed":
            fn = packed(
                model,
                [join(layer.mlp) for layer in model.model.layers],
                sf.model.layers[0].mlp.paths[1],
            )
    if args.backend == "inductor":
        fn = torch.compile(
            model, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
        )
    r = Runner(fn, model.config, cap, torch.device("cuda"), torch.bfloat16)
    if args.replay:
        r.setup_replay(ids)
    r.cache.reset()
    x = ids
    out = []
    for i in range(args.max_new_tokens):
        logits = r.forward(x, r.pos[:n]) if i == 0 else r.decode(x, r.pos[n + i - 1 : n + i])
        x = logits.argmax(-1, keepdim=True)
        t = x.item()
        out.append(t)
        if t == tok.eos_token_id:
            break
    print(
        json.dumps(
            dict(
                revision=REV,
                backend=args.backend,
                decode_replay=args.replay,
                input_tokens=n,
                generated_ids=out,
                text=tok.decode(out, skip_special_tokens=True),
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
