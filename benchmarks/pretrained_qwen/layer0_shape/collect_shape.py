"""Capture only three real positions per stage, with identical forced inputs and reverse orders."""

import argparse
import json
import sys
import time
from pathlib import Path
import torch
import transformers
from kvshape import ARMS, STAGES, need, sha, witness, official, probe
from attribution_review import source_panel, fileid, save


def orders(panel):
    keys = sorted(panel)
    for rep in (0, 1):
        for k in keys if not rep else keys[::-1]:
            for arm in ARMS if not rep else ARMS[::-1]:
                yield k, rep, arm


@torch.inference_mode()
def main():
    a = argparse.ArgumentParser()
    a.add_argument("weights", type=Path)
    a.add_argument("reference", type=Path)
    a.add_argument("capture", type=Path)
    a.add_argument("out", type=Path)
    z = a.parse_args()
    panel = source_panel(z.reference)
    h = fileid(z.capture / "index.json")
    need(
        h == "6d46f39acbf2b71387f1ea1ea00191380f74fad7b8d999969dfafa27a014ac6d"
        and sys.byteorder == "little",
        "exact parent required",
    )
    prior = json.loads((z.capture / "index.json").read_bytes())
    need(
        torch.cuda.get_device_name() == "NVIDIA GB10"
        and str(torch.__version__) == prior["meta"]["torch"]
        and torch.version.cuda == prior["meta"]["cuda"]
        and transformers.__version__ == prior["meta"]["transformers"],
        "runtime",
    )
    torch.set_num_threads(4)
    meta, lookup, table, norm, rope, w = official(
        z.weights, [t for v in panel.values() for t in v["ids"]]
    )
    meta.update(
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        transformers=transformers.__version__,
        device=torch.cuda.get_device_name(),
    )
    doc = dict(
        schema="sparkfuse.first-bank-probe.v1",
        meta=meta,
        parent_index_sha256=h,
        sources={
            n: fileid(Path(__file__).with_name(n))
            for n in ("kvshape.py", "collect_shape.py", "review_shape.py")
        },
        rows=[],
    )
    save(z.out / "index.json", doc)
    (z.out / "requests.started").touch()
    with (z.out / "samples.bf16").open("xb") as fp:
        for k, rep, arm in orders(panel):
            ids = panel[k]["ids"]
            x = table[torch.tensor([lookup[i] for i in ids], device="cuda")][None]
            v, rows = probe(x, len(ids) * 3 // 4, arm, norm, rope, w, 128)
            segments = {}
            for stage in STAGES:
                b = witness(v[stage])
                segments[stage] = dict(offset=fp.tell(), bytes=len(b), sha256=sha(b))
                fp.write(b)
            fp.flush()
            doc["rows"].append(
                dict(cell=list(k), repeat=rep, arm=arm, projected_rows=rows, stages=segments)
            )
            save(z.out / "index.json", doc)
    (z.out / "requests.finished").touch()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        q = z.out / "memory-guard.json"
        if q.is_file() and json.loads(q.read_bytes()).get("reason") == "workload-finished":
            break
        time.sleep(0.2)
    else:
        raise RuntimeError("no guard acknowledgement")


if __name__ == "__main__":
    main()
