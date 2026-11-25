"""Address-locality experiment: paired, alternating measurements for hot and rotating inputs/outputs."""

import argparse, json, math, random
from pathlib import Path
import torch
from torch.utils._pytree import tree_flatten
from examples.graphs import case
from sparkfuse import compile_export, Schedule
from sparkfuse.measure import compare, distribution
from sparkfuse.native import native_dispatch


def tensor_bytes(t):
    return t.numel() * t.element_size()


def capture(fn, pool, mode):
    out_live = []
    seen = {}
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for slot in range(len(pool)):
            ret = fn(*(pool[0] if mode == "hot" else pool[slot]))
            xs, _ = tree_flatten(ret)
            for x in xs:
                seen[x.data_ptr()] = tensor_bytes(x)
            if mode == "rotating":
                out_live.append(ret)
    if mode == "hot":
        out_live.append(ret)
    return (
        g,
        out_live,
        dict(unique_output_count=len(seen), unique_output_logical_bytes=sum(seen.values())),
    )


@torch.inference_mode()
def bench(name, rows, width, dtype, seed=550):
    m, a = case(name, rows, width, dtype, seed)
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    activ = sum(tensor_bytes(x) for x in a if x.ndim == 2)
    slots = max(4, math.ceil(3 * l2 / activ))
    if slots > 128 or slots * activ > 512 * 1024**2:
        raise ValueError("choose workload allowing 3xL2 within 128 slots / 512 MiB input budget")
    # The norm weight is model state and is intentionally shared across cases.
    pool = [a]
    for i in range(1, slots):
        _, b = case(name, rows, width, dtype, seed + i)
        pool.append(tuple(y if y.ndim == 2 else old for y, old in zip(b, a)))
    ep = torch.export.export(m, a)
    schedule = (
        Schedule(512, "block", False, 4) if name == "resnorm2" else Schedule(256, "flat", False, 4)
    )
    k = compile_export(ep, schedule=schedule)
    fs = {
        "eager": m,
        "inductor": torch.compile(m, dynamic=False, fullgraph=True),
        "sparkfuse": native_dispatch(k),
    }
    info = {}
    graphs = {}
    samples = {}
    for label, fn in fs.items():
        for slot in (0, slots - 1):
            compare(fn(*pool[slot]), m(*pool[slot]), exact=(1,) if name == "resnorm2" else ())
        for mode in ("hot", "rotating"):
            tag = f"{label}:{mode}"
            g, outputs, numbers = capture(fn, pool, mode)
            g.replay()
            graphs[tag] = (g, outputs)
            samples[tag] = []
            # Validate an output produced by the captured graph, not only the direct call.
            compare(
                outputs[-1],
                m(*(pool[0] if mode == "hot" else pool[-1])),
                exact=(1,) if name == "resnorm2" else (),
            )
            info[tag] = dict(
                **numbers, unique_activation_input_bytes=activ * (1 if mode == "hot" else slots)
            )
    torch.cuda.synchronize()
    rng = random.Random(seed)
    sequence = []
    events = []
    for block in range(15):
        order = list(graphs)
        rng.shuffle(order)
        sequence.append(order)
        for tag in order:
            g = graphs[tag][0]
            g.replay()
            g.replay()
            beg = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            beg.record()
            for _ in range(5):
                g.replay()
            end.record()
            events.append((tag, beg, end))
    torch.cuda.synchronize()
    for tag, beg, end in events:
        samples[tag].append(beg.elapsed_time(end) * 1000 / (5 * slots))
    for tag, xs in samples.items():
        info[tag]["device"] = distribution(xs)
    return dict(
        case=name,
        shape=[rows, width],
        dtype=str(dtype),
        device=torch.cuda.get_device_name(),
        l2_bytes=l2,
        ring_slots=slots,
        seed=seed,
        schedule=vars(schedule),
        order=sequence,
        paths=info,
        interpretation="Hot reuses inputs and allows allocator output reuse. Rotating holds distinct activation inputs and output buffers; norm weight stays shared. Logical footprint, not a DRAM hardware counter.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.write_text(
        json.dumps(
            [
                bench("resnorm2", 512, 1536, torch.bfloat16),
                bench("gate", 128, 8960, torch.bfloat16),
            ],
            indent=2,
        )
        + "\n"
    )
