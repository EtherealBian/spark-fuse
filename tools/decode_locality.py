"""Paired replay rings. Keep every output alive so only the input-reuse pattern changes."""

import math
import random
from statistics import median
import torch
from decode_sdpa_panel import cpu_reference, metrics, inputs
from token_quant_panel import digest


def footprint(requests):
    seen = {}
    for row in requests:
        for t in row:
            k = t.untyped_storage().data_ptr()
            n = t.untyped_storage().nbytes()
            if k in seen:
                assert n == seen[k]
            seen[k] = n
    return dict(storages=len(seen), bytes=sum(seen.values()))


def experiment(spec, seed, model, paths, *, panels=9):
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    sample = inputs(spec, seed)
    kv_bytes = sum(t.untyped_storage().nbytes() for t in sample[1:3])
    slots = max(4, math.ceil(3 * l2 / kv_bytes))
    total = footprint([sample])["bytes"] * slots
    if slots > 64 or total > 384 << 20:
        raise ValueError("ring exceeds experimental budget")
    pool = [sample] + [inputs(spec, seed + 137 * i) for i in range(1, slots)]
    rings = {
        "hot": [pool[0]] * slots,
        "shared_kv": [(q[0], *pool[0][1:]) for q in pool],
        "independent_kv": pool,
    }
    gr = {}
    gates = {}
    info = {}
    samples = {}
    orders = []
    oracle_cache = {}
    input_hashes = [[digest(t) for t in xs] for xs in pool]
    for mode, ring in rings.items():
        info[mode] = {
            "all_inputs": footprint(ring),
            "kv": footprint([row[1:3] for row in ring]),
            "query": footprint([row[:1] for row in ring]),
        }
        refs = []
        oracle = []
        for xs in ring:
            key = tuple(t.data_ptr() for t in xs)
            if key not in oracle_cache:
                oracle_cache[key] = cpu_reference(xs, spec[-1])
            oracle.append(oracle_cache[key])
            refs.append(model(*xs))
        for name, fn in paths.items():
            fn(*ring[0])
            tag = f"{name}:{mode}"
            g = torch.cuda.CUDAGraph()
            saved = []
            with torch.cuda.graph(g):
                for xs in ring:
                    saved.append(fn(*xs))
            g.replay()
            torch.cuda.synchronize()
            checks = [
                metrics(y, ref, expected, xs)
                for y, ref, expected, xs in zip(saved, oracle, refs, ring, strict=True)
            ]
            live = footprint([(t,) for t in saved])
            passed = live["storages"] == slots and all(
                z["structure"] and z["outside_tolerance"] == 0 for z in checks
            )
            gates[tag] = dict(admitted=passed, live_outputs=live, requests=checks)
            if passed:
                gr[tag] = (g, saved)
                samples[tag] = []
    rng = random.Random(seed)
    events = []
    torch.cuda.synchronize()
    for block in range(panels):
        order = sorted(gr)
        rng.shuffle(order)
        orders.append(order)
        for tag in order:
            g = gr[tag][0]
            g.replay()
            g.replay()
            a = torch.cuda.Event(enable_timing=True)
            b = torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(4):
                g.replay()
            b.record()
            events.append((tag, a, b))
    torch.cuda.synchronize()
    for tag, a, b in events:
        samples[tag].append(a.elapsed_time(b) * 1000 / (4 * slots))
    assert input_hashes == [[digest(t) for t in xs] for xs in pool]
    return dict(
        seed=seed,
        l2_bytes=l2,
        slots=slots,
        footprint=info,
        input_sha256=input_hashes,
        gates=gates,
        orders=orders,
        device_us=samples,
        median_us={tag: median(v) for tag, v in samples.items()},
    )
