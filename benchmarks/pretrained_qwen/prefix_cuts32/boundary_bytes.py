"""Content-addressed little-endian BF16 traces; independently replay actual words and greedy choices on CPU."""

from array import array
from hashlib import sha256
from pathlib import Path
import struct
import sys
from functools import lru_cache


def check(x, msg):
    if not x:
        raise ValueError(msg)


def hashes(data, steps, vocab):
    stride = 2 * vocab
    check(steps > 0 and vocab > 0 and len(data) == steps * stride, "invalid byte extent")
    return sha256(data).hexdigest(), [
        sha256(data[i : i + stride]).hexdigest() for i in range(0, len(data), stride)
    ]


def capture(run, data, steps, vocab):
    check(sys.byteorder == "little", "BF16 capture expects little endian")
    digest, rows = hashes(data, steps, vocab)
    root = run / "logits"
    root.mkdir(exist_ok=True)
    path = root / (digest + ".bf16le")
    if path.exists():
        check(path.read_bytes() == data, "inconsistent content address")
    else:
        tmp = path.with_suffix(".partial")
        tmp.write_bytes(data)
        tmp.replace(path)
    return digest, rows


@lru_cache(maxsize=1)
def values():
    return tuple(struct.unpack(">f", struct.pack(">I", i << 16))[0] for i in range(65536))


def words(b):
    a = array("H")
    check(a.itemsize == 2, "expected uint16")
    a.frombytes(b)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def describe(row):
    lookup = values()
    a = words(row)
    top = -float("inf")
    second = top
    idx = None
    for i, v in enumerate(a):
        check((v & 0x7F80) != 0x7F80, "nonfinite BF16 logit")
        f = lookup[v]
        if f > top:
            second, top, idx = top, f, i
        elif f > second:
            second = f
    return idx, top - second


def inventory(run, raw):
    root = Path(run) / "logits"
    expected = {d["logits_sha256"] + ".bf16le" for c in raw["cases"] for d in c["diagnostics"]}
    check(
        root.is_dir() and not root.is_symlink() and {p.name for p in root.iterdir()} == expected,
        "missing or extra captures",
    )
    memo = {}
    out = []
    rows = 0
    lookup = values()
    for c in raw["cases"]:
        by = {}
        for d in c["diagnostics"]:
            steps, v = d["shape"]
            h = d["logits_sha256"]
            p = root / (h + ".bf16le")
            check(not p.is_symlink() and p.is_file(), "invalid capture")
            data = p.read_bytes()
            hh, hs = hashes(data, steps, v)
            check(hh == h and hs == d["step_sha256"], "capture digest mismatch")
            stride = 2 * v
            split = [data[j : j + stride] for j in range(0, len(data), stride)]
            if h not in memo:
                memo[h] = [describe(row) for row in split]
            top = memo[h]
            check([e[0] for e in top] == d["tokens"], "captured argmax mismatch")
            by[d["arm"], d["branch"]] = (split, top)
            rows += steps
        for k in (0, 1):
            xs, xt = by["cold", k]
            ys, yt = by["snapshot", k]
            history = True
            detail = []
            for i, (x, y, (gx, mx), (gy, my)) in enumerate(zip(xs, ys, xt, yt, strict=True), 1):
                changed = 0
                max_delta = 0.0
                if x != y:
                    for a, b in zip(words(x), words(y), strict=True):
                        if a != b:
                            changed += 1
                            max_delta = max(max_delta, abs(lookup[a] - lookup[b]))
                detail.append(
                    dict(
                        step=i,
                        same_input_and_generated_history=history,
                        unequal_bf16_words=changed,
                        max_absolute_logit_delta=max_delta,
                        cold_greedy_id=gx,
                        snapshot_greedy_id=gy,
                        cold_top_two_gap=mx,
                        snapshot_top_two_gap=my,
                    )
                )
                history &= gx == gy
            out.append(dict(prefix=c["prefix"], family=c["family"], branch=k, steps=detail))
    return dict(
        format="bf16-little-endian",
        distinct_blobs=len(expected),
        validated_full_vocab_steps=rows,
        traces=out,
    )
