"""Recompute one published expert ledger with the pinned upstream GGUFReader and GGML ABI."""

import argparse
from collections import Counter, defaultdict
import ctypes
import hashlib
import json
from pathlib import Path
from audit_expert import MODELS, audit
from expert_ledger import BLOCKS, inspect, need


def sha(f):
    with f.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def verify_one(lock, weights, receipts, runtime_file, libpath):
    from gguf import GGUFReader

    pin = json.loads(lock.read_bytes())
    cp = json.loads((receipts / "checkpoint.json").read_bytes())
    rt = json.loads((receipts / runtime_file).read_bytes())
    article = inspect(pin, weights)
    need(
        cp["files"] == pin["files"]
        and cp["lock_sha256"] == sha(lock)
        and cp["total_bytes"] == article["source_object_bytes"]
        and cp["revision"] == article["revision"],
        "whole-object receipt/lock mismatch",
    )
    need(rt["runtime_lib_sha256"][libpath.name] == sha(libpath), "not the measured GGML runtime")
    norm = Counter()
    parts = []
    layer_experts = defaultdict(list)
    for item in article["shards"]:
        reader = GGUFReader(weights / item["name"])
        need(
            len(reader.tensors) == item["tensors"]
            and reader.data_offset == item["aligned_directory_bytes"],
            "upstream reader disagrees on directory",
        )
        total = 0
        for tensor in reader.tensors:
            num = int(tensor.n_bytes)
            norm[tensor.tensor_type.name.lower()] += num
            total += num
            if "_exps." in tensor.name:
                li = int(tensor.name.split(".")[1])
                n = int(tensor.shape[-1])
                need(n == article["experts_per_layer"] and num % n == 0, "wrong expert axis")
                layer_experts[li].append(num // n)
        parts.append(total)
        need(
            total == item["object_bytes"] - reader.data_offset - item["padding_or_trailer_bytes"],
            "upstream packed amount mismatch",
        )
        del reader
    need(
        sum(parts) == article["packed_tensor_bytes"]
        and {k.lower(): v for k, v in article["packed_bytes_by_type"].items()} == dict(norm),
        "upstream tensor bytes differ",
    )
    summed = [sum(v) for v in layer_experts.values()]
    k = article["selected_per_token_per_layer"]
    n = article["experts_per_layer"]
    if summed:
        need(
            len(summed) == article["layers"]
            and sum(summed) * n == article["packed_bytes_by_group"]["routed_expert"],
            "expert slice sizes disagree",
        )
        for count, bounds in article["routed_expert_union_bounds"].items():
            need(
                bounds["min_bytes"] == sum(x * k for x in summed)
                and bounds["max_bytes"] == sum(x * min(n, int(count) * k) for x in summed),
                "independent bounds mismatch",
            )
    else:
        need(
            not article["routed_expert_union_bounds"]
            and not article["packed_bytes_by_group"].get("routed_expert"),
            "dense model should not have routed tensors",
        )
    return article, dict(
        lock_sha256=sha(lock),
        prior_full_checksum_receipt_sha256=sha(receipts / "checkpoint.json"),
        runtime_receipt_sha256=sha(receipts / runtime_file),
        independent_reader_tensors=article["tensor_count"],
        independent_packed_tensor_bytes=sum(parts),
        abi_sha256=sha(libpath),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", choices=MODELS)
    ap.add_argument("weights", type=Path)
    ap.add_argument("ggml_base", type=Path, help="measured libggml-base.so.0.9.7")
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    args = ap.parse_args()
    src = Path(__file__).resolve().parent
    folder = args.reports / "expert-ledger"
    audit(folder, src, args.reports)
    proof = json.loads((folder / "verified.json").read_bytes())
    cfg = MODELS[args.model]
    libpath = args.ggml_base.resolve(strict=True)
    receipt = proof["models"][args.model]
    need(
        libpath.name == "libggml-base.so.0.9.7" and sha(libpath) == receipt["abi_sha256"],
        "supply the same measured GGML library",
    )
    import gguf

    pkg = Path(gguf.__file__).resolve().parent
    need(
        all(sha(pkg / f) == h for f, h in proof["readers"].items()),
        "upstream gguf-py source mismatch",
    )
    lib = ctypes.CDLL(str(libpath))
    lib.ggml_type_name.restype = ctypes.c_char_p
    lib.ggml_type_size.restype = ctypes.c_size_t
    lib.ggml_blck_size.restype = ctypes.c_int64
    for t, expected in BLOCKS.items():
        need(
            (lib.ggml_type_name(t).decode(), lib.ggml_blck_size(t), lib.ggml_type_size(t))
            == expected,
            "GGML ABI layout differs",
        )
    article, rec = verify_one(src / cfg[0], args.weights, args.reports / cfg[1], cfg[2], libpath)
    published = folder / (args.model + ".json")
    rec["ledger_sha256"] = sha(published)
    need(
        article == json.loads(published.read_bytes()) and rec == receipt,
        "independent reader or prior receipts disagree",
    )
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
