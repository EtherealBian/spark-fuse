"""Small, named reference panels: raw trials, three independent seeds and all outcomes in one portable file."""

import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANELS = {
    "scores": [
        ("scores", 1, 97, "bfloat16"),
        ("scores2", 47, 97, "bfloat16"),
        ("scores2", 257, 769, "float32"),
        ("scores", 512, 1024, "bfloat16"),
        ("scores2", 128, 4096, "float16"),
        ("scores", 12, 8192, "bfloat16"),
    ],
    "layernorm": [
        ("layernorm2", rows, width, dtype)
        for rows, width, dtype in (
            (1, 768, "bfloat16"),
            (32, 768, "bfloat16"),
            (128, 768, "bfloat16"),
            (17, 769, "float32"),
            (97, 1024, "float16"),
            (512, 1024, "bfloat16"),
        )
    ],
    "qwen": [
        (case, rows, 8960 if case == "gate" else 1536, "bfloat16")
        for case in ("resnorm2", "gate")
        for rows in (1, 32, 512)
    ],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("panel", choices=PANELS)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--budget", type=int, default=14)
    a = p.parse_args()
    if not a.run.replace("-", "").isalnum():
        p.error("invalid run id")
    folder = ROOT / ".runtime" / a.run
    folder.mkdir(exist_ok=False)
    records = []
    for ix, (name, rows, w, dtype) in enumerate(PANELS[a.panel]):
        j = dict(
            case=name,
            rows=rows,
            width=w,
            dtype=dtype,
            budget=a.budget,
            seed=8100 + ix * 23,
            profile=ix == 1,
        )
        path = folder / f"case-{ix}.pending"
        rejected = path.with_suffix(".rejected.json")
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/campaign_case.py"),
                json.dumps(j),
                str(path),
                str(folder / "dispatch.sqlite"),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode == 0 and path.is_file():
            saved = path.with_suffix(".json")
            path.replace(saved)
            result = json.loads(saved.read_text())
        elif rejected.is_file():
            result = json.loads(rejected.read_text())
        else:
            result = {"job": j, "status": "error"}
        if proc.returncode:
            result["worker_stderr"] = proc.stderr[-3000:]
        records.append(result)
        print(name, rows, w, result["status"], flush=True)
        doc = {
            "panel": a.panel,
            "expected_cases": len(PANELS[a.panel]),
            "records": records,
            "all_passed": len(records) == len(PANELS[a.panel])
            and all(x["status"] == "ok" for x in records),
        }
        tmp = a.output.with_suffix(".pending")
        tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        tmp.replace(a.output)
    return 0 if doc["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
