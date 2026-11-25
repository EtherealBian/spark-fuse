"""Seal one append-only campaign prefix and independently audit it without a GPU."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import tempfile

if __package__:
    from .verify_sources import verify_archive
else:
    from verify_sources import verify_archive

VERSION = "sparkfuse.campaign.v1"
BACKENDS = ("eager", "inductor", "sparkfuse")
HASH = re.compile("[0-9a-f]{64}")
ID = re.compile("[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}")
PHASES = {
    "qwen-reference": "general",
    "tail-reduction": "general",
    "broad-search": "general",
    "score-prefill-like": "attention",
    "score-decode-like": "attention",
}


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def encoded(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def portable(obj):
    # Change only free-text diagnostics. Never change a number, ID, measured GPU name or FX graph.
    if isinstance(obj, dict):
        return {
            k: (
                v.replace(str(Path.home()) + "/", "$HOME/")
                if k in ("diagnostic", "stderr", "worker_stderr") and isinstance(v, str)
                else portable(v)
            )
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [portable(x) for x in obj]
    return obj


def checked_median(d):
    samples = d["samples_us"]
    require(
        samples and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in samples),
        "invalid timing samples",
    )
    val = median(samples)
    require(val == d["median_us"], "recorded median differs from samples")
    return val


def check_quality(errors, name):
    require(isinstance(errors, list) and errors, "missing comparison evidence")
    if name in ("resnorm2", "scores2", "layernorm2"):
        require(len(errors) == 2, "missing observable output")
        obs = errors[1]
        require(
            obs["rtol"] == obs["atol"] == obs["unequal"] == 0 and obs["max_abs"] in (0, None),
            "observable residual/logits not exact",
        )


def check_tuning(q, job):
    check_quality(q["heldout_errors"], job["case"])
    trials = q["trials"]
    require(trials, "no discovery trials")
    times = [checked_median(v["device"]) for v in trials]
    require(
        any(t == min(times) and v["schedule"] == q["candidate"] for t, v in zip(times, trials)),
        "candidate did not win discovery",
    )
    windows = q["paired_holdout_us"]
    require(len(windows) == 5, "missing holdout round")
    for win in windows:
        require(
            all(
                type(win[k]) in (float, int) and math.isfinite(win[k]) and win[k] > 0
                for k in ("default", "candidate")
            ),
            "invalid holdout sample",
        )
    ratios = [x["default"] / x["candidate"] for x in windows]
    score = median(ratios)
    admit = q["candidate"] != q["default"] and score >= 1.03 and sum(x > 1 for x in ratios) >= 4
    require(
        q["admitted"] is admit
        and score == q["heldout_speedup_vs_default"]
        and q["selected"] == (q["candidate"] if admit else q["default"]),
        "holdout selection mismatch",
    )


def audit(records, panel):
    require(panel in PHASES.values(), "unknown campaign panel")
    groups = defaultdict(list)
    ids = set()
    sources = set()
    results = []
    for entry in records:
        r, w = entry["index"], entry["worker"]
        case = r["id"]
        require(ID.fullmatch(case) and case not in ids, "invalid/duplicate case ID")
        ids.add(case)
        try:
            require(PHASES.get(r["phase"]) == panel, "phase belongs to different panel")
            stat = r["status"]
            require(stat in ("ok", "error", "deadline-interruption"), "unknown index status")
            q = {} if w is None else w.get("tuning", {})
            key = q.get("compiler", r.get("compiler"))
            require(key is None or HASH.fullmatch(key), "invalid compiler hash")
            if "compiler" in r and q:
                require(r["compiler"] == q["compiler"], "index compiler mismatch")
            row = dict(
                id=case,
                phase=r["phase"],
                compiler=key,
                source=None,
                outcome=stat,
                backend_failures=[],
                legacy_implicit_parity=False,
            )
            if w is None:
                require(stat != "ok", "successful case missing worker evidence")
                row["outcome"] = (
                    "deadline-interruption"
                    if stat == "deadline-interruption"
                    else "failure-without-worker-evidence"
                )
            else:
                j = w["job"]
                require(
                    j["id"] == case
                    and j["phase"] == r["phase"]
                    and case == f"{j['case']}-{j['rows']}x{j['width']}-{j['dtype']}",
                    "worker identity mismatch",
                )
                if "provenance" in w:
                    pv = w["provenance"]
                    src = pv["source_tree_sha256"]
                    require(
                        HASH.fullmatch(src) and pv["archive"] == f"sources/{src}.tar.gz",
                        "bad source reference",
                    )
                    row["source"] = src
                    sources.add(src)
                state = w["status"]
                require((state == "ok") == (stat == "ok"), "index/worker outcome mismatch")
                if q:
                    check_tuning(q, j)
                    require(
                        q["seed"] == j["seed"] and q["compiler"] == key,
                        "discovery identity mismatch",
                    )
                if state in ("ok", "parity-failure"):
                    c = w["comparison"]
                    b = c["backends"]
                    require(set(b) == set(BACKENDS), "backend set mismatch")
                    require(
                        c["case"] == j["case"]
                        and c["shape"] == [j["rows"], j["width"]]
                        and c["dtype"] == "torch." + j["dtype"]
                        and c["seed"] == j["seed"] + 2000
                        and c["schedule"] == q["selected"],
                        "comparison not the selected schedule on the third request",
                    )
                    failed = []
                    times = {k: {} for k in ("device", "host_sync")}
                    for name in BACKENDS:
                        v = b[name]
                        parity = v.get("parity")
                        require(parity in (None, "pass", "fail"), "unknown parity value")
                        if parity == "fail":
                            require(not any(k in v for k in times), "failed backend has timings")
                            failed.append(name)
                            continue
                        check_quality(v["errors"], j["case"])
                        row["legacy_implicit_parity"] |= parity is None
                        for clock in times:
                            times[clock][name] = checked_median(v[clock])
                    require(
                        bool(failed) == (state == "parity-failure"),
                        "backend/worker parity mismatch",
                    )
                    row["backend_failures"] = failed
                    if "failed_backends" in w:
                        require(
                            set(w["failed_backends"]) == set(failed), "failure attribution mismatch"
                        )
                    if stat == "ok":
                        require(
                            r["us"] == times["device"], "index times differ from worker samples"
                        )
                        require(
                            len(q["trials"]) == r["trials"]
                            and len(q["invalid"]) == r["invalid"]
                            and type(r["admitted"]) is bool
                            and r["admitted"] == q["admitted"]
                            and r["holdout"] == q["heldout_speedup_vs_default"],
                            "tuning/index mismatch",
                        )
                        row.update(
                            times=times,
                            admitted=r["admitted"],
                            trials=r["trials"],
                            invalid=r["invalid"],
                        )
                else:
                    require(state.endswith("-error"), "unknown worker outcome")
                row["outcome"] = state if stat != "deadline-interruption" else stat
            groups[key or "unrecorded"].append(row)
            results.append(row)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{case}: {e}") from e
    by = {}
    for key, rows in sorted(groups.items()):
        good = [x for x in rows if x["outcome"] == "ok"]
        g = dict(
            outcomes=dict(Counter(x["outcome"] for x in rows)),
            failed_backend_counts=dict(Counter(k for x in rows for k in x["backend_failures"])),
            passed=len(good),
            indexed=len(rows),
            admitted=sum(x["admitted"] for x in good),
            discovery_timings_in_passed=sum(x["trials"] for x in good),
            rejected_schedules_in_passed=sum(x["invalid"] for x in good),
            missing_source=sum(x["source"] is None for x in rows),
            implicit_parity_in_passed=sum(x["legacy_implicit_parity"] for x in good),
            phase_counts=dict(Counter(x["phase"] for x in rows)),
        )
        for clock in ("device", "host_sync"):
            ratios = [x["times"][clock]["inductor"] / x["times"][clock]["sparkfuse"] for x in good]
            g[clock] = dict(
                paired_cases=len(ratios),
                sparkfuse_faster=sum(v > 1 for v in ratios),
                median_inductor_over_sparkfuse=median(ratios) if ratios else None,
            )
        by[key] = g
    return dict(
        indexed=len(records),
        passed=sum(x["outcome"] == "ok" for x in results),
        outcomes=dict(Counter(x["outcome"] for x in results)),
        by_compiler=by,
        sources=sorted(sources),
        missing_source=sum(x["source"] is None for x in results),
    ), results


def snapshot(run, status=None, panel=None):
    # The worker atomically renames its evidence before the supervisor appends the index.
    # Never include a pending/unindexed result or infer its outcome from a timer.
    require(ID.fullmatch(run.name), "invalid run ID")
    obs = json.loads(status.read_bytes()) if status and status.is_file() else {}
    blob = (run / "index.jsonl").read_bytes()
    lines = blob.split(b"\n")
    tail = lines.pop()
    records = []
    for line in lines:
        r = json.loads(line)
        case = r["id"]
        require(ID.fullmatch(case), "invalid case ID")
        evidence = (
            run / "evidence" / (case + (".json" if r["status"] == "ok" else ".rejected.json"))
        )
        records.append(
            dict(
                index=portable(r),
                worker=portable(json.loads(evidence.read_bytes())) if evidence.is_file() else None,
            )
        )
    if panel is None:
        panels = {PHASES.get(x["index"]["phase"]) for x in records}
        require(len(panels) == 1, "panel missing or mixed")
        panel = panels.pop()
    summary, _ = audit(records, panel)
    last = obs.get("last") or {}
    state = {
        k: obs.get(k) for k in ("updated_at_utc", "deadline", "stop_reason", "completed", "failed")
    }
    state["last_id"] = last.get("id")
    doc = dict(
        format=VERSION,
        run=run.name,
        panel=panel,
        captured_utc=datetime.now(timezone.utc).isoformat(),
        observed_supervisor=state,
        ignored_unterminated_index_bytes=len(tail),
        records=records,
        records_sha256=hashlib.sha256(encoded(records)).hexdigest(),
        summary=summary,
    )
    return doc


def completeness(doc):
    s = doc["summary"]
    obs = doc["observed_supervisor"]
    last = doc["records"][-1]["index"]["id"] if doc["records"] else None
    consistent = (
        obs["completed"] == s["passed"]
        and obs["failed"] == s["indexed"] - s["passed"]
        and obs["last_id"] == last
    )
    if not consistent:
        return "index prefix; supervisor counts/last ID unavailable or do not match"
    if doc["ignored_unterminated_index_bytes"]:
        return "index prefix; partial append excluded"
    return ("stopped: " + obs["stop_reason"]) if obs["stop_reason"] else "in progress at capture"


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        dest = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.close()
            dest.replace(path)
        finally:
            dest.unlink(missing_ok=True)


def verify_doc(doc, source_dir):
    require(doc["format"] == VERSION and ID.fullmatch(doc["run"]), "bad format or run")
    rec = doc["records"]
    require(
        hashlib.sha256(encoded(rec)).hexdigest() == doc["records_sha256"], "record digest mismatch"
    )
    s, _ = audit(rec, doc["panel"])
    require(encoded(s) == encoded(doc["summary"]), "summary differs from worker evidence")
    for key in s["sources"]:
        verify_archive(source_dir / (key + ".tar.gz"), key)
    return {
        "run": doc["run"],
        "panel": doc["panel"],
        "indexed": s["indexed"],
        "passed": s["passed"],
        "distinct_sources_verified": len(s["sources"]),
        "missing_source": s["missing_source"],
        "state": completeness(doc),
    }


def save(doc, run, output):
    require(output.name.endswith(".json.gz"), "output must end in .json.gz")
    source_dir = output.parent / "sources"
    for key in doc["summary"]["sources"]:
        dst = source_dir / (key + ".tar.gz")
        if not dst.is_file():
            orig = run / "sources" / dst.name
            verify_archive(orig, key)
            write_atomic(dst, orig.read_bytes())
    result = verify_doc(doc, source_dir)
    write_atomic(output, gzip.compress(encoded(doc) + b"\n", mtime=0))
    return result


def verify_file(path):
    with gzip.open(path, "rt") as f:
        d = json.load(f)
    return verify_doc(d, path.parent / "sources")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal")
    s.add_argument("run", type=Path)
    s.add_argument("output", type=Path)
    s.add_argument("--status", type=Path)
    s.add_argument("--panel", choices=["general", "attention"])
    v = sub.add_parser("verify")
    v.add_argument("files", type=Path, nargs="+")
    a = p.parse_args()
    if a.cmd == "seal":
        print(json.dumps(save(snapshot(a.run, a.status, a.panel), a.run, a.output)))
    else:
        for f in a.files:
            print(json.dumps(verify_file(f)))
