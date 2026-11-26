"""Offline audit of this release (Python standard library only; never imports torch or runs a kernel)."""

import argparse
from datetime import datetime
from contextlib import closing
from hashlib import sha256
from pathlib import Path
import gzip
import json
import re
import sqlite3
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from row_objectives import analyze as analyze_objectives
from campaign_snapshot import encoded, verify_doc
from import_campaign import import_bundle
from report import markdown
from verify_sources import verify_archive

STAGES = (
    "warp-decode",
    "protected-stack",
    "paged-decode",
    "ragged-page",
    "page-host",
    "slot-forward",
    "dispatch-replay",
    "catalog-rows",
)


CUTOFF_STAGE = "dispatch-replay"
SUPPLEMENT = "catalog-rows"

def must(value, message):
    if not value:
        raise ValueError(message)


def file_digest(p):
    return sha256(p.read_bytes()).hexdigest()


def source_digest(root):
    pats = (
        "sparkfuse/*.py",
        "sparkfuse/csrc/*.cpp",
        "examples/*.py",
        "tools/*.py",
        "pyproject.toml",
    )
    names = sorted({p.relative_to(root).as_posix() for pat in pats for p in root.glob(pat)})
    h = sha256()
    for name in names:
        h.update(name.encode() + b"\x00" + bytes.fromhex(file_digest(root / name)))
    return h.hexdigest()


def acceptance(data):
    checks = data["checks"]
    names = [x["check"] for x in checks]
    must(
        names and names[0] == "tests" and len(names) == len(set(names)),
        "missing or duplicate acceptance gate",
    )
    must(
        data["passed"] is True
        and all(q["status"] == "pass" and q["returncode"] == 0 for q in checks),
        "acceptance stage did not pass",
    )
    m = re.search(r"\b(\d+) passed\b", checks[0]["tail"])
    must(m and int(m[1]) > 0, "pytest result not recorded")
    return {"gates_passed": len(checks), "pytest_passed": int(m[1])}


def sanitizer_log(text, *, expected_init):
    lines = text.splitlines()
    expected = ["memcheck", *(["initcheck"] if expected_init else []), "racecheck", "synccheck"]
    must(len(lines) == 3 * len(expected), "missing sanitizer run")
    for i, tool in enumerate(expected):
        label, tests, summary = lines[i * 3 : i * 3 + 3]
        must(
            label.endswith(": " + tool) and re.match(r"^[1-9]\d* passed(?:,| in )", tests),
            "missing sanitizer results: " + tool,
        )
        suffix = (
            "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
            if tool == "racecheck"
            else "ERROR SUMMARY: 0 errors"
        )
        must(summary == "========= " + suffix, "sanitizer findings: " + tool)
    return expected


def deadline_gate(folder, stage_key, deadline):
    final = "gb10-overnight-acceptance"
    base = CUTOFF_STAGE + "-stage-acceptance"
    src = json.loads((folder / (base + ".json")).read_bytes())
    record = folder / (final + ".json")
    d = json.loads(record.read_bytes())
    facts = acceptance(d)
    must(
        [q["check"] for q in src["checks"]] == [q["check"] for q in d["checks"]]
        and facts == acceptance(src),
        "deadline rerun has different gates or test count",
    )
    ts = datetime.fromisoformat(deadline)
    must(
        ts.tzinfo is not None and 0 < ts.timestamp() - d["completed_epoch"] < 600,
        "acceptance not completed during final ten minutes",
    )
    files = {record.name: file_digest(record)}
    ref = {p.name[len(base) :] for p in folder.glob(base + "-*.json")}
    current = {p.name[len(final) :] for p in folder.glob(final + "-*.json")}
    must(ref and ref == current, "deadline smoke artifact set differs")
    proven = 0
    for suffix in sorted(ref):
        p = folder / (final + suffix)
        other = json.loads((folder / (base + suffix)).read_bytes())
        new = json.loads(p.read_bytes())
        prov = other.get("provenance") if isinstance(other, dict) else None
        if prov:
            expected = "sources/" + stage_key + ".tar.gz"
            v = new["provenance"]
            must(
                v["source_tree_sha256"] == prov["source_tree_sha256"] == stage_key
                and v["archive"] == prov["archive"] == expected,
                "deadline source mismatch: " + suffix,
            )
            proven += 1
        files[p.name] = file_digest(p)
    must(proven > 0, "no source-backed deadline output")
    return dict(
        stage=CUTOFF_STAGE, source_sha256=stage_key,
        **facts,
        completed_at=datetime.fromtimestamp(d["completed_epoch"], ts.tzinfo).isoformat(),
        smoke_outputs=len(ref),
        source_stamped=proven,
        files=files,
    )


def separate_windows(cutoff, supplement, deadline):
    """The state at 09:00 and a later verified increment are different claims."""
    limit = datetime.fromisoformat(deadline)
    must(limit.tzinfo is not None, "timezone required")
    a, b = acceptance(cutoff), acceptance(supplement)
    before, after = cutoff["completed_epoch"], supplement["completed_epoch"]
    must(before < limit.timestamp() < after, "stage on wrong side of the cutoff")
    ca, cb = ({x["check"] for x in j["checks"]} for j in (cutoff, supplement))
    must(ca < cb and len(cb - ca) == 1 and b["pytest_passed"] > a["pytest_passed"],
         "post-deadline supplement must extend the verified cutoff suite")
    return dict(stage=SUPPLEMENT, classification="post-deadline supplement", **b,
                completed_at=datetime.fromtimestamp(after, limit.tzinfo).isoformat(),
                added_gates=sorted(cb - ca))


def stage(folder, name):
    actual = folder / (name + "-gb10.json")
    b = folder / (name + "-stage-acceptance-" + name + ".json")
    g = folder / (name + "-stage-acceptance.json")
    san = folder / (name + "-sanitizer-gb10.txt")
    a = json.loads(actual.read_bytes())
    smoke = json.loads(b.read_bytes())
    key = a["provenance"]["source_tree_sha256"]
    must(
        not a["smoke"]
        and smoke["smoke"]
        and smoke["provenance"]["source_tree_sha256"] == key
        and a["provenance"]["archive"] == smoke["provenance"]["archive"] == f"sources/{key}.tar.gz",
        "full/smoke provenance differs: " + name,
    )
    verify_archive(folder / "sources" / (key + ".tar.gz"), key)
    tools = sanitizer_log(san.read_text(), expected_init=name not in STAGES[:3])
    return dict(
        **acceptance(json.loads(g.read_bytes())),
        measured_config_seed_cases=len(a["cases"]),
        source_sha256=key,
        sanitizers=tools,
        files={p.name: file_digest(p) for p in (actual, b, g, san)},
    )


def campaign(folder, panel):
    p = folder / f"campaign-final-{panel}.json.gz"
    m = p.with_suffix("").with_suffix(".md")
    d = json.loads(gzip.decompress(p.read_bytes()))
    v = verify_doc(d, folder / "sources")
    must(
        v["state"].startswith("stopped: ")
        and d["panel"] == panel
        and not d["ignored_unterminated_index_bytes"],
        "not a complete stopped campaign: " + panel,
    )
    if panel == "general":
        must(
            markdown(d, p.name).encode() == m.read_bytes(),
            "general Markdown does not match the sealed data",
        )
    return dict(
        **v,
        outcomes=d["summary"]["outcomes"],
        compiler_groups=len(d["summary"]["by_compiler"]),
        records_sha256=d["records_sha256"],
        compressed_sha256=file_digest(p),
        markdown_sha256=file_digest(m),
        supervisor_deadline=d["observed_supervisor"]["deadline"],
        implicit_parity_in_passed=sum(
            s["implicit_parity_in_passed"] for s in d["summary"]["by_compiler"].values()
        ),
    )


def logical_catalog(db):
    h = sha256()
    n = 0
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        for key, record in conn.execute("select key,record from schedules order by key"):
            obj = json.loads(record)
            must(obj["key"] == key, "catalog row key mismatch")
            h.update(encoded([key, obj]) + b"\n")
            n += 1
    return n, h.hexdigest()


def audit(root):
    folder = root / "reports"
    stages = {n: stage(folder, n) for n in STAGES}
    current = source_digest(root)
    must(
        current == stages[STAGES[-1]]["source_sha256"],
        "checked-out sources differ from the last measured stage",
    )
    panels = {k: campaign(folder, k) for k in ("general", "attention")}
    last = json.loads((folder / "catalog-rows-gb10.json").read_bytes())
    compiler = last["catalog"]["compiler"]
    general = folder / "campaign-final-general.json.gz"
    with TemporaryDirectory() as t:
        db = Path(t) / "dispatch.sqlite"
        q = import_bundle(general, db, compiler)
        size, key = logical_catalog(db)
        must(size == q["unique_keys"], "import count differs")
        dispatch = dict(**q, logical_schedules_sha256=key)
    objective_file = folder / "row-objectives-gb10.json"
    objective = analyze_objectives(folder / "catalog-rows-gb10.json")
    must(
        encoded(json.loads(objective_file.read_bytes())) == encoded(objective),
        "request-objective holdout summary mismatch",
    )
    cutoff = panels["general"]["supervisor_deadline"]
    cutoff_key = stages[CUTOFF_STAGE]["source_sha256"]
    must(cutoff_key != current, "cutoff and supplement unexpectedly share a source tree")
    deadline = deadline_gate(folder, cutoff_key, cutoff)
    load_gate = lambda n: json.loads((folder / (n + "-stage-acceptance.json")).read_bytes())
    supplement = separate_windows(load_gate(CUTOFF_STAGE), load_gate(SUPPLEMENT), cutoff)
    supplement["source_sha256"] = current
    return {
        "final-evidence-verification.json": dict(
            deadline_acceptance=deadline,
            post_deadline_supplement=supplement,
            request_objectives_sha256=file_digest(objective_file),
            format="sparkfuse.release-audit.v2",
            source_sha256=current,
            stages=stages,
            campaigns=panels,
        ),
        "final-dispatch-qualification.json": dispatch,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--emit",
        type=Path,
        help="create both new verification files in a separate existing directory; default compares with the published files",
    )
    args = p.parse_args()
    out = audit(ROOT)
    for name, value in out.items():
        if args.emit:
            with (args.emit / name).open("x", encoding="utf8", newline="\n") as f:
                f.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        else:
            must(
                encoded(json.loads((ROOT / "reports" / name).read_bytes())) == encoded(value),
                "manifest mismatch: " + name,
            )
    c = out["final-evidence-verification.json"]["campaigns"]["general"]
    d = out["final-dispatch-qualification.json"]
    print(
        f"Verified {len(STAGES)} GPU stages; general: {c['passed']}/{c['indexed']} passed; rebuilt {d['unique_keys']} exact dispatch keys. No GPU was executed."
    )


if __name__ == "__main__":
    main()
