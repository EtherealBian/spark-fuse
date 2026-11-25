"""Build a version-stratified report from a live index or already sealed offline bundle."""

import argparse
import gzip
import json
import os
from pathlib import Path

if __package__:
    from .campaign_snapshot import audit, completeness, snapshot, save, verify_doc, write_atomic
else:
    from campaign_snapshot import audit, completeness, snapshot, save, verify_doc, write_atomic


def markdown(doc, relative_bundle):
    s, rows = audit(doc["records"], doc["panel"])
    state = doc["observed_supervisor"]
    text = [
        f"# DGX Spark: {doc['run']} / {doc['panel']}",
        "",
        f"Captured {doc['captured_utc']} (UTC); **{completeness(doc)}**. Supervisor deadline: {state['deadline'] or 'not recorded'}.",
        "",
        f"{s['passed']} of {s['indexed']} indexed cases passed all three backends. {s['missing_source']} records lack an archived source tree. [Portable evidence]({relative_bundle}) retains indexed failures, discovery/holdout samples and both host/device samples where recorded.",
        "",
        "## Results by compiler",
        "",
        "Compare only within a row. Ratios are medians of paired case ratios (Inductor / SparkFuse), not model-weighted speedups. Device = warm CUDA Graph replay; host = synchronized Python invocation. Neither is serving throughput. Admitted means a held-out gain vs the compiler’s own default, not a win over Inductor.",
        "",
        "| compiler | passed / indexed | admitted | device wins | device ratio | host wins | host ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k, g in s["by_compiler"].items():

        def ratio(clock):
            v = g[clock]["median_inductor_over_sparkfuse"]
            return "—" if v is None else f"{v:.3f}×"

        text.append(
            f"| `{k[:14]}` | {g['passed']} / {g['indexed']} | {g['admitted']} | {g['device']['sparkfuse_faster']} | {ratio('device')} | {g['host_sync']['sparkfuse_faster']} | {ratio('host_sync')} |"
        )
    text += [
        "",
        "## Failures and interruptions",
        "",
        "Original attempts remain separate from any later replay. Missing attribution is not evidence that a particular backend failed.",
        "",
    ]
    for r in rows:
        if r["outcome"] != "ok":
            text.append(
                f"- `{r['id']}`: {r['outcome']}; backend: {', '.join(r['backend_failures']) or 'not attributed'}; compiler: `{(r['compiler'] or 'unrecorded')[:14]}`."
            )
    if s["passed"] == s["indexed"]:
        text.append("None in this prefix.")
    text += [
        "",
        "## Passing cases",
        "",
        "Per-backend device medians in µs; corresponding synchronized host samples are in the bundle.",
        "",
        "| case | compiler | eager | Inductor | SparkFuse |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        if r["outcome"] == "ok":
            t = r["times"]["device"]
            text.append(
                f"| `{r['id']}` | `{(r['compiler'] or 'unrecorded')[:14]}` | {t['eager']:.3f} | {t['inductor']:.3f} | {t['sparkfuse']:.3f} |"
            )
    return "\n".join(text) + "\n"


def render(run, output, *, status=None):
    if output.suffix != ".md":
        raise ValueError("expected .md output")
    if run.is_dir():
        status = status or Path(__file__).resolve().parents[1] / "reports" / (
            run.name + "-status.json"
        )
        doc = snapshot(run, status)
        bundle = output.with_suffix(".json.gz")
        save(doc, run, bundle)
    else:
        bundle = run
        with gzip.open(bundle, "rt") as f:
            doc = json.load(f)
        verify_doc(doc, bundle.parent / "sources")
    link = Path(os.path.relpath(bundle, output.parent)).as_posix()
    write_atomic(output, markdown(doc, link).encode())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--status", type=Path)
    a = p.parse_args()
    render(a.run, a.output, status=a.status)
