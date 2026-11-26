"""Ahead-of-request row multiversioning from exact, previously measured catalog entries."""

from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from .codegen import Schedule, generate
from .dynamic import DynamicRows
from .native import _native_dispatch
from .runtime import build
from .tuning import backend_id, signature


@dataclass(frozen=True)
class RowVariant:
    rows: int
    signature: str
    selected: Schedule | None
    source_sha256: str | None


@dataclass(frozen=True)
class CatalogRows:
    run: object
    bounds: tuple
    generic: Schedule
    fallback_source_sha256: str
    variants: tuple[RowVariant, ...]

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


def compile_catalog_rows(ep, catalog, rows, device="cuda"):
    """One finite leading export symbol; specialize only a verified catalog's exact keys.

    No interpolation, on-demand compilation, database reads or tensor staging during a request.
    The generic dynamic kernel handles every other in-range leading dimension. A missing or
    incompatible catalog is an error at setup; a missing row entry is a normal generic fallback.
    """
    if (
        type(rows) not in (list, tuple)
        or len(rows) > 32
        or any(type(n) is not int for n in rows)
        or len(set(rows)) != len(rows)
    ):
        raise ValueError("expected at most 32 distinct integer row hints")
    path = Path(catalog).resolve()
    dyn = DynamicRows(ep, device)
    lo, hi = dyn.bounds
    if any(not lo <= n <= hi for n in rows):
        raise ValueError("row hint outside export bounds")
    libs = []
    audit = []
    k = dyn.kernel
    code = backend_id()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        # One read snapshot across all rows. Only trust catalogs produced by import_campaign.
        db.execute("BEGIN")
        info = db.execute("select info from provenance").fetchall()
        if len(info) != 1:
            raise ValueError("expected one catalog provenance record")
        meta = json.loads(info[0][0])
        if meta.get("format") != "sparkfuse.measured-catalog.v1" or meta.get("compiler") != code:
            raise ValueError("incompatible measured catalog")
        for n in sorted(rows):
            plan = dyn._adapt(n).plan
            key = signature(plan, k.device)
            row = db.execute("select record from schedules where key=?", (key,)).fetchone()
            if row is None:
                audit.append(RowVariant(n, key, None, None))
                continue
            rec = json.loads(row[0])
            if rec["key"] != key or rec["compiler"] != code or not rec["evidence"]:
                raise ValueError("catalog record identity mismatch")
            sched = Schedule(**rec["selected"])
            src = generate(plan, sched)
            lib, _ = build(src, k.device)
            libs.append((n, lib))
            audit.append(RowVariant(n, key, sched, sha256(src.encode()).hexdigest()))
    return CatalogRows(
        _native_dispatch(k, dyn.bounds, specials=libs),
        dyn.bounds,
        k.schedule,
        sha256(k.source.encode()).hexdigest(),
        tuple(audit),
    )
