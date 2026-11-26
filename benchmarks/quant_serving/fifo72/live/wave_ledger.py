"""Decompose each measured wall interval; waiter CPU overlaps it and is never added."""
import json
import sys
from pathlib import Path
from statistics import mean, median
from live72_review import bundle

PHASES = ('blocker', 'handoff', 'after_grant')


def intervals(row):
    arrival, release, grant = row['arrival_ns'], row['blocker']['release_start_ns'], row['grant_ns']
    stops = [t['end_ns'] for t in row['targets']]
    if len(stops) != 2 or any(type(t) is not int or t <= 0 for t in (arrival, release, grant, *stops)):
        raise ValueError('expected integer clocks for both HTTP responses')
    if not (arrival < release <= grant < min(stops)):
        raise ValueError('non-nested wall boundary')
    return dict(blocker=release-arrival, handoff=grant-release, after_grant=max(stops)-grant, total=max(stops)-arrival)


def ledger(run, ref):
    # Never compute an attractive breakdown from rows rejected by the live replay.
    audited = bundle(run, ref)
    if audited != json.loads((run/'summary.json').read_bytes()):
        raise ValueError('stale upstream summary')
    rows = json.loads((run/'rows.json').read_bytes())
    pairs, groups = [], {}
    for p, checked in zip(rows['pairs'], audited['measurement']['pairs'], strict=True):
        first = p['arms'][0]['mode']
        arms = {a['mode']: intervals(a['row']) for a in p['arms']}
        for mode, costs in arms.items():
            if (sum(costs[k] for k in PHASES) != costs['total'] or costs['total'] != checked['arms'][mode]['arrival_to_last_ns']):
                raise ValueError('timestamps disagree with live audit')
        delta = {k: arms['poll-1ms'][k] - arms['fifo'][k] for k in (*PHASES, 'total')}
        if sum(delta[k] for k in PHASES) != delta['total']:
            raise ValueError('paired intervals do not add up')
        groups.setdefault(first, []).append(delta)
        pairs.append(dict(case=p['case'], repeat=p['repeat'], first=first, saving_ns=delta))
    return dict(schema=1, rows_sha256=audited['evidence_sha256']['rows.json'],
                by_first={first: dict(n=len(g), mean_paired_saving_ms={k: mean(z[k] for z in g)/1e6 for k in (*PHASES,'total')},
                                      median_paired_total_saving_ms=median(z['total'] for z in g)/1e6) for first,g in groups.items()},
                pairs=pairs, scope='Retrospective wall-clock accounting. Positive = polling minus FIFO; only means are additive. No assignment of causality or GPU time.')


if __name__ == '__main__':
    print(json.dumps(ledger(*map(Path,sys.argv[1:])), indent=2, allow_nan=False))
