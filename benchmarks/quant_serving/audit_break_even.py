"""Measured break-even for a four-request burst; include real preparation and guard output equality."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from audit_bundle import ROOT, attest
from audit_catalog import load_inputs, source_files
from heldout import audit, need
from prefix_replay import digest


MARGIN=.03


def crossing(valid):
    """Return the first positive observed prefix after which all remaining observed points pass."""
    if len(valid)!=4 or any(type(v) is not bool for v in valid):raise ValueError('four observed prefixes required')
    return next((k+1 for k in range(4) if all(valid[k:])),None)


def compare(case,baseline,contender):
    need(baseline in ('cold','on-demand') and contender in ('prepared','on-demand') and contender!=baseline,'invalid arms')
    d={(z['repeat'],z['arm']):z for z in case['runs']}; points=[]
    for j in range(4):
        gains=[]; net=[]; same=[]; stable=[]
        for t in range(3):
            a=d[t,baseline];b=d[t,contender];ms=a['cumulative_wall_ms'][j];cand=b['cumulative_wall_ms'][j];net.append(ms-cand);gains.append((ms-cand)/ms)
            stable.append(all(a['requests'][i]['ids']==d[0,baseline]['requests'][i]['ids'] for i in range(j+1)))
            same.append(all(a['requests'][i]['ids']==b['requests'][i]['ids'] for i in range(j+1)))
        timing=min(gains)>=MARGIN
        points.append(dict(completed_requests=j+1,paired_ms_saved=net,median_paired_ms_saved=median(net),paired_fraction_saved=gains,baseline_repeated_exact=all(stable),all_outputs_exact_in_each_pair=all(same),all_repeats_save_margin=timing,timing_and_outputs=timing and all(stable) and all(same)))
    return dict(baseline=baseline,contender=contender,points=points,timing_only_first_sustained_request=crossing([p['all_repeats_save_margin'] for p in points]),output_preserving_first_sustained_request=crossing([p['timing_and_outputs'] for p in points]))


def rebuild(run,ref):
    # Reuse published single-slot source attestation; also bind the independent environment & every byte below.
    validated=load_inputs(run,ref,include_rows=False); parent=source_files(ref);need(validated==parent['rows'],'reference mismatch');fp={}
    def read(label,p):
        data=p.read_bytes();fp[label]=hashlib.sha256(data).hexdigest();return json.loads(data)
    old=read('reference_rows',parent['rows']);old_cp=read('reference_checkpoint',parent['checkpoint']);old_rt=read('reference_runtime',parent['runtime']);rows=read('burst_rows',run/'rows.json');cp=read('burst_checkpoint',run/'checkpoint.json');rt=read('burst_runtime',run/'runtime.json');lock=ROOT/'qwen72b-q4km.lock.json';sig=attest(lock,cp,rt,81,1,128);need(sig==attest(lock,old_cp,old_rt,81,1,128) and cp==old_cp,'different verified source')
    basic=audit(rows,old,set(sig['eog_token_ids'])); parent_ids={digest(p[name]) for p in old for name in ('target_ids','primer_ids')}; need(all(parent_ids.isdisjoint(c['prompt_sha256']) for c in rows),'held-out target seen as reference target or primer')
    cases=[dict(domain=c['domain'],length=c['length'],comparisons=[compare(c,a,b) for a,b in [('cold','on-demand'),('cold','prepared'),('on-demand','prepared')]]) for c in rows]
    return dict(schema='sparkfuse.prefix-burst-break-even.v1',model=cp['model'],revision=cp['revision'],verified_weight_bytes=cp['total_bytes'],runtime=sig,evidence_sha256=fp,minimum_fraction_saved_in_each_repeat=MARGIN,target_requests=sum(len(a['requests']) for c in rows for a in c['runs']),actual_preparation_requests=sum(a['primer'] is not None for c in rows for a in c['runs']),baseline_panel=basic,cases=cases,scope='Serial warm backend, four branches, three matched repetitions. Preparation time is charged. A sustained crossing must meet the 3% margin in every repetition through the fourth observation. The protected result also requires all output token prefixes equal across arms and a repeatable baseline. Null means not demonstrated within four requests, not never. No extrapolation to other branches, natural traffic or longer bursts.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('reference',type=Path);v=p.parse_args();print(json.dumps(rebuild(v.run,v.reference),ensure_ascii=False,indent=2,allow_nan=False))
