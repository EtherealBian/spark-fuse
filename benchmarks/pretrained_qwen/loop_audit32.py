"""CPU-only reconstruction against two digest-pinned, already published 32B reports."""
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ARMS = ('queued', 'recurrent')
LENS = (128, 2048)
STEPS = 32
ORACLES = {'eager':'c271a47061af2aced3c41520e7a3b0c772e9706a55387b3786949099b3c978d3',
           'resident':'246c0657ed97aabd8d2d55faf01f95786a73cc8ecf64c8ad200b59ad49cf12c4'}


def demand(x, message):
    if not x:
        raise ValueError(message)


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def refs(folder):
    obj = {}
    for k, sha in ORACLES.items():
        p = folder / f'qwen32b-{k}.json'; demand(digest(p) == sha, 'oracle content changed'); obj[k] = json.loads(p.read_bytes())
    return obj


def schedule():
    # Cross prompt and first arm independently (eight paired windows per length).
    return [(i, ARMS if (r + i) % 2 else ARMS[::-1]) for r in range(4)
            for i in ((1,2) if r % 2 == 0 else (2,1))]


def reference(o,n,i):
    a,b = [next(x for x in next(c for c in o[k]['cases'] if c['length']==n)['checks'] if x['prompt']==i) for k in ORACLES]
    demand(a['input_ids']==b['input_ids'] and a['free']==b['free']==a['reference']==b['reference'] and a['reference_logits_sha256']==b['logits_sha256']==b['reference_logits_sha256'], 'unmatched published references')
    return a


def positive(x):
    return type(x) in (float,int) and math.isfinite(x) and x > 0


def summarize(raw,o):
    expected_source={k:digest(Path(__file__).with_name(k)) for k in ('device_greedy32.py','loop_audit32.py','measure_loop32.py')}
    demand(raw['experiment_source_sha256']==expected_source,'experiment source mismatch')
    resident = o['resident']; meta = raw['meta']; demand(raw['schema']=='sparkfuse.device-greedy32.v1' and raw['oracle_sha256']==ORACLES,'identity')
    for k in ('model','revision','parameters','device','torch','transformers','cuda','accelerate','layers','attention','dtype','model_storage_before','model_storage_after','gate_source_sha256','source_sha256'):
        demand(meta[k]==resident[k], 'runtime/source/checkpoint mismatch: '+k)
    demand(meta['all_cuda_bf16'] is True and raw['batch']==1 and raw['generated']==32,'backend placement or work')
    demand([v['length'] for v in raw['cases']]==list(LENS),'lengths')
    summary=[]
    for c in raw['cases']:
        n=c['length']; expected={i:reference(o,n,i) for i in (1,2)}
        demand(c['capacity']==n+STEPS and set(c['capture_ms'])==set(ARMS) and all(positive(t) for t in c['capture_ms'].values()),'graph setup')
        demand([(z['arm'],z['prompt']) for z in c['checks']]==[(k,i) for i in (1,2) for k in ARMS],'missing/duplicated diagnostic')
        for z in c['checks']:
            a=expected[z['prompt']]; demand(z['input_ids']==a['input_ids'] and z['tokens']==a['reference'] and z['logits_sha256']==a['reference_logits_sha256'] and z['shape']==[32,152064] and z['dtype']=='torch.bfloat16','full-vocab oracle or free token mismatch')
        rows=c['pairs']; demand(len(rows)==len(schedule()),'pair count'); splits={arm:[] for arm in ARMS}; prompts={i:[] for i in (1,2)}; vals={arm:[] for arm in ARMS}
        for j,((i,order),pair) in enumerate(zip(schedule(),rows,strict=True)):
            demand(pair['pair']==j and pair['prompt']==i and tuple(z['arm'] for z in pair['rows'])==order,'pair order or prompt')
            v={}
            for z in pair['rows']:
                demand(z['tokens']==expected[i]['reference'] and positive(z['wall_ms']),'invalid timed request'); v[z['arm']]=z['wall_ms']; vals[z['arm']].append(z['wall_ms'])
            saving=1-v['recurrent']/v['queued']; splits[order[0]].append(saving); prompts[i].append(saving)
        gain={k:median(v) for k,v in splits.items()}; by_prompt={str(k):median(v) for k,v in prompts.items()}
        summary.append(dict(length=n,pairs=len(rows),warm_wall_median_ms={k:median(v) for k,v in vals.items()},paired_saving_by_first=gain,paired_saving_by_prompt=by_prompt,passes=min(gain.values())>=.03 and min(by_prompt.values())>=.03))
    return dict(schema='sparkfuse.device-greedy32.summary.v1',full_logit_checks=8,timed_requests=32,cases=summary,recommend_for_tested_geometry=all(c['passes'] for c in summary),boundary='Fixed-budget warm batch-one request: prefill to 32 host-visible IDs. No token streaming, early EOS, queue wait, or TTFT measurement.')

if __name__=='__main__':
    import sys
    print(json.dumps(summarize(json.loads(Path(sys.argv[1]).read_bytes()),refs(Path(sys.argv[2]))),indent=2,allow_nan=False))
