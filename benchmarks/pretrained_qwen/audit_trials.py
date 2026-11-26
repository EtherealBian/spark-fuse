"""Replay matched fixed-address feedback-only submission records; no CUDA dependency."""
import hashlib,json,math,sys
from pathlib import Path
from statistics import median
WIDTHS=(32768,152064); BANKS=(1,16); STEPS=32;REPS=12;ARMS=('torch','native')
FILES=('vocab_feedback.cu','vocab_feedback.py','audit_trials.py','measure_feedback.py')
def hashes():return {k:hashlib.sha256(Path(__file__).with_name(k).read_bytes()).hexdigest() for k in FILES}
def need(ok,reason):
    if not ok:raise ValueError(reason)
def order(w,b,rep):return ARMS if (WIDTHS.index(w)+BANKS.index(b)+rep)%2==0 else ARMS[::-1]
def ids(w,b):return [((i%b)*9319+817)%w for i in range(STEPS)]
def audit(raw):
    need(raw['schema']=='sparkfuse.feedback-micro.v1' and raw['source']==hashes() and raw['device']=='NVIDIA GB10' and raw['cc']==[12,1] and raw['cuda'] and raw['torch'],'wrong implementation/device');need(raw['edge_checks']==['scalar','tile-minus-one','tile-exact','tile-plus-one','signed-zero','infinity','nan-cross-tile','tie-cross-tile','offset','bad-bind','overflow','reset-replay'],'missing device checks')
    out=[]; need(len(raw['trials'])==len(WIDTHS)*len(BANKS)*REPS,'pair count');rows=iter(raw['trials'])
    for w in WIDTHS:
        for b in BANKS:
            times={a:[] for a in ARMS};gains={a:[] for a in ARMS}
            for j in range(REPS):
                x=next(rows);seq=order(w,b,j);need((x['width'],x['bank'],x['rep'])==(w,b,j) and tuple(r['arm'] for r in x['runs'])==seq,'bad geometry or arm order');v={}
                for r in x['runs']:
                    a=r['arm'];us=r['graph_us'];need(type(us) in (int,float) and math.isfinite(us) and us>0,'invalid CUDA event time');need(all(type(i) is int for i in r['history']) and r['history']==ids(w,b) and r['pos']==100+STEPS and r['dst']==STEPS and r['token']==ids(w,b)[-1],'mismatched recurrence');v[a]=us;times[a].append(us/STEPS)
                gains[seq[0]].append(1-v['native']/v['torch'])
            saving={k:median(v) for k,v in gains.items()};out.append(dict(width=w,logit_rows=b,paired_reps=REPS,device_us_per_feedback={k:median(v) for k,v in times.items()},paired_fraction_saved_by_first=saving,passed=min(saving.values())>=.08))
    return dict(scope='Isolated fixed-address CUDA graph. Device events bracket 32 feedback operations on predetermined BF16 scores; includes state writes, not logits production, model inference, host request time or streaming. Hardware pilot only; not a default or an end-to-end speed claim.',matched_pairs=len(raw['trials']),cases=out,eligible_for_model_experiment=all(x['passed'] for x in out))
if __name__=='__main__': print(json.dumps(audit(json.loads(Path(sys.argv[1]).read_bytes())),indent=2,allow_nan=False))
