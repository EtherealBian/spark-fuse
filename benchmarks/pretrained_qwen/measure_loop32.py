"""Compare eager host enqueue with graph-owned token/position updates on full resident Qwen."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from time import monotonic, perf_counter_ns, sleep

import accelerate
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from checkpoint32b import verify, MODEL, REV, PARAMETERS
from device_greedy32 import RecurrentGraph, decode_request
from loop_audit32 import ARMS, LENS, STEPS, ORACLES, demand, digest, reference, refs, schedule, summarize
from resident import resident
from run import Runner, sync, timed
from scale7b import inputs, storage_bytes
from scale32b import SOURCES
import scale32b


def save(path, value):
    tmp=path.with_name(path.name+'.partial');tmp.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n');os.replace(tmp,path)


def panel(m,tok,n,oracle,raw,out):
    device=torch.device('cuda'); prompts=[inputs(tok,n,i) for i in range(3)]
    for i in (1,2): demand(prompts[i].tolist()[0]==reference(oracle,n,i)['input_ids'],'tokenizer differs')
    runners={}; c=dict(length=n,capacity=n+STEPS,capture_ms={},checks=[],pairs=[]);raw['cases'].append(c)
    for arm,cls in zip(ARMS,(Runner,RecurrentGraph),strict=True):
        runner=cls(m,m.config,n+STEPS,device,torch.bfloat16);runners[arm]=runner
        _,ms=timed(lambda:runner.setup_replay(prompts[0]) if arm=='queued' else runner.setup_loop(prompts[0],STEPS)); c['capture_ms'][arm]=ms
        runner.cache.reset(); sync(); decode_request(runner,prompts[0],STEPS)
    for i in (1,2):
        for arm in ARMS:
            r=runners[arm]; r.cache.reset(); sync(); tokens,scores=decode_request(r,prompts[i],STEPS,record=True)
            row=dict(arm=arm,prompt=i,input_ids=prompts[i].tolist()[0],tokens=tokens,logits_sha256=hashlib.sha256(scores.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),shape=list(scores.shape),dtype=str(scores.dtype));c['checks'].append(row);save(out/'raw.json',raw)
            ref=reference(oracle,n,i);demand(row['shape']==[STEPS,152064] and row['dtype']=='torch.bfloat16' and tokens==ref['reference'] and row['logits_sha256']==ref['reference_logits_sha256'],'full-model correctness gate failed; retained diagnostic');del scores
    for j,(i,order) in enumerate(schedule()):
        pair=dict(pair=j,prompt=i,rows=[]);c['pairs'].append(pair)
        for arm in order:
            r=runners[arm]; r.cache.reset(); sync(); start=perf_counter_ns(); ids,_=decode_request(r,prompts[i],STEPS); end=perf_counter_ns()
            pair['rows'].append(dict(arm=arm,wall_ms=(end-start)/1e6,tokens=ids));save(out/'raw.json',raw); demand(ids==reference(oracle,n,i)['reference'],'timed output changed')


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--weights',type=Path,required=True);parser.add_argument('--reports',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    oracle=refs(args.reports); old=oracle['resident']; verify(args.weights)
    meta=dict(model=MODEL,revision=REV,parameters=PARAMETERS,device=torch.cuda.get_device_name(),torch=str(torch.__version__),transformers=transformers.__version__,cuda=torch.version.cuda,accelerate=accelerate.__version__,attention='sdpa',dtype='bfloat16',source_sha256={k:digest(Path(scale32b.__file__).with_name(k)) for k in SOURCES})
    demand(all(meta[k]==old[k] for k in meta),'published runtime mismatch')
    torch.manual_seed(617);torch.set_num_threads(4); opts=dict(local_files_only=True,trust_remote_code=False);tok=AutoTokenizer.from_pretrained(args.weights,**opts)
    m=AutoModelForCausalLM.from_pretrained(args.weights,**opts,use_safetensors=True,device_map={'':0},torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval().requires_grad_(False)
    meta['layers']=m.config.num_hidden_layers; meta['all_cuda_bf16']=all(p.is_cuda and p.dtype==torch.bfloat16 for p in m.parameters()) and all(b.is_cuda for b in m.buffers()); demand(meta['layers']==64 and meta['all_cuda_bf16'] and sum(p.numel() for p in m.parameters())==PARAMETERS,'wrong architecture or placement')
    meta['model_storage_before']=storage_bytes(m);meta['gate_source_sha256']=resident(m);sync();meta['model_storage_after']=storage_bytes(m);demand(all(meta[k]==old[k] for k in ('model_storage_before','model_storage_after','gate_source_sha256')),'resident bank changed')
    raw=dict(schema='sparkfuse.device-greedy32.v1',oracle_sha256=ORACLES,meta=meta,batch=1,generated=STEPS,cases=[],experiment_source_sha256={k:digest(Path(__file__).with_name(k)) for k in ('device_greedy32.py','loop_audit32.py','measure_loop32.py')}); save(args.out/'raw.json',raw);(args.out/'requests.started').touch()
    for n in LENS:
        gc.collect();torch.cuda.empty_cache();sync();panel(m,tok,n,oracle,raw,args.out)
    summary=summarize(raw,oracle);save(args.out/'timing.json',summary);(args.out/'requests.finished').touch()
    if os.environ.get('SPARKFUSE_GUARD_HANDSHAKE')=='1':
        deadline=monotonic()+12
        while monotonic()<deadline:
            p=args.out/'memory-guard.json'
            if p.is_file() and json.loads(p.read_bytes())['reason']=='workload-finished': break
            sleep(.2)
        else: raise RuntimeError('guard did not acknowledge workload end')


if __name__=='__main__': main()
