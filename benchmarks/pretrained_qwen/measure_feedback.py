import json,sys
from pathlib import Path
import torch
from vocab_feedback import Feedback
from audit_trials import ARMS,BANKS,REPS,STEPS,WIDTHS,audit,hashes,ids,need,order

def snap(s):return dict(history=s.history.tolist(),token=s.token.item(),pos=s.pos.item(),dst=s.dst.item())
def edges():
    ok=[]
    for label,w,edits,expected in [('scalar',1,[(0,0xff80)],0),('tile-minus-one',511,[(510,0x3f80)],510),('tile-exact',512,[(400,0x7f80)],400),('tile-plus-one',513,[(512,0x7f80)],512),('signed-zero',520,[(14,0x8000),(16,0)],14),('infinity',515,[(490,0x7f80),(513,0x7f80)],490),('nan-cross-tile',1101,[(515,0x7f81),(1090,0xffc1),(1,0x7f80)],515),('tie-cross-tile',2049,[(789,0x3f80),(1704,0x3f80)],789),('offset',1503,[(1500,0x3f80)],1500)]:
        # Leave one-scalar storage offset including at the native 512-element boundary.
        a=torch.full((w+1,),0xff80,dtype=torch.int32);a[[i+1 for i,_ in edits]]=torch.tensor([x for _,x in edits],dtype=torch.int32); q=a.to(torch.uint16).view(torch.bfloat16).cuda()[1:].view(1,w)
        s=Feedback(w,1,'cuda');call=s.bind(q);call(); need(snap(s)==dict(history=[expected],token=expected,pos=1,dst=1) and q.argmax().item()==expected,label);ok.append(label)
    for q in (torch.zeros(1,1503),torch.zeros(1,1503,device='cuda'),torch.zeros(1,3006,device='cuda',dtype=torch.bfloat16)[:,::2],s.history.view(torch.bfloat16).view(1,-1)):
        try:s.bind(q)
        except ValueError:pass
        else:raise ValueError('invalid binding accepted')
    ok.append('bad-bind');call();need(snap(s)==dict(history=[1500],token=1500,pos=1,dst=-1),'overflow state corrupted');ok.append('overflow')
    s.reset(2);stream=torch.cuda.Stream();g=torch.cuda.CUDAGraph();torch.cuda.synchronize()
    with torch.cuda.graph(g,stream=stream):call()
    s.reset(9);g.replay();need(snap(s)==dict(history=[1500],token=1500,pos=10,dst=1),'reset replay');ok.append('reset-replay');return ok

def panel(raw,w,b):
    gen=torch.Generator(device='cpu').manual_seed(4711+w+b);v=torch.randn(b,w,generator=gen,dtype=torch.float32).clamp_(-5,5).to(torch.bfloat16);expected=ids(w,b)
    for j in range(b):v[j,expected[j]]=10
    scores=v.cuda(); states={};graphs={};stream=torch.cuda.Stream(); holder=[]
    for arm in ARMS:
        s=Feedback(w,STEPS,'cuda');states[arm]=s;fns=[]
        for j in range(b):
            row=scores[j:j+1];
            if arm=='native':fns.append(s.bind(row))
            else:
                def fallback(x=row,state=s):
                    state.token.copy_(x.argmax(-1,keepdim=True));state.history.index_copy_(0,state.dst,state.token.view(-1));state.pos.add_(1);state.dst.add_(1)
                fns.append(fallback)
        holder.extend(fns);s.reset(100);torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            for fn in fns:fn()
        torch.cuda.synchronize();s.reset(100);torch.cuda.synchronize();graphs[arm]=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[arm],stream=stream):
            for i in range(STEPS):fns[i%b]()
    for k in range(-2,REPS):
        rows=[]
        for a in order(w,b,k):
            s=states[a];s.reset(100);start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2));start.record();graphs[a].replay();end.record();end.synchronize();row=dict(arm=a,graph_us=1000*start.elapsed_time(end),**snap(s));need(row['history']==expected and row['token']==expected[-1] and row['pos']==100+STEPS and row['dst']==STEPS,'wrong measured output');rows.append(row)
        if k>=0:raw['trials'].append(dict(rep=k,width=w,bank=b,runs=rows))

@torch.inference_mode()
def main(root):
    need(torch.cuda.is_available() and torch.cuda.get_device_name()=='NVIDIA GB10','requires Spark');raw=dict(schema='sparkfuse.feedback-micro.v1',device=torch.cuda.get_device_name(),cc=list(torch.cuda.get_device_capability()),cuda=torch.version.cuda,torch=str(torch.__version__),source=hashes(),edge_checks=edges(),trials=[])
    for w in WIDTHS:
        for b in BANKS:panel(raw,w,b)
    report=audit(raw)
    for name,data in [('raw',raw),('summary',report)]:
        with (root/(name+'.json')).open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__':main(Path(sys.argv[1]))
