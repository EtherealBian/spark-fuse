import copy
from pathlib import Path
import unittest
import torch

from device_greedy32 import GreedyState, RecurrentGraph, decode_request
from loop_audit32 import ARMS,LENS,ORACLES,reference,refs,schedule,summarize,digest


def fixture():
    o=refs(Path('reports'));meta=copy.deepcopy(o['resident']);meta['all_cuda_bf16']=True; r=dict(schema='sparkfuse.device-greedy32.v1',meta=meta,oracle_sha256=ORACLES.copy(),batch=1,generated=32,cases=[],experiment_source_sha256={k:digest(Path(__file__).with_name(k)) for k in ('device_greedy32.py','loop_audit32.py','measure_loop32.py')})
    for n in LENS:
        c=dict(length=n,capacity=n+32,capture_ms={k:100. for k in ARMS},checks=[],pairs=[]);r['cases'].append(c)
        for i in (1,2):
            a=reference(o,n,i)
            for k in ARMS:
                c['checks'].append(dict(arm=k,prompt=i,input_ids=a['input_ids'],tokens=list(a['reference']),logits_sha256=a['reference_logits_sha256'],shape=[32,152064],dtype='torch.bfloat16'))
        for j,(i,order) in enumerate(schedule()):
            c['pairs'].append(dict(pair=j,prompt=i,rows=[dict(arm=k,wall_ms=90. if k=='recurrent' else 100.,tokens=list(reference(o,n,i)['reference'])) for k in order]))
    return o,r


class Tiny:
    def __init__(self,cap): self.pos=torch.arange(cap)
    def forward(self,x,p):
        y=(x.sum()+p.sum()+1).remainder(11).view(1,1);return torch.zeros((1,11)).scatter_(1,y,1.)
    decode=forward


class Replay:
    def __init__(self,r): self.r=r
    def replay(self): self.r.buf_y=self.r.state.advance(self.r.forward)


class Tests(unittest.TestCase):
    def test_device_recurrence_with_real_cpu_tensors(self):
        # A CPU step emulator, not a CUDA Graph or model performance test.
        for budget in (2,5,32):
            ref=Tiny(6+budget-1);g=object.__new__(RecurrentGraph);g.pos=ref.pos;g.forward=ref.forward;g.state=GreedyState(budget,'cpu');g.graph=Replay(g)
            ptr=[x.data_ptr() for x in (g.state.ids,g.state.pos,g.state.dst,g.state.token)]
            for seed in (0,1,0):
                inp=(torch.arange(6)+seed).view(1,-1);a,sa=decode_request(ref,inp,budget,record=True);b,sb=decode_request(g,inp,budget,record=True)
                self.assertEqual(a,b);self.assertTrue(torch.equal(sa,sb));self.assertEqual(g.state.pos.item(),6+budget-1);self.assertEqual(g.state.dst.item(),budget);self.assertEqual(ptr,[x.data_ptr() for x in (g.state.ids,g.state.pos,g.state.dst,g.state.token)])
            with self.assertRaises(ValueError):decode_request(g,inp,budget+1)

    def test_seed_rejects_without_writing(self):
        s=GreedyState(4,'cpu');x=torch.tensor([[7]]);s.seed(x,2,5)
        for y,n,cap in [(x.float(),2,5),(x.squeeze(1),2,5),(x,3,5),(x,0,8)]:
            with self.assertRaises(ValueError):s.seed(y,n,cap)
            self.assertEqual((s.token.item(),s.pos.item(),s.dst.item()),(7,2,1))

    def test_rebuild_and_reject(self):
        o,r=fixture();self.assertTrue(summarize(r,o)['recommend_for_tested_geometry'])
        edits=[lambda z:z['cases'][0]['checks'][0].update(logits_sha256='0'*64),lambda z:z['cases'][0]['checks'][0]['tokens'].__setitem__(0,-1),lambda z:z['cases'][1]['pairs'][0]['rows'].reverse(),lambda z:z['cases'][0]['pairs'][3]['rows'][0].update(wall_ms=float('nan')),lambda z:z['cases'][0]['pairs'].pop(),lambda z:z['meta'].update(model_storage_after=3),lambda z:z['cases'][1]['pairs'][0]['rows'][0]['tokens'].__setitem__(0,-1),lambda z:z['oracle_sha256'].update(eager='0'*64),lambda z:z['experiment_source_sha256'].update(**{'loop_audit32.py':'0'*64})]
        for edit in edits:
            z=copy.deepcopy(r);edit(z)
            with self.assertRaises(ValueError):summarize(z,o)

    def test_one_order_or_shape_gain_is_not_recommended(self):
        o,r=fixture()
        for q in r['cases'][1]['pairs']:
            if q['rows'][0]['arm']=='queued':q['rows'][1]['wall_ms']=105.
        s=summarize(r,o);self.assertTrue(s['cases'][0]['passes']);self.assertFalse(s['recommend_for_tested_geometry'])

if __name__=='__main__':unittest.main()
