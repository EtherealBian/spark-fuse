import random
from unittest import TestCase, mock
from prefix_index import scoped
from slot_policy import SlotRouter

class NoRead:
    def __len__(self): raise AssertionError('foreign prompt length read')
    def __getitem__(self, idx): raise AssertionError('foreign prompt contents read')

class Exact(TestCase):
    def test_cross_scope_and_short_empty(self):
        self.assertIsNone(scoped([('A',(1,)*20)], [('B',NoRead()),('A',(1,)*15)], 16))
        self.assertIsNone(scoped([], [('B',NoRead())], 16))
        self.assertEqual(scoped([('A',(1,)*16),('B',(2,)*17)], [('A',(1,)*19),('C',NoRead()),('B',(2,)*19)],16),[[16,0,0],[0,0,17]])
    def test_oracle_without_lcp_solver(self):
        rng=random.Random(260117)
        for t in (1,2,15,16):
            for rep in range(35):
                a=[(rng.choice('AB'), tuple([0]*rng.randrange(20)+[rng.randrange(4)]*rng.randrange(14))) for _ in range(19)]
                b=list(reversed(a[:8]))+[('A',(0,)*20)]*8+[('C',(0,)*19)]
                got=scoped(a,b,t); expected=[]
                for scope,ids in a:
                    row=[]
                    for owner,toks in b:
                        hit=max((i for i in range(t,min(len(ids),len(toks))+1) if scope==owner and ids[:i]==toks[:i]),default=0)
                        row.append(hit)
                    expected.append(row)
                self.assertEqual(got if got is not None else [[0]*len(b) for _ in a],expected)
    def test_full_api_duplicate_ties_failures_and_busy(self):
        r,s=SlotRouter(64,2048),SlotRouter(64,2048,batch_matcher='scoped16')
        def key(ls):return [(x.slot,x.prefix,x.reuse) for x in ls]
        for d in (0,1,15,16,17,5):
            qs=[('ABA'[i%3],(0,)*d+(100+i%7,)*(2048-d)) for i in range(64)]
            aa,bb=r.acquire_batch(qs),s.acquire_batch(qs);self.assertEqual(key(aa),key(bb));self.assertIsNone(s.acquire_batch(qs))
            for i in reversed(range(64)):
                for rt,ls in ((r,aa),(s,bb)):rt.finish(ls[i],success=i%11!=0)
        x=r.acquire('A',(0,)*2048);y=s.acquire('A',(0,)*2048); self.assertEqual(key([x]),key([y]));self.assertIsNone(s.acquire_batch(qs));self.assertEqual(key(r.acquire_batch(qs[:4])),key(s.acquire_batch(qs[:4])))
