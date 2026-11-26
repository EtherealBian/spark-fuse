import math, struct
from unittest import TestCase

def rank(x):
    mag=x&32767
    return 65536 if mag>32640 else 32768 if mag==0 else (x^65535 if x>=32768 else x^32768)

def asfloat(x):return struct.unpack('>f',struct.pack('>I',x<<16))[0]

class Order(TestCase):
    def test_all_bfloat16_bitpatterns(self):
        ordered=sorted(range(65536),key=rank);last=None
        for x in ordered:
            v=asfloat(x);key=(math.isnan(v),0.0 if math.isnan(v) else v)
            if last is not None:
                self.assertLessEqual(last,key);self.assertEqual(last==key,prev==rank(x))
            last,prev=key,rank(x)
    def test_combined_key_first_ties(self):
        for row,idx in [([32768,0],0),([0xff80,0xff80],0),([0,0x7fff,0x7fc0],1),([0xffc1,0x7f80],0),([0x8001,0x0001,0x8000],1)]:
            best=max((rank(x)<<32)|(0xffffffff-i) for i,x in enumerate(row));self.assertEqual(0xffffffff-(best&0xffffffff),idx)
