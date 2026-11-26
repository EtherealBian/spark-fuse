"""Owned batch-one fixed-address BF16 greedy feedback; enqueue on the current, caller-ordered stream."""
import ctypes as C
from pathlib import Path
import torch
from sparkfuse.runtime import build

class Feedback:
    def __init__(self, width, capacity, device):
        if type(width) is not int or not 1<=width<=1048576 or type(capacity) is not int or not 1<=capacity<=1048576:
            raise ValueError('unsupported vocabulary or output capacity')
        dev=torch.device(device)
        if dev.type!='cuda': raise ValueError('CUDA required')
        self.device=torch.device('cuda',torch.cuda.current_device() if dev.index is None else dev.index)
        self.lib,_=build(Path(__file__).with_name('vocab_feedback.cu').read_text(),self.device)
        self.lib.step.argtypes=[C.POINTER(C.c_void_p),C.c_int,C.c_int,C.c_void_p]; self.lib.step.restype=C.c_int
        self.width,self.capacity=width,capacity
        def alloc(n):return torch.empty(n,device=self.device,dtype=torch.int64)
        self._scratch=alloc((width+511)//512);self.token=alloc((1,1));self.pos=alloc(1);self.dst=alloc(1);self.history=alloc(capacity)
        self.reset(0)
    def reset(self,position):
        if type(position) is not int or not 0<=position<=2**63-1-self.capacity:raise ValueError('position out of bounds')
        self.token.fill_(-1);self.pos.fill_(position);self.dst.zero_();self.history.fill_(-1)
    def bind(self,scores):
        if not isinstance(scores,torch.Tensor) or scores.device!=self.device or scores.dtype!=torch.bfloat16 or scores.shape!=(1,self.width) or scores.layout!=torch.strided or not scores.is_contiguous() or scores.is_neg() or scores.is_conj() or scores.requires_grad:
            raise ValueError('expected one contiguous inference-only CUDA BF16 vocabulary row')
        args=(scores,self._scratch,self.token,self.pos,self.dst,self.history)
        # Attribute reassignment is unsupported; disallow storage aliases including dtype-reinterpreted logits.
        if len({a.untyped_storage().data_ptr() for a in args})!=len(args):raise ValueError('aliased state or logits')
        for t,shape in zip(args[1:],(((self.width+511)//512,),(1,1),(1,),(1,),(self.capacity,)),strict=True):
            if t.device!=self.device or t.dtype!=torch.int64 or tuple(t.shape)!=shape or not t.is_contiguous() or t.is_neg() or t.is_conj():raise ValueError('invalid owned scratch/state')
        ptrs=(C.c_void_p*len(args))(*(t.data_ptr() for t in args))
        def run():
            with torch.cuda.device(self.device):
                stream=torch.cuda.current_stream(self.device)
                if self.lib.step(ptrs,self.width,self.capacity,stream.cuda_stream): raise RuntimeError('CUDA feedback launch failed')
                for t in args:t.record_stream(stream)
        return run
