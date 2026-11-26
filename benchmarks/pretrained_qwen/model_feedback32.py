"""Fixed-budget batch-one greedy decoding. No streaming or data-dependent EOS exit."""
import torch
from run import Runner, sync


class GreedyState:
    """Fixed-address recurrence; host controls only the number of graph launches."""
    def __init__(self, steps, device):
        if type(steps) is not int or steps < 2:
            raise ValueError('expected a budget of at least two tokens')
        self.ids = torch.empty(steps, device=device, dtype=torch.long)
        self.token = self.ids.new_empty((1, 1))
        self.pos = self.ids.new_empty((1,))
        self.dst = self.ids.new_empty((1,))

    def seed(self, first, n, capacity):
        if (first.shape != (1, 1) or first.dtype != self.ids.dtype or first.device != self.ids.device
                or type(n) is not int or n < 1 or n + self.ids.numel() - 1 > capacity):
            raise ValueError('invalid seed or insufficient KV capacity')
        self.token.copy_(first)
        self.ids[:1].copy_(first.view(-1))
        self.pos.fill_(n)
        self.dst.fill_(1)

    def advance(self, forward):
        scores = forward(self.token, self.pos)
        self.token.copy_(scores.argmax(-1, keepdim=True))
        self.ids.index_copy_(0, self.dst, self.token.view(-1))
        self.pos.add_(1)
        self.dst.add_(1)
        return scores


class NativeState(GreedyState):
    """The token/KV position advance is recorded inside exactly the same model graph."""
    def __init__(self,steps,device):
        from vocab_feedback import Feedback
        self.feedback=Feedback(152064,steps,device)
        self.ids,self.token,self.pos,self.dst=self.feedback.history,self.feedback.token,self.feedback.pos,self.feedback.dst
        self.bound=None

    def advance(self,forward):
        scores=forward(self.token,self.pos)
        # Hold the most recently bound row and tensors alive, including the actual captured row.
        self.bound=self.feedback.bind(scores)
        self.bound()
        return scores


class RecurrentGraph(Runner):
    state_cls=GreedyState
    def setup_loop(self, sample, steps):
        self.cache.reset()
        n = sample.size(1)
        first = self.forward(sample, self.pos[:n]).argmax(-1, keepdim=True)
        self.state = self.state_cls(steps, sample.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.state.seed(first, n, self.pos.numel())
                self.state.advance(self.forward)
        torch.cuda.current_stream().wait_stream(side)
        self.state.seed(first, n, self.pos.numel())
        sync()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.buf_y = self.state.advance(self.forward)
        sync()
        self.cache.reset()


class NativeGraph(RecurrentGraph):
    state_cls=NativeState


def decode_request(r, ids, steps, record=False):
    # Callers reset KV before measuring. The two arms expose the same result boundary.
    n = ids.size(1)
    if not 2 <= steps <= r.pos.numel() - n + 1:
        raise ValueError('bad decode budget')
    if isinstance(r, RecurrentGraph) and r.state.ids.numel() != steps:
        raise ValueError('recording owns a different token budget')
    s = r.forward(ids, r.pos[:n])
    x = s.argmax(-1, keepdim=True)
    scores = [s.clone()] if record else []
    if isinstance(r, RecurrentGraph):
        r.state.seed(x, n, r.pos.numel())
        for _ in range(1, steps):
            r.graph.replay()
            if record:
                scores.append(r.buf_y.clone())
        out = r.state.ids
    else:
        tokens = [x]
        for j in range(1, steps):
            s = r.decode(x, r.pos[n + j - 1:n + j]); x = s.argmax(-1, keepdim=True)
            tokens.append(x)
            if record:
                scores.append(s.clone())
        out = torch.cat(tokens, dim=1).flatten()
    # One blocking read, after every dependent token has been enqueued (both arms).
    return out.cpu().tolist(), torch.cat(scores, dim=0).cpu() if record else None
