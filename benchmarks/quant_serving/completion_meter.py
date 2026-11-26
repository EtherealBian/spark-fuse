"""Local, content-free accounting for finished llama.cpp native non-streaming requests.

Requires return_tokens=true. Endpoint throughput here is per sum of observed request
latencies, not multi-client goodput, backend decode tok/s, or an API billing quantity.
"""

from dataclasses import dataclass, fields


def unsigned(n):
    return type(n) is int and n >= 0


def require(c, msg):
    if not c:
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Counts:
    requests: int = 0
    delivered: int = 0
    terminal_eog: int = 0
    predicted: int = 0
    proposed: int = 0
    accepted: int = 0
    wall_ns: int = 0
    nonidentical_counters: int = 0

    def __post_init__(self):
        require(all(unsigned(getattr(self, f.name)) for f in fields(self)), "invalid count")
        require(
            self.terminal_eog <= self.requests <= self.delivered <= self.predicted
            and self.nonidentical_counters <= min(self.requests, self.predicted - self.delivered)
            and bool(self.nonidentical_counters) == (self.predicted != self.delivered)
            and self.predicted - self.delivered
            <= self.accepted
            <= min(self.predicted, self.proposed)
            and bool(self.wall_ns) == bool(self.requests)
            and (self.requests > 0 or not any(getattr(self, f.name) for f in fields(self))),
            "inconsistent counts",
        )

    def __add__(self, other):
        if not isinstance(other, Counts):
            return NotImplemented
        return Counts(*(getattr(self, f.name) + getattr(other, f.name) for f in fields(self)))

    def report(self):
        return dict(
            requests=self.requests,
            delivered_tokens_including_eog=self.delivered,
            delivered_non_eog_tokens=self.delivered - self.terminal_eog,
            terminal_eog_tokens=self.terminal_eog,
            backend_predicted=self.predicted,
            backend_minus_delivered=self.predicted - self.delivered,
            requests_with_counter_difference=self.nonidentical_counters,
            proposed=self.proposed,
            accepted=self.accepted,
            proposal_acceptance=self.accepted / self.proposed if self.proposed else None,
            sum_request_wall_ns=self.wall_ns,
            delivered_per_sum_request_second=self.delivered * 1e9 / self.wall_ns
            if self.wall_ns
            else None,
        )


def native(response, *, wall_ns, n_predict, draft_max, vocab_size, eog):
    """Read only returned IDs, stop and counters; never retain prompts, IDs or output text.

    Scope: one pinned native endpoint, EOS or token limit, no stop strings/time limits.
    Reject missing tokens/counters instead of treating them as free or estimated tokens.
    draft_max=0 denotes plain decoding; use the configured n-gram lookahead otherwise.
    """
    require(
        unsigned(wall_ns)
        and wall_ns
        and unsigned(n_predict)
        and n_predict
        and unsigned(draft_max)
        and unsigned(vocab_size)
        and vocab_size,
        "invalid request envelope",
    )
    require(
        isinstance(eog, (list, tuple, set, frozenset))
        and eog
        and all(unsigned(t) and t < vocab_size for t in eog),
        "invalid EOG vocabulary",
    )
    end = frozenset(eog)
    require(
        isinstance(response, dict) and isinstance(response.get("timings"), dict),
        "native timings missing",
    )
    t = response["timings"]
    ids = response.get("tokens")
    pred = t.get("predicted_n")
    require(
        isinstance(ids, list)
        and unsigned(pred)
        and 0 < len(ids) <= pred <= n_predict
        and all(unsigned(i) and i < vocab_size for i in ids),
        "invalid delivered IDs or count",
    )
    if "tokens_predicted" in response:
        require(
            unsigned(response["tokens_predicted"]) and response["tokens_predicted"] == pred,
            "conflicting predicted counters",
        )
    stop = response.get("stop_type")
    require(
        stop in ("eos", "limit")
        and not any(i in end for i in ids[:-1])
        and ((ids[-1] in end) == (stop == "eos"))
        and (stop != "limit" or pred == n_predict),
        "unsupported or inconsistent stop",
    )
    require(("draft_n" in t) == ("draft_n_accepted" in t), "incomplete draft counters")
    proposed = t.get("draft_n", 0)
    ok = t.get("draft_n_accepted", 0)
    extra = pred - len(ids)
    require(
        unsigned(proposed)
        and unsigned(ok)
        and ok <= min(proposed, pred)
        and extra <= min(draft_max, ok)
        and (draft_max > 0 or proposed == 0),
        "draft counters do not reconcile",
    )
    return Counts(1, len(ids), int(stop == "eos"), pred, proposed, ok, wall_ns, int(extra > 0))
