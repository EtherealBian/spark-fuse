"""Explicitly admitted request shapes for one frozen eval model, sharing only model state."""

from contextlib import ExitStack
import torch
from .ir import Unsupported
from .resident import _capture_export, _get, _request


def _key(args):
    if not args or not all(type(t) in (torch.Tensor, torch.nn.Parameter) for t in args):
        raise Unsupported("shape-family requests require positional tensors")
    return tuple((tuple(t.shape), t.dtype) for t in args)


class ShapeFamily:
    """Route fixed shapes without export, compilation, capture or eviction on a request miss."""

    def __init__(self, handles, scopes, metadata):
        self._handles = handles
        self._scopes = scopes
        self.metadata = metadata

    def __call__(self, *args):
        if self._scopes is None:
            raise RuntimeError("closed shape family")
        key = _key(args)
        if key not in self._handles:
            raise Unsupported("shape/dtype not admitted by this family")
        return self._handles[key](*args)

    def __copy__(self):
        raise TypeError("shape families cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("shape families cannot be copied")

    def close(self):
        if self._scopes is not None:
            # Don't pop ExitStack callbacks when the caller is on a foreign stream/thread.
            next(iter(self._handles.values()))._guard()
            self._scopes.close()
            self._handles.clear()
            self._scopes = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@torch.inference_mode()
def capture_family(
    model, requests, *, backend="mixed", partition_options=None, max_static_bytes=1024**3
):
    """Prepare a bounded, explicit list of (example, audit) pairs on one owning stream.

    Each pair is an independent fixed-shape export of the same model. Inputs must be positional
    tuples and both requests must differ in contents. The aggregate cap covers single-copy state
    and every shape's input staging, not independent graph pools, outputs or CUDA workspaces.
    An admission failure closes already captured handles. No partial family is returned.
    """
    if not isinstance(model, torch.nn.Module) or any(m.training for m in model.modules()):
        raise Unsupported("eval model required")
    if not isinstance(requests, (tuple, list)) or not 1 <= len(requests) <= 16:
        raise ValueError("provide between 1 and 16 explicit request pairs")
    if type(max_static_bytes) is not int or max_static_bytes < 1:
        raise ValueError("positive aggregate byte limit required")
    keys = set()
    device = None
    stage = 0
    for pair in requests:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("expected example and audit pair")
        ex, audit = pair
        if (
            not isinstance(ex, tuple)
            or not ex
            or not isinstance(ex[0], torch.Tensor)
            or not ex[0].is_cuda
        ):
            raise Unsupported("CUDA tensor request expected")
        device = device or ex[0].device
        if _request(ex, device) != _request(audit, device):
            raise Unsupported("audit layout mismatch")
        key = _key(ex)
        if key in keys:
            raise ValueError("duplicate shape/dtype key")
        keys.add(key)
        stage += sum(t.nbytes for t in ex)
    if stage > max_static_bytes:
        raise ValueError("aggregate byte limit exceeded by staging")
    handles = {}
    members = []
    shared = None
    state_bytes = None
    with ExitStack() as scopes:
        for ex, audit in requests:
            ep = torch.export.export(model, ex)
            # Pre-reserve the staging of ALL shapes when admitting each member. The remaining
            # allowance is exactly one copy of state plus the current member's input staging.
            cap = max_static_bytes - stage + sum(t.nbytes for t in ex)
            run = scopes.enter_context(
                _capture_export(ep, ex, audit, backend, partition_options, cap, shared)
            )
            handles[_key(ex)] = run
            size = run.metadata["staging_and_state_bytes"] - run.metadata["input_bytes"]
            if shared is None:
                shared = {
                    n.target: _get(run._fn, n.target)
                    for n in run._fn.graph.nodes
                    if n.op == "get_attr"
                }
                state_bytes = size
            elif size != state_bytes:
                raise Unsupported("different state storage size in shape family")
            members.append(
                dict(
                    inputs=[{"shape": list(t.shape), "dtype": str(t.dtype)} for t in ex],
                    input_bytes=run.metadata["input_bytes"],
                    fresh_output_bytes=run.metadata["fresh_output_bytes"],
                )
            )
        meta = dict(
            backend=backend,
            shapes=len(members),
            named_state=len(shared),
            weight_snapshot_bytes=state_bytes,
            input_staging_bytes=stage,
            static_bytes=stage + state_bytes,
            members=members,
        )
        return ShapeFamily(handles, scopes.pop_all(), meta)
