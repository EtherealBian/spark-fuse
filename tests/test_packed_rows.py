"""Bounded native launches never specialize by observed shape or silently drop FX outputs."""

from unittest.mock import patch
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from sparkfuse.ir import Unsupported
from sparkfuse.packed_rows import PackedRows


class Gate(nn.Module):
    def __init__(self, form="single", reverse=False):
        super().__init__()
        self.form = form
        self.reverse = reverse

    def forward(self, x):
        a, b = x.split([x.shape[1] // 2] * 2, dim=-1)
        if self.reverse:
            a, b = b, a
        s = F.silu(a)
        y = b * s
        if self.form == "tuple":
            return (y,)
        if self.form == "visible":
            return y, s
        if self.form == "alias":
            return y, a
        if self.form == "reorder":
            return s, y
        if self.form == "nested":
            return {"gate": y}
        if self.form == "extra":
            return y + 1
        return y


def export(m, x, low=1, high=67):
    return torch.export.export(
        m, (x,), dynamic_shapes=({0: torch.export.Dim("tokens", min=low, max=high)},)
    )


@pytest.mark.parametrize(
    "dtype,width,schedule,form,reverse",
    [
        (torch.bfloat16, 32, "adaptive", "visible", False),
        (torch.float16, 7, "adaptive", "single", True),
        (torch.bfloat16, 14, "adaptive", "tuple", True),
        (torch.float16, 16, "scalar", "visible", True),
    ],
)
@torch.inference_mode()
def test_bounded_packed_rows_dispatch(dtype, width, schedule, form, reverse):
    m = Gate(form, reverse)
    example = torch.randn(7, 2 * width, device="cuda", dtype=dtype)
    r = PackedRows(export(m, example), schedule=schedule)
    assert r.record["gate_half"] == int(reverse) and r.record["visible_activation"] == (
        form == "visible"
    )
    with patch(
        "sparkfuse.index_program.build", side_effect=AssertionError("unexpected recompilation")
    ):
        for rows in (1, 2, 5, 33, 67, 2):
            for off in (0, 1, 2, 4):
                base = torch.randn(rows * 2 * width + off, device="cuda", dtype=dtype)
                x = base[off:].view(rows, 2 * width)
                before = base.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    out = r(x)
                    ref = m(x)
                torch.cuda.current_stream().wait_stream(stream)
                assert type(out) is type(ref)
                outputs = out if isinstance(out, tuple) else (out,)
                inputs = ref if isinstance(ref, tuple) else (ref,)
                assert len({t.data_ptr() for t in (*outputs, x)}) == len(outputs) + 1
                for a, b in zip(outputs, inputs, strict=True):
                    assert a.stride() == b.stride() and torch.equal(
                        a.view(torch.int16), b.view(torch.int16)
                    )
                assert torch.equal(before, base)


def test_native_packed_row_guards_and_replay():
    m = Gate()
    x = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)
    r = PackedRows(export(m, x, 3, 21))
    for invalid in [
        x[:2],
        x[:0],
        x.repeat(4, 1),
        x.T.contiguous(),
        x[:, ::2],
        x.float(),
        x.cpu(),
        x.view(1, 7, 32),
        x.detach().requires_grad_(),
    ]:
        with pytest.raises(ValueError):
            r(invalid)
    x = x.detach()
    wrong_stride = torch.randn(32, 7, device="cuda", dtype=x.dtype).T
    with pytest.raises(ValueError):
        r(wrong_stride)
    with pytest.raises((TypeError, ValueError)):
        r(x, x)
    with pytest.raises(TypeError):
        r(x=x)
    a = r(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = r(x)
    x.zero_()
    graph.replay()
    assert torch.count_nonzero(captured) == 0 and torch.count_nonzero(a) > 0


@pytest.mark.parametrize("form", ["alias", "reorder", "nested", "extra"])
def test_no_silent_elision_or_output_reordering(form):
    x = torch.randn(7, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(Unsupported):
        PackedRows(export(Gate(form), x))


def test_reject_noncanonical_domains():
    x = torch.randn(7, 32, device="cuda", dtype=torch.float16)
    m = Gate()
    with pytest.raises(Unsupported, match="symbol"):
        PackedRows(torch.export.export(m, (x,)))
    with pytest.raises(Unsupported, match="32-bit"):
        PackedRows(export(m, x, 1, 2**28))
    with pytest.raises(Unsupported, match="CUDA export"):
        PackedRows(export(m, x.cpu()))
    with pytest.raises(Unsupported):
        PackedRows(export(m, x.float()))
    with pytest.raises(ValueError, match="schedule"):
        PackedRows(None, schedule="typo")
