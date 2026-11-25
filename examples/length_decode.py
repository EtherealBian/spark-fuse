"""The explicit contiguous clone fixes otherwise backend-dependent singleton strides."""

import torch
from examples.single_decode import SingleDecode


class CanonicalDecode(SingleDecode):
    def forward(self, q, k, v, mask=None):
        return super().forward(q, k, v, mask).clone(memory_format=torch.contiguous_format)


def export_length_decode(model, example, *, lower=2, upper=2049):
    s = torch.export.Dim("kv_length", min=lower, max=upper)
    dims = ({}, {2: s}, {2: s}) + (({3: s},) if len(example) == 4 else ())
    return torch.export.export(model, tuple(example), dynamic_shapes=dims)
