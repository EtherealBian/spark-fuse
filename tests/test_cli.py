import hashlib
import json
import pytest
import torch
from torch import nn
from sparkfuse.__main__ import main


class Scores(nn.Module):
    def forward(self, x, b):
        s = x + b
        return s.softmax(-1), s


def test_offline_emit_and_input_manifest(tmp_path, capsys):
    args = (torch.randn(5, 17), torch.randn(17))
    src = tmp_path / "graph.pt2"
    torch.export.save(torch.export.export(Scores(), args), src)
    out = tmp_path / "artifact"
    opts = ["--vec", "4", "--layout", "warp", "--cache"]
    assert main(["inspect", str(src), *opts]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(report["plan"]["output_ids"]) == 2
    assert main(["emit", str(src), "--out", str(out), *opts]) == 0
    assert json.loads((out / "manifest.json").read_text()) == report
    assert report["source_sha256"] == hashlib.sha256((out / "kernel.cu").read_bytes()).hexdigest()
    assert report["input_archive_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert "softmax" in (out / "graph.txt").read_text()
    before = (out / "kernel.cu").read_bytes()
    with pytest.raises(SystemExit):
        main(["emit", str(src), "--out", str(out)])
    assert (out / "kernel.cu").read_bytes() == before


def test_unsupported_offline_graph_does_not_create_directory(tmp_path):
    class Matmul(nn.Module):
        def forward(self, a, b):
            return a @ b

    src = tmp_path / "mm.pt2"
    out = tmp_path / "new"
    args = (torch.randn(3, 3), torch.randn(3, 3))
    torch.export.save(torch.export.export(Matmul(), args), src)
    with pytest.raises(SystemExit) as e:
        main(["emit", str(src), "--out", str(out)])
    assert e.value.code == 2 and not out.exists()


def test_offline_compact_mean_emits_one_row_result(tmp_path):
    class RowMean(nn.Module):
        def forward(self, x):
            return x.square().mean(-1, keepdim=True)

    out = tmp_path / "cuda"
    src = tmp_path / "row-mean.pt2"
    torch.export.save(torch.export.export(RowMean(), (torch.randn(3, 769),)), src)
    assert (
        main(["emit", str(src), "--out", str(out), "--layout", "compact_mean", "--vec", "4"]) == 0
    )
    manifest = json.loads((out / "manifest.json").read_text())
    cu = (out / "kernel.cu").read_bytes()
    assert manifest["schedule"] == dict(threads=128, layout="compact_mean", cache=False, vec=4)
    assert manifest["source_sha256"] == hashlib.sha256(cu).hexdigest()
    assert b"fused_compact" in cu and len(manifest["plan"]["output_ids"]) == 1
