from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import pytest
import torch
from examples.graphs import case
from sparkfuse.runtime import compile_export
from sparkfuse.measure import compare
from sparkfuse.tuning import backend_id, compile_tuned
from tools.import_campaign import import_bundle


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path_factory.mktemp("replay") / "verified.sqlite"
    import_bundle(root / "reports/campaign-dispatch-sample.json.gz", out, backend_id())
    return out


@pytest.mark.parametrize(
    "name,rows,width,dt,hit",
    [
        ("gate", 32, 1536, torch.float16, True),
        ("resnorm2", 32, 768, torch.bfloat16, True),
        ("resnorm2", 7, 1009, torch.bfloat16, False),
    ],
)
@torch.inference_mode()
def test_replayed_code_uses_new_arguments_without_overwriting_prior_outputs(
    catalog, name, rows, width, dt, hit
):
    before = sha256(catalog.read_bytes()).digest()
    m, args = case(name, rows, width, dt, 17031)
    ep = torch.export.export(m, args, strict=True)
    baseline = compile_export(ep)
    dep = compile_tuned(ep, catalog)
    assert (asdict(baseline.schedule) != asdict(dep.schedule)) == hit
    keep = []
    for seed in (163197, 827011, 97531):
        _, a = case(name, rows, width, dt, seed)
        y = dep(*a)
        ref = m(*a)
        compare(y, ref, (1,) if name == "resnorm2" else ())
        out = y if isinstance(y, tuple) else (y,)
        if name == "resnorm2":
            assert torch.equal(y[1].view(torch.int16), ref[1].view(torch.int16))
        keep.extend((t, t.clone()) for t in out)
    assert (
        all(torch.equal(t, clone) for t, clone in keep)
        and len({t.data_ptr() for t, _ in keep}) == len(keep)
        and sha256(catalog.read_bytes()).digest() == before
    )
