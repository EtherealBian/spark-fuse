"""Build without touching the checkout's build tree; validate an installed wheel under python -I."""

import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = """
import sys,json,torch
from pathlib import Path
site=Path(sys.argv[1]).resolve(); sys.path.insert(0,str(site))
import sparkfuse
from sparkfuse.native import native_dispatch
from sparkfuse.calibration import calibrate_export
assert Path(sparkfuse.__file__).resolve().is_relative_to(site)
class M(torch.nn.Module):
    def forward(self,x,b):
        z=x*0.125+b
        return z.softmax(-1), z
with torch.inference_mode():
    torch.manual_seed(721); x=torch.randn(17,129,device='cuda',dtype=torch.bfloat16);b=torch.randn(129,device='cuda',dtype=torch.bfloat16);m=M()
    k=sparkfuse.compile_export(torch.export.export(m,(x,b)),schedule=sparkfuse.Schedule(128,'warp',True,4));y=native_dispatch(k)(x,b); ref=m(x,b)
    torch.testing.assert_close(y[0],ref[0],atol=0,rtol=.008);torch.testing.assert_close(y[1],ref[1],rtol=0,atol=0)
from sparkfuse.packed_rows import PackedRows
class Gate(torch.nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, -1)
        g=torch.nn.functional.silu(a)
        return b*g, g
with torch.inference_mode():
    gate=Gate(); x=torch.randn(5,30,device='cuda',dtype=torch.float16)
    rows=torch.export.Dim('rows',min=1,max=31)
    r=PackedRows(torch.export.export(gate,(x,),dynamic_shapes=({0:rows},)))
    for n in (1,3,31):
        x=torch.randn(n,30,device='cuda',dtype=torch.float16)
        assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(r(x),gate(x),strict=True))
from sparkfuse.partition import partition_export
class Copies(torch.nn.Module):
    def forward(self, x):
        return tuple(y.clone(memory_format=torch.contiguous_format) for y in x.split([16,8,8],-1))
with torch.inference_mode():
    m=Copies(); x=torch.randn(2,1,32,device='cuda',dtype=torch.bfloat16)
    g,rr=partition_export(torch.export.export(m,(x,)),split_copies=True,host_backend='native')
    assert [r.get('kind') for r in rr].count('split_copy3') == 1
    y=g(x); assert len({a.untyped_storage().data_ptr() for a in (x,*y)})==4
    assert all(a.stride()==b.stride() and torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(y,m(x),strict=True))
print(json.dumps({'split_copy_widths':[16,8,8],'bounded_packed_rows':[1,3,31],'import_origin_verified':True,'torch':str(torch.__version__),'device':torch.cuda.get_device_name(),'shape':[17,129],'dtype':'bfloat16','host':'native','seed':721,'passed':True}))
"""


def run():
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=runtime, prefix="wheel-smoke-") as td:
        t = Path(td)
        src = t / "src"
        src.mkdir()
        shutil.copy2(ROOT / "pyproject.toml", src)
        shutil.copytree(
            ROOT / "sparkfuse",
            src / "sparkfuse",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        env = {
            **os.environ,
            "PYTHONPATH": "",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
        }

        def cmd(a):
            return subprocess.run(
                [sys.executable, "-I", *map(str, a)],
                cwd=t,
                env=env,
                text=True,
                capture_output=True,
                check=True,
                timeout=200,
            ).stdout

        cmd(["-m", "pip", "wheel", "--no-deps", "--no-build-isolation", src, "-w", t])
        (f,) = t.glob("*.whl")
        site = t / "site"
        cmd(["-m", "pip", "install", "--no-deps", "--target", site, f])
        result = json.loads(cmd(["-c", SMOKE, site]))
        with zipfile.ZipFile(f) as w:
            result["wheel_members"] = w.namelist()
        result.update(
            wheel_name=f.name,
            wheel_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
            network_index_disabled=True,
        )
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(), indent=2) + "\n")
