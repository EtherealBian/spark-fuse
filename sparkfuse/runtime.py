import ctypes as C
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from functools import lru_cache
import torch
from torch.utils import _pytree as tree
from .ir import DT, Unsupported, lower
from .codegen import Schedule, generate


# Index programs may consume or produce integer storage without admitting integer scalar IR.
STORAGE_DT = {**DT, torch.int8: "i8", torch.int32: "i32"}


@lru_cache(None)
def toolchain():
    nvcc = os.environ.get("NVCC") or shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    return nvcc, subprocess.check_output([nvcc, "--version"], text=True)


def build(source, device):
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    nvcc, version = toolchain()
    flags = [
        "-O3",
        "--fmad=false",
        "-std=c++17",
        f"-arch={arch}",
        "--shared",
        "-Xcompiler=-fPIC",
        "--cudart=shared",
    ]
    key = hashlib.sha256(repr((source, version, flags)).encode()).hexdigest()
    root = Path(os.environ.get("SPARKFUSE_CACHE", Path.home() / ".cache/sparkfuse"))
    dest = root / key
    dest.mkdir(parents=True, exist_ok=True)
    cu = dest / "kernel.cu"
    so = dest / "kernel.so"
    with (dest / "build.lock").open("a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not so.exists():
            cu.write_text(source)
            tmp = dest / "building.so"
            run = subprocess.run(
                [
                    nvcc,
                    *flags,
                    "-Xlinker",
                    "-rpath",
                    "-Xlinker",
                    str(Path(nvcc).resolve().parents[1] / "lib64"),
                    str(cu),
                    "-o",
                    str(tmp),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if run.returncode:
                raise RuntimeError(f"{cu}: {run.stderr}")
            tmp.replace(so)
    lib = C.CDLL(str(so))
    lib.launch.argtypes = [C.POINTER(C.c_void_p), C.c_int, C.c_void_p]
    lib.launch.restype = C.c_int
    return lib, cu


class DenseKernelCall:
    """Guard inputs, allocate fresh dense outputs, and launch on the current stream."""

    def prepare(self, *args, **kwargs):
        flat, spec = tree.tree_flatten((args, kwargs))
        if self.ep.call_spec.in_spec != spec:
            raise ValueError("input pytree mismatch")
        it = iter(flat)
        tensors = [next(it) if v is None else v for v in self.bind]
        p = self.plan
        fixed = getattr(self, "input_strides", None)
        for j, (t, i) in enumerate(zip(tensors, p.inputs)):
            v = p.vals[i]
            stride = fixed[j] if fixed else None
            if (
                not isinstance(t, torch.Tensor)
                or tuple(t.shape) != v.shape
                or STORAGE_DT.get(t.dtype) != v.dt
                or t.device != self.device
                or t.layout != torch.strided
                or (tuple(t.stride()) != stride if stride else not t.is_contiguous())
                or t.is_neg()
                or t.is_conj()
            ):
                raise ValueError("input shape/dtype/device/layout mismatch")
            if torch.is_grad_enabled() and t.requires_grad:
                raise ValueError("inference only: run under torch.inference_mode()")
        dtype = {v: k for k, v in STORAGE_DT.items()}
        out = [
            torch.empty(p.vals[i].shape, dtype=dtype[p.vals[i].dt], device=self.device)
            for i in p.outputs
        ]
        all_t = tensors + out
        ptr = (C.c_void_p * len(all_t))(*(t.data_ptr() for t in all_t))
        ret = tree.tree_unflatten(out, self.ep.call_spec.out_spec)

        def launch():
            with torch.cuda.device(self.device):
                stream = torch.cuda.current_stream(self.device)
                err = self.lib.launch(ptr, p.rows, stream.cuda_stream)
                if err:
                    raise RuntimeError(f"CUDA launch error {err}")
                for x in all_t:
                    x.record_stream(stream)
            return ret

        return launch

    def __call__(self, *args, **kwargs):
        return self.prepare(*args, **kwargs)()


class Kernel(DenseKernelCall):
    def __init__(self, ep, device, schedule=None):
        self.plan = p = lower(ep)
        from .mean_tree import ordered_default, sensitive

        self.schedule = schedule or (
            ordered_default(p)
            if sensitive(p)
            else (
                Schedule(128, "block", True)
                if p.red is not None
                else Schedule(
                    256, "flat" if all(p.vals[i].shape == p.domain for i in p.outputs) else "block"
                )
            )
        )
        self.device = torch.device(device)
        self.device = (
            torch.device(
                "cuda",
                torch.cuda.current_device() if self.device.index is None else self.device.index,
            )
            if self.device.type == "cuda"
            else self.device
        )
        if self.device.type != "cuda":
            raise Unsupported("CUDA device required")
        self.ep = ep
        self.bind = []
        for spec in ep.graph_signature.input_specs:
            kind = spec.kind.name
            if kind == "USER_INPUT":
                self.bind.append(None)
            elif kind in ("PARAMETER", "BUFFER", "CONSTANT_TENSOR"):
                src = ep.state_dict if spec.target in ep.state_dict else ep.constants
                self.bind.append(src[spec.target])
            else:
                raise Unsupported(f"unsupported input kind {kind}")
        if any(s.kind.name != "USER_OUTPUT" for s in ep.graph_signature.output_specs):
            raise Unsupported("side effects unsupported")
        self.source = generate(p, self.schedule)
        self.lib, self.path = build(self.source, self.device)


def compile_export(ep, device="cuda", schedule=None):
    return Kernel(ep, device, schedule)
