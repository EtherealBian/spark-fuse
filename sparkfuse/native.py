"""Optional ATen host launcher. Mirrors the Python API; not a different CUDA algorithm."""

import ctypes
import fcntl
from functools import lru_cache
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sysconfig
import torch
from torch.utils import _pytree as tree


@lru_cache(None)
def _extension():
    from torch.utils.cpp_extension import include_paths, library_paths

    src = Path(__file__).with_name("csrc") / "host.cpp"
    resident_src = src.with_name("resident_io.cpp")
    length_src = src.with_name("length_io.cpp")
    append_src = src.with_name("append_io.cpp")
    cxx = shutil.which(os.environ.get("CXX", "c++"))
    if cxx is None:
        raise RuntimeError("native dispatch requires a C++ compiler")
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    include = include_paths("cuda") + [sysconfig.get_paths()["include"]]
    libs = library_paths("cuda")
    flags = [
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-fvisibility=hidden",
        f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
    ] + ["-I" + x for x in include]
    links = (
        ["-L" + x for x in libs]
        + ["-Wl,-rpath," + x for x in libs]
        + ["-ltorch_python", "-ltorch_cpu", "-lc10_cuda", "-lc10", "-lcudart"]
    )
    key = hashlib.sha256(
        src.read_bytes()
        + b"\0"
        + resident_src.read_bytes()
        + b"\0"
        + length_src.read_bytes()
        + b"\0"
        + append_src.read_bytes()
        + repr(
            (
                str(torch.__version__),
                torch.version.git_version,
                sysconfig.get_config_var("SOABI"),
                cxx,
                subprocess.check_output([cxx, "--version"]),
                flags,
                links,
            )
        ).encode()
    ).hexdigest()
    base = Path(os.environ.get("SPARKFUSE_CACHE", Path.home() / ".cache/sparkfuse"))
    folder = base / "host" / key
    folder.mkdir(parents=True, exist_ok=True)
    so = folder / ("_sparkfuse_host" + sysconfig.get_config_var("EXT_SUFFIX"))
    with (folder / "build.lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if not so.is_file():
            tmp = folder / "building.so"
            p = subprocess.run(
                [
                    cxx,
                    *flags,
                    str(src),
                    str(resident_src),
                    str(length_src),
                    str(append_src),
                    *links,
                    "-o",
                    str(tmp),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if p.returncode:
                raise RuntimeError(p.stderr)
            tmp.replace(so)
    spec = importlib.util.spec_from_file_location("_sparkfuse_host", so)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def native_dispatch(kernel):
    return _native_dispatch(kernel, None)


def _native_dispatch(kernel, row_bounds):
    p = kernel.plan
    desc = [p.vals[i] for i in p.inputs + p.outputs]
    out_spec = kernel.ep.call_spec.out_spec
    in_spec = kernel.ep.call_spec.in_spec
    num = {"f32": 0, "f16": 1, "bf16": 2, "i8": 3, "i32": 4}
    host = _extension()._Plan(
        kernel.lib,
        ctypes.cast(kernel.lib.launch, ctypes.c_void_p).value,
        kernel.device.index,
        p.rows,
        len(p.inputs),
        # DynamicRows rejects static rank-2 non-singleton leading extents;
        # representative rows >= 2 therefore distinguishes [R,W] from [1,W].
        [
            (-1, x.shape[1])
            if row_bounds is not None and len(x.shape) == 2 and x.shape[0] == p.rows
            else x.shape
            for x in desc
        ],
        [num[x.dt] for x in desc],
        out_spec.is_leaf(),
        *(row_bounds or (0, 0)),
        getattr(kernel, "input_strides", ()),
    )
    raw = host.run
    plain_out = out_spec.is_leaf() or (
        out_spec.type is tuple
        and all(
            s.is_leaf()
            for s in (
                out_spec.children() if hasattr(out_spec, "children") else out_spec.children_specs
            )
        )
    )
    plain_in = in_spec == tree.tree_flatten((tuple(range(len(kernel.bind))), {}))[1]
    if plain_in and plain_out and all(x is None for x in kernel.bind):
        return raw

    def call(*args, **kwargs):
        args_flat, actual = tree.tree_flatten((args, kwargs))
        if actual != in_spec:
            raise ValueError("input pytree mismatch")
        it = iter(args_flat)
        ret = raw(*(next(it) if x is None else x for x in kernel.bind))
        return ret if plain_out else tree.tree_unflatten(list(ret), out_spec)

    return call
