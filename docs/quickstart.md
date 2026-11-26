# 从一个残差子图开始 / A first subgraph

这两个例子只需要仓库里的代码，不下载模型权重。从仓库根目录执行 `source tools/env.sh`，使用已安装 PyTorch 的 Python。第一段可在 **Linux CPU** 上运行；第二段才使用 CUDA。它们应当分别运行，不依赖彼此留下的变量。

## 1. 先看编译器决定写什么

```python
import torch
from examples.graphs import ResidualNorm
from sparkfuse import Schedule, generate, lower

args = (torch.ones(7, 127), torch.ones(7, 127), torch.ones(127))
for visible in (False, True):
    ep = torch.export.export(ResidualNorm(expose=visible).eval(), args)
    plan = lower(ep)
    cuda = generate(plan, Schedule(layout="warp", threads=128))
    print(f"return residual={visible}: {len(plan.outputs)} output(s)")
    print(next(line for line in cuda.splitlines() if "__global__ void fused(" in line))
```

模型的加法和 RMSNorm 没变，只有 `return` 变了。第二次生成的 kernel 参数里多出 `out1`：对调用方可见的残差需要写回；第一次则不需要给它分配独立输出。这个例子用奇数宽度 127，源码中还可以看到列边界检查。完整源码示例见[生成结果](../examples/generated/)。

## 2. 让实测记录决定哪些动态行数值得特化

下例用一个有界 `torch.export.Dim` 表示 1–129 行，宽度固定。它从一个**小型、已公开的实测样本**创建临时调度库，在准备阶段编译 kernel。调用阶段只根据实际行数分发，不跑搜索。需要 CUDA 版 PyTorch、`nvcc`、C++17 编译器；所选 schedule 只适用于匹配的图、设备和工具链。请只使用可信的导出图与结果包。

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from examples.graphs import case
from sparkfuse.catalog_rows import compile_catalog_rows
from sparkfuse.tuning import backend_id
from tools.import_campaign import import_bundle

with torch.inference_mode(), TemporaryDirectory() as tmp:
    db = Path(tmp) / "catalog.sqlite"
    import_bundle(Path("reports/campaign-dispatch-sample.json.gz"), db, backend_id())
    model, inputs = case("resnorm2", 7, 768, torch.bfloat16, seed=19)
    rows = torch.export.Dim("tokens", min=1, max=129)
    ep = torch.export.export(model, inputs, dynamic_shapes=({0: rows}, {0: rows}, None))
    compiled = compile_catalog_rows(ep, db, rows=[1, 7, 32, 48])
    print("exact catalog hits:", [item.rows for item in compiled.variants if item.selected])
    live = []
    for n in (1, 32, 97, 129):
        _, xs = case("resnorm2", n, 768, torch.bfloat16, seed=97+n)
        value, residual = compiled(*xs)
        old = {t.data_ptr() for t in [*live, *xs]}
        assert value.data_ptr() != residual.data_ptr()
        assert not old.intersection((value.data_ptr(), residual.data_ptr()))
        live.extend((value, residual))
        ref_value, ref_residual = model(*xs)
        torch.testing.assert_close(value, ref_value, rtol=0.012, atol=0.0002)
        assert torch.equal(residual.view(torch.int16), ref_residual.view(torch.int16))
```

在本项目的 GB10 环境，这一样本对 32 行命中，97/129 行走通用 kernel。环境签名不匹配时不会把别的设备的时间当成本机实测；提示行数没有匹配记录是正常情况。超出导出范围会报错，不会静默截断。这里的断言检查正确性，不是计时。想比较性能，请看[有界行数的热设备与轮转请求实测](performance.md#catalog-rows)；想重建更大的调度库或只用 CPU 核验，请看[验收入口](acceptance.md#final-gb10-release)。

**English.** The first snippet shows how returning a residual changes the CUDA output list without changing the normalization operator. The second compiles one bounded export against an exact-match schedule catalog, checks fresh output allocations across four row counts, and requires identical bits for the observable residual. Neither example reports model-level throughput.
