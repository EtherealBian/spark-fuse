# SparkFuse

[English](#english) · [上手示例](docs/quickstart.md) · [生成的 CUDA](examples/generated/) · [性能记录](docs/performance.md) · [验收方法](docs/acceptance.md)

SparkFuse 是在 **NVIDIA DGX Spark（GB10）** 上开发的单 GPU Transformer 推理子图编译器。它读取 `torch.export` 图，把能融合的算子编译成 CUDA C++，其余算子交给 PyTorch。例如，残差只被 RMSNorm 读取时，融合 kernel 不单独写回残差；模型还要返回残差时，同一个 kernel 多写一个输出。可以直接对照[两份 CUDA 源码](examples/generated/)。

### 编译器做什么

- **生成 CUDA**：融合逐元素计算和末轴归约，支持 mean、stable softmax、LayerNorm 以及门控子图。可比较每行一个线程块、每个 warp 一行等布局；支持非对齐输入和奇数宽度。
- **保留模型语义**：根据图外使用者决定哪些值必须返回，区分视图与独占输出。[先写后读 KV 槽位](docs/performance.md#slot-forward)时，未被返回的缓存无需整块写回。BF16/FP16 的舍入顺序可能影响结果，因此会在独立输入上校验候选实现。
- **衡量请求成本**：分别测量 CUDA Graph 设备时间、完整单请求延迟和八请求排队耗时。对冻结权重的量化投影，可以[按实际请求选型](docs/performance.md#whole-request-policy)，下次只重建胜出的方案。

### DGX Spark 上的三个对照

| 工作负载 | 同机结果 | 口径 |
|---|---|---|
| [可见残差 RMSNorm](reports/accepted-resnorm2-32x1536-bf16.json)，BF16，`32 × 1536` | eager 2.961、Inductor 1.827、SparkFuse 1.850 µs | 热地址、设备重放时间；这组 Inductor 略快 |
| [四层 BERT 编码器](docs/performance.md#stacked-bert)，33 个 token | 233 → 64 次 kernel 发射 | 仅 Q/K/V 使用 W8A8；逐层返回隐藏状态和注意力概率，8 个输出与量化 eager 逐字节一致 |
| [21 输出诊断图](docs/performance.md#fused-slot-copy)，33 行 | 89.21 → 36.55 µs | 同步单请求，逐个写入预分配张量 → 一个 CUDA kernel 批量写入 |

[单 token 解码](docs/performance.md#warp-decode)用 warp 寄存器计算分块注意力；[分页 KV](docs/performance.md#paged-decode)可直接查物理页，省去两次稠密缓存重建，并[按每行实际长度跳过空页](docs/performance.md#ragged-page)；[原生主机入口](docs/performance.md#page-host)减少单请求提交开销。[Qwen2、Llama、Mistral 单层](docs/performance.md#decoder-families)支持动态 KV 长度和[双缓存写回](docs/performance.md#dynamic-append)。若调用方要求**多层 KV 缓存逐字节一致**，可在导出图中[标记需要保护的输出](docs/performance.md#protected-stack)。这些单层与堆叠实验使用随机权重，GEMM 仍交给 PyTorch。

[实测调度库](docs/performance.md#dispatch-replay)只复用图和运行环境都精确匹配的记录；未命中时使用默认实现。

同一张[有界动态行数图](docs/performance.md#catalog-rows)可以在命中行数时运行静态特化 kernel，其余合法行数走通用路径。

[09:00 截止验收](docs/acceptance.md#final-gb10-release)：1,183 项测试、63 道检查通过。截止后的动态行特化另有 64 道检查，单独计数。[通用搜索](reports/campaign-final-general.md)保留 2,490 个工单及负例；无 GPU 也可重建 1,237 个通过筛选的精确调度键。

[完整预训练 Qwen2 的逐 token 解码](docs/pretrained-qwen.md)：合并 24 层 MLP 的两次投影后，32 / 128 token 提示的中位延迟从 7.04 / 7.20 降至 6.93 / 7.08 ms。两边都使用 CUDA Graph；额外权重占 399 MiB，未设为默认。

### 复现

Linux、CUDA 版 PyTorch；构建 CUDA kernel 需要 `nvcc`，原生主机入口还需要 C++17 编译器。实测环境：GB10 / CUDA 13.0 / PyTorch 2.13.0+cu130。从仓库根目录运行：

```bash
source tools/env.sh
out=$(mktemp -d)
python examples/export_scores.py "$out/scores.pt2"
python -m sparkfuse emit "$out/scores.pt2" --out "$out/cuda" --layout warp --vec 4 --cache
# 实机三方对照；导出和查看 CUDA 源码可先单独完成
python tools/bench.py --case resnorm2 --rows 32 --width 1536 --dtype bfloat16 --profile
python -m pytest -q
```

## English

SparkFuse turns functional regions of a `torch.export` inference graph into readable CUDA C++ on NVIDIA DGX Spark. It uses values consumed outside a region to decide what a kernel must write. For example, compare [RMSNorm alone](examples/generated/resnorm.cu) with [RMSNorm that also returns the residual](examples/generated/resnorm2.cu). The same [output visibility rule for KV-slot writes](docs/performance.md#slot-forward) avoids building cache banks that no caller reads. The mixed execution path leaves unsupported operators and matrix multiplications in PyTorch.

[Warp-register decode](docs/performance.md#warp-decode) avoids KV-head expansion; [physical-page decode](docs/performance.md#paged-decode) also avoids two dense cache gathers and can [skip inactive pages per row](docs/performance.md#ragged-page). A [native caller](docs/performance.md#page-host) reduces submission overhead without changing the CUDA kernels. [Dynamic decoder layers](docs/performance.md#decoder-families) cover Qwen2, Llama and Mistral. A [multi-layer cache contract](docs/performance.md#protected-stack) preserves all returned KV bytes when later layers depend on earlier attention results. These layer-level experiments use random weights. A separate [pretrained Qwen2 generation benchmark](docs/pretrained-qwen.md) includes full-model Inductor, matched CUDA Graph replay, exact token checks and P95 latency. See the commands above for reproducible CUDA generation and three-way timing; the [performance notes](docs/performance.md) include regressions as well as wins.

An [exact-match schedule catalog](docs/performance.md#dispatch-replay) reuses admitted measurements; an unseen signature uses the default kernel.

A [bounded symbolic export](docs/performance.md#catalog-rows) selects exact-row kernels in C++ and keeps a generic fallback for other in-range requests.

[09:00 cutoff audit](docs/acceptance.md#final-gb10-release): 63 checks passed; the later bounded-row supplement is audited separately. Rebuild both campaign summaries and the 1,237-key measured catalog on a CPU with `python3 -I -S audit/release.py`.
