# SparkFuse

[English](#english) · [生成的 CUDA](examples/generated/) · [性能记录](docs/performance.md) · [验收方法](docs/acceptance.md)

SparkFuse 是在 **NVIDIA DGX Spark（GB10）** 上开发的单 GPU Transformer 推理子图编译器。它读取 `torch.export` 图，把能融合的算子编译成 CUDA C++，其余算子交给 PyTorch。例如，残差只被 RMSNorm 读取时，融合 kernel 不单独写回残差；模型还要返回残差时，同一个 kernel 多写一个输出。可以直接对照[两份 CUDA 源码](examples/generated/)。

### 编译器做什么

- **生成 CUDA**：融合逐元素计算和末轴归约，支持 mean、stable softmax、LayerNorm 以及门控子图。可比较每行一个线程块、每个 warp 一行等布局；支持非对齐输入和奇数宽度。
- **保留模型语义**：根据图外使用者决定哪些值必须返回，区分视图与独占输出。BF16/FP16 的舍入顺序可能影响结果，因此会在独立输入上校验候选实现。
- **衡量请求成本**：分别测量 CUDA Graph 设备时间、完整单请求延迟和八请求排队耗时。对冻结权重的量化投影，可以[按实际请求选型](docs/performance.md#whole-request-policy)，下次只重建胜出的方案。

### DGX Spark 上的三个对照

| 工作负载 | 同机结果 | 口径 |
|---|---|---|
| [可见残差 RMSNorm](reports/accepted-resnorm2-32x1536-bf16.json)，BF16，`32 × 1536` | eager 2.961、Inductor 1.827、SparkFuse 1.850 µs | 热地址、设备重放时间；这组 Inductor 略快 |
| [四层 BERT 编码器](docs/performance.md#stacked-bert)，33 个 token | 233 → 64 次 kernel 发射 | 仅 Q/K/V 使用 W8A8；逐层返回隐藏状态和注意力概率，8 个输出与量化 eager 逐字节一致 |
| [21 输出诊断图](docs/performance.md#fused-slot-copy)，33 行 | 89.21 → 36.55 µs | 同步单请求，逐个写入预分配张量 → 一个 CUDA kernel 批量写入 |

[单 token 注意力解码](docs/performance.md#decode-sdpa)直接计算 QK、softmax 与加权 V；在带掩码的 GQA 上更快，在较长的无掩码输入上慢于 PyTorch。在[真实 Qwen2 解码层](docs/performance.md#qwen-sdpa-sites)中，编译器还能消掉重复的 KV 头，同时保留两个公开缓存，并能[用一个 CUDA kernel 生成两个新缓存](docs/performance.md#dynamic-append)。[注意力分数扫描](reports/campaign-final-attention.md)的 290 个有效对照中，SparkFuse 有 125 个设备时间更短。调用方需要独占新输出时，可[批量写回新分配的张量](docs/performance.md#fresh-transport)，不必预先分配缓冲区。真实框架实验使用随机权重。子图提速不保证整层提速：[一个 Qwen2 长度反而变慢](docs/performance.md#qwen-packed-mean)。GEMM 由 PyTorch 执行；范围限于单 GPU 前向推理，支持静态形状、部分有界 token 行数，以及[Qwen2、Llama 和 Mistral 解码层](docs/performance.md#decoder-families)的动态注意力、旋转位置编码与残差融合。

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

SparkFuse turns functional regions of a `torch.export` inference graph into readable CUDA C++ on NVIDIA DGX Spark. It uses values consumed outside a region to decide what a kernel must write. For example, compare [RMSNorm alone](examples/generated/resnorm.cu) with [RMSNorm that also returns the residual](examples/generated/resnorm2.cu). The mixed execution path leaves unsupported operators and matrix multiplications in PyTorch.

[Single-token SDPA](docs/performance.md#decode-sdpa) fuses score and value accumulation without expanding KV heads; the report includes long contexts where PyTorch is faster. [A Qwen2 decode layer](docs/performance.md#qwen-sdpa-sites) eliminates repeated KV heads while preserving both returned caches; [a paired append](docs/performance.md#dynamic-append) writes them in one launch. [Qwen2, Llama and Mistral decoder layers](docs/performance.md#decoder-families) combine variable-context attention with fused rotary, residual and gate operations while preserving public KV caches. On the BF16 residual example, SparkFuse took 1.850 µs versus 1.827 µs for Inductor. In a four-layer BERT stack with W8A8 Q/K/V, packing each layer’s projections reduced launches from 233 to 64 at 33 tokens while retaining all eight observable outputs. These are random-weight inference experiments, not model quality measurements. Follow the commands above to export a graph, inspect its CUDA and benchmark all three backends. The [performance notes](docs/performance.md) include Qwen2, BERT, Phi-3 and T5 results, raw measurements, and cases where a change slowed the layer down.
