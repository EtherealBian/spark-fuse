# SparkFuse

[English](#english) · [快速上手](docs/quickstart.md) · [实测结论](docs/findings.md) · [生成的 CUDA 源码](examples/generated/)

SparkFuse 是在 **NVIDIA DGX Spark（GB10）** 上开发的单 GPU Transformer 推理子图编译器。它从 `torch.export` 图生成 CUDA C++，把其余算子交给 PyTorch。仓库也包含真实模型的推理实验：32B 用来检查权重布局与 KV 前缀复用；72B–355B 通过 llama.cpp 的 CUDA 后端研究量化模型的服务行为。

## 三项代表性结果

| 问题 | 同一台 DGX Spark 上的结果 |
|---|---|
| [编译器能省掉什么？](docs/performance.md#stacked-bert) | 四层 BERT 的 kernel 发射从 233 次降到 64 次。8 个返回值都与量化 eager 逐字节一致。该数字是发射次数，不是延迟加速比。 |
| [共享前缀能让请求更快吗？](docs/prefix32-closeout.md) | 32B BF16，两个 2,048-token 请求共用前 1,536 个 token：两组提示的总耗时均从约 9.44 秒降到 7.66 秒（每条生成 8 个 token，计入快照准备），四个分支的全词表 logits 逐字节一致。短输入反而变慢，仍默认从头计算。 |
| [并发能让谁更早收到答案？](docs/glm45-paired.md) | 355B 量化模型，两条 128-token 请求一起放行，两份完整响应收齐从 5.17 秒降到 4.34 秒；第一份完整响应却晚了 66%–69%，输出也有变化。保留原调度。 |

## 从哪里开始

- **看编译器：**对照[只返回归一化结果](examples/generated/resnorm.cu)和[同时返回残差](examples/generated/resnorm2.cu)的 CUDA。调用方是否还需要残差，决定 kernel 是否多写一份结果。[快速上手](docs/quickstart.md)的第一个例子可以在 Linux CPU 上运行。
- **核对实测：**在仓库根目录执行 `bash audit/replay-key-results.sh`，用仓库内的原始记录重建关键结果；推送和 PR 都会运行这项检查。需要 Bash、Python 3.11+ 和 `cmp`；无需 GPU 或模型权重。
- **重新测量：**[结论索引](docs/findings.md)列出每一项的模型、对照和复跑入口。[性能记录](docs/performance.md)区分设备时间、同步请求延迟和排队成本；[验收记录](docs/acceptance.md)保留各阶段的检查口径。

## English

SparkFuse compiles regions of `torch.export` inference graphs to readable CUDA C++ on a single NVIDIA DGX Spark. PyTorch runs unsupported operators and GEMMs. The dense 32B studies measure model execution and cache reuse. The quantized 72B–355B studies use the llama.cpp CUDA backend and report serving results separately from compiler performance.

Start with the [working example](docs/quickstart.md), inspect the [generated kernels](examples/generated/), or read the [findings and their limits](docs/findings.md#english). Run `bash audit/replay-key-results.sh` on Linux to reproduce the archived reports without downloading weights. Fresh GPU measurements use the instructions linked from each study.
