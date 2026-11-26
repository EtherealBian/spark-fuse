# SparkFuse

[English](#english) · [快速上手](docs/quickstart.md) · [实测与复现](docs/findings.md) · [项目主页](https://etherealbian.github.io/work/spark-fuse.html)

SparkFuse 是为 **NVIDIA DGX Spark（GB10）** 编写的单 GPU 推理子图编译器。它从 `torch.export` 图生成 CUDA C++，把不支持的算子交给 PyTorch。仓库还收录了同一台机器上的整模型推理实验。

## 代表性结果

| 实验 | 单机实测 |
|---|---|
| [四层 BERT · 编译器](docs/performance.md#stacked-bert) | Q/K/V 共用输入量化，再合并投影。33-token 前向的 CUDA kernel 发射从 **233 次减至 64 次**；8 个留出请求的全部返回值与量化 eager 逐字节一致。 |
| [Qwen2.5-32B · 前缀复用](docs/prefix32-closeout.md) | 两条 2,048-token 输入共享 1,536 个 token，各生成 8 个。计入快照准备，两组提示的双请求总耗时均从 **约 9.44 秒降至 7.66 秒**，全词表 BF16 分数逐字节一致。 |
| [GLM-4.5 355B · 单机推理](docs/glm45-355b.md) | 使用 llama.cpp CUDA 加载 UD-IQ1_S 量化权重。128-token 输入、固定生成 16 个 token，本机 HTTP 耗时中位数为 **2.588 秒**（模型已加载）。 |

## 阅读与复现

- **看编译器：**从[一个残差子图](docs/quickstart.md)开始，对照[只输出归一化结果](examples/generated/resnorm.cu)和[同时写回残差](examples/generated/resnorm2.cu)的 CUDA 源码。第一个例子只需 Linux CPU 和 PyTorch。
- **看实验：**[实测索引](docs/findings.md)按编译器、整模型和服务后端整理了测试条件、原始记录和复跑入口。
- **复核已发布结果：**在仓库根目录运行 `bash audit/replay-key-results.sh`。需要 Linux、Bash、Python 3.11+ 和 `cmp`，无需 GPU 或权重。

## English

SparkFuse compiles regions of `torch.export` inference graphs to CUDA C++ on a single NVIDIA DGX Spark. In a four-layer BERT forward pass, sharing input quantization and merging Q/K/V projections reduced kernel launches from **233 to 64**. A separate BF16 32B experiment reused 1,536 prefix tokens across two 2,048-token requests: pair time fell from **9.44 s to 7.66 s**, including snapshot preparation, with byte-identical logits across eight generation steps. The quantized 355B study uses the llama.cpp CUDA backend.

Start with the [example](docs/quickstart.md), inspect the [generated kernels](examples/generated/), or use the [experiment index](docs/findings.md#english) to replay the recorded results.
