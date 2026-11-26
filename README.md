# PyTorch → CUDA 推理编译器 · SparkFuse

[English](#english) · [项目图解](https://etherealbian.github.io/work/spark-fuse.html) · [实测与复现](docs/findings.md) · [图片来源](https://github.com/EtherealBian/etherealbian.github.io/blob/94bd5525d1e818a7f5a11e5d0691f27397a63ee7/ASSETS.md#sparkfuse-artwork)

SparkFuse 在 NVIDIA DGX Spark（GB10）上把 PyTorch 导出的推理子图编译成 CUDA C++。它读取 `torch.export` 计算图，判断哪些算子可以共用一次 GPU 启动、哪些中间张量必须写回。未被接管的算子仍由 PyTorch 执行。

[![DGX Spark 实测总览：四层 BERT 的 Q/K/V 投影打包、Qwen2.5-32B 长前缀复用与量化 GLM-4.5 355B 单机推理](https://raw.githubusercontent.com/EtherealBian/etherealbian.github.io/94bd5525d1e818a7f5a11e5d0691f27397a63ee7/assets/plates/sparkfuse-system.webp)](https://etherealbian.github.io/work/spark-fuse.html)

## 一层 BERT，三次投影怎样变成一次？

Query、Key、Value 读取同一份隐藏状态。SparkFuse 每层只量化一次输入，再把三个 `768 × 768` 的 INT8 权重沿列拼成 `768 × 2304`。一次整数矩阵乘法算完三个投影，一个 CUDA 后处理内核分别写回 Q、K、V。注意力和前馈网络继续走 PyTorch。

单机实测：四层 BERT，batch 1，33 个 token，仅 Q/K/V 使用 W8A8。下表统计的是整次前向的 GPU 内核启动次数。

| 量化 PyTorch 基线 | 三个投影独立量化 | 每层共享输入量化 | 再合并每层的三次矩阵乘法 |
| ---: | ---: | ---: | ---: |
| 233 | 88 | 72 | 64 |

每层都返回隐藏状态和注意力概率。8 个留出请求中，每个请求的 8 个返回张量均与量化基线逐字节一致。[实验方法](docs/performance.md#stacked-bert) · [导出图、内核名称和原始计数](reports/stacked-bert-gb10.json)

## 从哪里动手

- 想看生成的代码：打开[残差归一化示例](docs/quickstart.md)。只改模型的 `return`，对照[只返回归一化结果](examples/generated/resnorm.cu)和[同时返回残差](examples/generated/resnorm2.cu)两份 CUDA 源码，看第二个输出指针为什么出现。生成源码这一步只需 Linux CPU 和 PyTorch，不下载权重。
- 想核对上表：在仓库根目录运行 `python3 -m pytest -q tests/test_stacked_bert_evidence.py`（Python 3.11+、pytest，无需 GPU）。它检查两组权重的归档计数、每层的权重布局和输出校验记录，不重新计时。

## 同一台机器上的整模型实验

这些是独立的推理实验，不计入编译器加速成绩。

| 问题 | 测量结果 |
| --- | --- |
| [Qwen2.5-32B BF16：两条请求能否共用前缀？](docs/prefix32-closeout.md) | 每条输入 2,048 个 token，共享前 1,536 个，各生成 8 个。计入快照准备，两组提示的双请求总耗时都从约 9.44 秒降至 7.66 秒；各步全词表分数逐字节一致。 |
| [GLM-4.5 355B：量化模型在单机上的请求耗时？](docs/glm45-355b.md) | 使用 llama.cpp CUDA 和 UD-IQ1_S 权重。模型已加载，输入 128 个 token、生成 16 个，本机 HTTP 请求耗时中位数为 2.588 秒。 |

## English

SparkFuse compiles supported regions of a `torch.export` inference graph to CUDA C++ on one NVIDIA DGX Spark. In a four-layer BERT encoder, Q, K and V read the same activation. The compiler quantizes it once per layer, packs the three INT8 weights into one matrix, then writes three separate BF16 outputs. Attention and the feed-forward network stay in PyTorch.

At batch 1 and 33 tokens, a full forward pass went from 233 GPU kernel launches to 64. All eight returned tensors matched quantized eager execution byte for byte on each of eight held-out requests. [Inspect the measurements](docs/performance.md#stacked-bert) or [generate CUDA from a small residual block](docs/quickstart.md).

The [32B prefix-reuse study](docs/prefix32-closeout.md) and [355B quantized serving study](docs/glm45-355b.md) are separate runtime experiments. The [experiment index](docs/findings.md#english) distinguishes replaying the saved evidence on CPU from rerunning a benchmark on the GPU.
