# 实测与复现 / Experiment index

[返回 README](../README.md) · [English](#english)

| 想看什么 | 代表性结果与测量条件 | 从这里复核 |
|---|---|---|
| 编译器怎样减少 kernel 发射 | 四层 BERT，batch 1、33 token，仅 Q/K/V 用 W8A8。同一次前向的四种实现分别发射 233、88、72、64 次 kernel。 | [四种图改写与计数方法](performance.md#stacked-bert) · [生成的 CUDA](../examples/generated/) |
| 共享前缀省下多少时间 | Qwen2.5-32B BF16，每条输入 2,048 个 token，生成 8 个。共享 1,536 个 token 时，两组提示的双请求总耗时均由约 9.44 秒降至 7.66 秒，包含快照准备。对固定四条长输入再测，六个实测切点全部通过计时与全词表逐字节一致性验收。 | [原始双请求记录](../reports/qwen32b-fork/raw.json) · [六个切点与复跑方法](prefix32-closeout.md) |
| 355B 量化模型怎样在单机运行 | GLM-4.5 UD-IQ1_S，两个分片，共 97,047,546,752 字节。使用固定版本的 llama.cpp CUDA，在已加载模型上输入 128 个 token、生成 16 个，本机 HTTP 耗时中位数 2.588 秒。 | [分片校验与固定长度实验](glm45-355b.md) · [逐请求记录](../reports/glm45-355b/rows.json) |

## 复核还是重测？

**CPU 复核**读取已发布的观测，不重新计时。在仓库根目录执行：

```bash
bash audit/replay-key-results.sh
```

脚本从原始记录重建 32B 前缀、72B 排队、235B 流式响应和 355B 双请求报告，并与发布的 JSON 逐字节比较。较大的 32B 全词表观测另附在[对应版本](https://github.com/EtherealBian/spark-fuse/releases/tag/prefix32-closeout-v1)。

**DGX Spark 真机复跑**使用上表各方法页锁定的权重、后端与命令。编译器源码在 [`sparkfuse/`](../sparkfuse/)，实验驱动在 [`benchmarks/`](../benchmarks/)，读数在 [`reports/`](../reports/)。

## 技术档案

- **编译器：**[性能测量](performance.md) · [验收口径](acceptance.md)
- **32B 整模型：**[权重布局](qwen32b.md) · [前缀快照与逐对成本](qwen32b-fork.md) · [投影诊断](qwen32b-layer0-shape.md)
- **量化服务：**[72B 请求排队](qwen72b-fifo.md) · [120B 微批](gpt-oss-120b.md) · [235B 流式生成](qwen235b-natural.md) · [355B 双请求时间线](glm45-paired.md)

## English

The three studies measure different things: CUDA launches in a compiled BERT graph, end-to-end time and exact logits for paired 32B requests, and local HTTP latency for quantized 355B inference through llama.cpp. Follow each study’s method link for inputs and GPU commands. Use the replay script above on Linux to verify the archived observations without downloading model weights.
