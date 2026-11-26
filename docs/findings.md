# 单台 DGX Spark 的推理结论 / Findings

[返回 README](../README.md) · [English](#english)

## 编译器、整模型和服务各测了什么？

| 对照 | 实测结论 | 采用边界与证据 |
|---|---|---|
| 编译器：四层 BERT、33 token | 将每层 Q/K/V 投影共享输入量化，再合并矩阵乘法；一次前向的 kernel 发射由 233 次缩至 64 次。8 个留出请求的 8 个返回值逐字节匹配量化 eager。 | 仅 Q/K/V 使用 W8A8，权重随机；发射次数不是端到端加速比。[测量方法与原始结果](performance.md#stacked-bert) |
| 32B BF16：从头处理或复用 KV 前缀 | 输入/共享前缀为 2,048/1,536 时，两组提示各省约 19%，四个分支的完整 logits 一致；128/96 时变慢，一个分支的第 6 个输出 token 改变。 | 保留从头计算。短输入的首层采样在 K/V 投影处已出现差异。[两请求实测](qwen32b-fork.md) · [首层诊断](qwen32b-layer0-shape.md) |
| 355B：两个请求串行或并发放行 | 128-token 输入下，收齐两份响应从 5.17 秒降到 4.34 秒，第一份完整响应却晚了 66%–69%。1,024-token 输入下，收齐只省约 2%，第一份晚了约 92%–93%。 | 两种长度下都出现过输出变化。固定生成 16 个 token；第一份完整响应和首 token 是不同指标。[配对实验](glm45-paired.md) |

### 32B：从哪一条请求开始省时间？

原始记录把一次性准备和两个分支分开计时。对原来的 16 个配对重新核账，逐对相减，不拿不同组的中位数相减：

| 输入 / 共享前缀 | 快照准备 | 两条请求最终相差多少 | 逐对结果 |
|---|---:|---:|---:|
| 128 / 96 token | 346–349 ms | 复用慢 295–306 ms | 0/8 对更快 |
| 2,048 / 1,536 token | 1.831–1.838 s | 复用快 1.772–1.793 s | 8/8 对更快 |

长输入下，给第一条分支计入全部准备成本，仍比从头算多用 14–34 ms（不分摊外层循环的零碎开销）。第二条分支再次省去了前缀计算，两条请求合计才有净收益。短输入两个分支加起来也抵不了准备成本。以上都是同一份原始计时的分解，不是第三种调度或更高并发度的预测。[原始波次](../reports/qwen32b-fork/raw.json) · [分项核账](../reports/qwen32b-fork/paired-costs.json)。

### 其余实测资料

[32B 权重共享](qwen32b.md)、[72B 排队](qwen72b-fifo.md)、[120B 微批](gpt-oss-120b.md)、[235B 流式生成](qwen235b-natural.md)按各自的计时口径保存。单机容量见 [235B Q2_K](qwen235b.md) 与 [355B UD-IQ1_S](glm45-355b.md)；低比特模型能生成 token 不等于能以 BF16 运行同等规模，以上实验也没有测通用品质。

## 离线复核与真机复跑

```bash
# 从仓库根目录运行；只读取已发布的测量文件
bash audit/replay-key-results.sh
```

脚本重建 32B 前缀、72B 排队、235B 流式响应和 355B 双请求的主报告，并分别校验二请求成本、排队、流式输出和并发四份独立账本。重建文件必须与已发布 JSON 逐字节一致。它不会重新测量耗时。要做一次新的 GPU 实验，请使用表中各方法页的真机命令、同一检查点与相同验收条件。

`benchmarks/` 保存实验驱动与审计程序，`reports/` 保存测量和验收记录，`examples/generated/` 保存生成的 CUDA。历史报告保留原有路径和口径。正在运行或未经复核的候选项不列为发布结论。

## English

The compiler results report kernel or graph metrics. The dense 32B experiments add end-to-end timings and full-logit checks. Serving experiments use llama.cpp CUDA and compare native request traces on the same checkpoint. On the tested long prompts, 32B prefix reuse saved about 19% with exact logits; short prompts regressed. Across eight long-input pairs the second request paid back snapshot preparation; across eight short-input pairs it did not. The 355B two-slot experiment illustrates a separate tradeoff: a pair can finish earlier even though the first complete response arrives later. Neither result changes the default. The replay command above checks the published evidence on CPU; follow the individual methods to collect new GPU measurements.
