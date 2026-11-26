# 同一前缀分出四条请求：预制 KV 何时回本？

单台 DGX Spark 上的 Qwen2.5-72B Q4_K_M，单槽串行，同一服务进程。两种输入长度各取两份未参与原始实验的合成文档，每份各有四条不同的续问。三种策略各做三遍，轮换运行顺序；每条目标固定生成 24 个 token。

- **冷启动请求**：不复用提示 KV；不是每次重载模型。
- **按需复用**：第一条按冷请求运行，后三条允许复用共同前缀。
- **显式预制**：先单独提交共同前缀并生成一个 token，再依次处理四条目标；从预制前开始计时。

### 首次持续回本的分支序号

每格是「只看时间 / 同时保护输出」。要求到该分支为止的**累计**客户端时间，在三个重复中均减少至少 3%，且之后直到第 4 个分支持续满足。严格口径还要求比较双方截至该分支的每条输出逐 token 相同，且基线在三次重复间可复现。分支从 1 开始数。

| 文档 / 输入 token | 按需相对冷 | 预制相对冷 | 预制相对按需 |
|---|---|---|---|
| coast / 256 | 2 / 四次内未达到 | 3 / 四次内未达到 | 四次内未达到 / 四次内未达到 |
| kiln / 256 | 2 / 四次内未达到 | 3 / 四次内未达到 | 四次内未达到 / 四次内未达到 |
| coast / 2048 | 2 / 2 | 2 / 四次内未达到 | 四次内未达到 / 四次内未达到 |
| kiln / 2048 | 2 / 四次内未达到 | 2 / 四次内未达到 | 四次内未达到 / 四次内未达到 |

### 四条完成时的累计客户端中位数

单位：秒（三个完整重复）。含预制或按需首条的真实时间，不能与只计复用后目标请求的延迟直接比较。

| 文档 / 输入 token | 冷 | 按需 | 预制 | 按需 / 预制与同轮冷输出一致（各 12 对） |
|---|---:|---:|---:|---|
| coast / 256 | 23.794 | 22.358 | 22.618 | 9/12 · 6/12 |
| kiln / 256 | 23.811 | 22.380 | 22.642 | 9/12 · 9/12 |
| coast / 2048 | 45.284 | 27.991 | 28.225 | 12/12 · 9/12 |
| kiln / 2048 | 45.294 | 27.984 | 28.231 | 9/12 · 9/12 |

[原始记录、后端与权重校验](../reports/qwen72b/bursts/) · [各分支的配对累计时间与输出检查](../reports/qwen72b/bursts/break-even.json)。复核不需要模型：

```bash
python3 benchmarks/quant_serving/audit_break_even.py reports/qwen72b/bursts reports/qwen72b
```

真机复跑：按[量化运行说明](../benchmarks/quant_serving/README.md)准备 `WEIGHTS`、`SERVER`，设置新的 `OUT`，执行 `PANEL=burst bash benchmarks/quant_serving/run72.sh`。

**English.** The held-out panel includes 144 target responses and 12 actual preparation requests. All three matched repetitions must save at least 3% cumulatively up to a point and remain ahead through branch four. The stricter criterion additionally checks every generated token and baseline repeatability. A missing crossing means not demonstrated within four branches, not that the strategy never pays off. This serial, fixed-budget workload does not establish production throughput, natural-answer quality or a universal default.
