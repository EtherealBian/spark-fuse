# 235B：不加载第二个模型，查找已出现的文本能否加快生成？

单台 DGX Spark 上的 Qwen3-235B-A22B Q2_K。用同一 llama.cpp 后端启动四个独立进程，顺序为关闭、开启、开启、关闭。候选从已有 token 序列查找四 token 的键，至多提议八个后续 token；目标模型验证提议。每个进程执行一次预热和六条冷请求。目标自然结束或最多生成 64 个 token，不指定正确答案。

每种任务只有两条不同提示。下表按两种执行顺序分别汇总这两条请求的实际耗时，不是吞吐测试。`已接收/已提议` 是后端计数，`交付/计数` 分别是客户端实际收到的 token 数和后端 predicted 数（均为候选）。

| 任务 | 配对顺序 | 关闭 / 开启耗时之和（秒） | 已接收 / 已提议 | 交付 / 计数 | 输出相同 |
|---|---|---:|---:|---:|---:|
| 逐字复制 | 先关后开 | 6.855 / 3.847 | 68 / 72 | 80 / 83 | 2/2 |
| 逐字复制 | 先开后关 | 6.324 / 3.328 | 68 / 72 | 80 / 83 | 2/2 |
| 改词再抄写 | 先关后开 | 6.932 / 5.134 | 53 / 80 | 89 / 93 | 2/2 |
| 改词再抄写 | 先开后关 | 7.288 / 5.530 | 53 / 80 | 89 / 93 | 2/2 |
| 写新答案 | 先关后开 | 6.925 / 6.963 | 0 / 0 | 94 / 94 | 2/2 |
| 写新答案 | 先开后关 | 6.888 / 6.824 | 0 / 0 | 94 / 94 | 2/2 |

预先规定的单提示门槛是：两种顺序都至少节省 5% 的请求 wall time，都有被接受的提议，交付与候选后端计数相符，且两个实现及各自重复的 token、文本、停止原因均一致。通过的提示数：逐字复制 0/2，改词再抄写 0/2，写新答案 0/2。不改变默认解码；不能用提议接受率代替请求时延或回答质量。

### 离线复核与真机复跑

```bash
B=benchmarks/quant_serving; BASE=reports/qwen235b; R=$BASE/lookup
python3 -E -B -S "$B/audit_lookup235.py" "$B/qwen235b-q2k.lock.json" "$R" "$BASE" | cmp - "$R/summary.json"
python3 -E -B -S "$B/lookup_account.py" "$B/qwen235b-q2k.lock.json" "$R" "$BASE" | cmp - "$R/accounting.json"
```

在同一空闲机器准备[已验证的权重、后端、下载准入记录](qwen235b.md)，设置 `SERVER`、`WEIGHTS`、`TRANSFER_RECEIPT` 后，从仓库根目录逐块启动，每块使用新输出目录：

```bash
R=$(mktemp -d)
for i in 0 1 2 3; do LOOKUP_BLOCK=$i OUT="$R/block-$i" bash "$B/run_lookup_block.sh" || exit; done
python3 -B "$B/audit_lookup235.py" "$B/qwen235b-q2k.lock.json" "$R" "$BASE" > "$R/summary.json"
python3 -B "$B/lookup_account.py" "$B/qwen235b-q2k.lock.json" "$R" "$BASE" > "$R/accounting.json"
```

**English.** Four independently loaded processes compare no speculation with n-gram lookup. Inspect the two orders separately. All six synthetic cases, including those with no accepted drafts, contribute to the result. Delivered tokens, backend lookahead accounting and endpoint latency are distinct. The study does not estimate a production SLA.

### 实际交付的 token 怎么计？

普通解码的 12 个请求：客户端收到 526 个 token；后端计为 526。查找式解码的 12 个请求：收到 526 个（包含 12 个终止符），后端计为 540；8 个请求出现差额。这里不把差额当作交付量，也不据此更改原实验门槛。

`completion_meter.native` 只返回可合并的计数，不保留提示、回复文本或 token ID。它接收原生非流式 `return_tokens=true` 响应、请求配置和客户端耗时；未知的停止方式或缺失字段会报错。返回速率的分母是各次请求耗时之和，不是并发吞吐。对本实验存档可执行：

```bash
python3 -E -B -S "$B/audit_delivery_meter.py" "$B/qwen235b-q2k.lock.json" "$R" "$BASE" | cmp - "$R/delivery.json"
```
