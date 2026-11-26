# 235B 预填充：把物理微批从 128 调到 256 有用吗？

在单台 DGX Spark 上，固定同一份 Q2_K 权重、CUDA 后端、4,096-token 上下文和单个槽位。逻辑批大小一直是 512，只改每次送入后端的物理微批上限。4 次重新启动按 128 → 256 → 256 → 128 交叉排列。每次单独预热，再把 4 条提示按正序、反序各执行一次；共 32 个计时请求，每个生成 16 个 token，不复用提示 KV。

| 输入:组 | 客户端中位秒 128 → 256 | 预填充中位秒 128 → 256 | 两个启动顺序下的客户端节省率 | 跨设置输出 ID 一致 |
|---|---:|---:|---:|---|
| 128:0 | 2.074 → 2.063 | 1.108 → 1.111 | +6.3% / +0.3% | 是 |
| 128:1 | 1.880 → 1.815 | 0.904 → 0.901 | +7.0% / -0.2% | 是 |
| 1024:0 | 5.306 → 4.081 | 4.339 → 3.173 | +24.5% / +21.5% | 是 |
| 1024:1 | 5.197 → 4.017 | 4.232 → 3.118 | +24.1% / +21.2% | 是 |

时间门槛：**通过**。输出门槛：**通过**。这组固定请求支持使用 256；未修改默认值。

绝对秒数是每个提示、每种设置 4 次观测的中位数。门槛则按两个启动顺序、每条提示分别计算：长输入的客户端与预填充时间都至少节省 5%；短输入的客户端时间至多退化 3%。另核对同一进程内重复、跨微批及对照与已发布基线的输出。固定 token 数不代表自然生成质量，也没有测并发吞吐。

此前一次启动的 8 个请求全部排除：进度文本写到了标准输出，导致与离线 JSON 的逐字节校验失败。可用单独保存的原始输出和请求重建故障；没有挑选其中任何一次计时补足主实验。

## 复核与复跑

从仓库根目录运行，只需 Python 标准库即可复核归档：

```bash
set -euo pipefail
B=benchmarks/quant_serving; BASE=reports/qwen235b; R=$BASE/tiles; LOCK=$B/qwen235b-q2k.lock.json
python3 -E -B -S "$B/audit_tile235.py" "$LOCK" "$R" "$BASE" | cmp - "$R/summary.json"
python3 -E -B -S "$B/tile_timing_proof.py" "$LOCK" "$R" "$BASE" --excluded "$R/excluded-first" | cmp - "$R/timing-proof.json"
```

真机复跑需要基线指定的服务二进制、权重和下载前保存的准入回执。设置并导出 `SERVER`、`WEIGHTS`、`TRANSFER_RECEIPT`，在空闲 Spark 上执行：

```bash
set -euo pipefail
B=benchmarks/quant_serving; BASE=reports/qwen235b; LOCK=$B/qwen235b-q2k.lock.json
R=.runtime/tiles-fresh; mkdir -p .runtime; mkdir "$R"
for i in 0 1 2 3; do TILE_BLOCK=$i OUT="$R/block-$i" bash "$B/run_tile235_block.sh"; done
python3 -B "$B/audit_tile235.py" "$LOCK" "$R" "$BASE" > "$R/summary.json"
python3 -B "$B/tile_timing_proof.py" "$LOCK" "$R" "$BASE" > "$R/timing-proof.json"
```

**English.** Physical prefill microbatch size is the only intervention. Absolute medians describe each prompt; the decision uses paired order-specific timing and token equality. The first invalid stdout stream stays outside all timing cohorts. The report does not establish text quality, peak GPU memory or concurrent serving throughput.

[自然停止下的流式实测：更快的微批会不会改变答案？](qwen235b-natural.md)
