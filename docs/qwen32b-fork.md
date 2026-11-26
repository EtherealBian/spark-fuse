# 32B：两个请求共用 KV 前缀值得吗？

单台 DGX Spark，Qwen2.5-32B 的 64 层均为 BF16。两个相关提示各生成 8 个 greedy token。对照从头处理两次；候选先算一次公共前缀，将其 K/V 紧凑复制到独立存储，每个分支清零原静态缓存、还原前缀，再处理后缀。解码 CUDA Graph 仍使用原缓存地址。计时包含前缀计算、快照分配、两次还原以及两个输出回到主机的时间。

| 输入 / 公共前缀 | 提示组 | 两请求中位秒：从头 → 快照 | 输出 ID 一致 | 全词表 logits 逐字节一致 |
|---|---:|---:|---:|---:|
| 128 / 96 | 1 | 4.704 → 5.002 | 2/2 | 0/2 |
| 128 / 96 | 2 | 4.715 → 5.015 | 1/2 | 0/2 |
| 2048 / 1536 | 1 | 9.443 → 7.659 | 2/2 | 2/2 |
| 2048 / 1536 | 2 | 9.441 → 7.662 | 2/2 | 2/2 |

128-token 的四个分支都没有保持整条 logits 轨迹一致，一条的第 6 个输出 token 发生变化。2,048-token 的四个分支均一致，两请求总耗时在两组提示下均下降约 19%，每份快照需要 384 MiB。**保留默认从头计算**：这个结果不证明任意长提示可以安全拆分，也不是跨用户缓存或并发吞吐测试。

### 离线复核与真机复跑

```bash
B=benchmarks/pretrained_qwen; R=reports/qwen32b-fork
python3 -E -B -S "$B/audit_fork32.py" bundle "$R" reports | cmp - "$R/summary.json"
# 在空闲 Spark 上，从仓库根目录运行；依赖和固定权重见 32B 基线
WEIGHTS="$WEIGHTS" OUT="$NEW_OUTPUT_DIRECTORY" PYTHON=python3 bash "$B/run_fork32.sh"
```

**English.** Two related requests pay the entire snapshot preparation and restore cost. Both long-input groups improve, but the short-input panel includes a changed greedy token. Timings and per-trajectory digests come from the full model; the short-input failure is retained. No default or quality claim changes.
