# 355B 双请求：两人更早收齐，第一个人却等得更久

单台 DGX Spark 加载同一份 GLM-4.5 UD-IQ1_S 量化权重，在同一个双槽 llama.cpp CUDA 服务上比较两种做法：两个请求同时有资格运行，一个接一个放行，或同时放行。每槽 2048 上下文、每次固定生成 16 个 token、绕过提示缓存。两次独立启动，每次交换放行顺序和先发提示。共 16 波、32 个计时响应；另有 4 次不计时预热。下表是从两个请求同时具备运行资格，到完整响应抵达客户端的时间，**不是首 token 延迟**。

| 每个提示的长度 | 两人全部完成：串行 → 并发（中位数） | 各配对的总耗时节省 | 第一份完整响应变慢 | 并发对串行，逐 token 相同的配对 |
|---|---:|---:|---:|---:|
| 128 token | 5.165 → 4.338 秒 | 14.71%–16.74% | 65.76%–68.92% | 0/4 |
| 1024 token | 19.669 → 19.219 秒 | 2.15%–2.48% | 91.94%–93.04% | 2/4 |

短输入中，第二条提示在四个并发波次都改变了输出，最早出现在第 8 或第 10 个 token。长输入中，第一条提示在四个并发波次中的两次从第 1 个 token 起就不同。全部 **16 个串行响应**均与此前同输入的单槽结果一致。输出变化不说明哪个答案更好；本实验没有测质量或自然停止。

区别不只是两个 HTTP 请求发生了重叠。在每个并发波次，原生计数器都能证明**至少 15 次** `llama_decode` 调用采样到了两个处于处理状态的槽。审计按固定版本服务的浮点除法和六位有效数字输出，反推累积的整数槽位采样数，再用相邻空闲边界求差。若某波有 `d` 次调用、合计 `b` 个忙槽采样，则双忙调用数介于 `max(0,b-d)` 和 `floor(b/2)` 之间；无法从中断定每个 CUDA kernel 都同时处理了两条请求。

**验收：**每个配对的总耗时至少下降 5%、第一份响应不晚超过 10%、完整 token 序列一致、后端实有并发。短输入只通过总耗时和并发；长输入只通过并发。**不更改默认调度。**

## 复核 / Reproduce

无需模型或 GPU；从仓库根目录用 Python 3.11+ 重建主报告和独立时间线。

```bash
set -euo pipefail
B=benchmarks/quant_serving; T=$B/paired355; REF=reports/glm45-355b; OUT=$REF/paired
export PYTHONPATH="$PWD/$B"
python3 -B -S "$T/review_pair355.py" "$B/glm45-udiq1s.lock.json" "$OUT" "$REF" "$REF/tokenizer-contract.json" | cmp - "$OUT/summary.json"
python3 -B -S "$T/occupancy_ledger.py" "$OUT" "$REF" | cmp - "$OUT/occupancy-ledger.json"
```

For a new hardware run, use the [one-slot model and native-server prerequisites](glm45-355b.md). Supply `SERVER`, `WEIGHTS`, the original `TRANSFER_RECEIPT`, and `GLM_REF` (the verified one-slot evidence directory). Run `bash "$T/run.sh"` twice with `BLOCK=0` and `BLOCK=1` and new `OUT` directories named `block-0` and `block-1` under the same fresh parent; keep the weights, slot geometry and shared GPU lock unchanged. `review_pair355.py` rebuilds the summary without using the old two-slot timings. Store it in the new parent's `summary.json` before rebuilding the companion ledger. A completed negative result is not a failed run. This fixed-output, two-client workload does not establish a general throughput or service-level guarantee.
