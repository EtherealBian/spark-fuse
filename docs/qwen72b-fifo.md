# 72B 请求排队：省下等待线程的 CPU，不代表更早收到答案

两个请求同时需要全部两个推理槽位，其中一个槽位仍在输出。调度器应该每毫秒检查一次，还是等槽位释放时被唤醒？在单台 DGX Spark 上，`BatchAdmission` 用有界 FIFO（先到先服务）替代 1 ms 轮询，减少了等待线程消耗的 CPU。**在这组实测里，最后一个响应反而更晚完成。** 这项策略为可选组件，未改默认行为。

### 真实 72B 服务

同一份 Qwen2.5-72B-Instruct Q4_K_M 权重（44.01 GB），81/81 个后端层在 GPU 上，两个槽位。阻塞请求输出第一个流式 token 时提交一个两请求批次；批次获得两个槽位后并行执行。六组固定输入，变换策略先后顺序并反向重复。计时包含 24 个阻塞请求、48 个目标请求；两次预热不计入。两种策略的输出及与历史基线可比较的目标 ID 均一致。

|先运行的策略|等待线程 CPU 中位数：轮询 → FIFO|目标路由器调用：轮询 → FIFO|从到达到最后响应：成对中位节省率|
|---|---:|---:|---:|
|轮询|314.35 → 0.49 ms|18,142 → 6|−3.47%|
|FIFO|314.87 → 0.47 ms|18,249 → 6|−2.59%|

等待线程的 CPU 成对下降 99.83% / 99.85%，达到 10% 门槛。端到端耗时**没有达到节省 3% 的独立门槛**。按原始时间戳分段，FIFO 从获得槽位到最后响应结束，平均多用 335 / 230 ms；从开始释放到授权反而平均少用 0.14 / 0.29 ms。这里用平均值让各段之和等于总差值，不能把这些差值解释成 GPU 计算变慢，也不能把线程 CPU 时间加进墙钟。这里只测一个等待客户端，没有验证吞吐、功耗或多租户尾延迟。

### 独立的仅 CPU 对照

不加载权重，使用同六组已发布输入、四个受控占用时长：48 个配对，192 个目标租约均与历史决策一致。两种顺序下等待线程 CPU 分别节省 55.7% / 59.7%，释放到授权的 P95 从约 1.28 ms 降至 0.26–0.27 ms。这不是上述 72B 在线延迟。

### 使用与复核

`BatchAdmission.reserve()` 一次授予整个批次；可限制排队批数、总输入 token 和等待时间。由调用方在**后端实际停止使用**槽位后调用 `finish()`；不能把客户端断开当作后端已停止。关闭入口会唤醒等待者，不会回收仍在使用的槽位。先来的大批次可能阻塞后来的小批次；没有跨进程共享队列。后端重启后必须创建新实例。

从仓库根目录运行；离线复核仅需 Python 3.11+ 标准库：

```bash
set -euo pipefail
B=benchmarks/quant_serving; A=$B/fifo72; R=reports/qwen72b/fifo; REF=reports/qwen72b/assignment
export PYTHONPATH="$PWD/$A:$PWD/$B"
python3 -B -S "$A/admission_trace.py" audit "$REF" "$R/cpu/rows.json" | cmp - "$R/cpu/summary.json"
python3 -B -S "$A/live/live72_review.py" "$R/live" "$REF" | cmp - "$R/live/summary.json"
python3 -B -S "$A/live/wave_ledger.py" "$R/live" "$REF" | cmp - "$R/live/wave-ledger.json"
```

在空闲 Spark 上重新测量：指定与[基线](qwen72b-assignment.md)一致的 `SERVER`、`WEIGHTS` 和不存在的 `OUT` 目录，执行 `bash benchmarks/quant_serving/fifo72/live/run.sh`。它重新检查权重、运行库和内存准入。不得用离线或模拟结果替代实测。

**English.** A bounded in-process FIFO removed repeated router calls for a blocked two-request batch. In 12 paired waves on a real 72B server, waiter-thread CPU fell by more than 99%, while arrival-to-last-response time regressed in both order strata. The isolated CPU trace and the live service are separate experiments. No throughput claim or default policy change.
