# 72B：长前缀复用与双槽请求路由

DGX Spark 运行 Qwen2.5-72B-Instruct 的 Q4_K_M 版本。12 个 GGUF 分片共 44,010,465,472 字节，加载前逐一核对 SHA-256。后端报告 80 个 transformer 层及输出层均在 GPU；词嵌入或其他缓冲仍可能留在主机。量化计算与 attention 使用 llama.cpp CUDA；SparkFuse 实现请求路由与验收，不把上游内核算作自己的加速。

## 一、同一请求能少算多少前缀？

每对请求都先执行相同的先行请求。对同一条目标输入关闭或开启槽内 KV 复用，交换顺序，并在两组都显式屏蔽结束标记（EOG）后各生成 24 个贪心 token。初次试运行仅设置请求的 ignore_eos，256-token 的 indigo 复用臂在第 15 个 token 提前结束（冷臂为 24 个）；那次整轮不计入以下等长性能比较。前端从提示 ID 重算最长前缀，再与后端报告的已处理、已复用 token 数逐一核对。

| 输入 | 实际复用 token | 预填充中位数：关闭 → 开启（ms）| 每对端到端节省的中位数（ms） | 输出逐 token 相同 |
|---:|---:|---:|---:|---:|
| 256 | 239–241 | 753.3 → 279.0 | 480.0 | 3/4 对 |
| 2048 | 2031–2033 | 6072.0 → 285.5 | 5794.5 | 4/4 对 |

## 二、什么时候值得改变槽分配？

两个槽依次接收六个请求。轮询和最长前缀策略处理相同输入，两个独立重复交换策略顺序。交替流量 `ABABAB` 是对照：轮询本来就能把同系列送回原槽。折返流量 `ABBAAB` 则会打破轮询的对应关系。每个请求 1,024 token 输入、16 token 输出；不是并发吞吐测试。

| 流量 | 总处理 token：轮询 → 前缀路由 | 六请求端到端：轮询 → 前缀路由（s） | 输出逐 token 相同 |
|---|---:|---:|---:|
| 交替 | 2,088 → 2,088 | 27.70 → 27.74 | 12/12 对 |
| 折返 | 6,144 → 2,088 | 38.84 → 27.76 | 11/12 对 |

部分配对输出出现 token 分歧；即使有计时改善，也不能视为等价替换。单次请求走回环 HTTP，端到端时间含本机往返，不含权重加载或分词；后端预填充时间不等同于客户端流式首 token 延迟。所用提示为构造流量，不能推断其他任务的回答质量，也不能把这里的差异当作相对 32B BF16 的提速。

<a id="prepared-prefix"></a>
## 三、预先准备一段前缀，要命中几次才回本？

同一台机器、同一构建，再对上面八条原始目标输入各做两个冷对照。另三种做法先在单槽中运行一次前缀（生成 1 个 token），再提交完整输入（生成 24 个 token）：准备完整公共部分，或把准备长度向下对齐到 128 / 512 的倍数。256-token 组没有非零的 512 对齐前缀。这里的「完整」只指公共部分，不是完整提示。每个目标单独准备，各实验臂按预先确定的顺序交错执行。

| 输入 | 策略 | 准备往返中位数（ms） | 目标请求往返中位数（ms） | 与同组冷对照逐 token 一致 | 预计回本命中次数¹ |
|---:|---|---:|---:|---:|---:|
| 256 | 不复用 | — | 6,024 | 参照 | — |
| 256 | 完整公共前缀 | 775 | 5,520 | 4/4 | 2 |
| 256 | 128 对齐 | 405 | 5,630 | 4/4 | 2 |
| 2,048 | 不复用 | — | 11,719 | 参照 | — |
| 2,048 | 完整公共前缀 | 6,362 | 5,635 | 4/4 | 2 |
| 2,048 | 128 对齐 | 5,981 | 5,758 | 4/4 | 2 |
| 2,048 | 512 对齐 | 4,793 | 6,961 | 4/4 | 2 |

¹ 按每条配对请求计算 `ceil(准备往返 / (冷请求往返 − 命中请求往返))`，表中给出组内范围。假设命中成本可重复、期间没有驱逐；本实验没有连续命中多次。冷对照的先行调用仅用于统一槽状态，不计入无缓存服务的准备成本。表中延迟含解码与本机 HTTP，所以不能拿它和第一节的预填充延迟直接相除。

两次新冷对照在 256 / 2,048 组分别有 4/4、4/4 对输出相同；与第一节的原冷对照分别有 4/4、4/4 相同。这里的先行调用只发公共前缀，第一节则发送带不同后缀的完整先行请求。不能用其中一轮的命中结果抹掉另一轮的分歧，也不能外推到新提示。

[原始前缀预制记录](../reports/qwen72b/catalog/) · [逐臂、逐案例回本估算](../reports/qwen72b/catalog/summary.json)。CPU 复核：`python3 benchmarks/quant_serving/audit_catalog.py audit reports/qwen72b/catalog reports/qwen72b`。

**English.** The prepared-prefix panel measures one-time construction separately from each target request. It compares a full shared stem and shorter aligned stems against two cold controls on the same eight original prompts. The break-even hit counts are projections from paired wall times, not observed sustained hit rates. Each original and prepared request remains available for offline token and counter checks.

## 复核

[原始输入、输出与运行回执](../reports/qwen72b/) · [复算汇总](../reports/qwen72b/summary.json) · [重新运行实验](../benchmarks/quant_serving/README.md)。用 Python 3.11+ 在 CPU 上运行 `python3 benchmarks/quant_serving/audit_bundle.py 72b reports/qwen72b`。汇总从原始记录重算，不读取发布的 summary；同时核对权重指纹、实际映射的推理库、槽位、逐请求 token 和计时。

**English.** A single DGX Spark runs the dense 72B model in Q4_K_M using upstream llama.cpp CUDA. SparkFuse dispatches requests and audits the evidence. The paired single-slot test measures prefix reuse; the two-slot test contrasts round-robin and prefix affinity on an alternating control and a switchback trace. Results above are local serial request timings, not concurrent throughput or a quality benchmark. On the switchback trace, affinity reduced six-request prefill work from 6,144 to 2,088 tokens; 11 of 12 paired outputs matched. The single-slot 256-token cohort also contains a disagreement.

<a id="cold-streaming"></a>
## 两个请求同时就绪：串行发还是并发发？

单台 DGX Spark、同一个两槽服务。每波两条请求同时就绪，实验交换串行与并发策略的顺序，重复时交换串行先发的槽。每次固定输出 16 个 token、关闭前缀缓存。下表从两请求共同就绪时刻计时，包含串行时后发请求的等待。

| 输入长度（token） | 两请求都完成（秒）：串行 / 并发 | 较晚收到首 token（秒）：串行 / 并发 | 输出完全相同 |
|---|---:|---:|---:|
| 512 + 512 | 9.81 / 6.89 | 6.41 / 3.48 | 2/4 |
| 256 + 2,048 | 13.68 / 11.62 | 10.27 / 8.15 | 4/4 |

每格是两波的中位数。保留每个原生 SSE 输出的 token ID 和相对到达时间；可分别复核从共同就绪到首 token 的耗时、从单次发送到首 token 的耗时、客户端请求重叠和首个输出分歧。没有测生产尾延迟，也不能用客户端重叠证明 GPU 内核同时执行。

[原始流式时间线及环境回执](../reports/qwen72b/stream/) · CPU 重算：`python3 benchmarks/quant_serving/audit_stream.py reports/qwen72b/stream`。复跑：`PANEL=stream OUT=.runtime/stream-01 bash benchmarks/quant_serving/run72.sh`，另需同文所述的 `WEIGHTS` 与 `SERVER`。

**English.** Compare serial and concurrent dispatch of two ready, cold requests. Client timestamps distinguish time spent waiting for dispatch from time spent waiting after posting. Two repetitions per shape; this is not a saturated-throughput or P99 benchmark.

四组配对波次中，串行和并发的逐 token 输出共 6/8 对一致。 此结果不支持把并发发送当作保证逐 token 等价的默认策略。首轮采集因客户端错误地要求中间事件携带最终槽号而被排除；[采集记录](../reports/qwen72b/stream/collection.json)说明修正和排除范围。

[四条后续请求什么时候收回预制成本？](qwen72b-bursts.md) 这组实测将预制请求计入总时间，并单独检查每条输出。

[两条请求争同一槽时的联合分配与反序对照](qwen72b-assignment.md)。
