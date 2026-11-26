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

## 复核

[原始输入、输出与运行回执](../reports/qwen72b/) · [复算汇总](../reports/qwen72b/summary.json) · [重新运行实验](../benchmarks/quant_serving/README.md)。用 Python 3.11+ 在 CPU 上运行 `python3 benchmarks/quant_serving/audit_bundle.py 72b reports/qwen72b`。汇总从原始记录重算，不读取发布的 summary；同时核对权重指纹、实际映射的推理库、槽位、逐请求 token 和计时。

**English.** A single DGX Spark runs the dense 72B model in Q4_K_M using upstream llama.cpp CUDA. SparkFuse dispatches requests and audits the evidence. The paired single-slot test measures prefix reuse; the two-slot test contrasts round-robin and prefix affinity on an alternating control and a switchback trace. Results above are local serial request timings, not concurrent throughput or a quality benchmark. On the switchback trace, affinity reduced six-request prefill work from 6,144 to 2,088 tokens; 11 of 12 paired outputs matched. The single-slot 256-token cohort also contains a disagreement.
