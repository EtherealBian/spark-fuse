# 两条请求争用同一 KV 槽，先来先选合算吗？

同一台 DGX Spark，同一个 Qwen2.5-72B Q4_K_M 服务，两个槽。两条请求同时就绪时，贪心法按到达顺序选最长前缀；联合分配一次选定两个不同的槽，使合计可复用 token 最多。三个合成场景各做两个配对重复，交换策略执行顺序。每条目标输入 1,024 token，固定生成 16 token。

| 场景 | 实际重新处理的输入 token（贪心 → 联合，每波） | 两请求都完成的配对节省中位数¹ | 输出逐 token 一致 |
|---|---:|---:|---:|
| 首条先占竞争槽 | 1,088 → 768 | +1.288 秒 | 4/4 |
| 同一竞争关系，反转就绪次序 | 768 → 768 | -0.009 秒 | 4/4 |
| 两个请求各有所长 | 448 → 448 | -0.123 秒 | 4/4 |

¹ 从两条目标就绪、开始选槽到两个客户端都收到最终结果，包含分配和本地 HTTP；正数表示联合更快。每种策略之前都发两条真实预填请求建立相同槽状态，其开销不计入目标窗口。客户端请求时间重叠，不等于 CUDA kernel 同时运行。

只在贪心法把高价值槽提前给错请求时看到明确收益。反序与各取不同槽的对照没有显示提速；这是 12 组配对输出，不足以估计线上 P99。

复核不需要 GPU 或权重：

```bash
python3 benchmarks/quant_serving/audit_batch.py reports/qwen72b/assignment
```

真机复跑：配置[后端和权重](../benchmarks/quant_serving/README.md)，用新的 `OUT` 并执行 `PANEL=assignment bash benchmarks/quant_serving/run72.sh`。

**English.** Joint assignment maximizes the total eligible prefix across distinct slots. On a single Spark, the arrival-order trap required 320 fewer input tokens per two-request wave. Reversing the order or giving the requests distinct preferred slots removed the advantage. The raw artifact includes all 24 target responses and 24 preparation responses, not just the winning case.

[72B 在线 FIFO：等待线程 CPU 与最后一个响应时间分别验收](qwen72b-fifo.md)。
