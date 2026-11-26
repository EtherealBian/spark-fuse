# 64 个 KV 槽，能不能少算一些前缀？

此处测量 DGX Spark **单个 CPU 核**对整批请求分配缓存槽的耗时，不启动模型或 GPU。常规实现为每对候选请求和槽比对 token，然后求全局分配。新实现先按租户和前 16 个 token 分桶，只在可能命中的桶内计算精确公共前缀；租户、槽次序、平局处理和返回的租约保持不变。这里不储存跨请求的索引。

| 合成场景 | 槽数 | 常规扫描（ms） | 分桶后（ms） |
|---|---:|---:|---:|
| 同租户，所有前缀未命中 | 64 | 24.070 | 5.924 |
| 多个槽相同前缀 | 64 | 414.503 | 37.627 |
| 共同前缀恰好达到阈值或更长 | 64 | 11.937 | 8.923 |
| 两个槽，回退对照 | 2 | 0.400 | 0.403 |

最早的全排序方案在同租户未命中时从 23.985 ms **变慢到 28.894 ms**，未通过准入。它的原始 48 对数据和代码一并保留。分桶方案另测 80 对；九个 64 槽场景里，每个场景的两种运行先后次序分别计算，配对节省中位数均超过 10%。这些是完整 `acquire_batch` 的计时，不包括填充缓存、请求排队或生成 token。

默认仍使用 `scan`。在服务端对请求完成身份鉴别、提供可信的作用域后，可用 `SlotRouter(64, 2048, min_prefix=16, batch_matcher="scoped16")` 选择分桶路径；只对 64 条同时就绪、64 个可用槽、输入各 2,048 token、槽为空或同长、联合分配且阈值为 16 的组合启用，其余回退扫描。这个路由器本身不是认证边界。

复核原始结果无需权重：

```bash
B=benchmarks/quant_serving
python3 -B "$B/audit_slot_matchers.py" reports/slot-matchers | cmp - reports/slot-matchers/summary.json
# 重测分桶方案；写新的文件，不覆盖原始证据
PYTHONHASHSEED=0 python3 -B "$B/bench_profiled.py" run /tmp/slot-scoped-new.json
python3 -B "$B/bench_profiled.py" audit /tmp/slot-scoped-new.json
```

**English.** This measures complete joint slot reservation on one pinned Spark CPU core, using synthetic traces. Each lease was verified against an independent case oracle. An earlier ordered index failed the same-tenant miss control; the subsequent scope-and-prefix index passed the measured holdouts. Neither result is a throughput or latency measurement of an LLM service.
