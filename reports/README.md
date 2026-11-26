# 测量档案 / Measurement archive

先读[三项代表性结论](../docs/findings.md)。本目录保存的是实验记录；不同模型的时间不能直接排成性能榜。

| 要核对的量 | 从哪里读 |
|---|---|
| 一次前向发射了多少次 CUDA kernel | [四层 BERT 的图、源码和计数口径](../docs/performance.md#stacked-bert) |
| 两条请求都完成需要多久，快照何时回本 | [32B 原始波次](qwen32b-fork/raw.json) → [逐对成本](qwen32b-fork/paired-costs.json)；[相同输入改变共享切点](../docs/prefix32-closeout.md) |
| 同时放行是否拖慢先完成的请求 | [355B 双请求的两条时间线](../docs/glm45-paired.md) |

从仓库根目录运行 `bash audit/replay-key-results.sh`，用 CPU 从原始记录重建报告。`raw.json` 是采集记录，`timing.json` 和 `summary.json` 是推导结果；重放不会产生新的 GPU 耗时。较大的全词表观测放在对应实验的 release 附件中，不含权重。历史文件保持原路径，已有报告与校验程序仍可定位证据。

**English.** Start with the [findings](../docs/findings.md#english). This directory preserves measurements and derived reports, not a cross-model leaderboard. The CPU replay checks archived evidence; it is not a fresh performance run.
