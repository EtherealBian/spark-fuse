# Attention score campaign · DGX Spark

The planned score/mask/softmax sweep finished at **2026-11-26 01:32:59 UTC+08:00**, when the worklist ran out. This is the final attention panel only. The general Transformer panel runs independently.

| Workload | Indexed | All three backends passed | SparkFuse faster than Inductor, device | Median Inductor/SparkFuse, device | SparkFuse faster, synchronized host |
|---|---:|---:|---:|---:|---:|
| Prefill-like scores | 192 | 184 | 93 / 184 | 1.0002× | 0 / 184 |
| Decode-like scores | 108 | 106 | 32 / 106 | 0.9111× | 0 / 106 |
| All | 300 | 290 | 125 / 290 | 0.9785× | 0 / 290 |

A ratio above one favors SparkFuse. Medians use one paired ratio per passing case, not a ratio of pooled times. The complete synchronized-host median ratio was 0.3134×. These measurements use the campaign's subgraph benchmark; they do not measure a capture-resident serving stack or whole-model throughput. They span 22 compiler keys as development proceeded and should not be read as a fixed-release speedup.

The [compressed evidence](campaign-final-attention.json.gz) retains all 300 indexed records: 290 passing, seven parity failures, three discovery errors. No failure was converted into a speed comparison. All 69 referenced source snapshots are present; no successful case relies on implicit parity. The archive contains the original timings, candidate schedules, holdout decisions, rejection diagnostics and observed supervisor stop reason. JSON record digest: `a926887b735a5b59d2ce486dae4b38c2b6e370a5412f66f932b4245d48151f57`.

无需 GPU 即可从仓库根目录核验：

```bash
python tools/campaign_snapshot.py verify reports/campaign-final-attention.json.gz
```

Transformer 分数子图的 300 项任务已跑完。通过正确性校验的 290 项中，设备重放有 125 项快于同机 Inductor；同步主机计时没有胜例。完整结果包含失败项和对应源码，不能用最快的个例代替整组表现。
