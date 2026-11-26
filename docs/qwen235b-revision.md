# 清单改了一个值，235B 的提示缓存会不会给出旧答案？

在单台 DGX Spark 上，用已验收的 Qwen3-235B-A22B Q2_K 测了一个可精确判分的任务。给模型一份合成清单，指定一个记录编号，只要求返回它的代码。两版清单只差被查询的一个字段。分别替换靠前或靠后的字段，覆盖 16 行、96 行两种长度和两个替换方向，共八组配对。

每个目标之前都真实提交旧版清单。接着要么禁止复用，要么只复用相同的 token 前缀。两种策略交替先跑。没有给候选答案加偏置，也没有屏蔽结束 token。返回的 ID 重新解码后必须与服务返回文本相同；正确、过期和其他回答全部计入。

| 策略 | 新值 / 旧值 / 其他 | 自然结束 | 只计新版请求 | 旧版与新版耗时之和 |
|---|---:|---:|---:|---:|
| 不复用 | 8 / 0 / 0 | 8/8 | 3.519 秒 | 7.430 秒 |
| 复用共同前缀 | 8 / 0 / 0 | 8/8 | 0.837 秒 | 4.312 秒 |

两种策略在 8/8 组给出逐 token 相同的回答。表中时间是八条不同目标的中位数，每种策略每条只测一次；不估算尾延迟，也不能据此推断一般问答质量或与 BF16 等价。所用推理后端是 llama.cpp。

可离线从逐请求记录重新判分，不需要 GPU 或权重：

```bash
python3 benchmarks/quant_serving/audit_revision.py benchmarks/quant_serving/qwen235b-q2k.lock.json reports/qwen235b/revision reports/qwen235b
```

真机复跑按[235B 容量基线](qwen235b.md)准备相同权重、后端和原始下载准入回执。设置新的 `OUT`、`SERVER`、`WEIGHTS`、`TRANSFER_RECEIPT` 后，执行 `bash benchmarks/quant_serving/run_revision235.sh`；`REFERENCE` 默认用 `reports/qwen235b`。

**English.** Each revised target has a matched cold and prefix-reuse request, each preceded by the old document. The scorer compares the complete trimmed answer with the current and predecessor values and reconciles returned token IDs with decoded text. Latencies exclude tokenization and detokenization; the last column sums the two real request times. Eight synthetic pairs are a cache-regression check, not general accuracy or a production SLA.
