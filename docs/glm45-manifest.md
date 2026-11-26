# 355B 能完成推理，能按接口约定只返回一个代号吗？

让 GLM-4.5 355B（UD-IQ1_S）和 Qwen3-235B-A22B（Q2_K）分别读取同样的八份合成清单。每次只改一个字段，要求只输出该行的新代号。例如接口期望 `MAPLE` 时，多返回标签或标点也不合格。两个模型都用各自的原生聊天模板与分词器；沿用相同的消息、冷请求/复用策略、贪心解码和最多 12 个输出 token。不禁用结束 token，不强制答案格式。

|严格整段判分|冷请求|复用前一版清单的前缀|
|---|---:|---:|
|Qwen3-235B：完全正确|8/8|8/8|
|GLM-4.5 355B：完全正确|0/8|0/8|
|GLM-4.5：完整输出恰好等于旧值|0/8|0/8|
|GLM-4.5：自然结束|5/8|6/8|
|GLM-4.5：目标请求耗时中位数|5.38 秒|1.32 秒|

**0/8 是这个接口的真实失败，不等于模型八次都取错了字段。** 事后单独检查返回边界：只接受纯文本，或一个位于开头、完整闭合的 `<think>…</think>` 标签之后的文本；不从未闭合内容里搜索答案，不去标点。冷请求有 4 个最终字段完全正确、1 个其他、3 个未闭合并触及长度上限；复用时分别为 5、1、2。这个诊断没有更改上表。两种缓存策略只有 5/8 组生成 ID 完全一致。两个模型之间不比较 token ID。

这些结果来自真实单台 DGX Spark；GLM 再次完整校验两片权重并核对后端和缓存几何。目标耗时不含前一版清单的生成，原始数据另列 primer。这里只比较八份清单，不能据此拆分模型、量化与模板各自的影响，也不是通用能力排名。

### 复核：不用 GPU 或权重

```bash
set -euo pipefail
B=benchmarks/quant_serving; D=reports/glm45-355b/manifest; G=reports/glm45-355b; Q=reports/qwen235b/revision
python3 -E -B -S "$B/audit_glm_manifest.py" "$B/glm45-udiq1s.lock.json" "$D" "$G" "$Q" | cmp - "$D/summary.json"
python3 -E -B -S "$B/manifest_answer_boundary.py" "$D" | cmp - "$D/answer-boundary.json"
```

[逐项判分与两模型配对结果](../reports/glm45-355b/manifest/summary.json) · [输入、输出 ID 与逐请求记录](../reports/glm45-355b/manifest/rows.json) · [单独的输出边界检查](../reports/glm45-355b/manifest/answer-boundary.json)。

真机复跑沿用[容量基线](glm45-355b.md)的后端、两片权重和下载前准入回执。确保设备空闲；保留两个已验收参考目录：

```bash
GLM_REF=reports/glm45-355b QWEN_REF=reports/qwen235b/revision \
  SERVER="$SERVER" WEIGHTS="$WEIGHTS" OUT="$NEW_OUTPUT" TRANSFER_RECEIPT="$TRANSFER_RECEIPT" \
  bash benchmarks/quant_serving/run_glm_manifest.sh
```

**English.** The 355B model ran on one Spark, but none of its eight whole responses satisfied the code-only contract under either cache policy; Qwen-235B satisfied all eight in each arm. A separate retrospective answer-field diagnostic found four and five exact final fields respectively. It does not change the strict scores. The experiment uses native templates, natural stopping or a 12-token cap. This is a bounded serving-contract test, not a model ranking.
