# 32B：第 6 个 token 为什么变了？

[前一次测试](qwen32b-fork.md)中，一条 128-token 提示在复用前缀后改变了第六个输出。这一次固定每一步的输入，区分两个操作：**把预填充分两段计算**，以及**将前缀快照复制回缓存**。如果让输出自由回馈，第六步之后三路比较的就不再是同一输入。

在一台 DGX Spark 上加载相同的 Qwen2.5-32B-Instruct BF16 权重。8 条分支提示，每条比较整段预填充、原地分段、分段并恢复到刻意写脏的缓存；正反顺序各测一次，共 48 条轨迹。每条记录 8 步、每步 152,064 个 BF16 logits，同时取预填充结束时全部 64 层 K/V 的三个位置：前缀末尾、后缀开头、输入末尾。

| 输入 / 前缀 | 整段 vs 原地分段：8 步 logits 全相同 | 原地分段 vs 快照恢复：8 步 logits 全相同 | 整段 vs 原地分段出现不同贪心 token |
|---|---:|---:|---:|
| 128 / 96 | 0/4 分支 | 4/4 | 1/4，第 6 步 |
| 2,048 / 1,536 | 4/4 分支 | 4/4 | 0/4 |

短输入发生分歧时，整段计算中 ID `11639` 比 `5461` 高 0.125；仅改变预填充分段后，两者并列，选出 `5461`。原地分段与快照恢复的全词表 logits 和所抽取的 KV 字节在所有分支、两次顺序下均相同。**对这条提示，不拷贝快照也能重现第六步的变化。** 这不能定位到某个 CUDA 算子，三个采样位置也不代表缓存的每个位置。保持整段预填充为默认路径；本次额外导出数据的运行不作速度测试。

### 不用 GPU、权重或 PyTorch 复核

从 [GitHub Release 下载诊断数据](https://github.com/EtherealBian/spark-fuse/releases/download/qwen32b-fork-attribution-data-v1/fork32-attribution-bf16.zip)。这是测量得到的 BF16 输出和三个位置的 KV 抽样，不是模型权重。96 份原始观测中有 24 份不同的字节串，按内容去重；下载 30,144,989 字节，还原后的逻辑大小为 154,533,888 字节。仓库中的 `witness.json` 绑定索引、压缩包及大小。使用 Python 3.11+ 与 NumPy：

```bash
set -euo pipefail
B=benchmarks/pretrained_qwen; R=reports/qwen32b-fork; D=$R/attribution
python3 -B "$B/fork_witness.py" replay "$D" fork32-attribution-bf16.zip "$R" | cmp - "$D/summary.json"
```

脚本验证每份内容、重建 48 条轨迹、从 BF16 字节重新计算每一步的 argmax 和差值。[完整索引](../reports/qwen32b-fork/attribution/index.json) · [逐分支诊断](../reports/qwen32b-fork/attribution/summary.json)。要从头采集，沿用 32B 基线的固定环境，在空闲 Spark 的仓库根目录运行：

```bash
REFERENCE=reports/qwen32b-fork WEIGHTS="$WEIGHTS" OUT="$NEW_OUTPUT" PYTHON=python3 \
  bash benchmarks/pretrained_qwen/run_attribution32.sh
```

**English.** One short branch changed its sixth greedy ID under split prefill, even without a snapshot restore. With identical forced decode inputs, in-place splitting and restoring a dirty cache agreed bitwise in the full vocabulary and sampled KV at both lengths, for both run orders. This localizes the observed difference to the prefill split in this panel, not a specific kernel. The measured arrays are a separately downloadable, lossless release asset; the CPU replay does not load the model. No new speed or general equivalence claim is made.

[首层投影实测：短输入拆分后，哪些 BF16 字发生变化？](qwen32b-layer0-shape.md)
