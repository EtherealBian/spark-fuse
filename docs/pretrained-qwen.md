# 用预训练 Qwen2 实测逐 token 延迟 / Pretrained Qwen2 decode

**结论：在 DGX Spark 上，合并前馈网络的两次投影能缩短完整模型的解码时间，但相对已有融合实现只省约 1.6%，且要额外保留 399 MiB 权重。此实现不作为默认路径。**

模型是 [Qwen2-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/tree/c540970f9e29518b1d8f06ab8b24cba66ad77b6d)，24 层，BF16。比较的是同一检查点、词表、SDPA 实现、batch=1 和 137 槽静态 KV 缓存。以下全是单机、单请求、greedy 解码结果，不代表模型质量、长上下文或并发服务吞吐。

## 从 kernel 到完整请求

每个长度有 3 条未用于预热的提示，12 轮打乱后端顺序的配对测量。每次生成 9 个 token：首 token 单独计时，接下来的 8 个从模型自己的上一步输出继续生成，每步同步。`TTFT` 从 GPU 上的输入已就绪计到第一个 token 可用；不含分词、网络、缓存清零或首次编译。CUDA Graph 仅用于 decode，所有后端都给相同的重放机会。

| 提示长度 | 普通提交的首 token：eager / SparkFuse / Inductor (ms, P50) | 普通提交的 decode：同序 (ms/token, P50) | CUDA Graph decode：同序 (ms/token, P50) |
|---:|---|---|---|
| 32 | 9.56 / 9.52 / 7.21 | 9.26 / 9.16 / 7.25 | 7.07 / 7.03 / 6.19 |
| 128 | 10.59 / 10.63 / 8.07† | 9.40 / 9.31 / 7.29† | 7.23 / 7.21 / 6.26† |

SparkFuse 用现有 `torch.export` → CUDA 编译器生成 `SiLU(gate) × up`，在每个 MLP 中替换这两步，其余路径保留 Transformers 实现。以 32 token 为例，普通逐 token 解码剖析记录了 1,088 → 1,064 个 GPU 事件；eager 耗时最高的三个矩阵计算 kernel 约占 GPU 事件累计时间的 80%。这些比例不是端到端延迟的占比。

† Inductor 在 128 长度的一个提示上从第 5 个生成 token 起不同于 eager；仍公布时间，但不把这组算作等价结果。在同前缀比较中，它仍满足本实验的 logits 容差。SparkFuse 在两个长度上均通过逐 token 和 logits 校验。[原始计时、首调用耗时、logits 误差和剖析](../reports/pretrained-qwen-gb10.json) · [CPU 审计结果](../reports/pretrained-qwen-audit.json)

## 追踪矩阵投影：少发射一次值多少钱？

只改 **单 token 解码**：把同一输入的 gate/up 两个无偏置线性层权重按行拼接，每层运行一个矩阵乘，再把结果按列分成两段。预填充沿用两次投影；原权重和精度不变。这版为保留原有预填充路径额外存储 **399 MiB**。只用于已冻结的权重，换权重必须重新打包。

用 **4 条新提示 × 2 种长度**，每长度仍为 12 个随机顺序配对窗口。下表所有方法都重放完整 decode；每格为 **P50 / P95，ms/token**。

| 方法 | 32 token | 128 token |
|---|---:|---:|
| 分开投影 + 普通门控 | 7.07 / 7.47 | 7.24 / 7.90 |
| 分开投影 + SparkFuse 门控 | 7.04 / 7.43 | 7.20 / 7.80 |
| 合并投影 + 普通门控 | 6.97 / 7.36 | 7.13 / 7.68 |
| **合并投影 + SparkFuse 门控** | **6.93 / 7.31** | **7.08 / 7.44** |
| 整模型 Inductor + 同样的重放 | 6.20 / 6.52‡ | 6.26 / 6.93 |

‡ 本轮 Inductor 在 32 长度的一条新提示上未通过 token 一致性检查；128 长度通过。其余四种组合均通过这 8 个提示/长度组合。每步剖析中，从分开投影加融合门控到两者都融合，GPU 事件从 1,064 降至 1,040。首轮重新排布并逐元素检查全部权重约 61 ms，一次性支出，未摊入逐 token 时间。[独立提示、逐轮计时与剖析](../reports/pretrained-projections-gb10.json) · [第二组审计](../reports/pretrained-projections-audit.json)

**选型规则。** 两个长度都要满足：整段 greedy token 与 eager 完全相同，同前缀逐步 logits 的余弦相似度 ≥ 0.999 且最大绝对误差 ≤ 1；在可比的 Inductor 对照上，合并样本的 decode P50 降低 ≥ 10%，P95 恶化不超过 5%，至少 8/12 个配对窗口也达到 10%。两个实验均未达到部署要求。测试依据只覆盖给定提示，不是语义质量认证。

## 复现 / Reproduce

Use the published environment (Linux, NVIDIA GB10, CUDA 13.0, PyTorch 2.13.0+cu130, Transformers 4.51.3). Download the four pinned upstream files into a local, untracked directory; the driver checks the safetensors SHA-256 and three Git blob IDs before loading. No custom model code is loaded. The two reports include the executable driver hashes; `model_load_and_h2d`, first requests and replay setup are separate from warm timings. First-use measurements reflect load order and local compiler caches; they are not comparable cold-start benchmarks.

```bash
source tools/env.sh  # use a Python environment with the above packages
D=.runtime/pretrained-qwen-v1/weights; mkdir -p "$D"
REV=c540970f9e29518b1d8f06ab8b24cba66ad77b6d
for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do
  curl -fL --retry 2 "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/$REV/$f" -o "$D/$f" || exit 1
done
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false TORCHINDUCTOR_COMPILE_THREADS=4
B=benchmarks/pretrained_qwen
python "$B/generate.py" --weights "$D" --backend packed --replay --max-new-tokens 9 --prompt 'What is two plus two? Answer with one number.'
python "$B/run.py" --weights "$D" --output /tmp/qwen.json
python "$B/projection_panel.py" --weights "$D" --output /tmp/projections.json
```

**Verify without a GPU, model weights, or third-party packages:**

```bash
python3 -I -S benchmarks/pretrained_qwen/audit.py reports/pretrained-qwen-gb10.json
python3 -I -S benchmarks/pretrained_qwen/audit_projections.py reports/pretrained-projections-gb10.json reports/pretrained-qwen-gb10.json
```

**English result.** A frozen gate/up projection pair reduces median full-model decode latency by 1.6–1.7% relative to separate projections with the same generated SwiGLU kernel and CUDA Graph replay. The packed path passed the token and logit gates on all eight held-out prompt/length pairs. A same-device, whole-model Inductor control was faster on the eligible 128-token projection panel. The timings for incompatible controls are retained, not counted as wins. The extra 399 MiB and modest savings do not warrant a default change.
