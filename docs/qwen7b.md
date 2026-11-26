# Qwen2-7B：一份投影权重，两个推理阶段 / Single-bank 7B pilot

[固定检查点](https://huggingface.co/Qwen/Qwen2-7B-Instruct/tree/f2826a00ceef68f0f2b946d945ecc0477ce4450c)，原始 BF16、28 层、batch=1、SDPA，在 DGX Spark 上运行。问题是：原有解码优化如果用于 7B，需要多保留约 **7.08 GiB** 的 `gate_proj` / `up_proj`。能否让预填充和解码读取同一份权重？

**内存结果。** 每层把两块冻结权重按行拼接，校验内容，再释放旧的线性层；一次只转换一层。预填充仍做两次线性投影（矩阵是同一 bank 的两个行视图），单 token 解码则做一次投影并调用 SparkFuse 生成的 CUDA SwiGLU。逐地址去重统计所有 CUDA 参数与 buffer，转换前后均为 **15,231,233,280 字节**，常驻模型张量增量 **0 字节**。这不意味着 KV、图内存池、临时张量或 PyTorch 保留内存也是零。该实验实现仅用于冻结后的推理模型，不用于训练，也不改变发布检查点的格式。

### 单请求实测

| 输入 token | 后端（均重放 decode） | TTFT P50 (ms) | 后续 P50 / P95 (ms/token) | 热段峰值分配 (GiB) | 输出门槛 |
|---:|---|---:|---:|---:|---|
| 128 | eager | 86.4 | 79.09 / 79.92 | 14.352 | 通过 |
| 128 | SparkFuse 共享权重 | 86.5 | 78.33 / 79.20 | 14.352 | 通过 |
| 128 | Inductor | 79.1 | 78.29 / 79.15 | 14.352 | 未通过 |
| 2048 | eager | 532.1 | 87.03 / 87.92 | 14.726 | 通过 |
| 2048 | SparkFuse 共享权重 | 542.4 | 86.15 / 86.96 | 14.722 | 通过 |
| 2048 | Inductor | 453.1 | 85.72 / 86.61 | 14.727 | 未通过 |

与通过输出检查的 eager 对比，两个输入长度的解码 P50 分别降低 **0.96% 和 1.01%**；2,048-token 的首 token 反而慢 10.3 ms。共享权重版在 4 个「提示 × 长度」组合上，每个生成 token 相同，同前缀 128 步的最大 logits 绝对误差均为 **0**。这组数据最有把握的结论是避免重复权重，不能从约 1% 的差距推出普遍加速。

128/inductor/提示 1：首个不同输出 token = 10，同前缀最小 cosine 0.997710、最大绝对误差 0.469；128/inductor/提示 2：首个不同输出 token = 无，同前缀最小 cosine 0.998188、最大绝对误差 1.156；2048/inductor/提示 1：首个不同输出 token = 38，同前缀最小 cosine 0.996951、最大绝对误差 0.500；2048/inductor/提示 2：首个不同输出 token = 89，同前缀最小 cosine 0.998204、最大绝对误差 0.781。未通过的行只作诊断，不用来宣称等价加速。每个后端、每种长度各四个计时请求，覆盖两条不同提示；没有跨进程随机配对，不足以认定部署胜出或跨任务泛化，默认实现不变。

- 每请求固定生成 128 个 token：首个来自 prefill，127 个为依赖上一步输出的 decode。固定工作量，即使遇到 EOS 也继续；不是对话服务。128 和 2,048 长度的输入由两种固定文本按 token 重复填充，覆盖的是序列长度、不是长文理解质量。
- 每个后端/长度独立进程，同样的缓存容量（输入长度 + 128）、完整 decode 的 CUDA Graph 和 warm-up。Inductor 为整模型 `fullgraph=True`，禁用其内部 CUDA Graph，避免嵌套。prefill 不重放；每 token 同步计时，报告包含 argmax，排除首次编译、权重加载、KV 清零、分词和网络。
- 普通 eager、不重放的 128 步轨迹作独立参照。所有后端检查自由生成 token 完全一致，并在共同前缀上检查每一步的 **cosine ≥ 0.999、最大绝对误差 ≤ 1**。计时请求也必须产生同一串 token，计时前后图计数不得增加。原始记录保留不一致的位置与误差。
- 内存口径为 PyTorch CUDA allocator 的已分配峰值（非整机统一内存总用量）。保留池大小、含准备阶段的峰值、原始时间、token 和驱动源码哈希均可从记录查到。[CPU 可复核汇总](../reports/qwen7b-summary.json)；同目录 `qwen7b-{128,2048}-{eager,resident,inductor}.json` 为六份原始报告。

### 复现 / Reproduce

使用同一 CUDA / PyTorch / Transformers 环境（版本见原始记录），GPU 不应有其他工作负载。原始检查点约 15.23 GB；验证全部上游 Git blob 和 4 个 safetensors 文件的 SHA-256 后才加载，不执行远端 Python 代码。`WORK` 下生成的教师 logits、编译缓存不提交。

```bash
export WEIGHTS="$PWD/.runtime/pretrained-qwen-7b/weights"
mkdir -p "$WEIGHTS"
REV=f2826a00ceef68f0f2b946d945ecc0477ce4450c
for f in config.json tokenizer.json tokenizer_config.json model.safetensors.index.json model-0000{1,2,3,4}-of-00004.safetensors; do
  curl -fL --retry 2 "https://huggingface.co/Qwen/Qwen2-7B-Instruct/resolve/$REV/$f" -o "$WEIGHTS/$f" || exit 1
done
export PYTHON=python3
bash benchmarks/pretrained_qwen/run7b.sh
# Recompute the published summary in a CPU-only Python 3.11+ environment (no torch required):
python3 benchmarks/pretrained_qwen/audit7b.py reports
```

**English.** One frozen bank per layer serves both split prefill and packed decode. Model tensor storage stays at 14.185 GiB; this avoids a permanent 7.08 GiB second copy rather than claiming a reduction in whole-machine memory. All three implementations use the same external decode graph and BF16 weights. Compared with eligible eager, decode P50 fell by 0.96% and 1.01%; long-prompt TTFT rose by 10.3 ms. All resident-path teacher-forced logits matched eager exactly on the measured trajectories. This is a two-prompt-per-length pilot, not a deployment benchmark: rejected controls are retained and no default changes.
