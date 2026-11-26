# Qwen2.5-32B：单台 DGX Spark 的全模型 BF16 实测

[32B 原始检查点](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/tree/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd)含 32,763,876,352 个 BF16 参数。单台 GB10 运行全部 64 层，未量化，也没有把层卸载到 CPU。这里检验的是推理实现，不评测模型回答质量；此前的 7B 是 Qwen2，本轮是 Qwen2.5，不能把两轮的差别只归因于参数量。

**一次解码如何复用权重？** 每层把 `gate_proj` 与 `up_proj` 的冻结权重按行拼接，核对两半的内容后释放原权重。预填充用这份矩阵的两个视图，仍做两次 GEMM；单 token 时一次 GEMM 得到两份投影，交给 SparkFuse 生成的 CUDA SwiGLU，再做输出投影。GEMM 和 attention 仍由 PyTorch 执行。如果为 decode 单独留一整份合并权重，需要 `64 × 2 × 27648 × 5120 × 2` 字节，即 **33.75 GiB**。实际转换前后，模型 CUDA 张量按唯一存储去重为 **65,527,752,960 → 65,527,752,960 字节**；这不含 KV、图池和临时张量。

| 输入 token | 后端 | 首 token P50 (ms) | 后续 P50 / P95 (ms/token) | 热段峰值分配 (GiB) | 严格输出门槛 |
|---:|---|---:|---:|---:|---|
| 128 | PyTorch eager | 351.6 | 284.52 / 285.29 | 61.186 | 通过 |
| 128 | SparkFuse 共享权重 | 367.6 | 284.52 / 303.52 | 61.187 | 通过 |
| 2048 | PyTorch eager | 2642.6 | 310.84 / 311.88 | 62.066 | 通过 |
| 2048 | SparkFuse 共享权重 | 2521.4 | 310.39 / 311.38 | 62.065 | 通过 |

**验收：通过。** 每种长度选两条未参与预热的提示；每条自由生成 32 个 token，再在相同前缀上逐步比较 152,064 维的整份 logits，与不重放 CUDA Graph 的 eager 对照。计时另外执行四个请求，每个请求还要核对输出 token；decode P95 是这四个请求共 124 次后续 token 延迟的第 95 百分位。记录保留 token、logits 摘要与逐步最大绝对误差。小样本且后端顺序固定，不能据此断言速度胜负；不改变默认实现。

**测量边界。** Batch=1，SDPA。eager 与 SparkFuse 在不同进程依次加载，均在 GPU 上使用 BF16 权重和静态 KV，均只对 decode 重放 CUDA Graph。每个后端先 128 后 2,048，两个长度分开建图；每请求首 token 来自 prefill，后面 31 个依赖上一步输出。逐 token 同步，含 argmax，不含权重加载、构图、分词和网络。显存列为 PyTorch CUDA allocator 的峰值分配，不等同于整机统一内存。提示以重复文本构造固定长度，不代表长文理解、多用户吞吐或服务端并发。此次没有测整模型 Inductor，也不把本轮 32-token 延迟与 7B 的 128-token 生成直接对比。

[设备端 token 反馈：CUDA 算子与整请求对照](qwen32b-feedback.md)。

### 复核与复现

- [汇总](../reports/qwen32b-summary.json) · [eager 原始记录](../reports/qwen32b-eager.json) · [共享权重原始记录](../reports/qwen32b-resident.json) · [测试回执](../reports/qwen32b-validation.json)。汇总可用 Python 3.11+ 在 CPU 上复算：`python3 benchmarks/pretrained_qwen/audit32b.py reports`。
- 在空闲的同配置机器上，用 `huggingface-cli download Qwen/Qwen2.5-32B-Instruct --revision 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd --local-dir "$WEIGHTS"` 获取检查点（先设 `WEIGHTS` 为空目录）。在项目的 CUDA PyTorch 环境安装 `transformers==4.51.3` 和 `accelerate==1.6.0`。配置与每个分片均经过[固定摘要](../benchmarks/pretrained_qwen/qwen32b.lock.json)校验，驱动只加载本地 safetensors，不执行远端代码。
- 在仓库根执行 `WEIGHTS="$WEIGHTS" PYTHON=python3 bash benchmarks/pretrained_qwen/run32b.sh`。默认结果在私有工作目录，不覆盖发布记录。完整 BF16 权重约 65.53 GB，不进入 Git 仓库。

**English.** A single DGX Spark ran all 64 layers of the dense Qwen2.5-32B checkpoint in BF16, without CPU layer offload. Prefill and single-token decode read the same gate/up weights; a separate packed copy would cost 33.75 GiB. The table reports warm synchronized request timings under matched decode graph replay. The strict gate checks exact free-running tokens and byte-identical logits against non-replayed eager for two prompts per length. This is a bounded batch-one pilot, not a serving or model-quality benchmark.
