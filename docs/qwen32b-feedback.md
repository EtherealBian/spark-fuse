# Qwen2.5-32B：一个更快的 CUDA 算子，能省多少整请求时间？

单台 DGX Spark，完整 64 层 BF16 模型。把新 token、KV 位置和输出历史留在设备上，用两个 CUDA kernel 完成词表分块 argmax 以及选中 ID 和位置的更新。下表分开报告算子和整条请求，不换算两者的加速比。

| 隔离算子：词表宽度 / 轮换行数 | PyTorch 图（µs） | 自编 CUDA 图（µs） |
|---|---:|---:|
| 32,768 / 1 | 9.964 | 2.664 |
| 32,768 / 16 | 10.478 | 2.864 |
| 152,064 / 1 | 11.596 | 3.422 |
| 152,064 / 16 | 11.617 | 3.434 |

共 48 对 CUDA event 计时；含并列最大值、NaN、带偏移量的张量、越界保护与图重放等正确性检查。不是 tokens/s、不是流式首 token 延迟。

| 全模型对照 / 输入长度 | 对照中位数（ms） | 候选中位数（ms） | 通过预设门槛 |
|---|---:|---:|---|
| 主机排队 → 设备递归 / 128 | 9,152.440 | 9,157.749 | 否 |
| 主机排队 → 设备递归 / 2,048 | 12,099.365 | 12,110.390 | 否 |
| PyTorch 递归图 → CUDA 反馈递归图 / 128 | 9,299.207 | 9,306.181 | 否 |
| PyTorch 递归图 → CUDA 反馈递归图 / 2,048 | 12,299.136 | 12,308.544 | 否 |

每组整模型对照计时 32 条完整热请求：batch=1，两条留出提示、两种输入长度，固定生成 32 个 token；从预填充到最后一个 ID 对主机可见。另取 8 份完整 `[32, 152064]` BF16 logits 校验，与已发布 eager 参考逐字节散列一致。按配对顺序和提示分别检查至少 3% 的改进。没有改变默认路径。

原生 CUDA 对 PyTorch 递归图的对照已单独测量。 原生 CUDA 对主机排队基线没有直接配对测量；不同基线的结果不能相乘。

### 离线复核（无 GPU、权重或第三方依赖）

```bash
B=benchmarks/pretrained_qwen; R=reports/qwen32b-feedback
python3 -E -B -S "$B/audit_feedback_stack.py" --reports reports --micro "$R/micro" --loop "$R/loop" --native "$R/native" | cmp - "$R/summary.json"
```

### 在 DGX Spark 重跑

从仓库根目录、已配置 CUDA/PyTorch/Transformers 的环境运行；见[32B 检查点与环境要求](qwen32b.md)。为各次运行提供不同且尚不存在的输出目录。整模型启动前校验固定权重，获取独占锁，要求系统至少 84 GiB 可用内存和 2 GiB 可用磁盘；运行途中单独采样系统内存，不将其当作 GPU 峰值。

```bash
export PYTHON=/path/to/venv/bin/python WEIGHTS=/path/to/pinned-qwen32b
OUT=/path/to/new-micro bash "$B/run_feedback_micro.sh"
OUT=/path/to/new-loop bash "$B/runloop32.sh"
OUT=/path/to/new-native bash "$B/runmodel.sh"
```

**English.** A two-kernel BF16 greedy-feedback operator is compared with PyTorch under captured graphs. End-to-end requests use the same checkpoint, fixed output budget and reference logits. Read each paired comparison separately: a faster isolated operator is not a production serving speedup.
