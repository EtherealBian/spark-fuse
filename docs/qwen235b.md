# 单台 DGX Spark：Qwen3-235B-A22B 的容量和冷请求基线

这里回答两个问题：两个 Q2_K 分片能否在一台 Spark 上完整校验并装入 CUDA 后端？同一进程再次处理同一条冷请求时，输出 token 是否相同？计算来自固定版本的 llama.cpp CUDA；SparkFuse 负责容量检查、进程看护、请求和证据复核。

两片共 **85,691,001,472 字节**，逐文件 SHA-256 校验；重新读取文件头确认 94 层、每层 128 个专家选 8 个。后端确认 **95/95 层（含输出层）** 卸载到 GPU，报告 CUDA 权重缓冲 81,520.84 MiB。它不是 GPU 峰值。每秒采样系统 `MemAvailable`，装载阶段最低 35.88 GiB，请求阶段最低 35.55 GiB；两阶段分别采样 305、30 次。

| 输入 token | 预填充（秒） | 解码（秒） | 客户端请求（秒） | 两次 16-token 输出完全一致 |
|---:|---:|---:|---:|---:|
| 128 | 0.908 | 0.829 | 1.742 | 2/2 |
| 1,024 | 4.239 | 0.831 | 5.074 | 2/2 |

每个长度两条不同提示，按正、反次序各执行一遍。表中时间是四个请求的中位数；模型已加载，目标提示的 KV 不复用。为固定解码长度，显式屏蔽本模型的终止 token。这不是自然对话、长上下文压力测试、BF16 质量对照或其他模型的速度排名。

[预填充微批 128 与 256 的四次交叉实测](qwen235b-tiles.md)。

### 复核与复跑

无需 GPU 即可从[原始回执](../reports/qwen235b/)重算：

```bash
B=benchmarks/quant_serving
python3 "$B/audit235.py" audit "$B/qwen235b-q2k.lock.json" reports/qwen235b
```

真机复跑需要空闲 DGX Spark，以及[固定版本的后端和构建参数](../benchmarks/quant_serving/README.md)。先创建实际将存储分片的目录，**在下载前**执行准入检查并保存回执。在锁文件对应的公开模型版本的 `Q2_K/` 目录下载两个分片，原名放在 `WEIGHTS` 中。脚本不会自行联网或接管已有推理进程。

```bash
set -e
B=benchmarks/quant_serving
mkdir -p "$WEIGHTS" .runtime
python3 "$B/admit235.py" transfer "$WEIGHTS" > .runtime/before-fetch-235.json
# 此时下载固定的两个 GGUF 分片；不要在下载后伪造 before-fetch 回执。
SERVER="$SERVER" WEIGHTS="$WEIGHTS" TRANSFER_RECEIPT=.runtime/before-fetch-235.json \
  OUT=.runtime/235-baseline-01 bash "$B/run235.sh"
```

脚本在启动前再次检查资源，对完整权重重新哈希并比对分片目录。连续 3 次采样低于 6 GiB 时，看护程序只通知自己已验证身份的后端进程；缺少正常结束和两阶段观测就不验收。失败文件和部分请求不能替代整轮基线。

**English.** This is a single-Spark feasibility and greedy repeatability run for one pinned two-shard Q2_K model. Raw input/output IDs, server and client timing, the mapped inference-library hashes, admissions, and sampled system memory are archived for independent review. The 16-token fixed-length test does not measure answer quality or GPU allocation peaks.

[清单修改后的回答与缓存复核（八组配对）](qwen235b-revision.md)。

[235B 查找式推测解码：四次启动、三类提示](qwen235b-lookup.md)。
