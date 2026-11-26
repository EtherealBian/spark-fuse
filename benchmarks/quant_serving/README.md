# DGX Spark：量化模型的请求与缓存实验

这里研究的是**请求如何进入推理后端**。底层量化矩阵运算及 attention 来自固定版本的 llama.cpp CUDA 实现。SparkFuse 提供前缀路由、配对实验、验收和离线复核工具。它与 32B BF16 的 SparkFuse CUDA SwiGLU 实验不是同一条计算路径。

以下是复现工具覆盖的实验设计；已取得的真机结果以项目主 README 所链接的报告为准。

| 实验 | 改动与对照 | 独立检查 |
|---|---|---|
| 72B 单槽 | 同一提示、同一先行请求，关闭 / 开启槽内前缀复用；256 与 2,048 token 各四组 | 从原始输入 ID 重算最长前缀、核对后端实际处理量、24 个贪心输出 token 逐个比较 |
| 72B 双槽 | 两个空闲槽的轮询 / 最长前缀选择，比较交替流量和折返流量；请求串行，不代表并发吞吐 | 两种策略处理同一批 1,024-token 请求；独立重放每次槽选择，逐对比较 16 个输出 token |
| 72B 前缀预制 | 同一个新提示在两个冷对照、全长公共前缀与 128 / 512 对齐的前缀间比较；各臂独立准备 | 区分准备成本与命中延迟，核对实际命中边界、24 个输出及首个分歧；回本次数只是按该对请求估算 |
| [72B 联合分配](../../docs/qwen72b-assignment.md) | 两条同时就绪的请求、两个已暖槽 | 逐条输出、实际重算 token 数和完成时间 |
| [72B 四分支复用](../../docs/qwen72b-bursts.md) | 冷启动、按需和显式预制，共 144 条目标和 12 条预制请求 | 计入全部准备时间，逐分支复核回本条件和输出差异 |
| [72B 双客户端流式调度](../../docs/qwen72b.md#cold-streaming) | 两条请求同时就绪，比较串行与并发，每次保留逐 token 时间线 | 分开复核排队、首个和最后输出，以及配对输出是否一致 |
| 120B MoE | 固定逻辑 batch=512，用 128 / 512 / 512 / 128 四次独立启动改变物理 microbatch | 关闭跨请求前缀复用，对同一提示核对 token 并保留各次预填充、解码与端到端时间 |
| [235B 单机容量](../../docs/qwen235b.md) | 逐片核对 Q2_K 权重，两条 128-token 和两条 1,024-token 提示各执行两次 | 核对 95/95 层放置、装载与请求阶段的系统可用内存，以及两次输出是否一致 |
| [235B 旧值回归检查](../../docs/qwen235b-revision.md) | 同一字段的新旧清单，两种缓存策略 | 精确判分、自然结束、记录含预填的总时间 |

固定输出长度时，所有实验臂显式屏蔽本次后端识别的 EOG token；验收拒绝提前停止或包含被屏蔽 ID 的输出。这是性能控制条件，不代表自然结束的对话行为。

完整分片清单固定在相应 `*.lock.json` 中，运行前逐一校验 SHA-256。计时区间不含权重下载、加载或分词；`prompt_ms`、`decode_ms` 取自后端，`wall_ms` 是本机 HTTP 请求往返。它们不等同于流式首 token 延迟。输出只对本实验的提示及生成长度有效；没有评估回答质量或与原始 BF16 权重的接近程度。

[专家权重分布：72B / 120B / 235B 的逐张量核算](../../docs/gguf-expert-bytes.md)。

[64 槽的 CPU 路由开销与回退对照](../../docs/slot-matchers.md)。

### 复现

在同一台空闲 DGX Spark 上，用 CUDA 13.0 为 `sm_121` 编译 llama.cpp **b8183**（提交 `66d65ec29ba7c440cbc31b6f63b74a17b536ba65`）。从该提交的 GitHub 源码归档构建时，可核对归档 SHA-256 `42467a78960ed5eae1aa7a9231e9ce35bdb418f65eff5d7a9f266400fbf58f33`。指定版本标识可防止 CMake 误用外层项目的 Git 提交。

```bash
cmake -S "$LLAMA_SRC" -B "$LLAMA_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_NUMBER=8183 -DLLAMA_BUILD_COMMIT=66d65ec
cmake --build "$LLAMA_BUILD" --target llama-server -j 8
export SERVER="$LLAMA_BUILD/bin/llama-server"
```

从 SparkFuse 仓库根目录运行；`WEIGHTS` 指向 72B 的 12 个 GGUF 分片目录，`MODEL` 指向已下载的 120B MXFP4 GGUF。输出目录必须是新的；服务仅监听回环地址。前缀预制实验默认取已发布的基线；本机重新构建后，先跑 prefix，再把其目录设为 SOURCE_DIR，以匹配实际二进制。

```bash
mkdir -p .runtime
B=benchmarks/quant_serving
WEIGHTS="$WEIGHTS" PANEL=prefix OUT=.runtime/prefix-01 bash "$B/run72.sh"
WEIGHTS="$WEIGHTS" PANEL=routing OUT=.runtime/routing-01 bash "$B/run72.sh"
# 使用同一构建的已验收 prefix 记录；也可显式取本机新跑的配对基线：
SOURCE_DIR=.runtime/prefix-01 WEIGHTS="$WEIGHTS" PANEL=catalog OUT=.runtime/catalog-01 bash "$B/run72.sh"
python3 "$B/audit_catalog.py" audit .runtime/catalog-01 .runtime/prefix-01
MODEL="$MODEL" OUT=.runtime/moe-01 bash "$B/run120.sh"
python3 "$B/audit_replay.py" .runtime/prefix-01/rows.json
python3 "$B/audit_routes.py" .runtime/routing-01/rows.json .runtime/routing-01/trace-inputs.json
python3 "$B/moe_blocks.py" audit .runtime/moe-01
```

`gguf_budget.py` 可仅凭单文件 GGUF 的有界文件头读取张量目录，估算给定槽数、上下文长度下不截断历史的 FP16 KV 大小。这个估算不包含工作区、分配器或操作系统开销，也不能证明模型成功启动。独立 CPU 测试：`python3 -B -m unittest discover -s benchmarks/quant_serving`。

**English.** These reproducible pilots isolate single-slot prefix reuse, two-slot dispatch and physical prefill microbatch size. The CUDA inference backend is upstream llama.cpp; SparkFuse supplies the request policy and evidence checks. Timing and greedy-token equivalence are reported separately. The two-slot experiment submits requests sequentially and does not measure multi-client throughput. A favorable GGUF capacity estimate is not an inference result.
