# GB10 性能记录 / Performance evidence

### 相同 kernel，不同主机路径

`tools/host_bench.py` measures synchronized end-to-end calls, including fresh output tensors, with six alternating-order windows of 31 calls. All paths use the same examples and must pass eager parity. Median of the six window medians, µs:

| BF16 subgraph | eager | Inductor | SparkFuse Python | SparkFuse C++ |
|---|---:|---:|---:|---:|
| visible-residual RMSNorm, 32 × 1536 | 13.80 | 26.18 | 85.16 | 8.46 |
| SwiGLU pointwise part, 32 × 8960 | 11.75 | 24.34 | 80.35 | 7.99 |

[Raw timings](../reports/native-host-gb10.json). The RMSNorm device medians were 1.81 and 1.84 µs for the Python and C++ launchers: the change is in host overhead. With [actual Qwen2 library modules](../reports/qwen-native-gb10.json), the random-weight MLP measured 428.11 µs in mixed native mode vs 410.83 µs eager; this subgraph win did **not** make the full MLP faster.

### 地址局部性 / Hot vs rotating buffers

`tools/locality_bench.py` measures device replay, not synchronized host latency. Each backend captures a hot-address graph and a graph rotating through distinct activation inputs **and** retained distinct outputs. Weights remain shared; activation inputs alone cover at least 3× the 24 MiB L2. The six variants are interleaved in randomized order for 15 blocks. These are logical working sets, not measured DRAM traffic.

| BF16, device µs (hot / rotating) | eager | Inductor | SparkFuse |
|---|---:|---:|---:|
| visible-residual RMSNorm, 512 × 1536 | 6.30 / 31.02 | 10.08 / 29.93 | 6.53 / 30.05 |
| SwiGLU pointwise part, 128 × 8960 | 8.26 / 34.70 | 4.50 / 31.54 | 4.58 / 32.22 |

[Raw ordered samples, slot counts and footprints](../reports/locality-gb10.json). The difference between implementations on hot data does not predict their difference with rotating inputs.

### 不足一包的尾部 / Odd-width packets

`tools/tail_bench.py` fixes layout and threads; only vector width changes (1 vs 4). At 1, 47, 257 and 1024 rows, the median paired scalar/packed ratios were **0.75, 0.80, 1.32, 1.68** for width-1537 SwiGLU, and **1.29, 1.26, 0.84, 0.99** for width-769 RMSNorm. Ratios below 1 favor scalar. Vectors are candidates, not the default at every odd width. [All paired rounds](../reports/odd-vector-gb10.json).

Run from the repo after `source tools/env.sh`; each of the three benchmark scripts accepts `--output FILE`. Serialize runs when sharing the GPU with the campaign: `flock .runtime/gpu-execution.lock python tools/host_bench.py --output /tmp/sparkfuse-host.json`.

### Qwen2Attention：融合到真实 attention 的哪一步？

For BF16 batch 2, 12 query heads / 2 KV heads and sequence length 97, partitioning the actual `transformers==4.51.3` eager-attention implementation fused score scaling, a head-broadcast additive causal mask, FP32 softmax and a cast back to BF16. QK, PV, rotary embedding application and linear projections remained PyTorch operations. Both returned tensors (projected output and probabilities) passed the unchanged tolerance on seeds 64 and 939. Default mixed device replay was 165.09 µs vs 169.68 µs eager; synchronized host latency was 231.36 vs 210.63 µs. These are one experiment, not a general end-to-end win. Full-graph Inductor exceeded the same tolerance on both inputs, so its speed is not reported for this module. [Raw graphs, fallback reasons, samples and failure diagnostics](../reports/qwen-attention-gb10.json).

On four independent BF16 inputs to the **same real Qwen2Attention module**, activation-based calibration changed the fused score region from a 128-thread block per row to two 32-thread rows per 64-thread block, both retaining local values. The held-out **region** improvement was 2.33×. On the fourth request the whole module measured **162.72 / 158.02 / 155.03 µs** (eager / default mixed / tuned mixed) in device replay; synchronized host medians were **213.91 / 232.24 / 227.59 µs**. Each figure is the median of seven randomized-order windows. The 2.33× is not a speedup of attention as a whole. [Source inputs, applied dispatch, raw windows and audit errors](../reports/qwen-attention-calibrated.json).

### 长行分数：内存指令是否真的变宽？

With BF16 **12 × 8192** scores, holding `block/256/cache` fixed, eight-element packets were **2.21×** faster than scalar lanes in seven paired address-hot windows. On independent input seed 12572 the scalar kernel took 9.39 µs, packed 4.02 µs, eager 5.94 µs and Inductor 3.54 µs (packed comparison round). All passed parity. Inductor still won this three-way comparison. `cuobjdump` found 128-bit global loads/stores in the packed cubin, with scalar instructions also present for the alignment fallback; those are static code counts, not runtime transaction counts. At 47 × 97, the same eight-element policy was only 0.45× its scalar baseline at fixed warp layout. The compiler therefore offers packets to the tuner rather than making them mandatory. [All schedules and paired windows](../reports/softmax-packets-gb10.json) · [Independent input, generated CUDA, SASS counts and kernel events](../reports/long-score-heldout-gb10.json).

### Qwen 主尺寸：固定编译器的复验

Six BF16 primitive shapes, same compiler fingerprint `354c84411574`, separately recorded source-tree snapshots and independent tuning/confirmation/benchmark inputs. Warm CUDA Graph device medians in µs. These are constituent operations, not the complete Qwen model.

| Region | Rows × width | eager | Inductor | SparkFuse |
|---|---:|---:|---:|---:|
| resnorm2 | 1 × 1536 | 2.96 | 1.46 | 1.52 |
| resnorm2 | 32 × 1536 | 2.96 | 1.81 | 1.84 |
| resnorm2 | 512 × 1536 | 6.42 | 4.47 | 5.64 |
| gate | 1 × 8960 | 2.45 | 0.88 | 0.95 |
| gate | 32 × 8960 | 3.19 | 1.55 | 1.91 |
| gate | 512 × 8960 | 178.72 | 119.65 | 118.43 |

[All trials, source identifiers, raw device/host samples and third-input parity](../reports/qwen-current-panel-gb10.json).

### 三轴广播：因果位置偏置 / Causal position prior

A positional-bias subgraph generates both the unnormalized bias and its row-wise softmax from compact head slopes, query positions, key positions, and a head-broadcast causal mask. None has the whole output shape. This is an ALiBi-style *positional prior*, not attention with QK scores or a value product. The fused execution does not materialize the outer difference. FP32, mixed aligned/offset calibration inputs, exact bias including masked `-inf`, a third independent input and seven interleaved replay windows:

| Heads × queries × keys | eager | Inductor | SparkFuse default | SparkFuse selected |
|---|---:|---:|---:|---:|
| 8 × 1 × 513 | 6.30 | 1.56 | 2.07 | 1.73 |
| 8 × 97 × 129 | 5.61 | 3.22 | 3.87 | 2.44 |
| 12 × 128 × 1024 | 92.58 | 56.14 | 47.73 | 42.05 |

Device µs with warm addresses. Inductor is faster on the single-query case. [Exact schedules, input seeds, all raw windows, FX and parity](../reports/position-broadcast-gb10.json). The [pre-fix frontend probe](../reports/frontend-before-gb10.json) also shows why explicit mean-dtype conversion has to happen before accumulation: both narrowing cases previously yielded 0.375 rather than eager's 0.

### BERT：残差与带仿射参数的 LayerNorm

`tools/bert_output.py` exports the real Hugging Face `BertSelfOutput` with fixed random projection weights, BF16, batch 1, 97 tokens, hidden width 768, evaluation-mode dropout and epsilon 1e-12. The linear layer remains PyTorch. The add and affine LayerNorm are one generated kernel. The kernel subtracts a per-row origin before computing the mean, then accumulates squared centered values in FP32; it does not evaluate `E[x²] − E[x]²`.

On a fourth request, the entire module's **hot device** medians were **14.46 / 14.20 / 12.86 µs** for eager / default mixed / selected mixed. Its **synchronized host** medians were **27.52 / 47.09 / 45.92 µs**; eager was faster there. Mixed passed the unchanged tolerances on all four requests. Full-graph Inductor missed parity on those requests and is not assigned a speed. [Graphs, per-input errors, chosen schedule, timing order and windows](../reports/bert-self-output-guarded-gb10.json). No claim about end-to-end BERT throughput.

The intermediate implementation exposed bad GPU address calculations. We keep [the two failing generated kernels and Compute Sanitizer traces](../reports/layernorm-debug/README.md), not just the successful timings. [Stage regression: 231 tests plus installed wheel, Qwen and BERT checks](../reports/layernorm-guarded-acceptance.json).

### 完整 BERT 编码层：归约顺序与可见的舍入边界

在真实 `transformers.BertLayer` 导出图上，整块相连区域可能含两个归约，不能用一个内核替换。分区器在最多 48 个节点的候选区内尝试合法切分，这个编码层得到三个融合区：一处注意力分数除法 + Softmax，两处残差加法 + 仿射 LayerNorm。其余计算（含 GEMM 与 GELU）仍由 PyTorch 执行。

局部误差小不等于整层能过。对三个融合区做[八种开关组合](../reports/bert-layer-ablation-gb10.json)，并用原生结果作为下一区域的输入，分别隔离各处误差。固定顺序的 Softmax 用 warp XOR 归约和逐元素除法，LayerNorm 用四个 warp 合并 FP32 Welford 统计量。[第一轮固定顺序复验](../reports/bert-layer-ordered-boundaries-gb10.json)通过了 32 次整层比较，但在改变仿射参数后仍有 1 个整层元素不等。将**同一 LayerNorm 内部**的 `gamma * normalized + beta` 改为显式 `fmaf` 后，[原输入复验](../reports/bert-layer-affine-fma-dev-gb10.json)消除了剩余差异。独立的 `aten.mul` 和 `aten.add` 仍各自舍入。

[另换四组随机权重与八个新输入种子](../reports/bert-layer-affine-holdout-gb10.json)：33 和 97 tokens，含单位和非单位仿射参数，共 32 次运行、1,597,440 个 BF16 输出元素；固定顺序策略与 eager 逐元素一致，三个隔离区也一致。这里没有使用预训练权重或真实文本，不代表任意形状、权重、输入或 PyTorch 版本都能精确匹配。

另用四个不同输入种子运行性能对照，只有四个输入全过精度门槛的后端才计时。下表为第四输入、七轮交错测量的中位数；单位 µs，batch 1，hidden 768。

| tokens | eager 设备 | SparkFuse 设备 | eager 同步主机 | SparkFuse 同步主机 |
|---:|---:|---:|---:|---:|
| [33](../reports/bert-layer-affine-bench-33-gb10.json) | 81.54 | 75.84 | 155.35 | 164.48 |
| [97](../reports/bert-layer-affine-bench-97-gb10.json) | 139.74 | 137.47 | 186.15 | 201.60 |

这两个样例设备时间下降，普通主机调用反而变慢；97 tokens 的设备差距很小。默认布局、独立选型和此版本 Inductor 未通过全部输入的门槛，未列速度。校准现在要求 train、holdout、audit 三个请求均通过**整图**校验才允许更新 dispatch 数据库。固定顺序是显式策略，可用 `partition_export(ep, host_backend="native", schedule_policy=ordered_collectives)` 启用。

```bash
# 使用包含 transformers 的项目 Python 环境，GPU 上不要并行跑其他基准
source tools/env.sh
python tools/ordered_boundaries.py --weight-offset 6256 \
  --seeds 74929 88019 97247 55267 62423 47303 30577 21599 --output /tmp/bert-boundaries.json
python tools/bert_layer.py --seq 33 --seeds 11413 27617 31469 52783 \
  --db /tmp/bert-dispatch.sqlite --output /tmp/bert-33.json
bash tools/sanitize.sh
```

**English.** An opt-in fixed-order policy matches eager element for element on 32 held-out BF16 BERT encoder-layer runs, including non-identity LayerNorm weights and biases. The comparison is a single random-weight encoder layer, not full-model inference. The table separates warm CUDA Graph replay from ordinary synchronized host latency; the latter got worse at both lengths. Reports retain rejected schedules and baseline failures.

[阶段验收：262 个测试、隔离 wheel、Qwen 与 BERT 集成](../reports/affine-ordered-stage-acceptance.json) · [Compute Sanitizer 三项检查与额外无过滤 memcheck](../reports/ordered-reductions-sanitizer-gb10.txt)。

### 给共享调度键选一个所有调用点都通过的布局

同一 BERT 层的两处 residual + LayerNorm 拥有相同 IR、形状和设备签名，却处理不同激活。分开搜索后把它们写进同一 SQLite 键，不能代表两处都合格。`calibrate_export(..., sampled_exact=True)` 现在从**未融合 ATen** 的执行中截取各处输入；相同键的调用点联合筛选。候选必须在每个调用点的前两个请求上与 ATen 逐元素相等才计时。以第一个合格布局为基线，替代布局需在五次留出配对中至少四次更快，且中位加速至少 3%。选定后再测试第三请求，并比较三个请求的完整图；任一比较失败，不写目标数据库。建议每个模型/精度验收配置使用独立数据库；这些是样本门槛，不是任意输入的数值保证。

在三个不同序列长度上，每组各用一套新随机权重、非单位 LayerNorm 仿射参数与六个输入。前三个用于校准，后三个第一次用于额外比较，最后一个计时。新策略三组的后三个请求均与 eager 逐元素相等；原有逐区选型三组均未能发布。以下为**整层**七轮中位数（µs）：

| tokens | eager 设备 | 联合选型设备 | eager 同步主机 | 联合选型同步主机 |
|---:|---:|---:|---:|---:|
| [33](../reports/bert-layer-joint-exact-33-gb10.json) | 81.86 | 76.44 | 156.55 | 166.71 |
| [97](../reports/bert-layer-joint-exact-97-gb10.json) | 131.82 | 132.32 | 182.79 | 198.40 |
| [129](../reports/bert-layer-joint-exact-129-gb10.json) | 117.25 | 120.93 | 169.62 | 186.23 |

不能把联合搜索说成相对固定顺序的稳定提速。33、97 tokens 的两处 LayerNorm 在 12 个候选中都只留下 2 个，最终保留固定顺序基线。97 tokens 的整层窗口有较大波动，没有可靠的设备优势。公开 JSON 保留失败位置、每个候选的精度筛选、原始窗口与整图误差。复现第三组：

```bash
python tools/bert_layer.py --sampled-exact --seq 129 --weight-seed 99173 --affine perturbed \
  --seeds 12539 20747 32429 42899 57641 64489 --db /tmp/bert-joint.sqlite --output /tmp/bert-joint.json
```

**English.** Sampled-exact calibration tunes identical GPU dispatch keys across all call sites on unfused activations. A separate request audits the chosen schedules, and database updates are atomic. On the three BERT layer cases all nine extra requests matched eager exactly. Only the 33-token case showed a lower median than eager in warm device replay; eager returned sooner on the host at every length. [Regression suite and integration checks](../reports/joint-exact-stage-acceptance.json).

### T5 门控 FFN：分解 GELU 与原生 GELU 不是同一张图

同一个 Hugging Face `T5DenseGatedActDense` 类，默认 `gated-gelu` 会将 GELU 展开（含三次幂），`gated-gelu_pytorch_tanh` 则导出原生 `aten.gelu`。改动前，前者因 `pow(3)` 不支持、后者因乘法是单节点而无法融合，见[两张原始导出图的分区记录](../reports/t5-gate-before-gb10.json)。现在分别把 9 个和 2 个算子降为一个 CUDA kernel。三处线性投影仍由 ATen 执行。

数值上不能直接把低精度 `x.pow(3)` 替换成 FP32 连乘再转回。在 98,304 个 BF16 样本中，这样改有 [21,896 个结果与 eager 不同](../reports/t5-cube-rounding-probe-gb10.json)。IR 现在保留每次乘法的元素类型舍入；原生 GELU 的 erf/tanh 则在一个算子内部用 FP32 计算。对 FP16、BF16 **各 65,536 个可能输入编码**，在对齐标量和非对齐向量两条路径上，GELU-erf、GELU-tanh、cube 的[全枚举测试](../reports/activation-exhaustive-dev-gb10.json)均为零差异。比较区分正负零，双方均为 NaN 时不比较 NaN 载荷；结论仅限本次 GB10 软件栈上的这三个一元操作。

batch 1、33 tokens、768 → 3072 → 768、BF16 随机权重。在独立权重与输入上，**实际 profiler** 看到分解版整个 FFN 的 GPU 事件从 **12 降至 4**，PyTorch 统计的调用期间增量峰值分配从 811,008 降至 608,256 字节；原生版则是 **5 → 4**。两个混合图都只保留一个门控内核，不能把分解版的收益套在原生版上。[事件明细与分配](../reports/t5-gate-profiler-gb10.json)。

| 33 tokens，完整 FFN | eager 设备 µs | 选型后设备 µs | eager 同步主机 µs | 选型后同步主机 µs |
|---|---:|---:|---:|---:|
| [分解 GELU](../reports/t5-gated-gelu-dev-gb10.json) | 48.27 | 40.98 | 77.72 | 75.31 |
| [原生 tanh GELU](../reports/t5-gated-gelu_pytorch_tanh-dev-gb10.json) | 41.60 | 40.48 | 56.67 | 74.38 |

前三个输入用于校准与整图验收；用第四个新输入做七轮交错计时。两种导出下 Inductor 都未通过这一轮的原有数值门槛，不赋予速度排名。`python tools/activation_audit.py --output /tmp/all-encodings.json` 可复核编码穷举；`python tools/t5_trace.py --seq 33 --output /tmp/t5-trace.json` 可复核事件计数。这是 FFN，不是完整 T5 编码器或解码器。

**English.** Both decomposed and native-tanh GEGLU gate graphs now compile on the real T5 FFN. An independent trace counted 12→4 GPU events for the decomposed path and 5→4 for the native path. The host latency improvement did not carry over to the native variant. [Full suite, wheel, module integrations and exhaustive low-precision check](../reports/t5-gelu-stage-acceptance.json).

换一套权重与输入，再测短请求和较长请求（每种导出、每个长度各有两个不参与选型的输入）。八个额外输入的完整 FFN 输出都与各自 eager 参考逐元素相等。下表同为七轮中位数；同步主机列包含调用和等待，非单核时间。

| 导出 / tokens | eager → 选型后，设备 µs | eager → 选型后，同步主机 µs |
|---|---:|---:|
| [分解 / 1](../reports/t5-gated-gelu-fresh-1-gb10.json) | 24.42 → 18.11 | 69.79 → 56.14 |
| [分解 / 97](../reports/t5-gated-gelu-fresh-97-gb10.json) | 55.11 → 45.54 | 83.82 → 77.85 |
| [原生 / 1](../reports/t5-gated-gelu_pytorch_tanh-fresh-1-gb10.json) | 18.60 → 17.64 | 41.62 → 55.21 |
| [原生 / 97](../reports/t5-gated-gelu_pytorch_tanh-fresh-97-gb10.json) | 46.12 → 44.61 | 61.31 → 77.23 |

原生版在三个长度都没有同步主机优势。[Compute Sanitizer](../reports/t5-gelu-sanitizer-gb10.txt) 覆盖非对齐输入和向量尾部；用 `python tools/cube_rounding_probe.py` 可单独复核低精度乘法舍入。

### FX 主机调用：BERT 有收益，不给 T5 普遍提速的结论

融合内核不变时，哪些开销还留在 Python？`specialize_host_calls(gm)` 就地重写已知 ATen 重载的调用入口：线性投影和矩阵乘走 PyTorch 的直接绑定，静态 view/permute/transpose 走张量方法，只有合法常量概率且明确处于推理模式的 dropout 才直接返回输入。仍以 `gm(x)` 调用，**保留**导出 guard、模块 hooks、视图共享存储和参数更新。其他重载原样保留。这是显式可选的推理优化，仅针对普通张量；不保证两条路径显示相同的算子级 profiler 或张量子类回调。

每个模型形状用 5 个输入对比 eager，改写前后结果均逐元素相等。九轮随机交错，每轮各后端用同一个输入，包含未融合导出图的对照，并对混合图改写前后使用**相同的 CUDA 调度**。下面每格是同步主机调用中位数（µs），包含新输出与等待；胜场按同一轮比较，而不是比较两个中位数。

| 模块 / tokens | eager | 混合图原始 | 混合图改写后 | 改写后更快的轮数 |
|---|---:|---:|---:|---:|
| [BERT 编码层 / 33](../reports/host-call-bert-33-gb10.json) | 156.46 | 166.57 | 150.32 | 8/9 |
| [BERT 编码层 / 97](../reports/host-call-bert-97-gb10.json) | 182.13 | 197.55 | 191.60 | 9/9 |
| [T5 分解 GELU / 33](../reports/host-call-gated-gelu-33-gb10.json) | 77.69 | 74.61 | 73.85 | 4/9 |
| [T5 原生 GELU / 33](../reports/host-call-gated-gelu_pytorch_tanh-33-gb10.json) | 56.38 | 73.79 | 74.56 | 1/9 |
| [T5 原生 GELU / 1](../reports/host-call-gated-gelu_pytorch_tanh-1-gb10.json) | 41.76 | 54.30 | 52.03 | 9/9 |

BERT 33 的混合图改写了 22 个主机调用节点，设备热重放仍为 75.99 / 76.00 µs；同轮未融合导出图主机 180.97 → 164.59 µs，说明收益不能全算作 CUDA 融合。T5 只改写 4 个节点；33 tokens 的两组都没有可靠主机收益。五组中的 Inductor 均未通过原有误差门槛，报告对应误差而不计时。

```python
from sparkfuse.partition import partition_export, ordered_collectives
from sparkfuse.host_calls import specialize_host_calls
mixed, regions = partition_export(ep, host_backend="native", schedule_policy=ordered_collectives)
changes = specialize_host_calls(mixed)  # optional; use a separate module for before/after timing
```

**English.** Direct FX host calls cut the 33-token BERT layer's synchronized median from 166.57 to 150.32 µs without changing the device kernel schedule. The two 33-token T5 cases showed no consistent improvement. Repeat with `python tools/host_call_panel.py --case bert --seq 33 --output /tmp/host.json`. [Full regression and real-module checks](../reports/host-calls-stage-acceptance.json).

带有顶层 `provenance.archive` 的公开基准现在可从同一仓库直接校验[测量时的源码快照](../reports/sources/README.md)，无需账户或 DGX Spark。校验只需要 Python 标准库；执行 GPU 基准仍需报告对应的软件和硬件环境。

### 长行 Softmax：从夜间失败到可复测结果

在原注意力搜索中，两个已缓存的分块 Softmax 配置遇到了设备地址错误（FP16 24×8192、BF16 384×4096）；另一个 512 线程的多 warp 配置（FP16 24×2048，同时输出 logits）发射返回 CUDA 错误 701。三个原始失败都保留，没有从运行分母删除。新生成器在每线程缓存至少 32 个值时保留外层循环，不让编译器自动展开整条长行，包内的短循环仍展开。首次调整的 memcheck 已干净，但 racecheck 对单 warp 的分块归约仍报[八条警告](../reports/wide-score-intermediate-racecheck-gb10.txt)。在一个 32 线程块本身就是整行 warp 的 Softmax 中，现直接广播 lane 0 的结果，不再通过共享内存二次归约。我们验证了这两项代码改动，没有把硬件错误归因给某个未经证实的上游缺陷。

[三种原调度直接测试](../reports/wide-score-kernel-audit-gb10.json)：每种四个种子，含两次偏移 1 个元素的连续输入，暴露的 logits 必须精确匹配。[最终 Compute Sanitizer 日志](../reports/wide-score-sanitizer-gb10.txt) 显示 memcheck / synccheck 均零错误，racecheck 零警告。另以独立数据库重放原来的三项 16 候选搜索，三项均走完选型、留出检验和第三个种子的三后端对照。下表仅是最终选型布局的热设备时间，不等于发生过错误的候选均获选或更快。

| 子图、形状与类型 | eager µs | Inductor µs | SparkFuse µs |
|---|---:|---:|---:|
| [scores-24x8192-float16](../reports/wide-score-retune-scores-24x8192-float16-gb10.json) | 6.291 | 3.475 | 4.442 |
| [scores2-24x2048-float16](../reports/wide-score-retune-scores2-24x2048-float16-gb10.json) | 8.689 | 2.961 | 2.164 |
| [scores-384x4096-bfloat16](../reports/wide-score-retune-scores-384x4096-bfloat16-gb10.json) | 35.088 | 9.232 | 7.523 |

原始失败记录（gzip JSON）：[24×8192](../reports/wide-score-history-scores-24x8192-float16.json.gz)、[双输出 24×2048](../reports/wide-score-history-scores2-24x2048-float16.json.gz)、[384×4096](../reports/wide-score-history-scores-384x4096-bfloat16.json.gz)。运行 `python tools/wide_score_audit.py --output /tmp/wide.json` 可检验三个布局；[完整阶段验收](../reports/wide-score-stage-acceptance.json)另覆盖相邻奇数宽度和原有框架集成。

**English.** Two illegal accesses and one rejected kernel launch from the long-key search triggered this change. Keeping the outer cache loop and avoiding redundant shared storage for a single-warp score reduction passed the targeted memory, race, and synchronization checks. All three original workloads completed a fresh 16-candidate tuning run. This establishes correctness for the tested cases, not a blanket speedup.

### Variable token counts: one native descriptor and one CUDA binary

`DynamicRows(ep).native_dispatch()` binds one C++ launcher to a rank-2 export with a single bounded leading symbol. Each invocation reads the row count from the first dynamic input, checks the export bounds and every input (including static broadcasts), allocates the correctly sized outputs, and launches on the current stream. It neither re-exports the graph nor builds a native plan per row count. The original Python `prepare` path remains available. A C++17 toolchain is needed for the native path.

The [DGX Spark panel](../reports/dynamic-native-panel.json) exports at seven rows, validates 16 lengths from 1 to 129 against eager, and checks native vs Python outputs byte-for-byte. All four backends passed on all three cases, including the separately observable BF16 residual. Odd-numbered request indices use contiguous tensors offset by one element. Thirteen held-out lengths are then timed in mixed order (identical request sequence per backend; preallocated inputs, fresh outputs). Median synchronized host call in µs across nine randomized windows:

| exported region | width / type | eager | Inductor (`dynamic=True`) | SparkFuse Python | SparkFuse native |
|---|---|---:|---:|---:|---:|
| SiLU gate | 8960 / BF16 | 18.848 | 29.152 | 29.023 | 11.552 |
| visible residual + RMSNorm | 1536 / BF16 | 17.312 | 27.871 | 28.032 | 9.008 |
| squared mean, output `[R,1]` | 768 / FP32 | 13.216 | 26.159 | 22.992 | 7.280 |

Native had the lower median than each other frontend in 9/9 windows on these workloads. This is a host-dispatch improvement: on 33-row RMSNorm, Python/native fixed-shape CUDA Graph replay was 2.232/2.249 µs; at 129 rows, 2.771/2.707 µs. The two frontends launch the same CUDA binary. Replayed GPU time excludes the host and does not describe variable-shape graph capture. These warmed, single-subgraph calls are not serving throughput, cold compilation latency, or a claim about arbitrary dynamic axes. Reproduce with `python tools/dynamic_native_panel.py --output /tmp/rows.json` on an idle GPU, or add `--verify-only` for the parity matrix without timings/Inductor.

### 只写行结果的均值归约 / Compact row outputs

当下游只要每行的一个或多个标量时，没有必要让整块线程都拿到均值。`compact_mean` 每个 CTA 读一行，用寄存器累加向量包；跨 warp 时各 warp 只写一个部分和，最后由 lane 0 计算行级后处理并写回。多个按元素输入的地址必须全部对齐，才能走无尾部的向量读路径。带完整残差输出的 RMSNorm 不符合此布局的条件，也没有替换已有默认策略。

[DGX Spark 对照](../reports/compact-mean-panel.json)使用平方均值子图。每种布局同样搜索八种线程数和向量宽度组合，包括 1024 线程的普通 block。冻结每种布局的胜者，再用两个未参与选型的输入分别测量；一个对齐，另一个偏移一个元素。单位 µs，为七轮交错 CUDA Graph 设备时间的中位数，不含 Python，也不是整模型加速。

| 行 × 宽度 / 类型 | 偏移（元素） | eager | Inductor | 普通 block | compact |
|---|---:|---:|---:|---:|---:|
| 31 × 14336 BF16 | 0 | 3.852 | 1.421 | 1.912 | 1.677 |
| 31 × 14336 BF16 | 1 | 4.128 | 5.879 | 2.060 | 2.060 |
| 32 × 1536 BF16 | 0 | 3.085 | 1.037 | 1.165 | 1.143 |
| 32 × 1536 BF16 | 1 | 3.191 | 2.061 | 1.185 | 1.144 |
| 47 × 769 FP32 | 0 | 3.341 | 1.036 | 1.164 | 1.096 |
| 47 × 769 FP32 | 1 | 3.319 | 1.787 | 1.164 | 1.090 |
| 128 × 4096 FP16 | 0 | 5.773 | 1.464 | 1.793 | 1.610 |
| 128 × 4096 FP16 | 1 | 6.220 | 4.236 | 1.933 | 1.931 |

在第一行，两种布局都选中了 1024 线程、8 元素一包，compact 在七轮中都比普通 block 快；同形状的偏移输入上则没有可辨的收益。四个对齐输入上 Inductor 均更快。偏移与对齐使用不同种子，不把两行之差解读成仅由地址造成的因果效应。报告另含每个后端各 31 次普通同步主机调用，没有把这些时间混进上表。

[独立的生产选型回放](../reports/compact-mean-replay.json.gz)使用原搜索任务、空数据库和 24 个候选。它选中了 1024 线程的 `compact_mean`，在五轮留出复测中相对默认布局的配对中位比为 1.611；第三个种子的 eager / Inductor / SparkFuse 设备时间分别为 **3.858 / 1.428 / 1.683 µs**。三者均通过数值检查后才发布 SQLite 记录，未修改原搜索的记录。

**English.** The compact schedule removes the final block-wide broadcast when only row-sized results escape. Broadening the conventional block search to the same thread counts isolates the layout benefit. The aligned long BF16 case improved against the best conventional candidate, not against Inductor. Misaligned long rows were effectively tied. The table reports a single exported reduction, not Transformer latency. Reproduce the family comparison with `python tools/compact_mean_panel.py --output /tmp/compact-panel.json` on an idle GPU.

**What changed in the binary?** For the `31 × 14336` BF16 row above, the [ordinary](../examples/generated/compact-mean/block.cu) and [compact](../examples/generated/compact-mean/compact_mean.cu) emitted kernels use the same 1024 threads and eight-element pack. `cuobjdump --dump-sass` found two block-barrier instruction sites and two shared-load/store sites each in the ordinary kernel; each compact specialization has one of each. The compact aligned specialization has a 128-bit global load and no scalar BF16 global-load sites. These are [static counts per binary function](../reports/compact-mean-sass.json), not executed counts or a causal attribution of all the timing difference. After running the panel with the same build settings, use `python tools/inspect_reduction_sass.py reports/compact-mean-panel.json --cache .cache/kernels --output /tmp/reduction-sass.json` to inspect the selected binaries.

### 一个 BF16 舍入边界：先修数值，再看速度 / An ordered FP32 mean

夜间搜索在 `48 × 8960` BF16 的分解式残差 RMSNorm 上[拒绝了默认布局](../reports/mean-tree-history.json.gz)：留出种子 1753 的 430,080 个输出中有一个超出既定容差。`[25,3533]` 的 FP32 均值从 eager 的 `2.0016326904296875` 变成 `2.0016324520111084`，归一化值因而落到 BF16 舍入中点；乘权后的结果为 `−2.578125`，eager 为 `−2.546875`。误差来自归约加法次序；没有改容差，也没有删掉该样本。

`mean_tree` 在一个 CUDA kernel 中按本机 PyTorch eager 的次序计算 FP32 均值：四个独立累加器、降序 warp shuffle，宽行再归约 16 组部分和。这个**逻辑**次序不随物理线程块大小变化，归约结果可以继续喂给融合后处理，也可一同返回可见残差。匹配的对象是明确导出的、对已材化 FP32 行表达式的均值，不假定任意输入视图对齐，也不把该规则套到原生 `aten.rms_norm`。仅在已支持的至少 16 行、宽 128–65536 且整除 4、总元素数不超过 2²⁴ 的计划上开放；敏感低精度后处理默认走有序树。这些条件不保证不同 PyTorch 版本采用同一棵树。

[逐边界核验与计时](../reports/mean-tree-boundaries-gb10.json)将均值、逆平方根、FP32 归一化、窄化结果、乘权输出及残差分别与 eager 按位比较。四种形状 × 六个种子均一致，后面三个种子的三个输入分别偏移 1、2、3 个元素。另用第七个种子对已导出图的正常单输出调用（不返回诊断张量）计时。下表为七轮交错 CUDA Graph 设备时间的中位数，单位 µs；报告另列普通同步 Python 主机调用。

| 行 × 宽 / 类型 | eager | Inductor | 旧默认 | 有序默认 |
|---|---:|---:|---:|---:|
| 48 × 8960 BF16 | 20.520 | 3.301 | 9.124 | 4.389 |
| 17 × 8160 FP16 | 21.798 | 3.329 | 7.783 | 31.231 |
| 17 × 8164 FP16 | 12.732 | 3.366 | 8.229 | 3.487 |
| 32 × 1536 BF16 | 12.069 | 1.830 | 4.901 | 7.424 |

8160 和 8164 分别落在 eager 两种不同的行内归约策略上；表中的性能不能跨形状外推。特别是 `17 × 8160` 的修复明显比旧默认慢。历史错误发生在种子 1753，而不是上表的计时种子 91853。

[原生产任务重放](../reports/mean-tree-replay.json.gz)复用了原任务的三个种子和预算上限 24，实际测试全部六种有序布局（128/256/512 线程，各含缓存和重算）。SparkFuse 在留出输入与第三个输入均逐元素一致。第三个种子 3752 上，Inductor 在 `[43,4916]` 有一个值未通过相同门槛；未为它计时，整个重放按失败保存，空数据库未发布任何选型。这个结果不能算三后端通过，亦不代表 Inductor 在其他种子上不正确。

**English.** A one-element BF16 error motivated matching the observed eager FP32 reduction order, rather than relaxing parity tolerances. The six tapped results matched bitwise on the 24 sampled requests, including shifted inputs. The untapped seventh request demonstrates both gains and regressions relative to the previous schedule. Reproduce with `python tools/mean_tree_audit.py --output /tmp/mean-tree.json`; the report includes the installed PyTorch CUDA reduction-header digest, profiler geometry, raw timings and source snapshot.

### 保留加法次序，重排物理线程 / One thread per accumulation chain

上节的修复在 `17 × 8160` 上付出了明显的性能代价。对这一段宽度，eager 每行有 32 个逻辑 lane，各自维护 4 个有先后顺序的 FP32 累加器。[旧有序内核](../examples/generated/mean-cohort/mean_tree.cu)把一行交给一个 warp；[新有序内核](../examples/generated/mean-cohort/mean_cohort.cu)用一整个线程块处理一行，前 128 个线程各承担一条独立递推链，之后严格按原先的次序合并。缓存版本让整个 CTA 协作把可复用的逐元素值写进共享内存，并在后处理中复用；无缓存版本重算。这个选择只对原固定次序路径的子集开放：宽 1024–8160、四元素对齐的**行宽**，不要求输入基地址对齐。8164 及以上仍使用原宽行算法，低于 1024 的默认也未改。

[五形状独立对照](../reports/mean-cohort-panel-gb10.json)中，旧布局的六种配置和新布局的四种配置都在发现输入上计时，然后冻结各自最快配置。下表只比较两条**固定默认**路径；报告同时列出两类各自选出的最快配置、普通同步主机调用、每个输入的 6 个可见边界和七轮原始设备采样。对齐与偏移用了相同的留出数值，`1/2/3` 分别是三个输入以元素为单位的偏移。单位 µs，七轮交错 CUDA Graph 设备时间的中位数。

| 行 × 宽 / 类型 | 对齐：旧 → 新 | 偏移：旧 → 新 | Inductor：对齐 / 偏移 |
|---|---:|---:|---:|
| 17 × 8160 FP16 | 31.136 → 4.712 | 32.544 → 4.710 | 3.298 / 6.631 |
| 32 × 1536 BF16 | 7.205 → 3.109 | 7.398 → 3.109 | 未通过 / 未通过 |
| 128 × 4096 BF16 | 16.223 → 6.630 | 16.165 → 6.694 | 未通过 / 未通过 |
| 31 × 1024 FP16 | 4.864 → 2.341 | 5.093 → 2.341 | 1.254 / 3.429 |
| 63 × 4100 BF16 | 17.445 → 6.693 | 18.725 → 6.687 | 2.853 / 7.461 |

十个请求上，SparkFuse 两类默认及选出的布局在六个边界都与 eager 逐位一致。Inductor 的两种 BF16 形状没有通过原有容差，两个地址布局均未计时；不能把缺测当成速度优势。在 `17 × 8160` 的对齐输入上，新默认仍慢于 Inductor。[编译器资源报告](../reports/mean-cohort-binaries-gb10.json)记录该形状：旧默认为每线程 40 个寄存器、静态共享内存为 0；新默认为 23 个寄存器、每 CTA 33,796 字节静态共享内存。它们是 `cuobjdump --dump-resource-usage` 的结果，不是运行时占用率或访存计数。

[独立生产回放](../reports/mean-cohort-replay.json.gz)使用新种子 64333 和空数据库，遍历全部十种有序布局，留出复测保留 256 线程缓存默认配置；无额外调参收益。第三种子上 eager / Inductor / SparkFuse 分别为 **21.409 / 3.094 / 4.622 µs**，三者通过同一容差，SparkFuse 与 eager 逐元素一致。

**English.** `mean_cohort` changes which physical threads perform the 128 independent scalar recurrence chains, not the FP32 addition tree. For the formerly slow `17 × 8160` FP16 case the fixed default went from 31.136 to 4.712 µs on aligned inputs. Inductor remained faster on that aligned request. The ten held-out requests cover five shapes and two address layouts, not ten independent value seeds. Reproduce with `python tools/mean_cohort_panel.py --output /tmp/mean-cohort.json`.

### 同一处残差，两个下游使用者 / Splitting at a projection barrier

一个完整的 `Qwen2DecoderLayer` 暴露了旧分区器的盲点。注意力输出与输入相加后，RMSNorm 需要这个残差，MLP 后的跳连也需要它。如果把末端加法一起收进来，前端找不到一个同时满足所有生产者和消费者的插入位置。以前这种只有一个归约的连通块会整个退回 ATen。现在只在确有位置冲突时，对不超过 48 个节点的候选块寻找可独立生成的连续区间；每段分别检查输入可用性、输出用途、别名和可编译性。本例选中的 kernel 覆盖九个计算算子，写出两个结果：原残差保留给末端跳连，归一化并乘权后的张量送入门控和上投影。七个线性投影及 QK/PV 矩阵乘法仍走 PyTorch。

[首轮实测](../reports/qwen-decoder-barrier-gb10.json)使用 Hugging Face 的实际单层、随机非恒等归一化权重、BF16、batch 2、hidden 1536、MLP 8960、12 个查询头 / 2 个 KV 头，两条样本的位置和掩码不同。每个长度有六个验证种子，另用第七组输入计时；与建图输入不重叠。四处观察值（最终隐藏态、注意力概率、注意力后的残差、乘权归一化结果）逐位匹配 eager。遮住的注意力权重均为零。

| 序列长度 | eager 设备 µs | 旧分区 → 新分区：设备 µs | 旧分区 → 新分区：同步主机 µs | 旧 → 新实际 kernel 次数 |
|---:|---:|---:|---:|---:|
| 33 | 602.787 | 587.424 → 575.416 | 706.249 → 674.616 | 36 → 28 |
| 65 | 621.405 | 602.445 → 582.923 | 762.123 → 717.817 | 36 → 28 |
| 97 | 764.328 | 695.813 → 680.643 | 817.628 → 773.067 | 37 → 29 |

时间是七个交错测量窗口的中位数。两条混合路径共用同样的有序归约和原生调度，只改变一个分区选项。表的最后一列来自[第二次运行](../reports/qwen-decoder-kernel-census-gb10.json)：先重新计时，再分别记录一次普通调用中 Kineto `cat=kernel` 的 GPU 区间；不是 FX 节点数，也没有用 profiler 计算延迟。第二次整层设备时间旧 → 新分别为 589.491 → 580.541、601.280 → 583.789 和 704.541 → 684.133 µs。两次都少了八次发射，但这不等于整个模型有同样比例的加速。

Inductor 在本实验 21 组输入的隐藏态上没有通过原精度约定，因此不报告它的速度；两份 JSON 均保留逐种子的结果。例如长度 33、种子 30211 的隐藏态有 4,990 / 101,376 个元素超出容差，而该请求的注意力概率通过。未加载预训练权重，无 KV 缓存，不是生成吞吐基准。RoPE 表在计时外生成，旋转和所有投影在计时内。

**English.** A topological barrier used to send a whole single-reduction component back to ATen. The partitioner can now retain a legal two-output residual/RMSNorm prefix without moving work across a projection. On one real Qwen2 layer, this removed eight observed GPU kernel launches at each tested length. Four sampled tensors matched eager bitwise. The reports include both paired timing runs and the Inductor parity failures. Reproduce with `python tools/qwen_decoder.py --seq 33 65 97 --profile --output /tmp/qwen-decoder.json`.

### Qwen2 解码层：两支旋转位置编码合为一次发射 / Paired Q/K rotary

在同一次试验中只切换 `rotary_pairs`。两条路径都已经融合残差与 RMSNorm，使用相同的权重、输入、线程配置和主机调用路径。一个 CUDA kernel 分别读写 Q、K 的两半，复用同一组位置表，保留每次乘法的低精度舍入。单次普通 forward 的 profiler 记录显示，原先两支旋转的 10 次发射变为 1 次。切片视图不算 GPU kernel。

| 序列长度 | eager 设备 µs | SparkFuse 设备 µs，关 → 开 | 同步主机 µs，关 → 开 | 整层发射数，关 → 开 |
|---:|---:|---:|---:|---:|
| 33 | 605.973 | 584.728 → 559.120 | 676.330 → 636.617 | 28 → 19 |
| 65 | 622.867 | 584.349 → 568.739 | 756.251 → 685.898 | 28 → 19 |
| 97 | 743.696 | 681.557 → 653.856 | 804.540 → 735.035 | 29 → 20 |

[整层原始记录](../reports/qwen-decoder-rotary-gb10.json)含七个乱序计时窗口。BF16、batch 2、12 个 Q 头和 2 个 KV 头；三个长度各七组输入。Q/K 的值与步幅、可见残差、归一化结果及两个公开层输出均与 eager 对照。完整图 Inductor 的隐藏态未通过原容差，因此未计时。这是没有 KV cache 的单个随机权重解码层；计时包含所有投影及旋转应用，不含位置表生成。

The [isolated Q/K panel](../reports/paired-rotary-panel-gb10.json) also varies layout, input offsets and precision. For BF16 at length 97 in token-major physical storage, aligned device replay fell from 24.101 µs eager to 2.203 µs fused; with offset inputs it fell from 24.097 to 2.342 µs. In the small aligned FP32 case, eager / Inductor / SparkFuse took 12.581 / 1.574 / 1.062 µs. Inductor failed the unchanged tolerance on the four lower-precision cases, so their reports omit its timing. The paired pass is opt-in; an isolated speedup does not predict layer latency at other shapes or on other GPUs.

### 带 KV 缓存的两步解码 / Two cached Qwen2 decode steps

用一个真实的 Qwen2 解码层预填历史 K/V，各后端独立运行第一步并将自己的输出缓存交给第二步。编译器可用 `paired_appends=True` 将导出图里的两次长度轴拼接编译为一个 CUDA kernel，K、V 仍是分别分配的新张量。预填输出是物理 token-major，第一步拼接后是 head-major；按实际步幅导出两份静态图。下表的基线已经启用残差细分及 Q/K 旋转融合；两条 SparkFuse 路径只差缓存拼接。

| 预填长度 | 解码步 | eager 设备 µs | SparkFuse 设备 µs，关 → 开 | 同步主机 µs，关 → 开 |
|---:|---:|---:|---:|---:|
| 31 | 1 | 509.859 | 483.091 → 480.645 | 589.012 → 582.516 |
| 31 | 2 | 507.752 | 480.843 → 479.733 | 611.092 → 605.844 |
| 255 | 1 | 535.355 | 502.643 → 499.192 | 620.740 → 619.284 |
| 255 | 2 | 536.040 | 514.312 → 514.219 | 628.548 → 621.524 |
| 1023 | 1 | 609.821 | 593.595 → 586.509 | 683.845 → 672.661 |
| 1023 | 2 | 605.360 | 581.283 → 581.445 | 668.596 → 662.532 |

[逐次原始数据、FX 图与 profiler](../reports/qwen-decode-cache-gb10.json)。六次普通调用均从 19 次 GPU 发射变为 18 次；少一次发射不能保证每次设备延迟都下降。长度 1023 的第二步中位数略有回退，七个配对窗口仅四个有利。每个长度用五条轨迹核对隐藏态、注意力概率、缓存及其前缀；SparkFuse 所有输出与框架逐位相同，步幅也一致。完整图 Inductor 的隐藏态和 K/V 在这组输入上未通过既有容差，没有为它计时。

**Why choose copy width at launch?** In the [initial isolated panel](../reports/cache-append-initial-gb10.json), one thread owned a fixed-width packet. An unaligned address forced individual lane-strided scalar copies; copying the offset 1023-token prefill cache took 22.304 µs vs 16.017 µs eager. The new launcher checks all six addresses and chooses the widest legal element group before issuing a single kernel. Adjacent lanes copy adjacent elements when the safe width is one. In the [new same-run four-way comparison](../reports/cache-append-panel-gb10.json), that case measured **22.243 / 6.190 / 16.013 / 15.193 µs** (fixed fallback / adaptive / eager / Inductor). With aligned addresses the same order was **3.214 / 3.153 / 16.012 / 3.572 µs**. All isolated backends passed bitwise checks, including across input offsets. Profiler names confirm 8-, 1-, 2- and 4-element specializations for BF16 buffers offset by 0, 1, 2 and 4 elements. These are hot-address data-movement timings, not DRAM bandwidth or token throughput.

### GQA 解码：何时该拒绝一次融合 / GQA decode admission

Qwen2 的 12 个查询头共用 2 个 KV 头。一个看似能省去两次复制的办法，是把每组的 6 个查询头折叠进矩阵乘法的行维度。我们从真实 `torch.export` 图抓取 QK、PV 的操作数，并首先逐位复算原来的两次矩阵乘法。[两组权重、九种前缀长度和每组三条轨迹](../reports/grouped-kv-admission-gb10.json)共比较 108 个 BF16 张量：全部在原容差内，**20 个张量中共有 25 个元素与原图的位表示不同**。每处反例都记录了位置和 BF16 十六进制值（每张量至多四处）。这个改写没有进入编译器，也没有为它发布加速比。

选择保留原矩阵乘形状，只把两个 `unsqueeze → expand → reshape` 实际发生的头复制合为一个 CUDA kernel（`paired_head_repeats=True`）。每次加载的包写到各个组，K/V 仍是两个独立、连续的新张量。它**没有消除重复输出的显存占用**。在[独立复制对照](../reports/head-repeat-panel-gb10.json)中，32 个 KV 位置、地址对齐时，eager / Inductor / SparkFuse 分别为 **3.379 / 1.913 / 1.331 µs**；1025 个位置时则是 **65.301 / 65.807 / 66.227 µs**。后者需要写出 12,595,200 字节，少一次发射并没有让它更快。上述时间是热地址设备重放，不代表显存带宽。

[首轮整层对照](../reports/qwen-decode-head-repeat-initial-gb10.json)还暴露了另一个问题：预填 1024 时，即使关闭新的复制融合，旧的分数融合也在部分轨迹上改变了概率末位，并有一次影响隐藏态。不为未通过的数据计时。提供显式选项 `framework_softmax=True`：保留框架原 Softmax，仅融合其前后独立合法的区域。它适用于所有宽度，并非按反例或输入种子设白名单；不意味着其他归约自动逐位精确。

[在同一组权重、种子、掩码和前缀上复验](../reports/qwen-decode-head-repeat-boundary-gb10.json)：两步之间传递各后端自己的 K/V；3 个长度 × 5 条轨迹 × 2 步，四个可见输出和步幅在 SparkFuse 两条对照路径中均与 eager 一致。表中两条路径都保留框架 Softmax、都已融合缓存拼接，仅头复制不同。六组均由 **21 个 GPU kernel 降为 20 个**。完整图 Inductor 未通过原容差，不计时；这不是对整模型或 token 吞吐的测试。

| 预填长度 | 步 | 设备 µs，独立 → 融合 | 同步主机 µs，独立 → 融合 |
|---:|---:|---:|---:|
| 31 | 1 | 474.856 → 473.061 | 604.766 → 588.477 |
| 31 | 2 | 484.037 → 481.123 | 636.559 → 619.214 |
| 254 | 1 | 503.304 → 498.629 | 647.952 → 629.311 |
| 254 | 2 | 511.840 → 507.109 | 653.008 → 635.711 |
| 1024 | 1 | 582.747 → 575.664 | 676.592 → 665.311 |
| 1024 | 2 | 586.349 → 580.352 | 674.768 → 666.416 |

**Reproduce.** Use the same CUDA/PyTorch environment and optional Transformers 4.51.3. The first command audits a numerical counterexample; its successful exit means the audit ran, not that the candidate qualified. The second benchmarks two fresh physical copies vs. horizontal fusion. The last command exercises two ordinary cached layer forwards, not text generation.

```bash
source tools/env.sh
python tools/grouped_kv_audit.py --output /tmp/gqa-admission.json
python tools/head_repeat_panel.py --output /tmp/head-copy.json
python tools/qwen_cached_decode.py --head-repeats --framework-softmax --past 31 254 1024 \
  --seed-offset 44234 --profile --output /tmp/decode.json
```

[阶段验收](../reports/head-repeat-stage-acceptance.json)另外用不同的权重和五条新轨迹检查预填长度 255、1025，两步的四个可见输出均通过。此处报告样本范围内的逐位结果，而非对所有输入的等价性证明。生成的复制 kernel 也通过 [memcheck、racecheck、synccheck](../reports/head-repeat-sanitizer-gb10.txt)。

### 长行 Softmax：寄存器缓存与精度边界 / Register-resident score rows

`partition_export(ep, ordered_scores="slots")` 是可选策略，不改变默认调度。一个 warp 处理一行，每个 lane 按固定顺序累加，随后按 XOR 蝶形顺序合并。宽度 1025–2048 的已验证 FP32 归约把跨阶段的数组改为编译期下标；改变的是缓存位置，不是加法次序。`ordered_scores="loop"` 是同序循环对照。域外保留原 ATen Softmax。低精度最终输出不能代替对中间 FP32 概率的检查。

- **子图**：[独立对照](../reports/ordered-score-slots-gb10.json)包含 5 个种子、2 种地址偏移、7 个交错计时时窗，只对通过逐位与步幅检查的实现计时。对齐的 24×1537 BF16、三个输出可见时，eager / loop / slots 为 **10.801 / 26.161 / 5.683 µs**。实际二进制的每线程栈从 **200 B 降至 0 B**，寄存器从 **36 增到 128**。384×2047 时为 48.097 / 58.959 / 29.133 µs。更窄或少输出的图不一定选它：24×1025 仅保留 BF16 概率时，普通 block 在本组种子上比 slots 更快。
- **为什么有时回退？** [初次测量](../reports/ordered-score-slots-initial-gb10.json)发现，FP16 宽 2048 的掩码分数无一处变化，但首个种子的 FP32 概率有 **19,131** 个元素不同。[同一份 `z` 的独立复验](../reports/ordered-score-half-origin-gb10.json)区分了 `z.softmax(-1, dtype=float32)` 和 `z.float().softmax(-1)`：从 1025 开始，当前 ATen 对两者使用不同的 CUDA kernel。在这里仅前者触发 FP16→FP32 回退；不把它偷偷改成显式转换。显式转换图中的缩放值、掩码分数和两种概率输出逐位匹配。域和结论限于此 GB10 / PyTorch 构建。
- **真实子层**：[Qwen2 两步缓存解码](../reports/qwen-decode-ordered-scores-qualified-gb10.json)在相同配对头复制、KV 拼接、旋转和 GEMM 下仅替换分数路径。预填 1535 时，第一步的 eager / 框架 Softmax 分区 / loop / slots 为 **664.600 / 622.819 / 635.891 / 620.477 µs**；第二步为 **659.101 / 630.528 / 647.440 / 625.224 µs**。框架分区的 20 个 kernel 在此降为 17 个。预填 2047 的第二步宽 2049，三种分区均保留 ATen，计时差不能算 slots 加速。未通过既有数值门槛的整层 Inductor 不计时；这不是模型 token 吞吐。[旧反例的权重与输入种子](../reports/qwen-ordered-score-replay-gb10.json)也重跑了两步，检查各后端自己的 K/V、隐藏态和注意力概率。

**English.** This opt-in GB10 policy keeps the ordered warp reduction but makes wide per-lane cache slots statically addressable. The isolated register/stack measurements explain the loop comparison, not an end-to-end speedup. The profiler and a same-input witness distinguish native Half-to-Float Softmax from Softmax on an already-cast Float tensor. Unqualified paths retain ATen. See the [emitted CUDA](../examples/generated/ordered-scores), [sanitizer summary](../reports/ordered-score-sanitizer-gb10.txt), and [stage acceptance](../reports/ordered-scores-stage-acceptance.json).

```bash
source tools/env.sh
python tools/half_score_witness.py --output /tmp/half-origin.json
python tools/ordered_score_panel.py --emit-dir /tmp/score-kernels --output /tmp/scores.json
python tools/qwen_cached_decode.py --score-panel --past 1024 1535 2047 --seed-offset 52753 \
  --profile --emit-dir /tmp/decode-kernels --output /tmp/decode-scores.json
python tools/qwen_cached_decode.py --score-panel --past 1024 2046 --seed-offset 44234 \
  --verify-only --output /tmp/old-weights.json
```

### BF16 概率舍入：什么时候值得省下除法？ / Guarded BF16 division

若 FP32 Softmax 概率只经过立即 BF16 转换才可见，`ordered_scores="guarded"` 可以先乘舍入后的倒数，再转 BF16。接近 BF16 舍入中点、非正规乘积或非有限行仍按原方式除。若图还返回 FP32 概率、转 FP16 或在转换之前做运算，这条策略不适用。普通 `slots` 和默认调度不变。[舍入压力测试](../reports/bf16-quotient-rounding-gb10.json)对比实际生成的 helper：24,095,493 个中点附近的样本中，直接乘倒数有 609,628 个 BF16 位差，带保护的版本为 0；另有 6,291,456 个随机编码和 18 个特殊值。这些样本不是穷举，比较的是 **BF16 输出**，不是中间 FP32 值。

[从真实导出图截取的分数子图](../reports/qwen-score-slice-guarded-gb10.json)：对一个随机权重 Qwen2 层先做预填，截取第一步解码的 QK 结果和每个 batch 的实际掩码。用五个留出种子核对 HF 概率及整层四个输出，再对通过的后端交错计时。每个序列只屏蔽一个位置时，结果如下（GB10，设备时间 µs）。Inductor 在这三个子图上均有 BF16 位差，未进入计时。

| 键长度 | eager（5 个 kernel） | 精确 slots（1 个） | guarded（1 个） |
|---:|---:|---:|---:|
| 1025 | 7.987 | 3.121 | 3.233 |
| 1536 | 8.844 | 3.804 | 4.061 |
| 2048 | 9.360 | 4.659 | 4.744 |

**掩码的位置影响结论。** 保持同一份 QK 和相同的 CUDA 二进制，仅替换掩码。宽 1536 时，每条序列屏蔽 25 个分散的位置，slots / guarded 为 **5.937 / 3.889 µs**；屏蔽连续的前 24 个位置时则是 **3.785 / 3.805 µs**。这些是受控的替换输入，不代表原模型跑出了这个收益。

[真实两步整层](../reports/qwen-decode-guarded-scores-gb10.json)使用同一权重和轨迹。预填 1535 时，slots / guarded 第一步 **616.547 / 616.859 µs**，第二步 **623.515 / 624.485 µs**。三个前缀均未观察到整层加速，宽 2049 时两者都保留 ATen。策略保持显式选择，不根据这组数据变更默认值。参见[生成的 CUDA](../examples/generated/guarded-scores)、[三项 sanitizer](../reports/bf16-quotient-sanitizer-gb10.txt)与[独立权重验收](../reports/guarded-score-stage-acceptance.json)。

**English.** The opt-in schedule exploits BF16-only observability; it does not preserve approximate FP32 quotients for other uses. The isolated benchmark extracts the original score operators and captures real post-matmul operands; its emitted CUDA hashes match the whole-layer experiment. With the observed masks, guarded division was slower than exact slots at all three widths. Changing only mask positions can reverse that comparison. Neither the mask controls nor the rounding stress distribution establish a full-layer speedup.

```bash
source tools/env.sh
python tools/bf16_quotient_audit.py --output /tmp/quotient.json
python tools/qwen_score_slice.py --output /tmp/slice.json
python tools/qwen_cached_decode.py --score-panel --guard-panel --past 1024 1535 2047 \
  --seed-offset 64219 --profile --emit-dir /tmp/guarded-cuda --output /tmp/whole-layer.json
bash tools/sanitize_bf16_quotient.sh
```

### 一次读取同时生成缓存和 GEMM 输入 / Four-output KV cache fusion

`partition_export(ep, cache_fanout=True)` 识别两条 `cat → unsqueeze → expand → reshape` 分支。若更新后的 K/V 缓存仍有其他用户，它为每个输入数据包生成一次读取，同时写出 **K 缓存、V 缓存、展开后的 K、展开后的 V**；后面两个仍是原矩阵乘法需要的稠密输入。四个张量各自拥有新存储，保留对外可见的视图关系，不改 GEMM 的形状或乘加顺序。缓存仅在头复制内部使用、两分支有依赖或布局不合格时，不做这项合并。

[真实 Qwen2 输入的独立子图](../reports/qwen-cache-fanout-slice-gb10.json)核对了 HF 返回的缓存与原始两次 GEMM 收到的操作数。对预填 2047 后的第一步（B2、2 个 KV 头、12 个查询头、D128），两个复制 kernel → 一个，逻辑读取从 **8 MiB → 4 MiB**，写入仍为 **28 MiB**。这些是按索引与张量大小计算的有效载荷，不是硬件 DRAM 计数。对通过逐位、步幅、存储独立性和输入不变检查的后端，同机 GPU 热时间如下（µs，第一步）：

| 预填长度 | eager | Inductor | 分开复制 | 四输出融合 |
|---:|---:|---:|---:|---:|
| 31 | 7.267 | 3.592 | 2.552 | 1.643 |
| 1024 | 88.096 | 83.117 | 81.589 | 79.165 |
| 2047 | 179.392 | 167.525 | 171.688 | 165.717 |

[整层两步对照](../reports/qwen-decode-cache-fanout-gb10.json)固定旋转、分数调度、其余融合 CUDA 和两次 GEMM。预填 1024 时分开 / 合并的第一步 **574.488 / 571.637 µs**、第二步 **572.005 / 569.259 µs**；预填 2047 时却是 **687.376 / 692.317 µs**、**698.109 / 707.240 µs**。五个前缀分别用五条留出轨迹验证两步，隐藏态、概率、两份缓存均匹配。Inductor 通过独立复制子图的门槛，但未通过整层门槛，因此没有整层 Inductor 延迟。**单独复制更快，不足以把它设为整层默认。**

**English.** This opt-in functional rewrite reads one packet and writes the fresh cache plus each dense query-head replica. It does not alias a returned cache to a GEMM operand, compress the cache, or change either matmul. The warmed four-output copy region was faster than the two-kernel control at all six tested step/layout combinations, including both token-major and head-major cache inputs. Whole-layer timings did not consistently improve. See the [emitted CUDA](../examples/generated/cache-fanout), [Compute Sanitizer](../reports/cache-fanout-sanitizer-gb10.txt), and [independent-weight release checks](../reports/cache-fanout-stage-acceptance.json).

```bash
source tools/env.sh
python tools/cache_fanout_slice.py --output /tmp/fanout-slice.json
python tools/qwen_cached_decode.py --fanout-panel --past 31 255 1024 1535 2047 \
  --seed-offset 80831 --profile --emit-dir /tmp/fanout-cuda --output /tmp/fanout-layer.json
bash tools/sanitize_cache_fanout.sh
```

### KV 复制快了，时间花到哪去了？ / Attributing decode to FX nodes

For the [same Qwen2 weights, four visible outputs and FX graphs as the cache-fanout benchmark](../reports/qwen-decode-cache-fanout-gb10.json), we correlated ordinary kernel launches with FX ranges, then profiled **uninstrumented** single-step CUDA Graph replays. Each number below is in µs, **separate copies → fused fanout**. The first three columns are sums of profiler kernel durations per replay, summarized over windows; only the last is a CUDA-event measurement with the profiler off.

| Initial prefix · decode step | KV copy | QK matmul | MLP gate + up + down | Unprofiled layer |
|---|---:|---:|---:|---:|
| 1024 · 1 | 40.13 → 35.40 | 14.99 → 16.18 | 417.99 → 415.88 | 576.89 → 576.36 |
| 1024 · 2 | 39.24 → 34.98 | 13.02 → 13.31 | 414.93 → 416.33 | 573.87 → 575.71 |
| 2047 · 1 | 118.21 → 111.53 | 63.54 → 73.11 | 397.30 → 391.59 | 703.86 → 699.08 |
| 2047 · 2 | 113.49 → 109.50 | 79.48 → 82.08 | 392.26 → 393.00 | 709.69 → 716.02 |

复制在这四组都更快，整层却两胜两负。2047 的第二步，PV 也由 20.43 上升到 25.32 µs；第一步的整层方向与之前独立捕获的计时相反。**不能仅按上下文长度开启融合**，也不能把差异断言为 L2 命中率变化。三次 MLP 投影占去最多的 kernel 时间，是比继续微调复制更值得检验的方向。

[Raw report: all 384 replays](../reports/qwen-decode-phase-attribution-gb10.json) · [collector](../tools/qwen_decode_phases.py) · [offline checks](../tests/test_decode_phase_attribution.py). Eight order-balanced windows per case/variant, six profiled replays per window, with adjacent event samples from the **same captured graph**. Per-kernel signature includes symbol, grid, block, registers and shared memory. Replay boundaries must match graph-launch correlations and graph-node IDs must stay stable. Raw intervals start at each replay's first kernel. Small overlaps reported by Kineto on one stream are kept explicitly, not counted as negative idle time. Profiled duration sums, interval union and first-to-last span are distinct; independently computed medians need not add up. No hardware cache counters, allocator control across separate processes, or serving throughput claim.

### Qwen2 MLP：合并 gate 和 up 投影，值得吗？ / Packing admission study

We captured the actual normalized activation shared by `gate_proj` and `up_proj`, plus both projection outputs, the gated product and `down_proj` output. We compared two BF16 linears with one wider linear using row-concatenated weights, both as private split views and with contiguous output copies. **All four tensors matched bit-for-bit on 36 inputs** (two independent model initializations, prefixes 31 and 127, three input seeds, prefill and two decode steps). The split views share storage and have different strides; they cannot replace two independently visible projection outputs. This is an admission experiment, **not** a pass enabled in SparkFuse.

For the first decode step after prefix 127, unprofiled device medians in µs (seven shuffled windows, **MLP only**, including down projection):

| Weight seed | HF eager | Two linears | Prepacked, private views | Prepacked, contiguous copies | Repack every call |
|---:|---:|---:|---:|---:|---:|
| 141144 | 375.75 | 376.66 | 373.52 | 375.71 | 878.35 |
| 147096 | 375.98 | 376.67 | 374.94 | 376.23 | 878.43 |

Keeping the original weights plus the packed snapshot costs **52.5 MiB** per layer and about **514 µs** to pack here. Replacing one gate-weight row made the old snapshot produce two wrong BF16 gate elements; fresh packing recovered those bits. A hidden permanent weight cache would be wrong. The private-view path issued four kernels versus five for the two-linears path. A smaller kernel count did not make it universally faster than HF eager on prefill. Inductor failed this study’s **exact visible-down-output** gate on every case and was not timed; this is not a claim that its differences exceed ordinary model tolerances. [Raw windows, tensor strides, mismatch witnesses, kernel names and packing cost](../reports/qwen-mlp-projection-admission-gb10.json) · [reproduction](../tools/mlp_projection_audit.py). No complete-layer or generation speedup implied.

### 打包权重与门控 CUDA：子图收益能否带到整层？ / Packed SwiGLU

`PackedSwiGLUProjection.from_linears(gate, up)` 将两个无偏置投影显式转换为唯一注册的 `[gate; up]` 权重。导出的图仍用框架做矩阵乘法；`packed_gates=True` 识别最后一维等分、SiLU、乘法，生成一个读同一行两半的 CUDA kernel。SiLU 先舍入回原精度再相乘；若激活也是图输出，kernel 写出第二个独立结果。外部可见的切分视图仍指向输入。部署模型不再保留旧投影；转换本身需要暂存，测试还保留原模型作对照。同步请求之间可用 `reload_weights_` 原位更新，新值对已经捕获的 CUDA Graph 生效。

```python
from sparkfuse.packed_swiglu import PackedSwiGLUProjection
from sparkfuse.partition import partition_export

gate_up = PackedSwiGLUProjection.from_linears(mlp.gate_proj, mlp.up_proj)
ep = torch.export.export(gate_up, (example_hidden,))
mixed, decisions = partition_export(ep, packed_gates=True, host_backend="native")
```

On DGX Spark, batch 2, BF16, random-weight Hugging Face Qwen2 layer, after 1,024 prefix tokens. Each entry is the unprofiled hot device median in µs (seven shuffled windows); step 2 correctness carries each backend's own KV from step 1. The two right columns differ **only in gate scheduling**.

| Weight seed · step | HF eager | Hybrid, two projections | Packed, framework gate | Packed CUDA, adaptive | Packed CUDA, scalar |
|---|---:|---:|---:|---:|---:|
| 141144 · 1 | 614.14 | 577.22 | 575.77 | 573.55 | 573.66 |
| 141144 · 2 | 605.68 | 572.67 | 574.50 | 571.06 | 570.90 |
| 147096 · 1 | 614.49 | 578.79 | 581.45 | 579.94 | 577.58 |
| 147096 · 2 | 607.31 | 573.94 | 578.31 | 573.43 | 573.90 |

The gate replacement removed one kernel in each measured step (17 → 16 here). The checkpoint conversion alone was not a consistent win. At seed 147096, step 1, the adaptive packed layer was **slower** than the two-projection hybrid. Both generated schedules matched hidden state, probabilities, K and V bits and strides on five held-out trajectories per weight/prefix; Inductor did not meet the layer's predeclared numeric gate and was not timed. These are layer-forward times, not model tokens/s or measured DRAM traffic.

在相同的真实投影输出上独测门控：两行 decode 时，两组权重的标量调度均约 **1.47 µs**，一次最多 8 个元素的调度约 **1.99 µs**，eager 约 **4.03 µs**。62 行 prefill 时则反过来：标量约 **4.71 µs**，向量约 **3.52 µs**。子图不能替代整层测量；小 decode 的同步主机调用还由 eager 的约 14 µs 增至约 27 µs。因此不全局启用这个 pass，也不依行数自动切换整层策略。

[Layer windows: first](../reports/qwen-decode-packed-swiglu-gb10.json) · [second weight](../reports/qwen-decode-packed-swiglu-heldout-gb10.json) · [Epilogue: first](../reports/packed-swiglu-epilogue-gb10.json) · [second weight](../reports/packed-swiglu-epilogue-heldout-gb10.json) · [CUDA source](../examples/generated/packed-swiglu) · [Compute Sanitizer](../reports/packed-swiglu-sanitizer-gb10.txt) · [release checks](../reports/packed-swiglu-stage-acceptance.json).

```bash
source tools/env.sh
python tools/packed_gate_panel.py --seed-offset 86783 --output /tmp/epilogue.json
python tools/qwen_packed_decode.py --seed-offset 86783 --past 1024 --output /tmp/layer.json
bash tools/sanitize_packed_swiglu.sh
```

<a id="bounded-packed-rows"></a>

### 变长 token 门控：同一份二进制，少走一层 FX / Bounded packed rows

上面的两行门控在设备上仅约 2 µs，同步 Python 调用却约 27 µs。为分清时间花在哪里，我们用同一个静态 kernel 比较了完整 FX 调用和内部原生入口，并另做一个可变行数的原生入口。`PackedRows` 编译**整个独立门控子图**，不是跳过任意混合图的 hooks：只接受一个位置参数 `[rows, 2*width]`，由原生入口检查设备、精度、形状上界、连续布局和梯度。可返回门控乘积，或同时返回乘积和舍入后的激活。若切分视图也必须返回，它会拒绝；需要保留这些别名时用静态混合分区。

```python
from sparkfuse.packed_rows import PackedRows

# gate.forward(packed): a, b = packed.chunk(2, -1); return F.silu(a) * b
rows = torch.export.Dim("tokens", min=1, max=256)
ep = torch.export.export(gate, (example_2d,), dynamic_shapes=({0: rows},))
run = PackedRows(ep, schedule="scalar")   # or "adaptive": address-aligned packets
product = run(next_2d)                     # fresh allocation; no per-row compilation
```

Same DGX Spark, width 8,960, BF16. Synchronized **host latency** below includes fresh outputs. The static FX and static native controls use the very same compiled CUDA implementation; the bounded paths use one binary per schedule across all nine tested row counts. Input projections are from one random-weight HF Qwen2 layer; the two-row case is real decode and the other cases are slices of prefill, not independent model runs at different batch sizes.

| Rows · weight seed | Eager | Static FX | Static native | Bounded scalar | Bounded packets |
|---|---:|---:|---:|---:|---:|
| 2 · 141144 | 13.62 | 26.64 | 7.47 | 7.25 | 7.50 |
| 2 · 147096 | 13.74 | 26.61 | 7.49 | 7.25 | 7.58 |
| 62 · 141144 | 18.50 | 29.73 | 10.30 | 11.73 | 10.42 |
| 62 · 147096 | 18.62 | 29.58 | 10.38 | 11.55 | 10.37 |

All values in µs. At two rows the **device** medians of static FX and static native were both about 1.99 µs: most of the host difference is outside the kernel. A narrow native ABI avoids the exported FX entry without removing the native tensor guards. It is not a drop-in replacement for arbitrary module calls, a policy for all packet widths, or an end-to-end decoding speedup. The dynamically compiled Inductor control did not pass the exact product gate on these samples and has no timing entry.

[First weight: raw host/device data](../reports/bounded-packed-rows-gb10.json) · [Independent weight](../reports/bounded-packed-rows-heldout-gb10.json) · [Generated sources](../examples/generated/packed-rows) · [Sanitizer](../reports/bounded-packed-rows-sanitizer-gb10.txt) · [Installed-wheel and full regression](../reports/bounded-packed-stage-acceptance.json). Reproduce with `python tools/packed_rows_panel.py --seed-offset 86783 --output /tmp/packed-rows.json` and `bash tools/sanitize_packed_rows.sh`.
