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
