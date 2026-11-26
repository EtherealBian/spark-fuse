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

### Qwen2 有偏置的 Q/K/V：什么时候能合并？ / QKV packing admission

同一个归一化输入原本经过三个有偏置的线性层，输出宽度分别为 1,536、256、256。一次宽 GEMM 可以省两次启动，但不保证 BF16 输出逐位相同。我们从真实 HF Qwen2 层的 forward hook 取输入和参照值，用两组随机权重、每组三个输入种子，检查三个前缀的 prefill 和其后两步 decode（共 54 组）。除投影外也比较 RoPE 后的 Q/K，并与 HF 写入 cache 的 K/V 新片段交叉核对。

| First weight · projection-only, device µs | HF: three | Packed, private views | Packed, three copies | Q + packed KV | Inductor |
|---|---:|---:|---:|---:|---:|
| Prefix 127 · decode step 1 | 20.58 | 8.81 | 13.07 | 14.43 | 20.58 |
| Prefix 127 · prefill | 118.95 | rejected | rejected | rejected | 119.22 |
| Prefix 255 · prefill | 94.31 | 44.39 | 51.82 | rejected | 94.30 |

仅比较张量值时，可以使用切分视图；调用方若能观察三个独立输出，还要保留原来的 stride 和独立存储，不能把 8.81 µs 当作这个接口的结果。127-token prefill 首个输入的打包 K/V 分别有 **38,399 / 38,022 个 BF16 元素**与参考位型不同；复制输出不修复错误。第二组权重同样失败（38,264 / 38,242）。255-token prefill 则恰好相反：全打包在已测输入上通过，只打包 KV 却失败。因此**没有默认 QKV 改写**，也不按“单 token”或某个前缀向未测模型承诺等价。

Each timing is the median of seven shuffled windows on the last input, after all three inputs for that weight and case passed the private bitwise gate. Weight packing, RoPE and cache writes are outside these timings. The report retains first mismatching indices, BF16 encodings, output layouts and launched kernel names even for rejected candidates; the latter are not timed. These are projection measurements, not layer acceleration or generation throughput.

[Raw inputs' hashes, mismatches, traces and timing samples](../reports/qwen-qkv-projection-admission-gb10.json) · [Release checks](../reports/qkv-admission-stage-acceptance.json). Reproduce on DGX Spark with `python tools/qkv_projection_audit.py --output /tmp/qkv.json` in the Transformers-enabled project environment.

<a id="split-copy3"></a>

### 三路独立输出：一个 CUDA 拷贝 kernel / Three fresh outputs, one launch

Q/K/V 的两个问题可以分开：宽 GEMM 是否与原来三个 GEMM 数值相同，以及**已有宽张量**怎么交付给需要三块独立存储的后继。新 pass `partition_export(ep, split_copies=True)` 只做后者：识别末轴 `split([w0,w1,w2], -1)` 后三次显式 `clone(memory_format=torch.contiguous_format)`，生成一个按位复制的 CUDA kernel。它不改变任何矩阵乘法。三个目的张量分别分配，没有隐藏工作区；若图还返回未复制的切片，该切片仍与输入共享存储。地址及三段宽度允许时最多搬 16 字节，对偏移地址退回较窄的指令。保留默认关闭。

两行 BF16 解码、宽度 `[1536,256,256]`、第一组权重和前缀 127：只测三路拷贝，eager **4.88 µs / 3 次发射**，Inductor **2.82 µs / 3 次**，SparkFuse **1.73 µs / 1 次**。这是设备时间。同步主机时间次序不同：eager **25.09 µs**，完整 FX 路径 **33.70 µs**，同一 CUDA 的原生入口 **11.78 µs**。不能把原生入口与任意混合图的 Python 调用混为一谈。

Then we traced the copies through a real single-layer Qwen2 decode. The fixture holds one packed QKV weight and one packed bias, with no stale copy. Three independent `linear` calls over **row slices of those parameters** reproduced the HF projection bits and strides on every tested prefill and decode input, including the known 127-token counterexample to the wide GEMM. This is a tested fallback for the fixture, not a universal admission rule. Only the decode study uses a wide GEMM; HF prefill initializes the cache. Each backend carries its own first-step output into the second step, on five held-out trajectories per weight and prefix.

| Prefix 1,024 · device µs | HF eager | Original direct hybrid | Re-export control, original weights | One bank, 3 GEMMs | Wide GEMM, strided views | Wide GEMM, 3 clones | Wide GEMM, 1 copy kernel |
|---|---:|---:|---:|---:|---:|---:|---:|
| Weight 141144 · step 1 | 616.43 | 569.59 | 579.34 | 578.32 | 586.72 | 567.48 | 562.96 |
| Weight 141144 · step 2 | 609.72 | 577.56 | 580.06 | 580.53 | 581.38 | 566.28 | 563.72 |
| Weight 147096 · step 1 | 616.48 | 569.25 | 581.06 | 579.14 | 585.07 | 568.95 | 564.82 |
| Weight 147096 · step 2 | 609.15 | 577.39 | 579.73 | 580.46 | 582.39 | 568.47 | 564.22 |

Why copy at all? The private strided views miss both the dense-input rotary and cache-append fusions (38 kernels). Three clones restore them (31); horizontally fusing the same copies retains both passes (29). The fair isolation of the new pass is **31 vs 29**, not 48 vs 29. Re-exporting the unchanged graph itself alters partition coverage (30 launches vs the direct compiler’s 17), so both controls are reported. All four layer outputs met the exact bits/strides gate for SparkFuse. Layer-level Inductor did not meet its predeclared accuracy gate and has no latency entry; it did pass the independent copy-only gate. Layer-forward time is not model throughput.

[First weight, raw windows and FX](../reports/qkv-layout-layer-gb10.json) · [Independent weight](../reports/qkv-layout-layer-heldout-gb10.json) · [CUDA sources](../examples/generated/split-copy3) · [Sanitizer](../reports/split-copy3-sanitizer-gb10.txt) · [Release suite](../reports/split-copy3-stage-acceptance.json). Reproduce with `python tools/qwen_qkv_layout.py --seed-offset 86783 --past 127 1024 --output /tmp/qkv-layout.json` and `bash tools/sanitize_split_copy.sh`.

<a id="packed-rotary"></a>

### 省去 Q/K 的中间写回 / Packed Q/K RoPE and V copy

当 Q/K 的显式拷贝只用于分头和 RoPE 时，`partition_export(ep, packed_rotary=True)` 直接从原宽张量读两个半头，按原精度分别舍入两个乘积，再相加并写入 Q/K 的最终布局。同一次 launch 为 V 复制原始位型；三个输出各有独立存储。未复制的切片仍然指向原输入。若图返回 Q/K 的中间拷贝、暴露旋转内的乘积，或所需相位计算晚于第一个 V 用户，就保留原路径。此选项默认关闭。

| 已有宽张量上的后处理，设备 µs | eager | 原两个 kernel | 融合后一个 kernel |
|---|---:|---:|---:|
| 权重 141144 · 127-token prefill | 32.18 | 4.88 | 3.51 |
| 权重 141144 · 1,024-token prefill | 269.98 | 90.62 | 42.94 |
| 权重 147096 · 1,024-token prefill | 272.90 | 93.53 | 47.10 |
| 权重 141144 · 前缀 127，第一步解码 | 18.53 | 2.32 | 1.76 |

Both compiled paths run their full FX wrappers. At 1,024-token prefill, synchronized host time was **120.67 → 60.91 µs** for the first weight. These are epilogue measurements: every method receives the **same already-computed wide tensor**. In particular, the upstream wide GEMM still differs from the three original HF projections at 127-token prefill. Inductor’s epilogue changed some Q/K bits and was excluded from exact-gate timing; the report retains its mismatch counts.

For the full single-layer **decode**, we compared the same exported packed-checkpoint graph on two random weights, two prefixes and five held-out two-step trajectories per combination. The four outputs matched HF bits and strides, cache prefixes stayed intact, and every backend consumed its own cache on step two. The joint pass removed **one launch (29 → 28)**. The device-time change was small and inconsistent: at prefix 1,024, first weight, step one was slower (**564.82 → 571.17 µs**) and step two faster (**565.51 → 563.72 µs**). Synchronized host latency decreased on each of these eight measured weight/prefix/step combinations. Whole-layer Inductor did not meet its existing tolerance; it has no reported latency. No claim about full-model speed follows from these results.

[Raw samples and graphs](../reports/qwen-packed-rotary-gb10.json) · [Second weight](../reports/qwen-packed-rotary-heldout-gb10.json) · [Generated CUDA](../examples/generated/packed-rotary) · [Compute Sanitizer](../reports/packed-rotary-sanitizer-gb10.txt) · [Regression suite](../reports/packed-rotary-stage-acceptance.json). To reproduce with the Transformers-enabled environment, use `python tools/qwen_packed_rotary.py --past 127 1024 --seed-offset 80831 --output /tmp/joint.json`; run `bash tools/sanitize_packed_rotary.sh` for memory, race and synchronization checks.

<a id="phi3-native"></a>

### 原生 Phi-3：保留 V 的视图 / Native Phi-3, full and partial RoPE

Phi-3 原生就使用宽 QKV 投影，无需重新打包权重。`partition_export(ep, sliced_rotary=True)` 识别同一投影上的两段切片，读取共享相位，对 Q/K 旋转部分做两次独立舍入的乘法再相加，未旋转的尾部按位复制。Q/K 写入各自的新张量；V 保持为原张量的视图。以 127-token prefill 为例，返回的 V 在原宽张量中的偏移是 1,792 个 BF16 元素，stride 为 `(260096, 128, 2048, 1)`；换成 `.contiguous()` 虽然数值一样，接口就不对了。匹配条件不成立时保留 ATen。选项默认关闭。

| GB10 设备时间，µs | eager | 原混合分区 | 加入切片旋转融合 |
|---|---:|---:|---:|
| 整层：127-token prefill，旋转半头 | 444.07 | 372.85 | 344.74 |
| 整层：前缀 127，第一步解码，旋转半头 | 306.62 | 284.20 | 268.51 |
| 整层：前缀 1,024，第一步解码，旋转半头 | 397.65 | 382.22 | 365.83 |
| 仅 Q/K/V 后处理：1,024-token prefill，旋转半头 | 155.07 | 154.98 | 39.24 |
| 仅 Q/K/V 后处理：1,024-token prefill，旋转整头 | 278.39 | 274.51 | 34.99 |

These are medians of seven shuffled warm-address replay windows, not model throughput. Batch 2, 12 query heads, 2 KV heads, head width 128; the table uses random weight seed 184406. On independent weight seed 191640, the same 127-prefix first decode went from **294.06 to 277.83 µs** on device and **473.73 to 380.64 µs** with synchronized host calls. The epilogue went from **12 launches to 1**, the complete layer from **26 to 15**. Large (1,024-token) whole-layer prefill was profiled, **not timed**.

Across two weights, both rotary widths (64 and 128), two prefixes and five unseen inputs per combination, each backend carried its own prefill cache into two decode steps. We checked hidden states, probabilities, K/V bits and metadata (including the underlying storage size), mask zeros, unchanged old cache and exact prefixes against a persistent Hugging Face `DynamicCache`. For the isolated epilogue the upstream input was the real hooked projection. Inductor was evaluated on the same exports, but did not pass the layer tolerance or bitwise epilogue gate; its failure details remain in both reports, without a speed ranking.

[First weight: raw windows and FX](../reports/phi3-native-packed-gb10.json) · [Independent weight](../reports/phi3-native-packed-heldout-gb10.json) · [Generated CUDA](../examples/generated/sliced-rotary) · [Sanitizer](../reports/sliced-rotary-sanitizer-gb10.txt) · [Full regression](../reports/phi3-native-stage-acceptance.json). Reproduce with `python tools/phi3_native.py --seed-offset 124093 --past 127 1024 --rotary-dim 64 128 --output /tmp/phi3.json`; change the offset to `131327` for the independent weight. `bash tools/sanitize_sliced_rotary.sh` checks memory, race and synchronization errors.

<a id="phi3-cache"></a>

### Phi-3：用一次发射更新 KV 缓存 / Fused cache updates

在 `sliced_rotary=True` 的基础上开启 `sliced_cache=True`，编译器把宽 QKV 投影上的 Q/K 旋转和两个缓存拼接合为一次发射。Q、新 K 缓存、新 V 缓存各自拥有独立存储；如果图还返回未拼接的 V，它仍引用原投影。第一次更新直接从 prefill 留下的跨步 V 视图读取历史，第二次读取上一步产生的连续缓存。旧缓存不被覆盖。只有在图能证明切片不重叠、旋转结果 K 没有其他使用者、历史布局符合静态描述时才替换。

两组随机权重，旋转宽度 64/128，前缀 127/1024；每个组合测试连续单 token 更新 `(1,1)` 和带因果掩码的分块更新 `(4,3)`。五条独立输入轨迹中，各后端携带自己的缓存。两种 SparkFuse 路径都对齐 Hugging Face eager 的隐藏态、注意力概率和完整 K/V（BF16 逐位、步幅、偏移及存储大小）。以下是前缀 1024、旋转宽度 64、`(4,3)` 的热地址 CUDA Graph 设备中位数，单位 µs：

| 计时范围 / window | eager | 原融合 / rotation only | 加入 KV 更新 / fused | Inductor, cast emulation |
|---|---:|---:|---:|---:|
| 首组权重，4-token 子图 | 30.83 | 11.70 | 3.99 | 6.34 |
| 首组权重，4-token 整层 | 410.45 | 372.02 | 363.88 | — |
| 首组权重，接续 3-token 整层 | 425.66 | 388.38 | 381.86 | — |
| 第二组权重，4-token 整层 | 411.51 | 371.72 | 364.56 | — |
| 第二组权重，接续 3-token 整层 | 425.65 | 385.71 | 377.76 | — |

Across both weights the profiler counted **3 → 1** launches for the isolated update, **15 → 13** for the single-token layer and **16 → 14** for the chunked layer. All 32 measured whole-layer device medians improved against the rotation-only path (2.24–9.43 µs). These are two successive functional cache updates of one layer, not sustained generation throughput; allocating fresh caches still copies the prefix on every call.

Default Inductor failed the isolated contract; `emulate_precision_casts=True` qualified in 24 of 32 isolated windows. For full-width rotation with multiple new tokens, even the cast-aware output Q could have matching bits but a different stride. Neither Inductor variant passed the existing whole-layer numerical gate across all five trajectories for any configuration. The dash omits those timings, not the failed results. Both reports retain the per-output checks and all seven time windows for admitted methods (including synchronized host time).

[Primary report](../reports/phi3-sliced-cache-gb10.json) · [Independent weights](../reports/phi3-sliced-cache-heldout-gb10.json) · [Emitted CUDA](../examples/generated/sliced-cache) · [Sanitizer](../reports/sliced-cache-sanitizer-gb10.txt) · [Regression](../reports/sliced-cache-stage-acceptance.json). Run `python tools/phi3_cache.py --seed-offset 145091 --output /tmp/phi3-cache.json`; repeat with `152519`. Run `bash tools/sanitize_sliced_cache.sh` for memory, race and synchronization checks.

<a id="decomp-cast-replay"></a>

### BF16 舍入策略改变后，谁才是有效的对照？ / Replaying rejected comparisons

冻结通用搜索的前 773 条索引记录。19 条 `decomp` BF16 错误中，17 条保留了完整对照，能确认是 Inductor 默认设置超过容差；另有一条旧记录缺少 worker，及一条 [SparkFuse 自身归约次序导致的发现阶段失败](../reports/mean-tree-history.json.gz)，不能归为 Inductor 错误。保留原始判定，另取三组曾通过的同宽对照。在同一 DGX Spark 上重放原比较种子和当时选中的 SparkFuse 布局，17 条失败都复现了原先**超出容差的元素数**。两轮分别加入两个新种子，总计 20 种形状、100 个不同请求；这个刻意抽取的集合不能估计整个搜索的失败率。

分解式残差 RMSNorm 有两个 BF16 舍入点：残差相加之后、归一化值乘权之前。`31 × 1024` 的原请求在 `[24,424]` 处，eager 为 `−1.203125`；在独立 eager 算式里仅省略其中一次舍入得到 `−1.2109375`，两次都省略得到 `−1.21875`，恰好等于默认 Inductor。本请求有 10,858 个输出位模式不同，但仅该元素超出原容差。对全部 100 个请求，“两次都省略”的算式始终是四个候选里最接近默认 Inductor 的；只有 28 个请求的全部元素都与它相同。这里比较可观察的输出，不把它当作 Inductor 内核内部计算步骤的证明。

| 后端 / backend | 容差内的请求 / 100 | 全部 BF16 位相等 / 100 |
|---|---:|---:|
| Inductor 默认 | 39 | 0 |
| Inductor `emulate_precision_casts=True` | 100 | 95 |
| SparkFuse 历史搜索选中的布局，当前编译器重放 | 100 | 98 |
| SparkFuse 当前默认布局 | 100 | 100 |

The cast-aware comparator qualifies for timing on **all 20 shapes**. On `256 × 6144`, its warm device median was **7.62 µs** versus **10.92 µs** for current SparkFuse (7.61 vs 11.02 on the independent timing seed). On `512 × 14336`, current SparkFuse took 200.70 vs 201.91 µs for cast-aware Inductor in the first run, 193.03 vs 202.24 in the second. These are single exported regions. Backend admission is evaluated before timing the uninstrumented output. Diagnostics run separately and all failed comparisons remain in the reports.

[Selection and original comparisons](../reports/decomp-cast-catalog.json.gz) · [Replay](../reports/decomp-cast-replay-gb10.json) · [Independent inputs](../reports/decomp-cast-replay-heldout-gb10.json) · [Regression](../reports/decomp-cast-stage-acceptance.json). From the repository root run `python tools/decomp_casts.py --output /tmp/decomp-replay.json`; add `--seed-offset 19609` for the independent requests. `--case decomp-31x1024-bfloat16 --verify-only` runs just the small witness without timing.

<a id="packed-mean"></a>

### 残差 RMSNorm：低精度共享存储，保持 FP32 归约顺序 / Packed residual tile

这个布局只匹配明确导出的运算链：两个 FP16/BF16 矩阵相加并舍入，升为 FP32 后做平方、行均值和归一化，舍回原精度后再乘一维权重。匹配的是有类型的依赖关系，而不是节点名或输入位置。残差也被返回时，生成器按图指定的位置多写一个输出。当前范围是连续的二维输入、至少 16 行、宽 1024–8160、宽度能整除所选向量、总计最多 2²⁴ 个元素。原生 `aten.rms_norm`、不符合的广播或额外可见的中间值仍走原路径；默认布局未改。

一个线程块处理一行。先以两个窄元素为一组相加，把已舍入的残差留在共享内存里；再按原来的 128 条 FP32 累加序列计算平方和。每个 warp 自己完成最后 32 组求和，不需要第三次块级同步。最后从共享内存取回残差，升精度乘逆均方根，降精度，再做窄精度权重乘。非对齐但连续的视图有单独通路。对 `256 × 6144` 对齐 BF16，所选旧/新二进制的静态共享内存是 **25,732 / 13,440 字节**，SASS 有 **3 / 2 个块同步指令位置**。新对齐版本含 128 位全局读写；旧/新版本分别使用 20 / 28 个寄存器。这些是编译资源和静态指令，不能当作实际显存流量或单一加速原因。

两种布局分别选型：旧组包含全部 10 种有序树/分组配置，新组包含所有合法的线程数 × 向量宽度。每种形状和地址条件分别在独立输入上选出双方最快者，随后冻结配置，再更换输入和权重。两轮共 **108 个不同请求**；时间对照要求三组未参与选型的输入上，每个可见输出都逐位相同。下表仅列冻结配置后的复验，单位 µs；每一列都是**正常调用、分配新输出、热地址 CUDA Graph 设备时间**。同步主机时间和预分配输出重放在原始报告中分列。

| 形状、精度、输出 | 地址 | eager | 旧配置最优 | 新配置 | Inductor 保留精度转换 |
|---|---|---:|---:|---:|---:|
| 256 × 6144，BF16，归一化结果 | 对齐 | 135.56 | 11.02 | 8.97 | 7.75 |
| 31 × 1024，BF16，归一化结果 | 偏移 | 12.22 | 2.33 | 2.07 | 4.03 |
| 17 × 4100，FP16，同时输出残差 | 对齐 | 16.31 | 4.39 | 2.83 | 2.67 |
| 33 × 2048，BF16，同时输出残差 | 对齐 | 13.75 | 3.35 | 2.35 | 2.67 |

Across nine shape/dtype/output cases at two alignments, the frozen tile reduced the hot device median in **18/18** comparisons against the independently selected old schedules (**1.12–1.73×**, median **1.37×**); synchronized Python host medians improved in only **10/18**. Default Inductor passed no three-request *bitwise* gate in this repeat. Cast-aware Inductor qualified in **10/18** groups and was faster in **6** of those. Non-admission here is not necessarily an ordinary tolerance failure; every raw comparison includes both counts. These results describe a single exported region on GB10, not Transformer throughput.

[Selection, parity and timings](../reports/packed-mean-discovery-gb10.json) · [Frozen schedules on new inputs](../reports/packed-mean-heldout-gb10.json) · [CUDA examples](../examples/generated/packed-mean) · [Static SASS and resources](../reports/packed-mean-sass-gb10.json) · [Sanitizer: memory/race/sync](../reports/packed-mean-sanitizer-gb10.txt) · [Regression: 728 tests, 28 checks](../reports/packed-mean-stage-acceptance.json).

```bash
source tools/env.sh
python tools/packed_mean_panel.py --output /tmp/packed-primary.json
python tools/packed_mean_panel.py --selection /tmp/packed-primary.json --seed-offset 21061 --output /tmp/packed-repeat.json
python tools/inspect_packed_mean.py /tmp/packed-primary.json --cache "$SPARKFUSE_CACHE" --output /tmp/packed-sass.json
bash tools/sanitize_packed_mean.sh
```

<a id="qwen-packed-mean"></a>

### 一个更快的内核能给 Qwen2 整层省多少？ / A faster tile inside a real decoder layer

对随机权重的 Hugging Face `Qwen2DecoderLayer`，导出图中的 `o_proj` 产生 `[batch, seq, 1536]`。一次 CUDA 调用把它加到原输入，同时返回两份值：跨过 MLP 的 BF16 残差，以及供 gate/up 投影读取的归一化结果。新布局识别连续静态三维输入；只有两条残差流的每个维度都吻合，才把 batch 和 seq 合并为发射行数。广播和符号三维输入不走这条路径。

在 DGX Spark 上用第一组权重选型，在第二组权重上冻结调度复测。旧组试 10 种配置，新组试 9 种；只改这一个融合区，两个混合图的其他算子、调度及连接相同。下表为第二轮的热地址设备时间（µs），每栏取七个交错窗口的中位数。子图的输入由**未融合的 ATen 层**提供。

| 序列长（batch=2） | 子图：旧 → 新 | 整层：旧 → 新 | 整层 HF eager | 整层保留转换的 Inductor |
|---:|---:|---:|---:|---:|
| 33 | 3.17 → 2.15 | 561.16 → 558.42 | 609.07 | 未通过逐位门槛 |
| 65 | 3.52 → 2.67 | 564.60 → 570.78 | 623.50 | **544.87** |
| 129 | 5.56 → 3.35 | 691.87 → 690.26 | 803.35 | 未通过逐位门槛 |

两组权重、三个长度、共 18 个未参与选型的请求：SparkFuse 的残差、归一化结果、整层隐藏态和注意力概率都与 eager 逐位相同。新旧整层都是 19 次发射；已有混合图相对 eager 的 48 → 19 次不能归功给本次优化。长度 65 时新内核更快，整层却更慢；同组 cast-aware Inductor 既通过逐位检查，也比两种 SparkFuse 配置快。默认 Inductor 在六组整层对照中均未通过三请求逐位门槛，因此不列速度。原始数据保留逐位与普通容差检查，以及同步主机时间。

**English.** A static `[B,S,H]` residual tile improves the isolated post-attention region by **1.32–1.66×** on the second weight seed. End-to-end layer times move in both directions: changing the tile alone does not justify a new global default. The experiment uses one random-weight BF16 layer and measures forward latency; it does not measure trained-model generation throughput.

[Discovery](../reports/qwen-packed-mean-discovery-gb10.json) · [Frozen schedules, independent weights](../reports/qwen-packed-mean-heldout-gb10.json) · [Full regression](../reports/qwen-packed-mean-stage-acceptance.json) · [Memory, race and synchronization checks](../reports/qwen-packed-mean-sanitizer-gb10.txt).

```bash
source tools/env.sh  # use the interpreter with the optional Transformers integration
python tools/qwen_packed_mean.py --output /tmp/qwen-primary.json
python tools/qwen_packed_mean.py --selection /tmp/qwen-primary.json --seed-offset 98627 --output /tmp/qwen-repeat.json
```

<a id="context-site"></a>

### 要不要部署一个局部优化？用整图做决定 / Selecting a schedule in context

`select_site` 只改导出图中指定的一处 CUDA 调度；即便另一处生成了相同的 IR，它也保留原配置。在 DGX Spark 上用相同的规则测试了两类图：

| 图与输入 | 确认阶段采纳 | 独立请求上的设备时间（现有 → 入选，µs） |
|---|---:|---|
| 残差 RMSNorm → 64 维线性投影，BF16，128 × 2560 | 2/2 | 对齐 10.50 → 7.61；偏移一个元素 10.85 → 8.32 |
| 同一结构，FP16，63 × 8160 | 2/2 | 对齐 13.90 → 10.85；偏移一个元素 14.25 → 12.38 |
| Qwen2 解码层的 SwiGLU 门控，BF16，序列 33/65/129，每个长度两组随机权重 | 0/6 | 保留现有配置；确认阶段整层加速比中位数仅 1.003–1.014× |

投影组返回残差和投影结果。新旧调度都是 3 次 CUDA 发射，4 组独立复测的整图设备加速为 **1.15–1.38×**；同步主机延迟有一组完全相同。BF16 对齐时，保留精度转换的 Inductor 和入选调度同为 **7.61 µs**。Qwen2 的候选门控单测全都更快，但在序列长 33 的第二组权重上，强制采用候选反而把整层设备时间从 553.72 增至 557.22 µs。这里的基准是随机权重的一层前向，不能换算生成吞吐量。

**Decision protocol.** Start with the incumbent and up to 16 caller-specified schedules. Run each legal candidate on the first request, comparing the actual mixed-graph site inputs, outputs and composed result with the unmodified exported ATen graph. Strict checks include dtype, device, shape, stride and bits, plus shared-storage relationships and relative offsets within each checked boundary. On correctly rounded candidates, use five shuffled whole-graph windows to nominate one schedule. On different data require at least six of seven paired wins **and** a median incumbent/candidate device-time ratio of at least 1.02 (configurable before the run). The third request audits the decision without reselection. The published panels add two independent replays and separately measure CUDA Graph time and synchronized host dispatch. Setup, validation, hashing, compilation and profiling are outside both timers. One timing window uses seven groups of four graph replays, each containing three whole forwards at fixed addresses.

**When to use it.** Use a static, deterministic, functional export, positional tensor inputs on one GPU, and an ordinary generated region with traceable ATen boundaries. The graph returned after calibration is for the validated immutable model and layout; it does not revalidate weights on every forward. A receipt is specific to its site, state contents, code, input layout/alignment and recorded environment. Changing weights means calibrating again. Numerical rejection during discovery/confirmation keeps or falls back to the incumbent; failed audit, replay or CUDA execution raises. A new receipt file is installed without overwriting another; it is a checksummed experiment record, not a signed artifact. No timing entry is written to the shared IR dispatch database.

```python
from sparkfuse import Schedule
from sparkfuse.contextual import select_site, replay_site

# ep is a static ExportedProgram; site is the module name from partition_export's records.
# train, confirm, audit and unseen are four different, same-layout positional input tuples.
opts = {"host_backend": "native"}
graph, receipt = select_site(ep, site, [Schedule(128, "flat", False, 4)],
                             train, confirm, audit, partition_options=opts,
                             path="site-receipt.json")
replayed, checked = replay_site(ep, receipt, unseen, partition_options=opts)
```

[Qwen2 decisions, raw windows and Inductor comparisons](../reports/qwen-context-gb10.json) · [Residual/projection decisions and timings](../reports/context-projection-gb10.json) · [Release checks](../reports/context-stage-acceptance.json) · [Sanitizer checks](../reports/context-sanitizer-gb10.txt). Both reports retain numerical failures for Inductor even when those paths are excluded from strict timings. The regular and cast-aware Inductor paths are different baselines; cast-aware Inductor passed and beat the retained Qwen2 dispatch in all four 33/65-token panels.

```bash
source tools/env.sh
python tools/context_projection.py --output /tmp/context-projection.json
# Requires the optional Transformers dependency:
python tools/qwen_context.py --output /tmp/qwen-context.json
python -m pytest -q tests/test_contextual.py tests/test_contextual_evidence.py
```

<a id="resident"></a>

### 一层 Qwen2 的请求延迟：常驻 CUDA Graph / Serving fresh results from a resident graph

`capture_export` 捕获固定形状的 ATen 图或 SparkFuse 混合图。每个请求可以传入新地址：先把所有输入复制到私有缓冲，重放一次图，再将需要独占的结果复制给调用者。下一次重放不会覆盖上一次返回的结果。原图若返回输入本身、输入切片，或同一输出的多个视图，返回值仍保持这些关系。捕获时复制权重，换模型后需重新捕获。

**什么时候值得用？** 下表是在 DGX Spark 上交替输入八个不同请求的 BF16 Qwen2 解码层结果。每个长度测试两组随机权重；普通混合图和常驻混合图使用相同 CUDA 调度。每次同步等待完整响应时，六组都变快；同时排队八个请求时，长度 65 和 129 的普通混合图更快。

| 序列长度 | 权重种子 | 单请求同步：普通 → 常驻 (µs) | 八请求排队：普通 → 常驻 (µs/次) |
|---:|---:|---:|---:|
| 33 | 813157 | 633.12 → 594.52 | 595.35 → 585.51 |
| 33 | 821641 | 637.40 → 593.69 | 603.40 → 586.45 |
| 65 | 813157 | 646.38 → 614.11 | 593.49 → 602.68 |
| 65 | 821641 | 646.23 → 611.06 | 596.80 → 599.75 |
| 129 | 813157 | 761.48 → 745.16 | 728.24 → 734.59 |
| 129 | 821641 | 760.86 → 755.20 | 728.52 → 744.37 |

CPU 提交从 194–203 µs 降至 35–39 µs。提交耗时不含随后等待 GPU 的时间；表中的同步延迟包含该等待。各组先用八次请求逐位核对两个可见输出，检查每个请求的输出存储独立、布局和别名匹配、屏蔽位置为零、概率行和误差不超过 0.004。默认 Inductor 未通过六组的逐位门槛；保留精度转换的 Inductor 有一组八次全部通过（长度 33、种子 813157）：单请求 603.27 µs，八请求排队 **578.18 µs**，后者快于常驻混合图的 585.51 µs。其余组保留差异，但不作同精度速度排名。

**Contract.** Use nonempty whole contiguous tensor inputs on one CUDA device, with independent storage and the same layout on every request. Use the creating thread and stream. Arrange dependencies on an upstream producer before invoking the handle; `record_stream` protects lifetime, not readiness. Call `close()` or use a context manager to wait for pending replay before releasing resources. The executable supports static functional operators from an explicit allowlist; it rejects training dropout, mutations, dynamic dimensions, alias-only graphs and fresh output views without a visible whole owner. Autocast and nested capture are rejected. Model state and precision policy are those used for capture. Qualification compares against the unmodified export on two different requests, not every possible value.

```python
from sparkfuse.resident import capture_export
from sparkfuse.partition import ordered_collectives

opts = dict(host_backend="native", schedule_policy=ordered_collectives, rotary_pairs=True)
with capture_export(ep, example, audit, backend="mixed", partition_options=opts) as run:
    first = run(*request_a)
    second = run(*request_b)  # first's owned results remain valid
```

**Costs and interpretation.** The six mixed captures took 47–157 ms with warm artifact caches, including partitioning and correctness qualification. Inputs plus privately copied weights require 93.84–94.59 MB (decimal); this excludes graph intermediates and library workspaces. PyTorch allocator deltas in the report are process counters and may include release of earlier allocations; they do not estimate total executable memory. Inputs and fresh outputs are copied on every call: at length 129 this is 0.99 MB in and 1.59 MB out. Neither a new global default nor an autoregressive tokens/s result follows from this experiment.

[Eight-request gates, setup, seeds and raw timings](../reports/resident-qwen-gb10.json) · [Release checks](../reports/resident-stage-acceptance.json) · [Generated-kernel sanitizer checks](../reports/resident-sanitizer-gb10.txt).

```bash
source tools/env.sh  # use a Python environment with the optional Transformers dependency
python tools/resident_qwen.py --output /tmp/resident-qwen.json
python -m pytest -q tests/test_resident.py tests/test_resident_evidence.py
```

<a id="shape-family"></a>

### 三种序列长度，共用一份模型权重 / Multiple shapes, one frozen weight snapshot

Qwen2 的输入长度会变，投影和归一化的权重并不变。`capture_family` 对同一个 eval 模型分别导出指定形状，在 DGX Spark 上为每种形状捕获一张独立 CUDA Graph，但只复制一次权重。后续形状绑定快照前，逐项核对实际使用的状态名、布局、共享存储关系和每个字节。旧请求返回的结果在切换形状后仍然有效。未登记的形状直接报错，不在请求路径编译。

**实测省下 187,191,296 字节（178.52 MiB）**。长度 33、65、129 的 BF16 Qwen2 层共有 12 个权重张量。独立捕获三次会保留 36 块权重存储，共享后只有 12 块。下表是**相对准备前的 PyTorch 活动分配增量**，不是系统总内存或显存预留量。每组随机权重、每种方案都在独立进程测量，预热流程相同：

| 三种长度全部就绪时 | 各自捕获 | 共享权重 |
|---|---:|---:|
| 权重（字节） | 280,786,944 | 93,595,648 |
| 输入暂存（字节） | 1,714,956 | 1,714,956 |
| 活动分配增量，含图私有分配（字节） | 385,619,456 | 198,428,160 |

两组权重的分配结果相同；关闭三张图并清理临时对象后，四个进程各自回到准备前的计数。形状切换实验交替输入三种长度，各保留四个请求的两个输出。共 24 个请求中，路由路径的隐藏状态、注意力概率、布局与别名都与 eager 一致。

**速度基本持平。** 两组权重上，预先指定独立图与调用共享路由的同步响应分别是 **605.67 → 605.54 µs**、**615.94 → 616.60 µs**。在同一组共享图内，路由比预先指定增加了约 1.9 和 2.5 µs 的 CPU 提交时间。12 个请求排队时，共享路由和普通混合图在两组权重上互有胜负。默认 Inductor 在两组均未达到逐位一致；保留精度转换的 Inductor 分别通过 7/12、9/12 个请求，未纳入整个混合序列的同精度排名。

```python
from sparkfuse.family import capture_family

# One eval model, initialized on the owning GPU; two distinct input tuples per shape.
with capture_family(model.eval(), [(ex33, audit33), (ex65, audit65), (ex129, audit129)],
                    partition_options=opts, max_static_bytes=100_000_000) as serve:
    out65 = serve(*req65)
    out33 = serve(*req33)  # out65 is still valid
```

**Limits.** Admission is explicit, all-or-nothing, for 1–16 distinct shape/dtype keys. Each graph has its own input and intermediate storage. The byte cap covers one frozen model and all input staging, not graph pools, temporaries, driver allocations or library workspaces. In the example it fits under a 100 MB *static* cap; it does not run with 100 MB total memory. All handles use one device, stream and thread. Direct timings pick the handle before the clock; routed timings pay for selection. The memory reduction is verified for these shapes and does not imply faster model execution.

[Independent process counters, storage census, request gates and timing windows](../reports/shape-family-qwen-gb10.json) · [Release checks](../reports/family-stage-acceptance.json) · [Generated-kernel sanitizer checks](../reports/shape-family-sanitizer-gb10.txt).

```bash
source tools/env.sh
python tools/shape_family_panel.py --output /tmp/shape-family.json
python -m pytest -q tests/test_family.py tests/test_shape_family_evidence.py
```

<a id="token-quant"></a>
### 残差与逐 token INT8 量化 / Residual per-token quantization

`partition_export(ep, token_quant=True, host_backend="native")` 识别 FP16/BF16 的 `z=x+r` 和一段固定公式：先将窄精度残差升至 FP32，沿最后一维求绝对值最大值，以 `max(amax,1e-5)/127` 为尺度，除以尺度、舍入到最近偶数、限制到 `[-127,127]`，再写有符号 INT8。rank 为 2 或 3，末维 1–8192。每个线程块处理一行，warp 内 shuffle、warp 间共享内存归约；仅为图外可见的残差或尺度分配输出。[生成的 CUDA](../examples/generated/token-quant/residual_bf16_33x1536.cu) 和 [原始报告](../reports/token-quant-gb10.json) 可逐项核对。

| 片段（新输入，种子 279239） | eager 设备 µs | 融合后设备 µs | 实测 kernel 次数 |
|---|---:|---:|---:|
| FP16 `17×1025`，仅量化与可见残差 | 13.07 | 3.41 | 10 → 1 |
| BF16 `33×1536`，同上 | 14.44 | 4.18 | 10 → 1 |
| BF16 `1×65×8160`，同上 | 41.07 | 14.43 | 10 → 1 |
| BF16 `2×16×1536 → 128`，含 INT32 GEMM 和反量化 | 25.36 | 15.11 | 14 → 5 |
| FP16 `33×1536 → 64`，同上 | 51.30 | 40.38 | 14 → 5 |

两组种子、三组地址偏移的输出都与 eager 逐字节一致。投影的 INT32 累加另与 CPU 整数乘法核对。inductor 默认和启用 `emulate_precision_casts` 的版本都在每组至少一个请求中产生不同编码，因此保留诊断、不作为同精度速度基线。一处 FP16 样本的 eager 商为 −63.499996，舍入到 −63；另一条路径给出 −64。报告记录实际输出和重算商，不据此推断编译器内部指令。边界回归分别枚举两种 dtype 的全部 65,536 个源编码、四种配对策略，并测试半整数附近；并未枚举所有输入对。

The table measures warm device replay on DGX Spark; synchronized host samples are separate. Each projection uses PyTorch's existing `torch._int_mm` with synthetic INT8 weights, not a generated GEMM or a model-quality experiment. Unsupported formulas and escaping intermediate values stay in the ordinary partitioner. Run from the repository root after `source tools/env.sh`:

```bash
python tools/token_quant_panel.py --output /tmp/token-quant.json
bash tools/sanitize_token_quant.sh
```

<a id="int32-epilogue"></a>
### 整数投影的反量化尾部 / Ordered INT32 epilogue

上一组实验的五次发射里仍有三次用于反量化。`projection_epilogue=True` 将 INT32 累加结果升至 FP32，先乘每个 token 的尺度，再乘每个输出通道的尺度。两次乘法不能重新结合。累加器保留在框架中，可见的中间结果按需写回。`token_quant=True, projection_epilogue=True` 可同时启用两端；中间的 INT8 GEMM 始终由 PyTorch 执行。

| 投影片段，种子 903757 | eager | 只融合量化 | 只融合反量化 | 两端融合 |
|---|---:|---:|---:|---:|
| BF16 `2×16×1536 → 128` | 25.50 | 15.10 | 22.63 | 11.71 |
| FP16 `33×1536 → 64` | 51.48 | 40.38 | 48.55 | 37.64 |
| BF16 `2×32×3072 → 256` | 37.06 | 23.31 | 33.72 | 20.22 |

单位为热 CUDA Graph 设备 µs。两组种子 × 三个尺寸中，投影的发射计数分别为 14、5、12、3；每组四次独立请求与 eager 逐字节一致，整数累加另经 CPU 核对。只测反量化时，Inductor 也通过了准入：两者都是一次发射，生成的 CUDA 设备中位数为 1.47–1.74 µs，Inductor 为 3.51–3.52 µs。不过这个小片段的混合 FX **同步主机调用慢于直接 eager**。整段投影没有这个结论；两种 Inductor 配置均因量化边界输出不一致，未参与整段等精度计时。

The [ablation report](../reports/projection-epilogue-gb10.json) separates isolated dequantization, each pass on its own, and the combined projection. It includes per-output byte diagnostics, shuffled raw device/host samples and a profiler census. This remains a static synthetic W8A8 projection experiment, not a pretrained-model result. Run `python tools/projection_epilogue_panel.py --output /tmp/dequant.json` and `bash tools/sanitize_projection_epilogue.sh` after sourcing the environment.

<a id="frozen-rhs"></a>
### 静态 INT8 权重的存储顺序 / Frozen projection weight layout

同一份 `K×N` 权重可以有两种物理步长：按行存放为 `(N, 1)`，按列存放为 `(1, K)`。这不改变矩阵的形状、内容或运算。在 GB10 上实测，PyTorch 对这两种步长选用了不同的整数 GEMM kernel。`capture_export(..., int8_rhs_layout="column_major")` 只在建立图时分配并填充一份私有权重；`"preserve"` 做等量快照但保留原布局。两者均不修改调用方的权重。只接受直接读取、占满存储的静态 INT8 右矩阵；同存储但几何不兼容的视图、矩阵乘以外的读者和作为请求参数传入的权重不走这条路径。相同布局的共享权重只计一次。GB10 此路径保守限定为 `M > 16`、`K` 是 64 的倍数、`N` 是 8 的倍数；窄 `K` 在库的不同布局上支持不一致，并不默认补齐。

| 输入 → 输出通道（种子 987143） | 同一投影设备回放，行 → 列 µs | 常驻图，轮换请求同步延迟 µs | 常驻图，8 个请求排队，每请求 µs |
|---|---:|---:|---:|
| FP16 `33×1536 → 64` | 37.64 → 11.70 | 65.64 → 40.98 | 56.12 → 34.58 |
| BF16 `2×16×1536 → 128` | 39.01 → 12.39 | 72.25 → 48.78 | 58.28 → 47.29 |
| BF16 `65×3072 → 256` | 78.53 → 20.58 | 113.28 → 54.69 | 103.09 → 47.90 |

设备回放列用普通融合图测量；两列请求时间包含输入暂存、图回放、返回五个独立结果和等待 GPU，不再外套 CUDA Graph。三种形状的投影仍各发射三个 kernel。控制组用**本来就是列布局**的 FP16 权重，逻辑字节与表中首行完全相同；轮换请求为 48.87 → 48.89 µs，第二个种子为 48.72 → 48.61 µs，没有收益。两种策略的建图时间单列于报告，不是纯重排耗时，也不计入稳态请求。若原图不在准入范围内，默认接口不会自动改变权重布局。

The [paired report](../reports/frozen-rhs-gb10.json) includes two seeds, eight held-out requests per case, and every raw timing window. INT32 accumulators match an independent CPU integer multiplication. Both Inductor modes are shown in the accuracy gates but excluded from equivalent-output timings because some quantized codes differ. This is a frozen, single-shape W8A8 projection benchmark; it neither implements a new GEMM nor measures a complete trained model. After `source tools/env.sh`:

```python
from sparkfuse.resident import capture_export

with capture_export(ep, example, audit, backend="mixed",
                    partition_options={"host_backend": "native", "token_quant": True,
                                       "projection_epilogue": True},
                    int8_rhs_layout="column_major") as run:
    residual, codes, scales, accumulator, output = run(*request)
```

Run `python tools/quant_state_panel.py --output /tmp/frozen-rhs.json` and `bash tools/sanitize_quant_state.sh` for the matched panel and an unfiltered sanitizer replay. The fresh outputs and the frozen snapshot remain valid if the original module's weights are later overwritten.

<a id="grouped-epilogue"></a>
### 共享 token 尺度的多个投影 / Sibling projection epilogues

一个量化激活可以同时进入宽度不同的 Q、K、V 三路整数投影。`group_epilogues=True` 在导出图上匹配使用**同一个 FP32 行尺度**的 2–4 个独立反量化尾部。不同线程块处理不同分支，分别访问各自的 INT32 累加器和通道尺度；不转置、不拼接矩阵。每路的 FP32 转换、第一次乘法和最终乘积只有在图外可见时才写回，相互独立的输出不会改成共享视图。共享中间节点，或在所有输入就绪前就有外部消费者，不能合并。另开 `projection_epilogue=True` 可以继续处理剩下的单支。

| 三路投影，种子 1191169 | 只测三路尾部：分开 → 合并 µs | 含量化和三次 GEMM：分开 → 合并 µs |
|---|---:|---:|
| `M=17, K=1536, N=(128,64,64)` | 2.83 → 1.48 | 26.72 → 25.36 |
| `M=33, K=1536, N=(512,256,256)` | 3.35 → 1.75 | 28.75 → 27.40 |
| `M=129, K=3072, N=(1536,256,256)` | 4.41 → 3.08 | 57.34 → 55.24 |

两个种子的 profiler 均为：只看尾部 **3 → 1** 次发射，计入共享量化和三个原样的框架 GEMM 后是 **7 → 5**。表内是热设备回放，不是完整注意力层加速。九个可见输出与 eager 逐字节相同，三个整数累加器各用独立的 CPU 整数乘法复核。

The [matched experiment](../reports/grouped-epilogue-gb10.json) compares separate versus grouped tails and includes both Inductor modes subject to the same output gate. Both paths receive identical, already column-major weights. Inspect the [generated CUDA](../examples/generated/token-quant/group_17x128_64_64.cu) or reproduce with `python tools/group_epilogue_panel.py --output /tmp/grouped.json` and `bash tools/sanitize_group_epilogue.sh` after sourcing the environment. These are three random-weight projections, not a pretrained attention layer.

<a id="packed-projection"></a>
### 冻结的多路整数投影 / Packing frozen signed projections

对同一份逐 token INT8 编码，`capture_export(..., int8_rhs_layout="packed_fanout", partition_options={"token_quant": True, "group_epilogues": True, "host_backend": "native"})` 可把 2–4 份独立权重一次复制到列主序矩阵。请求路径只做一次 PyTorch 整数矩阵乘法。随后生成的 CUDA kernel 按列段取数，分别写出图中仍要使用的 INT32 累加器、FP32 转换值或缩放值。返回给调用方的累加器仍有独立的紧致存储。仅接受固定形状、冻结权重和同一份编码；共享权重或提前读取累加器的图会被拒绝。

三路对照的原始权重**已经是列主序**。两条路径都返回九个张量：残差、编码、逐 token 尺度，以及三对累加器和反量化结果。下表使用第一组随机种子，数值为七个打乱顺序的测量窗口的中位数，单位 µs。箭头左侧使用三个整数 GEMM 与一个共享尾部，右侧使用一次整数 GEMM 与分段尾部。

| M × K → 分支宽度 | 整段设备时间 | 轮换请求同步延迟（含拷贝） | 连续八请求，每请求 |
|---|---:|---:|---:|
| 17 × 1536 → 128/64/64 | 25.35 → 11.70 | 68.07 → 66.84 | 65.88 → 66.32 |
| 33 × 1536 → 512/256/256 | 27.58 → 13.07 | 70.38 → 67.21 | 66.41 → 66.26 |
| 129 × 3072 → 1536/256/256 | 53.74 → 30.22 | 117.08 → 92.87 | 113.73 → 91.61 |

The [six-case report](../reports/packed-projection-gb10.json) records the three widths and two independent seeds, including the small queued workload with no benefit. Profiled device launches fall from five to three: residual/token quantization, a framework integer GEMM, and the [generated segmented epilogue](../examples/generated/token-quant/packed_17x128_64_64.cu). Every case compared all nine outputs byte for byte across eight held-out requests and checked each signed accumulator against CPU INT32 multiplication. The two Inductor modes missed the byte-level gate and are excluded from same-precision timings. These results cover static, random-weight projections; they do not measure complete attention or model quality. Reproduce with `python tools/packed_projection_panel.py --output /tmp/packed.json` and `bash tools/sanitize_packed_projection.sh`.

<a id="resident-request-io"></a>
### 同一张图，两种请求提交路径 / Resident request I/O

`capture_export(ep, example, audit, request_backend="native", ...)` 把输入检查、复制到暂存区、CUDA Graph 回放和结果复制合在一次 C++ 调用里。它不改图里的 kernel。此选项要求返回单个独占张量，或由独占张量组成的一层 tuple；需要保留视图、重复引用或嵌套结构时，使用默认的 `request_backend="python"`。两种路径都允许下一次请求换地址，之前返回的结果仍属于调用方。`partition_options["host_backend"]` 控制图内算子的启动器，是另一个选项。

| GB10，八组轮换输入（种子 1481849） | eager 同步 µs | Python → C++ 同步 µs | Python → C++，每批 8 个请求、摊到每请求 µs |
|---|---:|---:|---:|
| 三路投影，17 行，九个输出 | 127.46 | 68.48 → 61.26 | 67.36 → 58.41 |
| 三路投影，129 行，九个输出 | 148.66 | 91.31 → 88.31 | 92.00 → 91.53 |
| 单输出 ATen 对照 | 13.78 | 28.91 → 23.33 | 26.51 → 20.43 |

The table measures whole requests, not device replay. The short projection also improved on a second seed (67.21 → 59.47 µs synchronized). For the 129-row case, queued cost moved from 91.45 to 92.08 µs on that seed; there is no consistent throughput gain there. Eager was fastest on the single-output control. Matched graph profiles have the same device launch counts; the change is in request handling. Both backends passed fresh-output, geometry and byte-level checks on eight held-out requests; integer accumulators were also checked against CPU INT32 matrix multiplication.

[Raw randomized windows, setup costs and checks](../reports/resident-io-gb10.json) · [Compute Sanitizer results](../reports/resident-io-sanitizer-gb10.txt). After `source tools/env.sh`, run `python tools/resident_io_panel.py --output /tmp/resident-io.json` and `bash tools/sanitize_resident_io.sh`.

<a id="packed-family"></a>
### 三种请求长度，共用一组打包权重 / Shared packed weights across fixed shapes

在 `capture_family(model, pairs, int8_rhs_layout="packed_fanout", ...)` 中，第一张图打包 Q/K/V 权重；后续图先核对每个分支的权重名、形状、步长、列偏移和逐元素 INT8 值，再绑定同一份私有存储，不重新打包。通道尺度也只保留一份。每种输入长度仍有自己的 CUDA Graph、暂存输入和静态内核。没有登记的长度会报错，不会在请求中重新编译。其他选项与单图打包相同；本实验两组都用 `request_backend="native"`。

| DGX Spark，17 / 33 / 129 行，K=1536，Q/K/V 宽 512/256/256 | 三张独立图 | 共享权重的三张图 |
|---|---:|---:|
| 私有打包矩阵：份数 / 字节 | 3 / 4,718,592 | 1 / 1,572,864 |
| 模型状态 + 三套输入暂存（字节） | 5,830,656 | 2,676,736 |
| 建图后 PyTorch 分配器净增（字节） | 8,126,976 | 4,973,056 |
| 轮换形状同步请求，两个种子（µs） | 49.10 / 50.79 | 48.83 / 48.96 |
| 8 个请求排队，每请求，两个种子（µs） | 47.07 / 46.90 | 45.68 / 47.56 |

Each allocator measurement used a separate, symmetrically warmed process; after closing the graphs, allocated bytes returned to baseline. Sharing saved **3,153,920 bytes** of retained allocations; it did not consistently improve queued throughput. Checking the shared state also took extra setup time (111 vs 102 ms; 109 vs 97 ms). Both serving paths use identical per-shape CUDA code and the same request router. Eight interleaved held-out requests per seed passed byte-level checks on all nine outputs, including independent CPU INT32 accumulator checks. Eager synchronized medians were 102.56 and 102.10 µs. Default and cast-aware Inductor failed the byte gate and were not timed as equivalent implementations.

[Physical inventories, isolated allocator readings and raw timings](../reports/packed-family-gb10.json) · [Compute Sanitizer](../reports/packed-family-sanitizer-gb10.txt). Reproduce with `python tools/packed_family_panel.py --output /tmp/packed-family.json` and `bash tools/sanitize_packed_family.sh` after `source tools/env.sh`.

<a id="affine-projection"></a>
### 带偏置的窄精度投影 / Ordered bias and narrow outputs

常见的 Q/K/V 投影不返回 FP32 反量化结果，而是加上每通道偏置，再转为 BF16 或 FP16。`partition_options={"token_quant": True, "affine_epilogues": True, "group_epilogues": True, "host_backend": "native"}` 识别这个顺序。生成的 CUDA 先将 INT32 累加器转为 FP32，依次乘 token 尺度、乘通道尺度、加偏置，最后做一次窄精度转换。乘法与加法分别舍入，不合为 FMA；导出图额外返回哪个中间结果，就为它单独写回。只识别精确匹配的加法和显式转换，不改写带不同 `alpha`、不同广播轴或未知断言的路径。

| 三路带偏置投影（种子 1661987） | 同一导出图设备时间：原有融合 → 合并尾部 → 打包 GEMM（µs） | 轮换请求同步延迟：原有融合 → 打包（µs） |
|---|---:|---:|
| 17 行，3 个 BF16 输出 | 28.77 → 24.68 → 10.86 | 54.36 → 34.91 |
| 33 行，21 个输出，保留中间值 | 32.87 → 28.09 → 13.45 | 115.88 → 116.09 |
| 129 行，3 个 BF16 输出 | 53.42 → 48.63 → 22.29 | 80.46 → 52.07 |

On GB10 the ordinary graph launches **28 / 10 / 5 / 3** kernels (eager / previous fusion / grouped affine / packed affine). All three resident paths start with column-major weights, return fresh outputs, and use the same native request interface. The packed path additionally requires `int8_rhs_layout="packed_fanout"`. The 21-output diagnostic case still paid to return 21 separate tensors: faster device replay did **not** reduce its request latency. Across six cases, two independent seeds and eight held-out requests per case, returned results matched eager byte for byte and the tested narrow outputs matched an oracle built from CPU INT32 accumulators. Default and cast-aware Inductor did not pass the same byte gate. These are random-weight projections, not a complete attention layer.

[Generated three-branch CUDA](../examples/generated/token-quant/affine_17x128_64_64.cu) · [Per-output errors, profiler counts and timing windows](../reports/projection-affine-gb10.json) · [Compute Sanitizer](../reports/projection-affine-sanitizer-gb10.txt). Run `python tools/projection_affine_panel.py --output /tmp/affine.json` and `bash tools/sanitize_projection_affine.sh` after `source tools/env.sh`.

<a id="output-slots"></a>
### 服务方预分配输出 / Caller-owned output slots

`run(inputs...)` 的每次结果都有独立存储。需要排队处理请求时，服务方也可以先分配八组结果缓冲区，再用 `run.run_into(inputs, slot)` 写入其中一组。它返回传入的 `slot` 本身；再次使用同一组会覆盖前一次结果。跨 CUDA stream 使用时，调用方须安排好等待，特别是写入前要等上一次消费者读完。此接口只用于原生请求路径已验证的单个张量或由独立结果组成的扁平元组。写入前检查所有输入、输出的布局和地址范围，拒绝重叠、模型权重、采样用输入和内部图缓冲区作为目的地。

| 相同的常驻打包投影，种子 1661987 | 轮换请求同步时间：独占新结果 → 预分配（µs） | 8 请求排队，平均每请求（µs） | 8 组预分配的 CUDA allocator 字节数 |
|---|---:|---:|---:|
| 17 行，3 个结果 | 34.90 → 34.71 | 29.55 → 26.80 | 77,824 |
| 33 行，21 个结果 | 115.28 → 88.70 | 112.76 → 85.30 | 7,168,000 |
| 129 行，3 个结果 | 49.17 → 54.37 | 44.96 → 44.64 | 4,227,072 |

The same graph runs three compute kernels in both modes. At 21 outputs, the per-request trace still shows 23 device copies (two inputs plus 21 results); the new path removes the 21 clones/allocations. In a queue of eight this is **168 → 0 hot CUDA allocations**, after paying for the ring up front. At the second seed, rotating latency was 116.03 → 88.72 µs. The longer three-output case was slower for a synchronous request at both seeds. The caller controls buffer reuse; this does not change the default fresh-result API. Seven shuffled windows include the complete request; the retained outputs of all eight requests match the archived arithmetic oracle byte for byte.

[Allocator counters, copy traces and raw clocks](../reports/output-slot-gb10.json) · [Compute Sanitizer](../reports/output-slot-sanitizer-gb10.txt). Reproduce after `source tools/env.sh` with `python tools/output_slot_panel.py --output /tmp/slots.json` and `bash tools/sanitize_output_slots.sh`. The plain-process test separately checks backing storage expires when the caller drops it; the instrumentation can retain Python arguments. Stream ordering and destination version increments are tested under all three sanitizer tools.

<a id="bert-w8a8"></a>
### 同一份输入，三条量化投影 / W8A8 Q/K/V inside a BERT layer

实验使用 Hugging Face 的 `BertLayer`（768 维、12 个头、FFN 3072 维）。只把 Q/K/V 三个线性层换成独立的 W8A8 模块：权重按输出通道量化，激活按 token 量化。导出图里三个模块分别读取同一个激活。若编码和尺度不对外可见，编译器可以只计算一份，供三个整数 GEMM 使用；冻结权重后还可把三个 GEMM 打包成一个。注意力、FFN 和 LayerNorm 仍按原图执行。

| 长度，第一组权重 | 整层 CUDA 发射：量化 eager / 独立融合 / 共享量化 / 再打包 | 热图设备时间：独立 → 共享 → 再打包（µs） | 轮换输入、同步完整请求：独立 → 共享 → 再打包（µs） |
|---|---|---|---|
| 17 | 58 / 22 / 18 / 16 | 79.97 → 75.88 → 67.01 | 101.66 → 97.54 → 120.64 |
| 33，含外部残差 | 59 / 22 / 18 / 16 | 81.43 → 76.56 → 67.85 | 104.80 → 100.53 → 91.46 |
| 129 | 58 / 22 / 18 / 16 | 109.48 → 105.06 → 102.51 | 131.50 → 128.99 → 122.22 |

17 个 token 时，第一次打包让同步请求变慢；换一组权重和请求后，打包的同步延迟从 97.10 降到 87.45 µs，热图设备时间却从 76.03 增到 99.06 µs。不能用发射次数代替请求耗时。33 和 129 两个长度的打包路径在两组权重下都改善了这两项指标。

`activation_quant=True` recognizes read-only activation quantization; `share_activation_quant=True` additionally shares hidden quantizers of eligible sibling projections. Observable intermediate codes or scales, unrelated inputs and unknown intervening operations prevent sharing. The panel uses `min_ops=100000` to keep unrelated generic arithmetic in PyTorch. The optional `packed_fanout` weight layout is selected at capture.

Across two random-weight seeds, three lengths and eight held-out requests per case, both the final hidden states and attention probabilities matched **quantized eager** byte for byte. Each Q/K/V narrow result was checked using independent CPU INT32 accumulation. Quantization itself changes the original BF16 model: case-wise maximum hidden-state differences ranged from 0.03125 to 0.0390625. Both default and cast-aware Inductor missed the byte gate and are excluded from equivalent-precision timings. Batch size was one, dropout was off, and no mask or pretrained weights were used.

[Raw timing windows and per-output checks](../reports/bert-w8a8-gb10.json) · [Compute Sanitizer](../reports/bert-w8a8-sanitizer-gb10.txt). To reproduce, source `tools/env.sh`, then run `python tools/bert_w8a8_panel.py --output /tmp/bert-w8a8.json` in a CUDA PyTorch environment with Transformers 4.51.3; run `bash tools/sanitize_shared_quant.sh` for the focused kernel checks.

<a id="fused-slot-copy"></a>
### 一次写回 21 个结果 / One launch for caller-owned outputs

同一个带中间结果的投影已经只用 3 个计算 kernel，诊断接口却要返回 21 个张量。使用 `capture_export(..., request_backend="native", slot_copy_backend="cuda")`，编译器根据每个结果的字节数分配 4 KiB 的块，并生成一个[专用拷贝 kernel](../examples/generated/slot-copy/diagnostic_21.cu)。`run_into` 在每次请求中只传实际地址：能对齐时按 16 字节搬运，其余地址和尾部按字节处理。结果仍是服务方提供的各个张量。普通 `run(*inputs)` 继续返回独占结果，默认配置也继续使用 ATen 拷贝。

| 预分配同一组缓冲区，默认 stream，第一组种子 | 完整同步请求：逐张量拷贝 → 一个 CUDA kernel（µs） | 8 请求排队，摊到每请求（µs） |
|---|---:|---:|
| 17 行，3 个结果 | 34.69 → 31.60 | 27.05 → 21.82 |
| 33 行，21 个结果 | 89.21 → 36.55 | 85.96 → 24.63 |
| 129 行，3 个结果 | 54.45 → 50.65 | 44.35 → 39.74 |

在第二组种子上，21 个结果的同步延迟为 89.49 → 36.61 µs。在显式 stream 上用同一组输入比较，直接在原生代码中发起 21 个 `cudaMemcpyAsync`、或合为一次 `cudaMemcpyBatchAsync` 调用，同步延迟分别为 76.13 和 75.63 µs，生成的 kernel 为 36.48 µs。profiler 解释了区别：批处理减少了 API 调用，但设备端仍记录 23 次拷贝（含 2 个输入）；生成路径只留下 2 次输入拷贝，另用一个 kernel 写回所有结果。要搬运的字节并没有变少。

The API accepts up to 32 independent, full-storage output tensors. It checks all destinations before staging, records allocator stream use and increments regular tensors' version counters. Both transports made zero new CUDA allocations across eight queued requests after the ring was created. The two seeds, three projection shapes and two stream types matched the archived quantized eager outputs byte for byte. The [timing report](../reports/fused-slot-gb10.json) includes all seven shuffled windows per case, setup costs, and direct CUDA controls. In a single-output ReLU control with a roughly 16 MiB result, queued latency moved from 397.53 to 399.01 µs on one seed, and from 405.27 to 388.34 µs on the other. These results support opting in for multi-output workloads; they do not establish a universal improvement for a single large copy.

After `source tools/env.sh`, run `python tools/fused_slot_panel.py --output /tmp/fused-slot.json`. The explicit-stream driver control uses CUDA 13's batch API; `--smoke` checks the two product paths without that control. Run `bash tools/sanitize_fused_slots.sh` for the [sanitizer checks](../reports/fused-slot-sanitizer-gb10.txt), including unaligned pointers, protected bytes around the destinations, tile boundaries and the 32-output limit.

<a id="whole-request-policy"></a>
## 应该按哪个请求目标编译？ / Selecting a lowering for a complete request

`select_request_policy` 面向固定权重、固定形状的 W8A8 服务。三个候选分别独立计算每个投影的量化、让各投影共享一次量化、或在共享后把 Q/K/V 打包成一次整数 GEMM。计时从传入新地址的张量开始，到原生入口写回独占结果并同步结束。可选择每次等待（`latency`），或连续提交八次再等待（`queue8`，报告每请求微秒）。这与测一个 CUDA kernel 的耗时不同。

| BERT，33 个 token，包含残差加法 | 权重种子 | 独立量化 → 选中的打包方案，独立留出集（µs/请求） |
|---|---:|---:|
| 同步一次 | 1840921 | 99.93 → 86.01 |
| 八次排队 | 1840921 | 90.85 → 77.75 |
| 同步一次 | 1851289 | 100.29 → 85.53 |
| 八次排队 | 1851289 | 92.86 → 78.55 |

12 组正式测量都选中了打包，但没有把它写成默认方案。每组先用 5 个窗口提名；另一批输入上 7 个窗口至少赢 6 次，且配对中位加速达到 1.03× 才可换掉基线。第三组 8 个输入只验收、不参与决策。每一组都逐字节核对 hidden state 和 attention probabilities，并检查输出不共享存储。相同图但权重、形状、设备、精度设置或优化目标发生变化，不能使用旧记录。报告里保留了所有候选的样本。

```python
from sparkfuse.serving_policy import select_request_policy, replay_request_policy

# Every request is a tuple of tensors with the same static descriptor, different storage and content.
exe, receipt = select_request_policy(ep, example, check, train8, confirm8, audit8,
                                     objective="queue8", receipt_path="choice.json")
with exe:
    result = exe(*live_request)
with replay_request_policy(ep, new_example, new_check, "choice.json", objective="queue8") as cached:
    result = cached(*live_request)
```

The [12-case report](../reports/request-policy-gb10.json) contains the timings, actual rewrites, selected arms, paired votes, held-out results and setup cost. On this panel selection took 220–603 ms; winner-only recapture took 62–160 ms. Using the observed holdout savings, the extra tuning cost over building the independent arm breaks even at roughly 13k–85k requests. These are estimates for a warm process, not a latency guarantee. Reusing a receipt still captures a new graph, validates two previously unseen inputs, and needs the same static context. Receipts are checksummed local records, not an authentication mechanism; pass only trusted files. Run `python tools/request_policy_panel.py --output /tmp/policy.json` in the Transformers environment and `bash tools/sanitize_request_policy.sh` for the regression checks.

<a id="fresh-transport"></a>
### 返回 21 个独占结果，而不要求调用方预分配 / Fresh outputs in one copy launch

`capture_export(..., request_backend="native", fresh_copy_backend="cuda")` 为每个结果单独分配张量，再用一次 CUDA 发射从常驻图的私有缓冲区写回。返回值保留设备、dtype、形状、步长以及独立存储；下一次请求或关闭执行句柄都不会影响已返回的结果。该选项仅用于 1–32 个独占、连续的整存储输出。已有的 `slot_copy_backend` 只控制 `run_into`；两个选项互不改变对方的默认行为。

| 33 行、21 个可见输出，默认 stream | 逐个 ATen clone | 原生分配 + 一个 CUDA kernel | Python 分配 + CUDA `run_into` |
|---|---:|---:|---:|
| 单请求，种子 1661987（µs） | 120.05 | 63.98 | 71.90 |
| 八请求排队，每请求（µs） | 117.77 | 49.78 | 59.54 |
| 单请求，种子 1674131（µs） | 119.53 | 64.57 | 71.94 |
| 八请求排队，每请求（µs） | 117.23 | 49.99 | 60.11 |

三条路径都为八次请求分配了 **168** 个 CUDA 张量，保留了同样的 7,168,000 字节。收益来自减少逐张量拷贝发射，不能声称减少分配。profiler 中，ATen 的 23 次设备拷贝变为 2 次输入拷贝和 1 个输出 kernel。三个大输出的 prefill 例不总是受益：在显式 stream 上，同步延迟 50.98 → 53.37 µs，排队时则是 43.86 → 41.02 µs（种子 1674131）。需要最低分配开销的调用方仍可使用前述预分配接口。

All three arms return fresh, independently owned tensors. The [complete panel](../reports/fresh-transport-gb10.json) covers two seeds, three projection graphs, default and explicit streams, and six single-output controls. The projections reuse the archived CPU INT32 oracle's matching inputs and outputs; all public tensors are checked before and after the handles close. Run `python tools/fresh_transport_panel.py --output /tmp/fresh.json` and `bash tools/sanitize_fresh_transport.sh` to reproduce the [memory and synchronization checks](../reports/fresh-transport-sanitizer-gb10.txt).

<a id="stacked-bert"></a>
### 四层 BERT 的 Q/K/V：按层打包，在一次请求中执行 / Packed groups across encoder layers

以前 `packed_fanout` 只接受一组投影。新的 `int8_rhs_layout="packed_groups"` 从同一个静态导出图中识别 2–8 组投影；每组 2–4 个独立、冻结的有符号整数矩阵乘必须读取同一份量化激活。按 FX 拓扑顺序为每组生成一个紧凑权重、一次库 GEMM 和一个 CUDA 后处理 kernel。不跨层拼接权重，输出也不是大张量上的切片。编译器先检查所有组是否完整覆盖整数 GEMM、权重是否不共享存储、是否存在过早读取累加值的使用者，再开始打包。单组接口和已有的形状族接口不变。

DGX Spark 上用两组不同的随机权重测试真实的 Hugging Face `BertLayer` 堆叠。输入为 BF16，batch 1；仅 Q/K/V 用 W8A8。四层、33 个 token 时，量化 eager 的一次普通前向有 **233** 次 kernel 发射；独立量化为 **88** 次，共享每层输入量化为 **72** 次，再把每层三次 GEMM 合为一次后为 **64** 次。四层的隐藏状态及注意力概率全部对外可见；8 个留出请求的 8 个输出逐字节匹配量化 eager。

| 四层，33 个 token；轮换请求，同步延迟（µs） | 量化 eager | 层内共享量化 | 逐层单独打包并调用 | 整图分组打包 |
|---|---:|---:|---:|---:|
| 权重种子 1981121 | 1749.32 | 459.51 | 451.50 | 431.24 |
| 权重种子 1993243 | 1755.40 | 494.20 | 488.02 | 462.44 |

逐层调用也有胜例：两层、17 个 token、第一组权重的同步延迟为 261.90 µs，整图为 264.76 µs。两种 Inductor 配置都运行了，但没有在这些用例的所有可见输出上达到逐字节一致，未列入计时。这里不据此声称快于 Inductor。库不提供相应矩阵尺寸的算法时，整图分组模式会明确拒绝，不会把 OOM 或其他执行错误当成可回退的能力限制。

The [full panel](../reports/stacked-bert-gb10.json) records all eight cases, both Inductor gates, two CPU INT32 Q/K/V audits per layer, weight and output digests, setup cost, launch traces and seven shuffled timing windows. Layerwise and whole-graph paths expose exactly the same intermediate hidden states and probabilities. The control captures the residual in the first layer, rather than charging it an extra host operation. Only the fixed-shape single-export path supports multi-group packing; this is not an end-to-end BERT task or accuracy benchmark. Reproduce with `python tools/stacked_bert_panel.py --output /tmp/stacked.json` in the Transformers environment and `bash tools/sanitize_packed_groups.sh`.

<a id="decode-sdpa"></a>
### 单 token 解码：一次读取一段 KV / One-token SDPA

这条可选路径只接收**整张图为一个 ATen SDPA 调用**的静态导出。每个 CUDA block 为一个 query head 计算一段 KV：warp 归约 Q·K，在共享内存中求最大值并计算相对权重，同时累积 V。跨段时，第二个 kernel 用各段的最大值重新缩放分子与分母，合并结果。它不分配完整注意力矩阵，也不把 KV 头复制到 query 头数。单段直接写回；多段的显式 FP32 工作区为 `4 × B × H × ceil(S/tile) × (D+2)` 字节。

设备重放中位数（µs），DGX Spark，同一台机器、首个种子，固定 `tile=32`，未按每行挑最快参数：

| 单步输入 | eager | Inductor | SparkFuse | kernel 次数（同序） |
|---|---:|---:|---:|---|
| B1，16/4 头，S=257，D=64，FP16 | 7.61 | 7.61 | 5.39 | 2 / 2 / 2 |
| B2，12/3 头，S=1025，D=128，BF16，逐批 FP32 偏置 | 357.38 | 119.39 | 12.39 | 17 / 8 / 2 |
| B1，16/4 头，S=8193，D=64，BF16，无掩码 | 19.73 | 19.73 | 35.60 | 2 / 2 / 2 |

第二行包含一个完全被遮蔽的批次。该配置默认 eager 有 17 次发射，不能把提速倍数外推到已经使用高效 fused attention 的配置。最后一行所有自定义 tile 的设备时间都慢于 eager。两套种子、七种形状；每套另留三份输入与**逐 KV 头的 CPU FP64** 参考比对。结果有误差容限，不承诺与 eager 逐位相同。实际输出的 stride（包括长度为 1 的轴）在原导出上独立核验，所有请求返回独占张量。

```python
from sparkfuse.decode_attention import compile_decode_attention

with torch.inference_mode():
    ep = torch.export.export(model, examples)
    decode = compile_decode_attention(ep, examples, independent_audit, tile=32)
    y = decode(*request)               # 每次扫描数值范围并同步
    y = decode.run_bounded(*request)   # 仅当上游已保证数值合同
```

The fast entry requires finite `|Q,K,V| ≤ 64` and either finite `|bias| ≤ 10000` or `-inf`. The checked entry scans these limits every request. For the masked S=1025 case above, the separate synchronized host comparison was **141.17 µs checked vs 23.63 µs bounded**. Both entries guard the qualified backend configuration, descriptors and alignment. Supported inputs: one query token, D=64/128, dense FP16/BF16, regular or grouped query heads, optional FP32 per-batch or broadcast GQA bias, dropout zero, noncausal. No prefill, mutable KV cache, arbitrary mask, dynamic shape, or general-graph replacement is implied. All timings use newly allocated outputs; GPU-graph timings reuse captured addresses. Clocks exclude export, qualification and compilation.

[All gates, kernel names, cubin resources and shuffled timing samples](../reports/decode-sdpa-gb10.json) · [sanitizers](../reports/decode-sdpa-sanitizer-gb10.txt) · [exportable example](../examples/single_decode.py). Reproduce with `python tools/decode_sdpa_panel.py --output /tmp/decode-gb10.json` after sourcing `tools/env.sh`.

<a id="grouped-decode"></a>
### 多个查询头复用 K/V：热地址与独立请求 / Grouped decode

`compile_decode_attention(..., head_group=R)` 可让一个线程块内的 1、2 或 4 个相邻查询头共用 K/V 读取；它们仍独立计算点积、softmax 和加权和。组不能跨 KV 头；第二阶段归并不变。五种形状、两套种子；每种候选另留八个输入检查 CPU FP64 参考。在同一 tile 下，R=2/4 的 FP32 分块中间量和最终窄类型输出都与 R=1 **逐位相同**。默认仍为 R=1。

以下为首套种子的设备重放中位数（µs）。表中每对数字都是 R=1 → 4，tile 不变。地址实验另行捕获，九个随机顺序窗口，对每个请求保留并检查独占的新输出。

| 输入 / tile | 重放同一 QKV | 每个请求独立 QKV | 独立 QKV：Inductor |
|---|---:|---:|---:|
| B2，12/3 头，S=1025，D128，BF16，带偏置 / 32 | 11.98 → 9.57 | 21.16 → 19.08 | 130.00 |
| B1，32/8 头，S=2049，D128，BF16 / 32 | 23.38 → 17.55 | **42.19 → 45.48** | 43.01 |
| B1，32/4 头，S=16385，D128，FP16 / 128 | 219.40 → 159.16 | 227.93 → 172.99 | **152.58** |

The middle row reverses direction when KV storage rotates. In the separate synchronized, rotating-input host test it also regressed (53.91 → 58.06 µs). In the S=257 hot test R=4 lost even without rotating (5.39 → 6.92 µs). Thus a hot-only winner should not become a universal default. The ring panel distinguishes same Q/K/V, independent Q with shared KV, and independent Q/K/V. The latter has unique KV storage totalling at least three times the reported L2 capacity; **this is allocated address footprint, not measured DRAM traffic or a guarantee of cache misses**. CUDA Graph numbers exclude graph construction, qualification and value scans.

[Raw timings, address footprints and FP64 gates](../reports/grouped-decode-gb10.json) · [Compute Sanitizer](../reports/grouped-decode-sanitizer-gb10.txt). Reproduce: `python tools/grouped_decode_panel.py --output /tmp/grouped-decode.json`.

<a id="qwen-sdpa-sites"></a>
### 实际解码层：在保留公开缓存的前提下消除 KV 头重复 / SDPA inside Qwen2

`partition_decode_sites` 在导出图中识别单 token SDPA，以原始 ATen 调用的**实际输出与步幅**验证替代 kernel。遇到 `unsqueeze → expand → reshape` 的 K/V 头重复，可让 CUDA 直接读取较少的物理头，不再为注意力物化两份重复数据。其他消费者仍能看到需要的重复张量；两步解码返回的完整 K/V 缓存不删、不覆写。支持 BF16/FP16 同类型共享头偏置，包括全遮蔽行。只接受已识别的无副作用静态图，不改传入的导出图；无法降低的注意力留在 ATen，未授权的周边算子在执行前拒绝。`domain="checked"` 默认检查范围；以下 CUDA 性能使用显式 `domain="bounded"`，前提是上游已经保证数值范围。

随机权重 **Hugging Face Qwen2DecoderLayer**（B=2、隐藏宽度 512、8/2 查询/KV 头、D=64）。历史缓存来自真正的预填；两组模型权重，每组每种形状八条独立轨迹，各后端把自己第一步的缓存传入第二步。三条 CUDA 路径的完整 K/V 均与 stateful Transformers 参考逐字节相同，隐藏输出满足逐元素 `atol=.012, rtol=.02`，不要求逐位相同。另用 CPU FP64 在算子边界核验独立请求。

| 第一步 KV 长度（首套权重） | HF eager | 原导出 ATen | CUDA，保留重复 | CUDA，消除重复 | Inductor |
|---|---:|---:|---:|---:|---:|
| 65，BF16 偏置 | 74.92 | 74.74 | 71.09 | **67.51** | 38.48 |
| 257，无偏置 | 75.78 | 75.78 | 73.05 | **68.61** | 39.70 |
| 1025，BF16 偏置 | 129.72 | 129.71 | 87.38 | **75.61** | 84.67 |

All entries are **whole-layer CUDA Graph device replay µs**, not throughput. Both CUDA ablations in the table fix tile=32 and one query head per thread block. At 1025 tokens removal eliminates two materializations totalling **4,198,400 logical bytes**, taking launches from 43 to 41; this is not a DRAM byte counter. The separately measured rotating, synchronized request still favors Inductor: HF eager **317.60**, original export **397.67**, retained-repeat CUDA **392.39**, compact CUDA **365.44**, Inductor **192.34 µs**. Inductor passed the numerical, prefix and fresh-storage checks in all six panels, but its returned full caches were not byte-identical. Its times are provided as a numerical baseline, not a byte-exact substitute. Grouping four query heads was slower than one at each tested layer length. These tests do not measure pretrained model quality or an entire autoregressive model.

[Executable real-model boundary](../examples/qwen_sdpa_step.py) · [raw gates, held-out hashes, kernel traces and timing windows](../reports/qwen-sdpa-sites-gb10.json) · [sanitizers](../reports/qwen-sdpa-sites-sanitizer-gb10.txt). Reproduce: `python tools/qwen_sdpa_sites.py --output /tmp/qwen-sdpa-sites.json` with the optional Transformers environment.

<a id="segmented-cache"></a>
### 两步解码：把缓存追加并入注意力是否更快？ / Cache append inside attention

`partition_segmented_decode` 让注意力直接从历史缓存和新 token 的四段 K/V 读取。调用者需要完整新缓存时，每个物理 KV 头由指定查询头的线程块写出一次；若缓存仅用于注意力，则不分配公开副本。存在必须提前读取拼接结果的依赖时保留原图。下面使用真实 Qwen2 层、两步各自传递缓存，并同时保留**两步的六个输出**，不是单个 kernel 的计时。首步历史来自预填（token-major），第二步读第一步返回的 head-major 缓存。

| 初始历史（首套随机权重） | HF eager | 导出 ATen | 两次 cat + 注意力 | 成对追加 + 注意力 | 追加并入注意力 | Inductor |
|---|---:|---:|---:|---:|---:|---:|
| 64，带偏置 | 147.46 | 147.30 | 133.12 | 129.75 | 138.62 | **76.56** |
| 256，无偏置 | 148.11 | 147.98 | 134.27 | 130.14 | 138.42 | **78.51** |
| 1024，带偏置 | 252.41 | 252.75 | 143.89 | **135.75** | 142.29 | 170.01 |

Time is CUDA Graph device replay in µs **for both steps together**, without qualification or capture. At history 1024 the six paths launch 88/88/82/80/78/36 kernels. Fusing removed launches but lost to paired append in **all six** seed/length cells. On rotating inputs, synchronized request medians at 1024 were 625.30/782.39/729.83/719.68/707.98/363.36 µs. The fused path does not give a material peak-allocation saving when both new caches must remain visible. These are different clocks and different storage lifetimes; launch count alone does not select a winner.

Eight held-out trajectories per cell, two independently initialized layers. The three CUDA paths agree byte-for-byte on all six final tensors, and checks at both attention sites cover FP32 partials and a CPU FP64 reference. Inductor meets the numerical gates but does not produce byte-identical full caches. [Raw results, allocation peaks and timing windows](../reports/segmented-qwen-gb10.json) · [sanitizer transcripts](../reports/segmented-qwen-sanitizer-gb10.txt). Reproduce: `python tools/segmented_qwen_panel.py --output /tmp/segmented-qwen.json` in the optional Transformers environment.

<a id="symbolic-kv"></a>
### 一个导出图适配不同的 KV 长度 / A bounded symbolic KV axis

`compile_length_decode` 直接读取 `torch.export.Dim` 的上下界。Q 固定为一个 token；K、V 和可选同精度 additive mask 共用长度符号。导出图必须显式把输出 `clone(memory_format=torch.contiguous_format)`，以约定新存储及单例维度的步幅。编译器将长度传给同一份 CUDA 代码：长度 ≤ tile 时只发射一次，较长时才分配当前分块所需的 FP32 中间量并归并。数值、推理模式、设备、布局和界限均有检查；`run_bounded` 仅省略由调用者保证的数值范围检查。

验证用长度 17、65；**不重新编译**运行 2、31、32、33、129、513、1025、2049，每个长度另取三个输入。两种几何、两套种子；每个配置只用 **2 个机器码对象**，同 tile 的逐长度静态 CUDA 用到 13 个不同内核缓存键。每个留出输出与静态 CUDA 逐字节一致，且另行通过 CPU FP64 对照。B2/8 个查询头、D64，S=32/33/2049 的中间量分别是 **0/8,448/274,560 字节**，不是按最大长度预分配。

| 带 BF16 偏置，首套种子 | eager | SparkFuse | Inductor dynamic | Inductor static |
|---|---:|---:|---:|---:|
| S=33，设备重放 µs | 23.99 | **6.25** | 31.50 | 8.98 |
| S=1025，设备重放 µs | 93.21 | **8.12** | 199.34 | 30.99 |
| 八种长度轮换，主机同步请求 µs | 108.70 | **31.86** | 91.78 | 76.50 |

The two Inductor settings invoked their compiler backend 3 and 8 times on the masked workload (not counts of CUDA binaries). The result does **not** transfer unchanged to unmasked FP16: at S=2049 eager beat SparkFuse on device, **14.61 vs 15.10 µs**; with rotating lengths it also won synchronized latency, **27.42 vs 30.18 µs**. Both Inductor modes produced a different singleton stride in that family, so we excluded them from contract-equivalent timing rather than assigning a speedup. First-seen request times may include persistent-cache hits; they are not clean-build latency. This is an isolated SDPA region, not a dynamic full Qwen layer.

[Example and export signature](../examples/length_decode.py) · [raw shape traces, CPU comparisons, binary hashes and clocks](../reports/symbolic-kv-gb10.json) · [sanitizers](../reports/symbolic-kv-sanitizer-gb10.txt). Reproduce: `python tools/length_decode_panel.py --output /tmp/symbolic-kv.json`.

<a id="length-host"></a>
### 相同的动态 CUDA，更短的主机调用路径 / Same kernels, different host dispatch

`compile_length_decode(..., host_backend="native")` 将形状、步幅、地址校验，新输出及临时分配，以及一到两次发射放入一次 ATen C++ 调用。它仍检查当前推理模式与 SDPA 策略，沿调用方的 CUDA stream 执行，并为跨流输入登记存储生命周期。默认 `ctypes` 仍可使用；两个入口加载**逐字节相同的 CUDA 二进制**。四组独立实验中，每组八种长度 × 三个留出请求，输出的结构、逐字节摘要和 FP64 误差均一致。

| 八种长度轮换，首套种子，µs | eager | ctypes | native | dynamic Inductor |
|---|---:|---:|---:|---:|
| BF16 带偏置，同步单请求 | 82.85 | 29.26 | **15.42** | 84.64 |
| BF16 带偏置，八请求排队后每请求 | 99.52 | 20.62 | **9.69** | 138.62 |
| FP16 无偏置，同步单请求 | 27.25 | 30.28 | **17.06** | 步幅不符，未计时 |

The reduced request time is a host dispatch result, not a faster attention kernel. At S=2049 without a mask, eager was still faster on the device: **14.44 vs 15.10 µs** for native. The fast entry assumes a bounded numerical domain. With actual per-request value scans enabled, the masked S=2049 native call took **142.26 µs**, versus **23.47 µs** without scans; neither cost is hidden inside the other timer. The installed-wheel smoke exercises this host entry at four lengths, including the one-to-two launch transition.

[Raw results, held-out hashes, checked calls and timing windows](../reports/length-host-gb10.json) · [stream lifetime and malformed-descriptor tests](../tests/test_length_host.py) · [sanitizers](../reports/length-host-sanitizer-gb10.txt). Reproduce: `python tools/length_host_panel.py --output /tmp/length-host.json`.

<a id="symbolic-qwen"></a>
### 一张真实 Qwen2 导出图，连续解码四步 / Four decode steps from one symbolic layer

对随机权重的 Hugging Face `Qwen2DecoderLayer` 做 strict export：过去的 K/V 共用一个符号，mask 的最后一维是它加一。每个 mask 配置仅导出一张图，两个 CUDA 二进制供全部长度和步骤使用。仅当 SDPA 的布局无法越过 `transpose → reshape → linear` 暴露，才旁路两段 KV 头复制。残留读者、公开缓存和输入检查不删除。第一步接受真实 prefill 产生的 token-major 缓存，后续步骤消费各自上一步返回的 head-major 缓存；没有为匹配步幅而添加拷贝。

两套权重种子 × 有/无 mask × 八个初始长度（2–2045）× 四步，共 **128 个留出步骤**。每步用 CPU FP64 独立验证注意力；每个后端独立传递自己的缓存，保活一条请求的全部 12 个返回值。两种 SparkFuse host 路径每步的公开 K/V 与 HF 逐字节一致；隐藏状态按列出的容差比较。Inductor 每配置触发三次 backend 编译，通过数值、步幅和旧缓存前缀检查，但并非所有新 K/V 字节都和 HF 一致，下面只将它作为数值等价基线。

| 有 mask，首套种子，**四步合计** µs | HF eager | 导出 ATen | SparkFuse native | dynamic Inductor |
|---|---:|---:|---:|---:|
| 过去 2 token，设备重放 | 279.82 | 279.62 | 250.73 | **144.05** |
| 过去 2045 token，设备重放 | 822.64 | 815.57 | **327.68** | 679.47 |
| 八种初始长度轮换，主机同步 | 1266.43 | 1600.49 | 1456.79 | **722.78** |

Only the attention subgraph is replaced. The long-context device win did **not** make ordinary rotating requests faster than HF eager or Inductor. Four dependent decode steps, the other layer operators, Python submission, allocation and synchronization are inside the request timer. Not a pretrained-model throughput or accuracy claim.

[Strict export and benchmark](../tools/symbolic_qwen_panel.py) · [raw steps, hashes, FP64 checks and timing windows](../reports/symbolic-qwen-gb10.json) · [sanitizers](../reports/symbolic-qwen-sanitizer-gb10.txt). Reproduce with the optional Transformers environment: `python tools/symbolic_qwen_panel.py --output /tmp/symbolic-qwen.json`.

<a id="hybrid-qwen"></a>
### 动态注意力与静态融合区一起执行 / Static islands in a dynamic layer

KV 长度是符号，单 token 的隐藏维度却是静态的。`partition_hybrid_decode` 只为形状和步幅都不依赖符号的算子块生成 CUDA；在两次独立请求中，逐个核对每个离开算子块的张量的字节、步幅和存储关系。跨不过下游 GEMM 的残差组件可以拆分：在同一个 kernel 中返回**残差和归一化结果**，让两条下游路径都拿到所需的值。校验失败的候选保留原 ATen 运算；抽样通过不等于对所有输入的数值证明。

| BF16 Qwen2，带 mask，首套种子，四步合计 | HF eager | 仅动态注意力 | + 静态融合 | + 残差拆分 | + 主机调用简化 | dynamic Inductor |
|---|---:|---:|---:|---:|---:|---:|
| 初始 2 token，GPU launches | 176 | 160 | 112 | 80 | 80 | 72 |
| 初始 2045 token，设备重放 µs | 819.55 | 342.00 | 283.22 | 246.74 | 246.98 | 688.70 |
| 八种初始长度轮换，同步 µs | 1271.37 | 1441.61 | 1144.77 | 889.95 | 851.33 | **712.92** |
| 八请求排队，每请求 µs | 1260.75 | 1435.52 | 1136.31 | 885.18 | 841.92 | **705.92** |

Across two weight seeds and both mask modes, six escaping static outputs per step were byte-identical to the attention-only path on all 128 held-out steps; every SparkFuse variant also returned the exact HF K/V caches. Dynamic Inductor passed the numerical and prefix contract but not exact new-cache bytes. The last optimization specializes 14 literal ATen host call sites; it does **not** reduce the launch count again. At initial length 2 the fastest SparkFuse variant still lost on device (159.92 vs 144.21 µs for Inductor). Each timing is for four autoregressive cache updates with fresh outputs, not for a whole model or a single token.

[Static-island pass](../sparkfuse/hybrid_decode.py) · [raw ablations, boundary bytes and clocks](../reports/hybrid-qwen-gb10.json) · [negative and public-residual tests](../tests/test_hybrid_decode.py) · [sanitizers](../reports/hybrid-qwen-sanitizer-gb10.txt). Reproduce: `python tools/hybrid_qwen_panel.py --output /tmp/hybrid-qwen.json`.

<a id="decoder-families"></a>
## Decoder families: fuse Q/K rotary updates across three real HF layers

A shared FX pass replaces the 14-node Q/K rotary pair with one CUDA call. Before rewriting, it checks shape, stride, aliases and bytes on two independent executions. If an intermediate escapes, a later rewrite invalidated a live-in, or the second execution disagrees, the original ATen nodes stay. The dynamic attention program, GEMMs, five static fusion islands, public residual semantics and cache contracts are unchanged in the ablation.

**What was tested.** Real Transformers 4.51.3 `Qwen2DecoderLayer` (8:2 heads, D64), `LlamaDecoderLayer` (8:4, D64) and `MistralDecoderLayer` (6:2, D128), random weights, batch 2, BF16, one layer. Mistral uses full context (`sliding_window=None`). Each model has two weight seeds and masked/unmasked graphs. The exporter sees past 17; a second trace at 65 qualifies each site. At eight other initial lengths (2–2045), four successive calls feed back their own fresh caches. All 384 SparkFuse steps passed the hidden-state tolerance and preserved both HF caches byte for byte; all 768 rotary tensors matched original ATen bytes and strides. The attention boundary has an independent CPU FP64 comparison. Dynamic Inductor met the numerical/geometry/prefix gate in all 12 workloads but was not byte-exact for every appended cache token.

Masked first seed (3734911), **µs per four-step request**, not per token. Short/long are CUDA Graph device replays; rotating is ordinary synchronized wall time over eight different contexts. Columns are HF eager / SparkFuse before rotary pairing / SparkFuse after pairing / dynamic Inductor:

| Layer | Short (past 2) | Long (past 2045) | Rotating sync | Queued 8, per request |
|---|---|---|---|---|
| Qwen2 | 279.47 / 160.27 / 134.50 / 144.05 | 810.51 / 232.97 / 207.02 / 682.94 | 1254.82 / 850.00 / 693.18 / 714.36 | 1246.76 / 837.95 / 685.17 / 709.02 |
| Llama | 284.18 / 159.58 / 133.16 / 141.57 | 894.05 / 307.22 / 282.02 / 731.50 | 1249.85 / 830.98 / 683.43 / 693.74 | 1237.10 / 828.75 / 671.62 / 691.26 |
| Mistral | 328.46 / 196.27 / 169.13 / 181.26 | 1214.89 / 427.18 / 402.92 / 1091.07 | 1269.52 / 845.04 / 691.57 / 731.15 | 1256.08 / 839.99 / 685.65 / 727.57 |

Paired rotary reduced the short four-step launch count from 80 to 60 for all three models; these masked eager/Inductor baselines used 176/72. The paired variant had lower rotating-sync medians than both its pre-pair ablation and dynamic Inductor in all 12 measured workloads. The smallest margins should not be generalized to other devices, batches, serving conditions, full checkpoints, or model quality. [Full report and individual timing windows](../reports/decoder-family-gb10.json) · [sanitizer](../reports/decoder-family-sanitizer-gb10.txt) · [regression gates](../reports/decoder-family-stage-acceptance.json).

<a id="dynamic-append"></a>
## One variable-length CUDA append for two public caches

`partition_hybrid_decode(..., paired_rotary=True, paired_append=True, host_calls=True)` recognizes the two `cat` inputs of a qualified attention site. It proves each new token has length one and each old cache has length `S−1`. The generated kernel copies both public results in one launch; separate allocations preserve aliasing and liveness. One native call selects head-major or prefill token-major access independently for K and V, and chooses copy width from actual pointer alignment. Unrecognized strides take the original two concatenations. A failed two-trace byte/ownership check leaves both nodes in the graph. No spare capacity, in-place cache update or paging is assumed.

The full-stack ablation holds fixed the attention, GEMMs, five static islands, rotary pair and other host calls. These are still **single-layer** models, four sequential decode steps per request, all 12 outputs outstanding. Eight independently generated prefixes reach past 8192 on the fourth step. Across three HF families × two weight seeds × two masks, the 384 compared steps had identical full outputs between separate and joined paths, and both preserved every HF cache byte. The hot launch count fell by four (60 → 56 at initial past 2; 64 → 60 at 8189).

| Masked first seed (3823703) | Long CUDA, µs | Rotating sync, µs | Queued 8, µs/request |
|---|---|---|---|
| Qwen2 | 2941.87 / 544.44 / 517.29 / 2737.08 | 1281.90 / 711.73 / 684.97 / 729.05 | 1346.20 / 695.28 / 678.30 / 943.99 |
| Llama | 3433.81 / 912.97 / 888.33 / 3222.95 | 1264.26 / 690.70 / 662.12 / 708.69 | 1440.43 / 689.03 / 666.55 / 1003.31 |
| Mistral | 3929.27 / 1109.84 / 1067.15 / 3741.56 | 1280.86 / 698.57 / 675.33 / 721.96 | 1595.29 / 696.69 / 675.54 / 1203.84 |

Order: HF eager / two ATen appends / paired CUDA append / dynamic Inductor. The isolated masked Qwen2 prefill append (past 8189) fell from 66.59 to 50.79 µs; the next already-contiguous append went from 48.78 to **49.09** µs. The long-request peak allocator increase was identical for the two implementations (35,649,536 bytes on that Qwen2 workload); this is a launch/layout optimization, not a memory-saving cache. [Full report and per-layout clocks](../reports/dynamic-append-gb10.json) · [sanitizer](../reports/dynamic-append-sanitizer-gb10.txt) · [regression gates](../reports/dynamic-append-stage-acceptance.json).

<a id="warp-decode"></a>
### Keeping split-KV scores in warp registers

`compile_length_decode(..., warp_heads=4, tile=32)` gives one warp a (query head, KV chunk). Eight groups of four lanes load 128-bit Q/K packets, accumulate token dots, then exchange scores and weights with shuffle instructions before reading V. One or four independent warps share a block; neither schedule uses a block barrier inside the partial. The default remains the existing CTA kernel. The final merge, output ownership, numerical domain and native launcher stay the same. This is single-token attention, not an end-to-end decoder result.

Device replay medians on GB10, µs, seed 3921719, **8,193** KV entries. All algorithms include the merge; dynamic Inductor is timed only when its result passes both the numerical and storage contract.

| B / H:G / D | bias, dtype | eager | CTA / tile 32 | four warps / tile 32 | four warps / tile 64 | dynamic Inductor |
|---|---|---:|---:|---:|---:|---:|
| 2 / 8:2 / 64 | per batch, BF16 | 1114.03 | 38.33 | 19.37 | 18.18 | 1959.29 |
| 2 / 6:2 / 128 | per batch, BF16 | 1815.39 | 41.71 | 25.36 | 27.83 | 3034.18 |
| 1 / 16:4 / 128 | none, FP16 | **25.34** | 49.24 | 28.77 | 30.62 | not admitted |
| 2 / 8:4 / 64 | none, BF16 | 24.67 | 35.51 | 19.22 | 18.85 | not admitted |

For masked H8:D64 at tile 32, the partial goes from 40 to 44 registers per thread, 1,160 to zero static shared bytes and four to zero static barrier sites. The new kernel contains `LDG.E.128`; neither uses local memory. Both need 1,085,568 bytes of global partials at 8,193 entries; tile 64 halves the chunk count (544,896 bytes). Vectorization isn't a universal win: masked H6:D128 at length 2 takes 2.67 µs on CTA32 versus 3.34 µs on warp4/32. On an eight-length rotation the same pair takes 18.70 vs 19.80 µs synchronous wall time. The unmasked FP16 long case still loses to eager. Both seeds' unmasked Inductor runs stayed within FP64 error tolerances but failed the output structure gate; those times are omitted rather than ranked.

[Raw windows, kernel resources, SHA-256s and all 192 held-out requests](../reports/warp-decode-gb10.json) · [Acceptance](../reports/warp-decode-stage-acceptance.json) · [Sanitizers](../reports/warp-decode-sanitizer-gb10.txt). Reproduce with `python tools/warp_decode_panel.py --output /tmp/warp-decode.json`; use `--smoke` for an untimed check. Qualification uses two separate lengths and storages; CPU FP64 is independent of SDPA. All six generated schedules pass the held-out gate in all eight cases.

<a id="protected-stack"></a>
### When later layers must return *exactly* the eager KV caches

A decode step returns a hidden state and two caches per layer. Replacing an early attention with a numerically close implementation can change a later layer's K/V projection. `partition_hybrid_decode(..., exact_outputs=(...))` takes flat public output indices: the partitioner freezes their transitive FX dependencies, including arguments, target and wiring, before rewriting other regions. With all caches protected, the last attention is still fused; earlier attentions stay in PyTorch. Protection applies to the selected outputs, **not** the whole model. `exact_outputs=()` remains the unconstrained path.

On GB10 we ran two actual HF Qwen2 decoder layers and three Llama layers (random weights), with/without bias, two weight seeds, eight initial context lengths and four sequential steps. Each path fed its **own** resulting K/V back to the next step. All five paths passed the ordinary numerical gate, but **768 of 1,280** unprotected cache tensors differed in at least one byte from HF; dynamic Inductor differed on **1,024**. Protected and exported ATen each differed on **zero**. The independent CPU FP64 check is for the last attention, not a claim of bit-exact model hidden states. The exported graph, weights, inputs, prefixes and output ownership were checked separately.

| Masked stack, seed 3998173 | HF eager | exported ATen | unprotected | protected | dynamic Inductor |
|---|---:|---:|---:|---:|---:|
| Qwen2 ×2, 4-step device replay, initial past 2045 (µs) | 1705.75 | 1701.82 | 440.49 | 1218.04 | 1400.87 |
| Qwen2 ×2, rotating synchronized request (µs) | 2514.11 | 3072.82 | 1219.19 | 2655.76 | 1200.86 |
| Llama ×3, 4-step device replay, initial past 2045 (µs) | 2924.56 | 2922.14 | 1249.80 | 2434.84 | 2479.79 |
| Llama ×3, rotating synchronized request (µs) | 3684.54 | 4420.94 | 2827.33 | 3984.80 | 1562.52 |

On masked Qwen2, protecting 113 FX ancestors reduces the 348 HF launches to 294, not the unconstrained 116. Even with lower long-context device time the protected path is **slower than HF** on these rotating host requests. Choose the contract first, then compare timings among paths that meet it. [Raw per-output hashes and every window](../reports/protected-stack-gb10.json) · [sanitizers](../reports/protected-stack-sanitizer-gb10.txt) · [release gates](../reports/protected-stack-stage-acceptance.json). With the optional Transformers environment, run `python tools/protected_stack_panel.py --output /tmp/protected-stack.json` (`--smoke` for a short check).
