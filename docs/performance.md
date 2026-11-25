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
