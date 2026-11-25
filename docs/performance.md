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
