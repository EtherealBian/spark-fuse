# SparkFuse

在 **NVIDIA DGX Spark（GB10）** 上开发的单 GPU 推理子图编译器。输入是 `torch.export` 导出图，输出是可阅读的 CUDA C++。重点是决定哪些值能留在寄存器里、哪些结果必须写回显存，以及哪种线程布局在这张卡上确实更快。

- **前端与融合**：将逐元素运算与一个末轴归约区域（均值、稳定 Softmax 或 LayerNorm）降为标量 DAG；根据图外使用者决定输出。只返回 RMSNorm 时不写回残差；残差也是输出时，同一个 kernel 多写一个结果。生成示例在 `examples/generated/`。
- **后端与实测选型**：按行分配线程块、每个 warp 独立处理一行，或线性遍历逐元素图。对只输出行统计量的均值归约，新增[单线程写回的 CUDA 布局](examples/generated/compact-mean/compact_mean.cu)。向量读写支持非对齐地址和奇数宽尾部。搜索线程数、向量宽度和跨归约复用，使用另一组输入交错复测后才采纳配置。
- **真实框架集成**：在 Hugging Face 单层上验证融合结果。Qwen2 的可选 Q/K 旋转融合把同一解码层的发射数从 [28 降到 19](reports/qwen-decoder-rotary-gb10.json)（BF16，序列长 65）。[带 KV 缓存的两步解码](reports/qwen-decode-cache-gb10.json)也验证了隐藏态、注意力概率和新 K/V。BERT 的残差 + LayerNorm 在 [32 组独立输入与权重](reports/bert-layer-affine-holdout-gb10.json)上逐元素匹配 eager；T5 门控 GELU 把 FFN 发射数从 [12 降到 4](reports/t5-gate-profiler-gb10.json)。矩阵乘法仍交给 PyTorch。这些实验使用随机权重。

[分解式 RMSNorm 的 CUDA 布局](docs/performance.md#packed-mean)把窄精度残差留在共享内存，保持 FP32 求和次序；冻结选型后，18 组子图对照加速 1.12–1.73×。[放进 Qwen2 整层](docs/performance.md#qwen-packed-mean)后，三个长度中有一个变慢。现在也可以[按整图耗时选择单个融合区域的布局](docs/performance.md#context-site)，其他区域保持原样。[常驻 CUDA Graph](docs/performance.md#resident) 支持请求换地址，返回独占结果；Qwen2 单请求延迟下降，但长序列排队时没有收益。独占输出可用 [C++ 请求路径](docs/performance.md#resident-request-io)减少提交开销。三个固定长度可[共用一份权重快照](docs/performance.md#shape-family)。

[Q/K/V 布局实验](docs/performance.md#packed-rotary)保留图要求的拷贝；在[原生 Phi-3 解码层](docs/performance.md#phi3-native)上，V 必须保留为视图。将 Q/K 旋转与[两次 KV 缓存拼接](docs/performance.md#phi3-cache)合成一个 CUDA kernel 后，单 token 更新的整层发射数从 15 次降到 13 次；不改权重。

[逐 token INT8 量化](docs/performance.md#token-quant)在一个 kernel 中计算残差、编码和尺度；配合[有序反量化](docs/performance.md#int32-epilogue)，三个投影片段从 14 次发射降到 3 次，整数矩阵乘法仍由 PyTorch 执行。对不再修改的整数权重，可以在[建立常驻图时按列存放](docs/performance.md#frozen-rhs)，不需要每次请求转置。[三路投影](docs/performance.md#grouped-epilogue)可共用反量化；权重冻结时还能[合为一次整数矩阵乘法](docs/performance.md#packed-projection)，每路累加器仍可单独读取；[偏置加法和 BF16/FP16 转换](docs/performance.md#affine-projection)也可在同一尾部完成。[多长度服务](docs/performance.md#packed-family)只保留一份打包权重。

### 跑一个例子

需要 Linux、支持目标 GPU 的 CUDA 工具链（含 `nvcc`）和 CUDA 版 PyTorch；实测环境为 GB10 / CUDA 13.0 / PyTorch 2.13.0+cu130。以下命令从仓库根目录运行：

```bash
source tools/env.sh
python -m pytest -q
# 也可先离线导出一个子图，检查 manifest 与生成的 kernel.cu
python examples/export_scores.py /tmp/scores.pt2
python -m sparkfuse emit /tmp/scores.pt2 --out /tmp/scores-cuda --layout warp --vec 4 --cache
python tools/bench.py --case resnorm2 --rows 32 --width 1536 --dtype bfloat16 --profile
python tools/tune_once.py --case resnorm2 --rows 32 --width 1536 \
  --db /tmp/dispatch.sqlite --output /tmp/tuning.json
```

### 初始验收

BF16、`32 × 1536`、残差可见的 RMSNorm。单位为 µs，报告的是**热地址 CUDA Graph 设备时间**，不是 Python 调用耗时：

| 独立对照轮次 | eager | TorchInductor | SparkFuse |
|---|---:|---:|---:|
| 默认布局 | 2.963 | 1.845 | 4.881 |
| 选型后，新输入 | 2.961 | 1.827 | 1.850 |

首轮 profiler 实测发射数分别为 2、1、1。选型明显改善了自己的基线；第二轮相对 Inductor 的差距不足 2%，不能据此宣称整体超越它。两轮 JSON 包括原图、精度误差、编译/首次调用时间、同步主机延迟和原始采样：`reports/baseline-resnorm2.json`、`reports/accepted-resnorm2-32x1536-bf16.json`。

可选的 C++ 调用路径复用同一份 CUDA 代码。该形状下，每次分配新输出并等待 GPU 完成的主机延迟从 **85.16 降到 8.46 µs**；设备时间分别为 1.81 和 1.84 µs。不同缓存条件、奇数宽度上未获益的配置和复现命令见 [性能记录](docs/performance.md)。

**形状和范围**：支持连续的 FP32/FP16/BF16 推理张量。静态图之外，`DynamicRows` 支持 rank-2 输入共享一个有界的动态 token 行数，无需按行数重新编译。原生调度在同一份 CUDA 二进制上处理变长输入；可见残差 RMSNorm 的[变长测试](reports/dynamic-native-panel.json)中，同步主机延迟由 28.03 降至 9.01 µs。[打包 SwiGLU 门控](docs/performance.md#bounded-packed-rows)也可用一份 CUDA 二进制处理有界变长行。暂不生成 GEMM、完整 Attention、反向传播或通用动态形状。独立编译拒绝不支持的图；混合分区只替换通过检查的区域。缓存热的微基准不能外推为模型吞吐。

真实 Qwen2Attention 的三次请求可直接用于选型和整图校验，第四次请求单独计时。在 `batch=2, seq=97` 的一轮对照中，分数子图比自身默认布局快 2.33×，但整个 Attention 的设备时间仅从 158.02 降至 155.03 µs。详见[实验记录](docs/performance.md)。

## English

SparkFuse compiles functional PyTorch inference regions into inspectable CUDA on NVIDIA DGX Spark. Its typed IR supports elementwise DAGs and one last-axis collective region (mean, stable softmax, or affine LayerNorm). Output liveness controls global writes. The tuner can choose a single-writer reduction when only row statistics escape; [paired measurements](reports/compact-mean-panel.json) include aligned and offset inputs. The heterogeneous path keeps unsupported operations in PyTorch. A complete Qwen2 decoder layer, a BERT encoder layer and T5 gated feed-forward blocks are tested with random weights, not a pretrained checkpoint. The [cached Qwen2 decode experiment](reports/qwen-decode-cache-gb10.json) follows two single-token calls with real K/V carried between them. Device replay and synchronized host latency are reported separately. The table above reports warm device replay, not Python latency or end-to-end model speedup.

An optional ATen host launcher supports static and bounded variable-row exports without changing the generated kernel. A separate bounded-row entry point compiles an entire packed SwiGLU gate to a guarded native call. See [performance evidence](docs/performance.md) for fresh-output host latency, address rotation, and odd-width vector results, including regressions.

The [packed QKV study](docs/performance.md#packed-rotary) preserves explicit copies. An [unmodified Phi-3 layer](docs/performance.md#phi3-native) instead exposes V as a view: fusing Q/K rotation preserves that alias. [Combining rotation and the two KV cache appends](docs/performance.md#phi3-cache) then reduces single-token updates from 15 to 13 whole-layer launches.

For [decomposed residual RMSNorm](docs/performance.md#packed-mean), a packed shared-memory tile preserves both narrow rounding points and the FP32 sum order. Across 18 held-out shape/alignment comparisons it was 1.12–1.73× as fast as the best measured previous schedule; it does not always beat cast-aware Inductor. [Inside Qwen2](docs/performance.md#qwen-packed-mean), one of three layer lengths regressed. An opt-in [whole-graph tuner](docs/performance.md#context-site) can now change one site without retuning its neighbors. [Resident graph execution](docs/performance.md#resident) returns fresh outputs for rotating requests; the Qwen2 panel separates single-request latency from queued throughput. For owning outputs, the [C++ request path](docs/performance.md#resident-request-io) reduces host overhead on short projections. [Multiple fixed shapes](docs/performance.md#shape-family) can share the frozen weights.

[Per-token INT8 quantization](docs/performance.md#token-quant) fuses the residual, codes and scales into one kernel. With the [ordered dequantization epilogue](docs/performance.md#int32-epilogue), three tested projections take three launches each, including the unchanged PyTorch integer GEMM. [Frozen weights](docs/performance.md#frozen-rhs) can be snapshotted in column-major order at capture; the serving path does not repack them. [Sibling projections](docs/performance.md#grouped-epilogue) can share a dequantization kernel. With frozen weights, [one integer GEMM](docs/performance.md#packed-projection) can also serve all three branches while preserving their separate accumulators. The [biased epilogue](docs/performance.md#affine-projection) can also return BF16 or FP16 directly. [Multiple fixed lengths](docs/performance.md#packed-family) can share the same packed weights.
