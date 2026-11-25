# SparkFuse

在 **NVIDIA DGX Spark（GB10）** 上开发的单 GPU 推理子图编译器。输入是 `torch.export` 导出图，输出是可阅读的 CUDA C++。重点是决定哪些值能留在寄存器里、哪些结果必须写回显存，以及哪种线程布局在这张卡上确实更快。

- **前端与融合**：将逐元素运算与一个末轴归约区域（均值、稳定 Softmax 或 LayerNorm）降为标量 DAG；根据图外使用者决定输出。只返回 RMSNorm 时不写回残差；残差也是输出时，同一个 kernel 多写一个结果。生成示例在 `examples/generated/`。
- **后端与实测选型**：按行分配线程块、每个 warp 独立处理一行，或线性遍历逐元素图。向量读写支持非对齐地址和不足一包的奇数宽尾部。搜索线程数、向量宽度和跨归约复用，使用另一组输入交错复测后才采纳配置。
- **真实框架集成**：已导出 Hugging Face 的 `Qwen2RMSNorm`、`Qwen2MLP`、`Qwen2Attention` 及 BERT 输出块与完整单个编码层。BERT 编码层可切出一处 Softmax 与两处残差 + LayerNorm 区域；固定归约顺序在 [32 次独立输入与权重组合](reports/bert-layer-affine-holdout-gb10.json) 上逐元素匹配 eager。分区器在 MLP 中自动识别 SiLU 与乘法，三个线性层仍交给 PyTorch；Attention 中可融合分数缩放、加性掩码、Softmax 和类型转换，QK/PV 矩阵乘法仍由 PyTorch 执行。验证使用随机权重，不代表整模型推理性能。

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

**形状和范围**：支持连续的 FP32/FP16/BF16 推理张量。静态图之外，`DynamicRows` 支持 rank-2 输入共享一个有界的动态 token 行数，无需按行数重新编译。暂不生成 GEMM、完整 Attention、反向传播或通用动态形状。独立编译拒绝不支持的图；混合分区只替换通过检查的区域。缓存热的微基准不能外推为模型吞吐。

真实 Qwen2Attention 的三次请求可直接用于选型和整图校验，第四次请求单独计时。在 `batch=2, seq=97` 的一轮对照中，分数子图比自身默认布局快 2.33×，但整个 Attention 的设备时间仅从 158.02 降至 155.03 µs。详见[实验记录](docs/performance.md)。

## English

SparkFuse compiles functional PyTorch inference regions into inspectable CUDA on NVIDIA DGX Spark. Its typed IR supports elementwise DAGs and one last-axis collective region (mean, stable softmax, or affine LayerNorm). Output liveness controls global writes; a measured dispatcher selects among block-, warp- and vector-based schedules. The heterogeneous path keeps unsupported operations in PyTorch. Real Qwen2 modules and a complete BERT encoder layer are tested with random weights, not a pretrained checkpoint. Fixed-order reductions matched eager elementwise across 32 held-out BERT layer runs; GPU replay and synchronized host latency are reported separately. The table above reports warm device replay, not Python latency or end-to-end model speedup.

An optional ATen host launcher reduces call overhead without changing the generated kernel. See [performance evidence](docs/performance.md) for fresh-output host latency, address rotation, and odd-width vector results, including regressions.
