# 32B：将前缀拆开以后，第一层从哪里出现差异？

在[整模型诊断](qwen32b-attribution.md)中，短输入不用快照也会改变第 6 个输出。这里在同一台 DGX Spark 上，用固定的 Qwen2.5-32B-Instruct 第一层权重，将预填充拆开，单独检查 K/V 投影。只计算这一层的输入归一化、投影和旋转位置编码，不运行注意力或语言模型输出头。先同之前独立采集的整模型首层 K/V 样本逐字节核对；整段和分段路径在两个运行顺序、所有抽查位置都对得上。

| 输入 / 前缀 | 整段与分段 | 将两段的投影行数分别补到整段长度 |
|---|---|---|
| 128 / 96 | 4/4 个分支在后缀首位置及输入末位置的 K/V 投影出现差异；前缀末位置相同 | 4/4 个分支七个阶段的抽样均与整段逐位相同 |
| 2,048 / 1,536 | 4/4 个分支的七个阶段均未观察到差异 | 4/4 同样逐位一致 |

例如第一组第一个短分支：在后缀第一个位置，1,024 个 BF16 元素中，旋转前的 K 有 **450** 个不同，V 有 **415** 个不同；词嵌入、归一化、旋转正余弦都相同。这里统计不同的 16 位字，不是误差幅度。每个分支比较三种形状，正反序各一次，共 48 条记录。在隔离实验里补齐会做两倍的投影行数；没有把空行写入缓存。这个结果不能定位具体 CUDA 内核、说明全模型已修复、或证明补齐速度更快。默认行为不变。

### 无权重复核

下载[已发布的整模型 BF16 观测包](https://github.com/EtherealBian/spark-fuse/releases/download/qwen32b-fork-attribution-data-v1/fork32-attribution-bf16.zip)，从仓库根目录执行（Python 3.11+，仅标准库）：

```bash
set -euo pipefail
B=benchmarks/pretrained_qwen; D=$B/layer0_shape; R=reports/qwen32b-fork; P=$R/attribution; L=$R/layer0-shape
export PYTHONPATH="$PWD/$B:$PWD/benchmarks/quant_serving${PYTHONPATH:+:$PYTHONPATH}"
Z=fork32-attribution-bf16.zip
python3 -B -S "$D/shape_evidence.py" replay "$L" "$P" "$Z" "$B/qwen32b.lock.json" | cmp - "$L/compact-replay.json"
```

56 组「分支 × 阶段」中，44 组的整段、分段和补齐抽样在两个顺序下都具有同一 SHA-256。不同的段按字比较：旋转后 K 和 V 来自整模型观测包，另附 8 段旋转前 K，共 **49,152 字节**。复核器还检查测量源码、权重修订、内存准入和守护记录，并同原始审核逐个分支对照。它不重建或发布所有词嵌入及归一化输出，不能用它声称重新哈希了原始 3,907,584 字节的采样文件。

### 在空闲 Spark 重新采集

沿用 [32B 基线](qwen32b-fork.md)环境，用上面的变量，在有空间的目录指定两个**尚不存在**的路径 `CAPTURE` 和 `OUT`。`WEIGHTS` 指向固定版本的本地权重，`PYTHON` 为已安装相应依赖的解释器。先还原整模型参照，再运行隔离实验；不需要重新下载模型。

```bash
python3 -B -S "$D/shape_evidence.py" materialize-parent "$P" "$Z" "$CAPTURE"
REFERENCE="$R" CAPTURE="$CAPTURE" WEIGHTS="$WEIGHTS" OUT="$OUT" PYTHON="$PYTHON" bash "$D/run.sh"
# 从本次采集的完整原始样本重建本次报告（需要 NumPy）
"$PYTHON" -B "$D/review_shape.py" "$OUT" "$CAPTURE" "$B/qwen32b.lock.json" | cmp - "$OUT/summary.json"
```

**English.** On this measured panel, the first sampled difference for short split inputs was visible at layer-0 K/V projection, not at the sampled embedding, RMSNorm or rotary tables. Matching projection shapes removed the sampled differences in the isolated first layer. This is not a full-model repair or a speed result. The compact verifier reproduces every reported per-position mismatch count without redistributing embedding rows.
