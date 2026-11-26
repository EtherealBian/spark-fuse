# 235B 自然生成：微批 256 更快，但会改变一个答案

在单台 DGX Spark 上，用同一个 Qwen3-235B-A22B-Instruct-2507 Q2_K 检查点和固定的 llama.cpp CUDA 服务，改变处理输入提示时的物理微批（128 或 256）。四次重新启动，顺序为 128 → 256 → 256 → 128。每次测四条提示，正序和反序各执行一次，另用四个非流式请求核对输出；共 32 条冷流、16 个独立见证，4 次预热不计时。最多生成 48 个 token，允许自然停止。所有 32 条实际自然结束，且输出 token 序号完整，与各自设置下的独立见证逐个相同。

| 输入 token 数 / 任务 | 首个可见字符延迟节省 | 请求结束延迟节省 | 两种微批的完整输出相同？ |
|---|---:|---:|---|
| 512 / 提取标签 | 27.9% / 27.8% | 26.8% / 26.6% | 是，`AMBER` |
| 512 / 解释维修原因 | 29.0% / 28.9% | 22.2% / 22.2% | 否，19 → 18 个 token（含结束 token） |
| 2048 / 提取标签 | 29.8% / 29.8% | 29.3% / 29.4% | 是，`AMBER` |
| 2048 / 解释维修原因 | 29.4% / 29.4% | 25.9% / 25.8% | 是 |

斜线两侧分别为两个交叉启动顺序的配对节省率中位数。首个可见字符指客户端首次收到非空白文字，不是 GPU 核心计时。短解释结尾在 128 下是 `accurate diagnosis and effective repair.`，在 256 下是 `accurate diagnosis and repair.`，交换启动顺序后仍然如此。答案不同，不等于质量变差。从这项测量不能推断浮点差异最早出现在哪一层。

**判定：延迟通过；自然停止通过；逐 token 一致性失败；保留原默认值。** 这扩展了[固定输出长度的实验](qwen235b-tiles.md)，没有覆盖并发吞吐或一般模型质量。不要把两个实验的百分比相加。

## 复核

从仓库根目录执行。无需模型或 GPU 即可用 Python 标准库从 36 份进程、内存和请求证据重建两个报告。完整交付账本区分已观察到的采样序号与缺失序号，不推断漏掉的 token。

```bash
set -euo pipefail
B=benchmarks/quant_serving; X=$B/stream235; BASE=reports/qwen235b; R=$BASE/natural-stream
export PYTHONPATH="$PWD/$B"; LOCK=$B/qwen235b-q2k.lock.json
python3 -B -S "$X/review_natural.py" "$LOCK" "$R" "$BASE" | cmp - "$R/summary.json"
python3 -B -S "$X/delivery_ledger.py" "$LOCK" "$R" "$BASE" | cmp - "$R/delivery-ledger.json"
```

独立重建冻结的提示 token：令 `LLAMA_SOURCE`、`LIBLLAMA`、`WEIGHTS` 指向基线版本的源码、已核验库和权重目录；只读首分片文件头，不载入权重。真机复跑还需 `SERVER` 和下载前保存的 `TRANSFER_RECEIPT`；两者及 `WEIGHTS` 需导出。`DEST` 必须是新目录，不要覆盖原证据。

```bash
python3 -B -S "$X/verify_vocab.py" "$LLAMA_SOURCE" "$LIBLLAMA" "$WEIGHTS" "$BASE/header-budget.json" "$BASE/runtime.json" | cmp - "$X/input-contract.json"
mkdir "$DEST"
for b in 0 1 2 3; do TILE_BLOCK=$b OUT="$DEST/block-$b" bash "$X/run.sh"; done
python3 -B -S "$X/review_natural.py" "$LOCK" "$DEST" "$BASE" > "$DEST/summary.json"
python3 -B -S "$X/delivery_ledger.py" "$LOCK" "$DEST" "$BASE" > "$DEST/delivery-ledger.json"
```

**English.** Across four cold process launches, microbatch 256 cut time to first visible text by 27.8–29.8% on these four prompts. All 32 streams ended naturally and matched their own non-stream token witnesses. One 512-token explanation consistently lost the word “effective” under the alternative microbatch. Timing passed; exact output across settings did not. No default change.
