# 32B 前缀复用：同样的请求，在哪里拆开？

[返回项目结论](findings.md) · [运行前固定的实验设计](../benchmarks/pretrained_qwen/prefix_cuts32/PROTOCOL.md) · [English](#english)

原实验说明，在 2,048-token 输入中复用前 1,536 个 token，两条请求合计可以省约 19%。但原来的短输入和长输入连提示长度都不同，不能用它们推断切点规律。这次固定原来的四条输入，只改变复用多少个 token。**从头计算仍是默认。**

**六个切点全部通过。** 即使只共享 511 个 token，两条请求也已回本。复用 1,535 和 1,536 个 token 时，最保守的分层节省率分别为 18.7% 和 19.0%，四条分支的完整输出分数都与从头算一致。在这批输入上，好结果并不依赖恰好切在 1,536。

| 共享 token | 最小分层节省率¹ | 生成 ID 一致 | 全词表逐字节一致 | 综合验收 |
|---:|---:|---:|---:|:---|
| 511 | 6.1% | 4/4 | 4/4 | 通过 |
| 512 | 7.3% | 4/4 | 4/4 | 通过 |
| 1,023 | 9.0% | 4/4 | 4/4 | 通过 |
| 1,024 | 9.3% | 4/4 | 4/4 | 通过 |
| 1,535 | 18.7% | 4/4 | 4/4 | 通过 |
| 1,536 | 19.0% | 4/4 | 4/4 | 通过 |

¹ 每个切点有两组提示，每组四个配对。按方法谁先运行、哪条分支先运行分别分层，表中取两组共八个配对节省率中位数中的最小值。计时从准备快照开始，到两份输出都回到主机结束。通过需要这个最小值 ≥3%、四个分支的 8 步 BF16 logits 逐字节相同、所有计时请求的 token 与独立诊断相同。分层结果不是置信区间。六个切点之间没有插值，也没有据此确定 CUDA 内核选择的机制。

**测了多少。** 单台 DGX Spark，完整 Qwen2.5-32B-Instruct、64 层 BF16、SDPA、静态容量 2,080。两组各两条请求，每条生成 8 个 token。96 个计时波次、192 条计时请求，另取 48 条完整诊断轨迹；内容相同的轨迹去重后只保存 4 份原始字节，384 个解码步分别核对。每一步检查 152,064 个 BF16 分数、贪心 token 和前两名分差。从头算及 1,536 切点还要匹配前次采集的整条轨迹摘要。[逐波计时](../reports/qwen32b-fork/cuts/raw.json) · [按顺序分层的结果](../reports/qwen32b-fork/cuts/timing.json) · [每一步的字差与分差](../reports/qwen32b-fork/cuts/summary.json)

## 为什么在这里收尾

- **没把融合恢复设为默认。** 两次独立加载模型，分别比较普通恢复与融合 CUDA 恢复；两次均为 0/4 个「长度 × 提示组」达到 3% 的双请求端到端门槛。各顺序分层的变化绝对值均不足 0.32%。这是两种恢复实现之间的比较，不是复用对从头算的 19%。[两次运行对照](../reports/qwen32b-fork/cuts/restore-comparison.json)
- **也没采用短输入的补齐方案。** 对 128/96 短面板，仅补第 0 层或补全部 64 层的 K/V 投影，都让四个分支在固定解码历史下的贪心 ID 和首层抽样对齐。但两种办法的首个剩余抽样差异都出现在第 1 层，全词表分数仍不一致。32 次采集、正反顺序一致、历史对照匹配。没有测补齐的请求速度，不能把它当作生产修复。[干预结论](../reports/qwen32b-fork/cuts/shape-verdict.json)

这组实验足以选出一个可展示的工作点，也足以拒绝全局开关。如果以后要改默认，值得增加的证据是真实请求中共享前缀的长度、复用次数及输出长度分布，而不是继续在这四条输入上寻找更好看的数字。

## 复核与复跑

从仓库根目录执行。Linux，Python 3.11+；快速路径不加载模型，也不下载文件。

```bash
bash audit/replay-key-results.sh
```

要核对全部 BF16 字节，下载[固定版本的观测包（45.5 MiB）](https://github.com/EtherealBian/spark-fuse/releases/download/prefix32-closeout-v1/sparkfuse-prefix32-evidence.zip)。脚本先检查[已记录的 SHA-256](../reports/qwen32b-fork/cuts/evidence-archive.json)，在临时目录还原观测，结束后清理。切点复核只用标准库；完整复核中的短输入干预另需 NumPy。附件不含权重。

```bash
python3 audit/replay-prefix32.py sparkfuse-prefix32-evidence.zip --only cuts
# 同时复核切点、融合恢复和补齐实验：
python3 audit/replay-prefix32.py sparkfuse-prefix32-evidence.zip
```

要在空闲 Spark 重新**测量**切点，使用[原 32B 面板的固定环境与权重](qwen32b-fork.md)，从仓库根目录运行下列命令。`WEIGHTS` 是已核验的本地检查点；`PYTHON` 是该环境的解释器；`OUT` 必须是与仓库同盘且尚不存在的目录。新测量不要求墙钟时间与历史完全相等。

```bash
WEIGHTS="$WEIGHTS" PYTHON="$PYTHON" OUT="$OUT" bash benchmarks/pretrained_qwen/prefix_cuts32/run.sh
```

## English

With the same four 2,048-token inputs, **6/6** tested reuse cuts met both the ≥3% order-stratified timing gate and exact full-vocabulary BF16 trajectories. Even 511 shared tokens paid back preparation. The smallest order-stratified savings were 18.7% at 1,535 and 19.0% at 1,536; exactness did not require that last shared token on these prompts. The two-request workload charges snapshot preparation and generates eight tokens per request. Two independent model loads did not qualify fused restore. Padding K/V projections restored the forced-history greedy IDs, but did not restore exact full-model logits. Cold prefill remains the default. Revisit the policy with a different workload distribution, not another sweep on these four requests.
