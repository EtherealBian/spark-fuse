# 单台 DGX Spark 上的 gpt-oss-120b

在一台 DGX Spark 上加载 MXFP4 混合量化的 gpt-oss-120b。这个稀疏 MoE 每层从 128 个专家中选 4 个；不能用模型名中的 120B 与其他稠密模型直接比较吞吐量。

固定文件 63,387,346,208 字节，加载前重新校验完整 SHA-256。四个 llama.cpp CUDA 进程都报告 36 个模型层及输出层放在 GPU。首次启动记录了 59,851.68 MiB 的 CUDA 模型缓冲；这是后端报告值，不是系统统一内存的峰值。

## 增大预填充的物理微批有帮助吗？

逻辑 batch 固定 512。按 128、512、512、128（ABBA）次序独立启动四次，每次预热后测四条输入、关闭槽内复用。每种长度有两条构造提示，每请求固定生成 24 个贪心 token（屏蔽结束 token）。表中每种配置的延迟是四次请求的中位数。

| 输入 token | 预填充 ms：128 / 512 | 端到端 ms：128 / 512 | 四次输出均相同 | 同设置内两次均可复现 |
|---:|---:|---:|---:|---:|
| 256 | 527.2 / 427.8 | 1025.6 / 941.6 | 0/2 | 2/2 |
| 2048 | 2235.2 / 1035.2 | 2740.7 / 1555.2 | 1/2 | 2/2 |

### 同一条提示、交换先后次序后，差异还在吗？

下表每格是同一提示的两次配对变化（首轮 128→512 / 次轮 512→128）。时间差统一计算为 128 减 512，正数表示 512 较快。输出列为两个配对中的首个分歧位置，从 1 开始；「同」仅指该对的 24 个 token 全部相同。

| 输入 / 案例 | 预填充节省（ms） | 端到端节省（ms） | 首个分歧 token |
|---|---:|---:|---:|
| 256 / 1 | +24.4 / +134.0 | +11.9 / +110.3 | 1 / 1 |
| 256 / 2 | +114.8 / +115.3 | +96.7 / +115.2 | 1 / 1 |
| 2048 / 1 | +1180.7 / +1166.5 | +1171.9 / +1128.2 | 同 / 同 |
| 2048 / 2 | +1223.8 / +1217.1 | +1218.9 / +1192.5 | 1 / 1 |

### 已知流量应选哪一种启动配置？

用四条提示各出现一次作为部署流量，工具建议**保留 128**。两种测试次序中，候选微批 512 相对 128 的加权端到端节省分别为 +33.2% / +33.9%。未通过的门槛：四次输出逐 token 一致。

推荐只对已测输入的 token 指纹和这份运行环境生效。有新输入时，工具拒绝按长度猜测，保留基线。加权总耗时是依据四轮原始记录的推算，不是重放新流量。工具只输出下次启动的建议，不会修改运行中的进程。三道门槛为四轮 token 一致、两个次序加权均节省至少 3%、任一提示在任一次序回退不超过 2%。

[部署选择及逐项门槛](../reports/gpt-oss-120b/profile.json)；CPU 上运行 `python3 benchmarks/quant_serving/profile120.py reports/gpt-oss-120b`。可选 `--mix counts.json`，键为完整输入 token 序列的 SHA-256，值为该输入的正整数请求数。可用 `--target <dir>` 提供目标实例的权重校验和运行回执；权重、GPU、驱动、后端二进制、动态库或上下文几何不匹配时工具拒绝给出该部署的建议。默认离线报告不验证新的部署实例。

将短、长两组已测输入分别视为独立部署（各两条、各出现一次），结果如下。不会把这个选择套用到长度相同的新提示。

| 已测输入组 | 建议的物理微批 | 两个次序的加权往返节省 |
|---|---:|---:|
| 256 token 的两条提示 | 128 | +5.3% / +10.8% |
| 2048 token 的两条提示 | 128 | +43.6% / +42.7% |

有提示在四次启动间产生不同的 token。因此不能把这些设置视为数值等价。测量不含模型加载和分词；端到端包括回环 HTTP。小样本只能说明这台机器在这些输入上的行为，不构成模型质量测试。

## 复核

[原始四轮记录、权重指纹及后端回执](../reports/gpt-oss-120b/) · [运行命令](../benchmarks/quant_serving/README.md)。用 Python 3.11+ 在 CPU 上重算：`python3 benchmarks/quant_serving/audit_bundle.py 120b reports/gpt-oss-120b`。不需要下载权重。重跑 GPU 计时则需要完整、校验相符的权重。

**English.** Single-Spark warm inference with a pinned MXFP4 MoE checkpoint. SparkFuse varies the prefill microbatch across four independent server starts, while keeping the logical batch, prompts and requested output length fixed. The CUDA implementation is upstream llama.cpp. For the two 2,048-token prompts, median end-to-end latency changed from 2.74 s to 1.56 s. Only 1 of four prompts matched across both settings; the deployment gate selected microbatch 128. These figures are not saturated throughput or a quality comparison.
