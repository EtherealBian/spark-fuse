# 单台 Spark 上，MoE 的专家权重占多少字节？

检查的是已经在 DGX Spark 上运行过的三个 GGUF 文件集。先按张量名、形状和量化块大小重建逐层目录，再独立使用固定版本的上游 `GGUFReader` 和实际加载过的 GGML 库逐张量核对。72B 稠密模型作为对照；它没有路由专家，但每个 token 仍需运行稠密前馈层。

| 模型 | 张量数 | 全部张量打包字节 | 其中路由专家字节 | 单 token 对应的专家切片字节¹ |
|---|---:|---:|---:|---:|
| Qwen2.5-72B Q4_K_M | 963 | 44,004,474,880 | 0 | 0 |
| gpt-oss-120b MXFP4 | 687 | 63,374,323,968 | 61,073,326,080 | 1,908,541,440 |
| Qwen3-235B-A22B Q2_K | 1,131 | 85,684,996,096 | 82,204,164,096 | 5,137,760,256 |

¹ 沿专家维度均分打包字节，再按每层选择数累计。120B 每层从 128 个专家中选 4 个；235B 从 128 个中选 8 个。区分**必须驻留的完整权重**与**某个 token 逻辑上选中的专家切片**。后者不含注意力、路由器、嵌入等非专家权重，也不等于 GPU 实际 DRAM 读写、显存峰值或每秒吞吐。

同一层并发处理多个 token 时，专家可能重合。报告给出 1、2、8、32 个 token 的重合与分离边界，不假定随机选择或均匀分布。独立复核读取真实目录并关联先前的全文件校验回执，没有在这一步重新散列全部权重。

## 复核

从仓库根目录执行；第一条命令无需模型、GPU 或第三方 Python 包。第二条增加有界文件头复核。第三条需要与报告 SHA-256 相符的上游 `gguf-py` 和实测 GGML 共享库。

```bash
B=benchmarks/quant_serving
python3 -B "$B/audit_expert.py" reports | cmp - reports/expert-ledger/summary.json
python3 -B "$B/audit_expert.py" reports --weights235 "$WEIGHTS235"
PYTHONPATH="$LLAMA_SRC/gguf-py" python3 -B "$B/expert_reference.py" 235 "$WEIGHTS235" "$LLAMA_BUILD/bin/libggml-base.so" --reports reports
```

**English.** This is a tensor-directory accounting tool, not a memory-bandwidth benchmark. It reconciles packed expert tensors across dense, MXFP4 and Q2_K layouts and reports bounds on the logical expert slices for a group of simultaneous tokens. It does not infer the actual routing, cache-hit rate or hardware traffic of a request.
