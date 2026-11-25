# Benchmark source snapshots

带有顶层 `provenance.archive` 的基准报告按其相对路径引用这里的源码包。文件名是**内容树 SHA-256**，而非压缩文件 SHA。每个不同的源码树只存一份；没有该字段的早期报告不宣称能还原当时的工作树。源码来自基准执行时的工作树，包括当时尚未提交的改动，不能仅凭 `git_head_at_capture` 定位。包不包含权重、虚拟环境、CUDA 缓存或私有机器配置。

From the repository root, verify before comparing reports or reusing the captured compiler:

```bash
python tools/verify_sources.py reports/host-call-bert-33-gb10.json reports/t5-gated-gelu-fresh-97-gb10.json
```

The standard-library verifier streams regular source members without extracting them. It recomputes the recorded tree digest from sorted file names and per-file SHA-256 values; gzip headers and archive timestamps are not part of this digest. A match identifies source bytes, not the identity of the publisher, the system libraries, or a universal performance guarantee. The runtime environment, seeds and measurement scope belong to each report.
