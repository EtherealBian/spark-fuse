# DGX Spark: gb10-overnight / general

Captured 2026-11-25T16:07:35.898398+00:00 (UTC); **in progress at capture**. Supervisor deadline: 2026-11-26T09:00:00+08:00.

224 of 230 indexed cases passed all three backends. 87 records lack an archived source tree. [Portable evidence](campaign-mid-general.json.gz) retains indexed failures, discovery/holdout samples and both host/device samples where recorded.

## Results by compiler

Compare only within a row. Ratios are medians of paired case ratios (Inductor / SparkFuse), not model-weighted speedups. Device = warm CUDA Graph replay; host = synchronized Python invocation. Neither is serving throughput. Admitted means a held-out gain vs the compiler’s own default, not a win over Inductor.

| compiler | passed / indexed | admitted | device wins | device ratio | host wins | host ratio |
|---|---:|---:|---:|---:|---:|---:|
| `024db5d8579f67` | 2 / 2 | 2 | 1 | 0.932× | 0 | 0.315× |
| `0f5957192edff3` | 55 / 55 | 46 | 6 | 0.845× | 0 | 0.314× |
| `29ab755b687d33` | 7 / 7 | 7 | 2 | 0.948× | 0 | 0.316× |
| `2ae32d6bd5bd00` | 5 / 5 | 5 | 0 | 0.848× | 0 | 0.319× |
| `351981d945009d` | 21 / 22 | 20 | 1 | 0.863× | 0 | 0.318× |
| `354c8441157449` | 18 / 18 | 17 | 1 | 0.895× | 0 | 0.318× |
| `4b50fb03bd796a` | 1 / 1 | 1 | 0 | 0.985× | 0 | 0.318× |
| `55d647ceb02340` | 2 / 2 | 2 | 1 | 1.049× | 0 | 0.471× |
| `6049812be63603` | 25 / 25 | 25 | 0 | 0.837× | 0 | 0.318× |
| `7e06bedbffaf67` | 4 / 4 | 4 | 1 | 0.827× | 0 | 0.321× |
| `83cb7e50f948fb` | 9 / 10 | 9 | 1 | 0.780× | 0 | 0.316× |
| `8804f866e0fd99` | 1 / 1 | 1 | 0 | 0.694× | 0 | 0.315× |
| `8bdb1aee47d4da` | 14 / 14 | 13 | 0 | 0.847× | 0 | 0.314× |
| `9b98496740e225` | 25 / 27 | 24 | 8 | 0.933× | 0 | 0.317× |
| `a1606ea0e2fda0` | 5 / 5 | 5 | 1 | 0.903× | 0 | 0.321× |
| `acdf484aa79dc2` | 3 / 3 | 3 | 2 | 1.053× | 0 | 0.414× |
| `baf67a5aa8ee62` | 1 / 1 | 1 | 0 | 0.897× | 0 | 0.309× |
| `beefdd4cc4f63a` | 4 / 4 | 3 | 0 | 0.897× | 0 | 0.310× |
| `c0587d2a6b8f3f` | 1 / 1 | 1 | 0 | 0.789× | 0 | 0.308× |
| `c6bb87358bfe3e` | 1 / 1 | 1 | 0 | 0.751× | 0 | 0.316× |
| `d7ea7af237cdee` | 16 / 17 | 15 | 4 | 0.869× | 0 | 0.320× |
| `f9f6b2632665c3` | 4 / 4 | 4 | 1 | 0.883× | 0 | 0.314× |
| `unrecorded` | 0 / 1 | 0 | 0 | — | 0 | — |

## Failures and interruptions

Original attempts remain separate from any later replay. Missing attribution is not evidence that a particular backend failed.

- `decomp-512x1536-bfloat16`: failure-without-worker-evidence; backend: not attributed; compiler: `unrecorded`.
- `decomp-128x2560-bfloat16`: parity-failure; backend: inductor; compiler: `83cb7e50f948fb`.
- `decomp-384x14336-bfloat16`: parity-failure; backend: inductor; compiler: `d7ea7af237cdee`.
- `decomp-16x3584-bfloat16`: parity-failure; backend: inductor; compiler: `9b98496740e225`.
- `decomp-1024x3072-bfloat16`: parity-failure; backend: inductor; compiler: `9b98496740e225`.
- `decomp-48x6144-bfloat16`: parity-failure; backend: inductor; compiler: `351981d945009d`.

## Passing cases

Per-backend device medians in µs; corresponding synchronized host samples are in the bundle.

| case | compiler | eager | Inductor | SparkFuse |
|---|---|---:|---:|---:|
| `resnorm2-32x1536-bfloat16` | `0f5957192edff3` | 2.961 | 1.827 | 1.850 |
| `gate-32x8960-bfloat16` | `0f5957192edff3` | 3.186 | 1.555 | 1.938 |
| `decomp-32x1536-bfloat16` | `0f5957192edff3` | 11.924 | 1.713 | 1.812 |
| `resnorm-32x1536-bfloat16` | `0f5957192edff3` | 2.963 | 1.684 | 1.812 |
| `mean-32x1536-bfloat16` | `0f5957192edff3` | 3.118 | 1.044 | 1.171 |
| `resnorm2-1x1536-bfloat16` | `0f5957192edff3` | 2.963 | 1.460 | 1.459 |
| `gate-1x8960-bfloat16` | `0f5957192edff3` | 2.639 | 0.883 | 1.043 |
| `decomp-1x1536-bfloat16` | `0f5957192edff3` | 9.776 | 1.331 | 1.555 |
| `resnorm-1x1536-bfloat16` | `0f5957192edff3` | 2.963 | 1.331 | 1.554 |
| `mean-1x1536-bfloat16` | `0f5957192edff3` | 2.449 | 1.012 | 1.172 |
| `resnorm2-128x1536-bfloat16` | `0f5957192edff3` | 3.571 | 2.195 | 2.451 |
| `gate-128x8960-bfloat16` | `0f5957192edff3` | 7.310 | 4.784 | 4.534 |
| `decomp-128x1536-bfloat16` | `0f5957192edff3` | 14.627 | 2.067 | 2.666 |
| `resnorm-128x1536-bfloat16` | `0f5957192edff3` | 3.571 | 2.067 | 2.194 |
| `mean-128x1536-bfloat16` | `0f5957192edff3` | 3.475 | 1.171 | 1.427 |
| `resnorm2-7x1536-bfloat16` | `0f5957192edff3` | 2.963 | 1.808 | 1.650 |
| `gate-7x8960-bfloat16` | `0f5957192edff3` | 2.305 | 1.012 | 1.170 |
| `decomp-7x1536-bfloat16` | `0f5957192edff3` | 10.646 | 1.651 | 1.556 |
| `resnorm-7x1536-bfloat16` | `0f5957192edff3` | 2.993 | 1.613 | 1.554 |
| `mean-7x1536-bfloat16` | `0f5957192edff3` | 2.633 | 1.043 | 1.170 |
| `resnorm2-512x1536-bfloat16` | `0f5957192edff3` | 6.291 | 4.624 | 5.522 |
| `gate-512x8960-bfloat16` | `0f5957192edff3` | 171.520 | 120.805 | 121.638 |
| `resnorm-512x1536-bfloat16` | `0f5957192edff3` | 6.231 | 4.243 | 3.956 |
| `mean-512x1536-bfloat16` | `0f5957192edff3` | 4.883 | 1.811 | 1.937 |
| `resnorm2-1x769-float32` | `0f5957192edff3` | 4.659 | 1.210 | 1.459 |
| `resnorm2-1x769-float16` | `0f5957192edff3` | 4.498 | 1.170 | 1.682 |
| `resnorm2-1x769-bfloat16` | `0f5957192edff3` | 4.500 | 1.169 | 1.716 |
| `gate-1x1537-float32` | `0f5957192edff3` | 2.322 | 0.819 | 1.043 |
| `gate-1x1537-float16` | `0f5957192edff3` | 2.451 | 0.879 | 1.043 |
| `gate-1x1537-bfloat16` | `0f5957192edff3` | 2.644 | 0.881 | 1.043 |
| `decomp-1x769-float32` | `0f5957192edff3` | 7.284 | 1.152 | 1.651 |
| `decomp-1x769-float16` | `0f5957192edff3` | 9.971 | 1.136 | 1.427 |
| `decomp-1x769-bfloat16` | `0f5957192edff3` | 10.000 | 1.128 | 1.779 |
| `resnorm-1x769-float32` | `0f5957192edff3` | 4.628 | 1.170 | 1.427 |
| `resnorm-1x769-float16` | `0f5957192edff3` | 4.498 | 1.104 | 1.428 |
| `resnorm-1x769-bfloat16` | `0f5957192edff3` | 4.497 | 1.087 | 1.428 |
| `mean-1x769-float32` | `0f5957192edff3` | 2.579 | 1.011 | 1.266 |
| `mean-1x769-float16` | `0f5957192edff3` | 2.579 | 1.011 | 1.267 |
| `mean-1x769-bfloat16` | `0f5957192edff3` | 2.579 | 0.946 | 1.267 |
| `resnorm2-47x769-float32` | `0f5957192edff3` | 4.883 | 1.451 | 1.811 |
| `resnorm2-47x769-float16` | `0f5957192edff3` | 4.595 | 1.299 | 1.651 |
| `resnorm2-47x769-bfloat16` | `0f5957192edff3` | 4.557 | 1.299 | 1.684 |
| `gate-47x1537-float32` | `0f5957192edff3` | 2.578 | 1.139 | 1.299 |
| `gate-47x1537-float16` | `0f5957192edff3` | 2.355 | 1.043 | 1.171 |
| `gate-47x1537-bfloat16` | `0f5957192edff3` | 2.324 | 1.043 | 1.202 |
| `decomp-47x769-float32` | `0f5957192edff3` | 8.211 | 1.428 | 1.652 |
| `decomp-47x769-float16` | `0f5957192edff3` | 11.329 | 1.299 | 1.555 |
| `decomp-47x769-bfloat16` | `0f5957192edff3` | 11.347 | 1.299 | 1.811 |
| `resnorm-47x769-float32` | `0f5957192edff3` | 4.852 | 1.425 | 1.651 |
| `resnorm-47x769-float16` | `0f5957192edff3` | 4.537 | 1.298 | 1.738 |
| `resnorm-47x769-bfloat16` | `0f5957192edff3` | 4.499 | 1.299 | 1.807 |
| `mean-47x769-float32` | `0f5957192edff3` | 3.347 | 1.011 | 1.195 |
| `mean-47x769-float16` | `0f5957192edff3` | 2.963 | 0.959 | 1.203 |
| `mean-47x769-bfloat16` | `0f5957192edff3` | 2.964 | 1.011 | 1.269 |
| `resnorm2-97x769-float32` | `0f5957192edff3` | 6.196 | 1.937 | 2.199 |
| `resnorm2-97x769-float16` | `c0587d2a6b8f3f` | 5.651 | 1.684 | 2.134 |
| `resnorm2-97x769-bfloat16` | `2ae32d6bd5bd00` | 5.687 | 1.683 | 2.292 |
| `gate-97x1537-float32` | `2ae32d6bd5bd00` | 3.443 | 1.427 | 1.683 |
| `gate-97x1537-float16` | `2ae32d6bd5bd00` | 2.963 | 1.523 | 1.779 |
| `gate-97x1537-bfloat16` | `2ae32d6bd5bd00` | 2.963 | 1.488 | 1.907 |
| `decomp-97x769-float32` | `2ae32d6bd5bd00` | 8.819 | 1.684 | 1.811 |
| `decomp-97x769-float16` | `6049812be63603` | 11.703 | 1.427 | 1.682 |
| `decomp-97x769-bfloat16` | `6049812be63603` | 11.780 | 1.427 | 1.811 |
| `resnorm-97x769-float32` | `6049812be63603` | 6.163 | 1.683 | 1.811 |
| `resnorm-97x769-float16` | `6049812be63603` | 5.651 | 1.425 | 1.811 |
| `resnorm-97x769-bfloat16` | `6049812be63603` | 5.708 | 1.427 | 1.812 |
| `mean-97x769-float32` | `6049812be63603` | 3.475 | 1.140 | 1.298 |
| `mean-97x769-float16` | `6049812be63603` | 2.963 | 1.043 | 1.267 |
| `mean-97x769-bfloat16` | `6049812be63603` | 2.963 | 1.043 | 1.299 |
| `resnorm2-257x769-float32` | `6049812be63603` | 9.643 | 3.520 | 4.339 |
| `resnorm2-257x769-float16` | `6049812be63603` | 8.852 | 2.833 | 3.345 |
| `resnorm2-257x769-bfloat16` | `6049812be63603` | 8.848 | 2.817 | 3.364 |
| `gate-257x1537-float32` | `6049812be63603` | 5.520 | 2.784 | 2.932 |
| `gate-257x1537-float16` | `6049812be63603` | 3.987 | 2.547 | 2.816 |
| `gate-257x1537-bfloat16` | `6049812be63603` | 4.223 | 2.547 | 2.900 |
| `decomp-257x769-float32` | `6049812be63603` | 11.398 | 2.943 | 3.060 |
| `decomp-257x769-float16` | `6049812be63603` | 14.221 | 2.196 | 2.832 |
| `decomp-257x769-bfloat16` | `6049812be63603` | 14.323 | 2.288 | 2.835 |
| `resnorm-257x769-float32` | `6049812be63603` | 9.681 | 3.035 | 3.089 |
| `resnorm-257x769-float16` | `6049812be63603` | 8.724 | 2.196 | 2.804 |
| `resnorm-257x769-bfloat16` | `6049812be63603` | 8.848 | 2.275 | 2.801 |
| `mean-257x769-float32` | `6049812be63603` | 4.083 | 1.427 | 1.487 |
| `mean-257x769-float16` | `6049812be63603` | 3.162 | 1.300 | 1.427 |
| `mean-257x769-bfloat16` | `6049812be63603` | 3.218 | 1.304 | 1.459 |
| `mean-1x8960-bfloat16` | `6049812be63603` | 2.834 | 1.170 | 1.652 |
| `resnorm2-64x3584-float32` | `6049812be63603` | 5.619 | 3.475 | 4.244 |
| `gate-1x4096-float16` | `baf67a5aa8ee62` | 1.938 | 0.819 | 0.913 |
| `gate-32x2048-float32` | `024db5d8579f67` | 2.674 | 1.207 | 1.171 |
| `mean-48x768-bfloat16` | `024db5d8579f67` | 2.674 | 0.975 | 1.170 |
| `mean-2x3072-float32` | `83cb7e50f948fb` | 2.707 | 1.137 | 1.393 |
| `mean-512x2048-float16` | `83cb7e50f948fb` | 5.927 | 2.063 | 2.101 |
| `decomp-8x6144-bfloat16` | `83cb7e50f948fb` | 14.644 | 1.875 | 2.835 |
| `decomp-8x5120-bfloat16` | `83cb7e50f948fb` | 13.204 | 1.795 | 2.739 |
| `resnorm-32x6144-bfloat16` | `83cb7e50f948fb` | 5.010 | 2.164 | 2.773 |
| `gate-256x8192-bfloat16` | `83cb7e50f948fb` | 36.385 | 10.420 | 14.255 |
| `mean-4x3584-bfloat16` | `83cb7e50f948fb` | 2.835 | 1.043 | 1.298 |
| `resnorm2-512x4096-float32` | `83cb7e50f948fb` | 166.766 | 156.240 | 154.961 |
| `resnorm2-4x5120-float32` | `83cb7e50f948fb` | 4.498 | 2.323 | 3.059 |
| `resnorm2-31x11008-float32` | `c6bb87358bfe3e` | 8.857 | 4.627 | 6.163 |
| `resnorm-256x6144-float16` | `354c8441157449` | 16.911 | 7.379 | 7.892 |
| `gate-96x6144-bfloat16` | `354c8441157449` | 4.434 | 2.703 | 2.961 |
| `resnorm2-2x8960-float16` | `354c8441157449` | 6.481 | 2.289 | 3.570 |
| `gate-16x4096-bfloat16` | `354c8441157449` | 2.193 | 1.014 | 1.169 |
| `mean-1024x3584-bfloat16` | `354c8441157449` | 15.085 | 4.628 | 5.311 |
| `gate-2x14336-float16` | `354c8441157449` | 1.970 | 0.915 | 1.043 |
| `mean-2x1536-float16` | `354c8441157449` | 2.420 | 1.042 | 1.169 |
| `gate-2x2048-float16` | `354c8441157449` | 1.939 | 0.820 | 0.913 |
| `gate-1x8192-bfloat16` | `354c8441157449` | 1.939 | 0.859 | 0.947 |
| `resnorm-192x768-float16` | `354c8441157449` | 3.122 | 1.427 | 1.779 |
| `resnorm2-1024x8192-float16` | `354c8441157449` | 351.532 | 312.284 | 314.332 |
| `mean-31x8960-float16` | `354c8441157449` | 3.347 | 1.425 | 1.937 |
| `gate-384x2560-bfloat16` | `354c8441157449` | 6.911 | 3.219 | 3.987 |
| `resnorm2-128x3584-float16` | `354c8441157449` | 5.170 | 2.833 | 3.699 |
| `decomp-48x8192-float32` | `354c8441157449` | 14.895 | 4.396 | 4.687 |
| `resnorm2-32x3584-float32` | `354c8441157449` | 4.752 | 2.707 | 2.835 |
| `gate-4x1536-bfloat16` | `354c8441157449` | 1.939 | 0.822 | 0.915 |
| `resnorm2-192x8960-float16` | `354c8441157449` | 36.736 | 35.999 | 35.425 |
| `resnorm-384x3584-float16` | `acdf484aa79dc2` | 11.443 | 5.652 | 6.967 |
| `resnorm-384x8192-float16` | `acdf484aa79dc2` | 88.288 | 48.705 | 46.273 |
| `resnorm-256x8192-float32` | `acdf484aa79dc2` | 145.362 | 97.902 | 88.091 |
| `resnorm2-8x5120-float16` | `8804f866e0fd99` | 4.339 | 1.811 | 2.610 |
| `mean-1x768-float16` | `beefdd4cc4f63a` | 2.451 | 0.947 | 1.139 |
| `gate-48x1536-float16` | `beefdd4cc4f63a` | 2.195 | 1.073 | 1.170 |
| `mean-96x4096-float32` | `beefdd4cc4f63a` | 8.674 | 1.683 | 1.779 |
| `decomp-32x2560-float16` | `beefdd4cc4f63a` | 13.682 | 1.810 | 2.066 |
| `mean-128x1024-float32` | `a1606ea0e2fda0` | 3.858 | 1.202 | 1.331 |
| `resnorm2-4x4096-float32` | `a1606ea0e2fda0` | 3.858 | 2.036 | 2.580 |
| `gate-16x8960-float32` | `a1606ea0e2fda0` | 3.184 | 1.425 | 1.523 |
| `mean-16x14336-float32` | `a1606ea0e2fda0` | 4.242 | 1.780 | 2.193 |
| `gate-32x5120-float32` | `a1606ea0e2fda0` | 3.380 | 1.613 | 1.555 |
| `resnorm-32x2560-float32` | `4b50fb03bd796a` | 3.826 | 2.163 | 2.196 |
| `mean-31x1024-bfloat16` | `8bdb1aee47d4da` | 2.834 | 0.968 | 1.170 |
| `resnorm2-32x11008-float16` | `8bdb1aee47d4da` | 7.568 | 2.833 | 4.724 |
| `resnorm-32x2048-bfloat16` | `8bdb1aee47d4da` | 2.961 | 1.811 | 1.843 |
| `decomp-31x4096-float16` | `8bdb1aee47d4da` | 16.288 | 1.922 | 2.293 |
| `gate-64x11008-bfloat16` | `8bdb1aee47d4da` | 4.978 | 3.058 | 3.090 |
| `mean-256x8960-float32` | `8bdb1aee47d4da` | 34.961 | 5.405 | 5.910 |
| `gate-384x768-float32` | `8bdb1aee47d4da` | 4.435 | 2.195 | 2.195 |
| `mean-64x2560-bfloat16` | `8bdb1aee47d4da` | 4.112 | 1.169 | 1.555 |
| `mean-96x5120-float16` | `8bdb1aee47d4da` | 6.320 | 1.428 | 1.777 |
| `resnorm2-4x8960-float32` | `8bdb1aee47d4da` | 6.162 | 3.187 | 4.243 |
| `decomp-16x14336-float32` | `8bdb1aee47d4da` | 11.680 | 4.451 | 5.009 |
| `resnorm-64x768-float32` | `8bdb1aee47d4da` | 3.074 | 1.523 | 1.779 |
| `mean-31x1024-float16` | `8bdb1aee47d4da` | 2.803 | 0.968 | 1.171 |
| `gate-512x768-float16` | `8bdb1aee47d4da` | 3.571 | 2.036 | 2.195 |
| `resnorm2-128x14336-float16` | `55d647ceb02340` | 41.968 | 48.562 | 40.498 |
| `decomp-8x2560-float32` | `55d647ceb02340` | 9.048 | 1.811 | 2.012 |
| `gate-192x11008-float16` | `29ab755b687d33` | 59.297 | 12.447 | 10.864 |
| `mean-96x5120-bfloat16` | `29ab755b687d33` | 6.417 | 1.427 | 1.812 |
| `resnorm-31x3584-float32` | `29ab755b687d33` | 4.881 | 2.202 | 2.449 |
| `decomp-256x2048-float16` | `29ab755b687d33` | 23.573 | 2.933 | 3.089 |
| `decomp-256x768-float32` | `29ab755b687d33` | 10.957 | 2.324 | 2.451 |
| `resnorm2-1024x1024-float32` | `29ab755b687d33` | 55.889 | 65.154 | 53.749 |
| `mean-48x1024-float16` | `29ab755b687d33` | 2.818 | 1.011 | 1.170 |
| `mean-31x8960-float32` | `d7ea7af237cdee` | 4.211 | 1.555 | 1.907 |
| `mean-16x2560-bfloat16` | `d7ea7af237cdee` | 3.828 | 1.043 | 1.396 |
| `gate-96x8960-bfloat16` | `d7ea7af237cdee` | 5.811 | 2.962 | 3.703 |
| `decomp-128x11008-bfloat16` | `d7ea7af237cdee` | 107.284 | 8.291 | 8.979 |
| `resnorm2-31x1536-bfloat16` | `d7ea7af237cdee` | 2.963 | 1.811 | 1.680 |
| `resnorm2-2x11008-bfloat16` | `d7ea7af237cdee` | 6.768 | 2.454 | 3.858 |
| `resnorm-96x11008-bfloat16` | `d7ea7af237cdee` | 9.774 | 6.167 | 6.655 |
| `decomp-512x1536-float32` | `d7ea7af237cdee` | 62.011 | 7.442 | 6.952 |
| `resnorm-384x2560-float32` | `d7ea7af237cdee` | 29.362 | 12.707 | 9.998 |
| `resnorm-128x2048-float16` | `d7ea7af237cdee` | 3.730 | 2.291 | 2.323 |
| `resnorm-1024x1536-bfloat16` | `d7ea7af237cdee` | 18.482 | 8.278 | 8.035 |
| `resnorm2-64x8960-float32` | `d7ea7af237cdee` | 15.966 | 11.280 | 14.147 |
| `resnorm2-96x5120-float16` | `d7ea7af237cdee` | 5.677 | 3.230 | 4.499 |
| `decomp-512x14336-float32` | `d7ea7af237cdee` | 1311.648 | 408.387 | 534.387 |
| `mean-384x6144-float32` | `d7ea7af237cdee` | 39.680 | 5.707 | 5.714 |
| `resnorm2-4x11008-float32` | `d7ea7af237cdee` | 7.135 | 3.731 | 5.011 |
| `resnorm-96x8960-float32` | `9b98496740e225` | 20.025 | 9.744 | 8.479 |
| `resnorm2-384x5120-bfloat16` | `9b98496740e225` | 51.383 | 69.642 | 46.003 |
| `decomp-512x6144-float16` | `9b98496740e225` | 387.459 | 43.642 | 39.267 |
| `resnorm-32x3584-float32` | `9b98496740e225` | 4.752 | 2.663 | 2.450 |
| `mean-2x5120-float16` | `9b98496740e225` | 2.708 | 1.086 | 1.553 |
| `mean-1x5120-float16` | `9b98496740e225` | 2.450 | 1.068 | 1.395 |
| `decomp-1024x5120-float32` | `9b98496740e225` | 881.262 | 289.979 | 276.093 |
| `resnorm2-1024x3584-float32` | `9b98496740e225` | 300.817 | 272.224 | 273.154 |
| `decomp-4x768-float32` | `9b98496740e225` | 7.058 | 1.139 | 1.395 |
| `decomp-4x1536-bfloat16` | `9b98496740e225` | 10.499 | 1.650 | 1.553 |
| `resnorm2-16x8192-float16` | `9b98496740e225` | 5.619 | 2.192 | 3.072 |
| `mean-1024x1024-float32` | `9b98496740e225` | 8.851 | 2.763 | 2.963 |
| `decomp-512x768-float32` | `9b98496740e225` | 15.457 | 3.798 | 3.899 |
| `gate-1024x6144-bfloat16` | `9b98496740e225` | 245.329 | 167.651 | 167.120 |
| `resnorm-384x768-bfloat16` | `9b98496740e225` | 3.731 | 2.195 | 2.291 |
| `resnorm2-48x11008-bfloat16` | `9b98496740e225` | 8.080 | 3.475 | 5.108 |
| `mean-96x2048-bfloat16` | `9b98496740e225` | 3.858 | 1.170 | 1.427 |
| `decomp-31x8960-float32` | `9b98496740e225` | 12.444 | 3.924 | 4.113 |
| `decomp-384x1536-float16` | `9b98496740e225` | 26.898 | 3.406 | 4.056 |
| `decomp-8x3584-float16` | `9b98496740e225` | 12.176 | 1.554 | 1.843 |
| `gate-48x11008-float32` | `9b98496740e225` | 6.931 | 4.152 | 3.443 |
| `resnorm2-96x2560-bfloat16` | `9b98496740e225` | 3.916 | 2.164 | 2.707 |
| `mean-1024x3072-bfloat16` | `9b98496740e225` | 12.688 | 4.113 | 4.463 |
| `resnorm2-8x3072-float32` | `9b98496740e225` | 3.698 | 2.083 | 2.355 |
| `resnorm2-48x4096-float16` | `9b98496740e225` | 4.244 | 2.193 | 2.576 |
| `decomp-192x3072-float32` | `7e06bedbffaf67` | 25.378 | 6.064 | 5.907 |
| `gate-64x4096-bfloat16` | `7e06bedbffaf67` | 3.270 | 1.522 | 1.906 |
| `resnorm-31x6144-bfloat16` | `7e06bedbffaf67` | 5.223 | 2.163 | 2.963 |
| `resnorm-32x3072-float32` | `7e06bedbffaf67` | 4.337 | 2.098 | 2.452 |
| `decomp-128x6144-float32` | `f9f6b2632665c3` | 64.098 | 7.988 | 7.341 |
| `mean-1024x768-float32` | `f9f6b2632665c3` | 7.096 | 2.323 | 2.578 |
| `mean-4x2560-float16` | `f9f6b2632665c3` | 2.707 | 1.041 | 1.203 |
| `resnorm2-8x11008-float32` | `f9f6b2632665c3` | 7.411 | 3.827 | 5.007 |
| `mean-512x8960-float16` | `351981d945009d` | 37.923 | 5.541 | 5.778 |
| `gate-96x11008-float32` | `351981d945009d` | 60.465 | 13.408 | 18.558 |
| `mean-31x768-float32` | `351981d945009d` | 3.091 | 1.011 | 1.171 |
| `gate-192x1536-float32` | `351981d945009d` | 4.432 | 2.195 | 2.195 |
| `gate-16x14336-bfloat16` | `351981d945009d` | 2.932 | 1.427 | 1.779 |
| `resnorm-1x2560-bfloat16` | `351981d945009d` | 3.186 | 1.489 | 1.650 |
| `resnorm2-512x11008-float32` | `351981d945009d` | 540.873 | 417.424 | 509.458 |
| `mean-128x2560-float32` | `351981d945009d` | 6.579 | 1.554 | 1.780 |
| `decomp-128x3584-float32` | `351981d945009d` | 20.140 | 4.794 | 4.502 |
| `mean-31x14336-bfloat16` | `351981d945009d` | 3.858 | 1.425 | 2.419 |
| `mean-512x1024-float32` | `351981d945009d` | 5.745 | 2.036 | 2.036 |
| `gate-4x5120-bfloat16` | `351981d945009d` | 1.939 | 0.883 | 1.044 |
| `resnorm-1x1536-float32` | `351981d945009d` | 3.151 | 1.427 | 1.429 |
| `resnorm2-32x5120-bfloat16` | `351981d945009d` | 4.626 | 2.194 | 2.931 |
| `resnorm2-512x8960-bfloat16` | `351981d945009d` | 186.975 | 170.513 | 172.097 |
| `gate-256x3072-bfloat16` | `351981d945009d` | 6.524 | 3.347 | 3.568 |
| `resnorm2-128x4096-bfloat16` | `351981d945009d` | 5.325 | 3.090 | 4.467 |
| `resnorm-96x5120-bfloat16` | `351981d945009d` | 5.695 | 2.931 | 3.825 |
| `resnorm-8x1024-bfloat16` | `351981d945009d` | 2.579 | 1.043 | 1.427 |
| `resnorm2-384x6144-float32` | `351981d945009d` | 189.057 | 175.373 | 177.666 |
| `resnorm2-16x8960-bfloat16` | `351981d945009d` | 6.164 | 2.450 | 3.862 |
