# DGX Spark: gb10-attention / attention

Captured 2026-11-25T17:32:59.911172+00:00 (UTC); **stopped: worklist exhausted**. Supervisor deadline: 2026-11-26T08:40:00+08:00.

290 of 300 indexed cases passed all three backends. 0 records lack an archived source tree. [Portable evidence](gb10-attention-report.json.gz) retains indexed failures, discovery/holdout samples and both host/device samples where recorded.

## Results by compiler

Compare only within a row. Ratios are medians of paired case ratios (Inductor / SparkFuse), not model-weighted speedups. Device = warm CUDA Graph replay; host = synchronized Python invocation. Neither is serving throughput. Admitted means a held-out gain vs the compiler’s own default, not a win over Inductor.

| compiler | passed / indexed | admitted | device wins | device ratio | host wins | host ratio |
|---|---:|---:|---:|---:|---:|---:|
| `02bba4166db6de` | 40 / 41 | 37 | 15 | 0.968× | 0 | 0.314× |
| `29ab755b687d33` | 6 / 6 | 6 | 3 | 1.015× | 0 | 0.310× |
| `351981d945009d` | 24 / 24 | 22 | 7 | 0.924× | 0 | 0.313× |
| `354c8441157449` | 11 / 11 | 10 | 5 | 0.999× | 0 | 0.318× |
| `4aa3fee495d695` | 72 / 75 | 67 | 32 | 0.981× | 0 | 0.313× |
| `4b50fb03bd796a` | 1 / 1 | 1 | 0 | 0.715× | 0 | 0.309× |
| `55d647ceb02340` | 2 / 2 | 2 | 1 | 1.071× | 0 | 0.314× |
| `5e2df4ae6b63e8` | 10 / 10 | 10 | 7 | 1.081× | 0 | 0.311× |
| `7c9241c1e568bc` | 13 / 13 | 11 | 4 | 0.941× | 0 | 0.312× |
| `7e06bedbffaf67` | 3 / 3 | 3 | 0 | 1.000× | 0 | 0.310× |
| `8804f866e0fd99` | 1 / 1 | 1 | 1 | 1.035× | 0 | 0.315× |
| `8bdb1aee47d4da` | 14 / 15 | 14 | 5 | 0.933× | 0 | 0.313× |
| `924612bf2fc197` | 1 / 1 | 1 | 0 | 0.869× | 0 | 0.328× |
| `9b98496740e225` | 24 / 25 | 24 | 12 | 1.013× | 0 | 0.312× |
| `a1606ea0e2fda0` | 5 / 5 | 4 | 3 | 1.011× | 0 | 0.311× |
| `acdf484aa79dc2` | 3 / 3 | 3 | 2 | 1.013× | 0 | 0.316× |
| `b9df74c8b7c340` | 7 / 7 | 7 | 5 | 1.041× | 0 | 0.317× |
| `beefdd4cc4f63a` | 3 / 3 | 3 | 1 | 0.867× | 0 | 0.319× |
| `d3a2b2ba8d045f` | 28 / 29 | 27 | 13 | 0.988× | 0 | 0.313× |
| `d7ea7af237cdee` | 17 / 17 | 15 | 8 | 0.977× | 0 | 0.314× |
| `f9f6b2632665c3` | 5 / 5 | 5 | 1 | 0.918× | 0 | 0.314× |
| `unrecorded` | 0 / 3 | 0 | 0 | — | 0 | — |

## Failures and interruptions

Original attempts remain separate from any later replay. Missing attribution is not evidence that a particular backend failed.

- `scores-384x128-bfloat16`: parity-failure; backend: inductor; compiler: `8bdb1aee47d4da`.
- `scores-24x8192-float16`: discovery-error; backend: not attributed; compiler: `unrecorded`.
- `scores2-24x2048-float16`: discovery-error; backend: not attributed; compiler: `unrecorded`.
- `scores2-1536x2048-float16`: parity-failure; backend: inductor; compiler: `9b98496740e225`.
- `scores-384x4096-bfloat16`: discovery-error; backend: not attributed; compiler: `unrecorded`.
- `scores-1164x1024-bfloat16`: parity-failure; backend: inductor; compiler: `d3a2b2ba8d045f`.
- `scores2-384x257-bfloat16`: parity-failure; backend: inductor; compiler: `02bba4166db6de`.
- `scores-384x128-float16`: parity-failure; backend: inductor; compiler: `4aa3fee495d695`.
- `scores2-1536x512-float16`: parity-failure; backend: inductor; compiler: `4aa3fee495d695`.
- `scores2-1536x769-float16`: parity-failure; backend: inductor; compiler: `4aa3fee495d695`.

## Passing cases

Per-backend device medians in µs; corresponding synchronized host samples are in the bundle.

| case | compiler | eager | Inductor | SparkFuse |
|---|---|---:|---:|---:|
| `scores-96x128-float16` | `354c8441157449` | 4.827 | 1.200 | 1.558 |
| `scores2-1536x769-float32` | `354c8441157449` | 66.367 | 58.850 | 53.554 |
| `scores2-96x97-bfloat16` | `354c8441157449` | 4.015 | 1.223 | 1.428 |
| `scores2-1164x2048-float16` | `354c8441157449` | 73.712 | 58.725 | 52.878 |
| `scores2-12x257-float16` | `354c8441157449` | 3.955 | 1.299 | 1.396 |
| `scores2-1536x128-float16` | `354c8441157449` | 4.627 | 2.942 | 1.938 |
| `scores-96x769-float32` | `354c8441157449` | 4.372 | 2.068 | 2.195 |
| `scores2-1536x97-bfloat16` | `354c8441157449` | 4.626 | 2.999 | 2.998 |
| `scores2-384x1024-bfloat16` | `354c8441157449` | 8.083 | 3.475 | 3.635 |
| `scores2-1536x2048-float32` | `354c8441157449` | 252.463 | 179.893 | 180.012 |
| `scores2-1164x2048-bfloat16` | `354c8441157449` | 74.288 | 56.082 | 53.618 |
| `scores-1164x257-bfloat16` | `acdf484aa79dc2` | 6.643 | 2.904 | 2.867 |
| `scores2-24x1024-bfloat16` | `acdf484aa79dc2` | 5.823 | 1.428 | 1.811 |
| `scores2-24x128-float32` | `acdf484aa79dc2` | 3.091 | 1.841 | 1.169 |
| `scores2-384x128-float16` | `8804f866e0fd99` | 4.064 | 1.344 | 1.298 |
| `scores-24x512-bfloat16` | `beefdd4cc4f63a` | 4.371 | 1.267 | 1.462 |
| `scores2-1164x769-float16` | `beefdd4cc4f63a` | 14.340 | 9.843 | 11.535 |
| `scores2-12x128-bfloat16` | `beefdd4cc4f63a` | 3.859 | 1.201 | 1.171 |
| `scores2-1164x97-float32` | `a1606ea0e2fda0` | 4.372 | 2.835 | 2.804 |
| `scores-384x97-float16` | `a1606ea0e2fda0` | 3.895 | 1.346 | 1.651 |
| `scores2-1536x257-float16` | `a1606ea0e2fda0` | 7.632 | 5.011 | 4.850 |
| `scores-12x2048-float16` | `a1606ea0e2fda0` | 8.595 | 2.835 | 1.811 |
| `scores2-1164x769-bfloat16` | `a1606ea0e2fda0` | 14.167 | 9.492 | 11.057 |
| `scores2-12x8192-bfloat16` | `4b50fb03bd796a` | 5.906 | 3.378 | 4.722 |
| `scores2-1536x1024-bfloat16` | `8bdb1aee47d4da` | 31.315 | 26.721 | 24.962 |
| `scores2-1164x257-float16` | `8bdb1aee47d4da` | 6.643 | 4.130 | 5.011 |
| `scores-12x1024-float16` | `8bdb1aee47d4da` | 5.875 | 1.338 | 1.554 |
| `scores-96x2048-float32` | `8bdb1aee47d4da` | 7.186 | 3.346 | 2.963 |
| `scores2-96x1024-float16` | `8bdb1aee47d4da` | 6.035 | 1.811 | 2.067 |
| `scores2-12x8192-float16` | `8bdb1aee47d4da` | 5.907 | 3.380 | 4.657 |
| `scores-1536x2048-float16` | `8bdb1aee47d4da` | 95.711 | 29.699 | 28.878 |
| `scores2-384x8192-float16` | `8bdb1aee47d4da` | 115.954 | 87.076 | 78.805 |
| `scores2-24x769-float32` | `8bdb1aee47d4da` | 3.992 | 1.681 | 1.779 |
| `scores2-96x8192-float32` | `8bdb1aee47d4da` | 33.636 | 24.456 | 26.801 |
| `scores-12x97-float32` | `8bdb1aee47d4da` | 3.220 | 1.170 | 1.139 |
| `scores-24x769-float32` | `8bdb1aee47d4da` | 3.987 | 1.583 | 1.906 |
| `scores2-24x769-bfloat16` | `8bdb1aee47d4da` | 5.395 | 1.549 | 1.679 |
| `scores2-96x512-float32` | `8bdb1aee47d4da` | 3.603 | 1.779 | 1.810 |
| `scores2-1164x128-float16` | `55d647ceb02340` | 4.498 | 2.323 | 1.794 |
| `scores-24x769-float16` | `55d647ceb02340` | 5.433 | 1.427 | 1.683 |
| `scores2-12x4096-float32` | `29ab755b687d33` | 4.371 | 2.652 | 2.963 |
| `scores2-24x128-bfloat16` | `29ab755b687d33` | 3.731 | 1.600 | 1.153 |
| `scores-1536x512-float32` | `29ab755b687d33` | 23.597 | 5.138 | 5.418 |
| `scores-384x8192-float32` | `29ab755b687d33` | 241.946 | 114.265 | 95.741 |
| `scores-96x512-float32` | `29ab755b687d33` | 3.603 | 1.683 | 1.554 |
| `scores-24x769-bfloat16` | `29ab755b687d33` | 5.420 | 1.425 | 1.555 |
| `scores2-1164x2048-float32` | `d7ea7af237cdee` | 185.762 | 137.219 | 140.512 |
| `scores2-1536x769-bfloat16` | `d7ea7af237cdee` | 21.507 | 15.653 | 17.826 |
| `scores-12x2048-bfloat16` | `d7ea7af237cdee` | 8.593 | 2.931 | 1.811 |
| `scores-1164x512-float32` | `d7ea7af237cdee` | 14.946 | 4.372 | 4.281 |
| `scores2-12x8192-float32` | `d7ea7af237cdee` | 5.939 | 4.211 | 4.684 |
| `scores2-96x512-bfloat16` | `d7ea7af237cdee` | 4.480 | 1.427 | 1.555 |
| `scores2-12x1024-float32` | `d7ea7af237cdee` | 4.211 | 1.553 | 1.811 |
| `scores-1164x97-float32` | `d7ea7af237cdee` | 4.372 | 2.704 | 1.682 |
| `scores2-12x2048-bfloat16` | `d7ea7af237cdee` | 8.592 | 3.015 | 2.067 |
| `scores-24x2048-bfloat16` | `d7ea7af237cdee` | 8.593 | 2.957 | 1.936 |
| `scores-12x512-float32` | `d7ea7af237cdee` | 3.403 | 1.299 | 1.426 |
| `scores-1536x1024-float16` | `d7ea7af237cdee` | 27.507 | 6.289 | 6.035 |
| `scores2-96x97-float16` | `d7ea7af237cdee` | 3.826 | 1.266 | 1.491 |
| `scores-96x4096-bfloat16` | `d7ea7af237cdee` | 8.579 | 3.859 | 3.765 |
| `scores-384x257-bfloat16` | `d7ea7af237cdee` | 4.371 | 1.907 | 2.061 |
| `scores2-96x128-bfloat16` | `d7ea7af237cdee` | 4.018 | 1.170 | 1.235 |
| `scores-12x8192-float16` | `d7ea7af237cdee` | 5.911 | 3.347 | 2.963 |
| `scores-1536x769-float16` | `9b98496740e225` | 18.112 | 7.794 | 7.841 |
| `scores2-1536x128-float32` | `9b98496740e225` | 5.267 | 2.960 | 2.675 |
| `scores2-96x257-bfloat16` | `9b98496740e225` | 3.955 | 1.427 | 1.652 |
| `scores-1536x257-float32` | `9b98496740e225` | 8.949 | 3.860 | 3.473 |
| `scores-1164x257-float16` | `9b98496740e225` | 6.578 | 2.961 | 2.472 |
| `scores2-24x4096-bfloat16` | `9b98496740e225` | 4.799 | 2.509 | 3.344 |
| `scores2-1164x1024-float16` | `9b98496740e225` | 19.120 | 12.720 | 11.823 |
| `scores2-24x4096-float16` | `9b98496740e225` | 4.851 | 2.547 | 2.836 |
| `scores2-384x257-float32` | `9b98496740e225` | 4.339 | 2.488 | 2.336 |
| `scores-96x2048-float16` | `9b98496740e225` | 9.134 | 3.187 | 2.450 |
| `scores-12x512-bfloat16` | `9b98496740e225` | 4.372 | 1.204 | 1.457 |
| `scores-1164x97-bfloat16` | `9b98496740e225` | 4.372 | 2.291 | 1.779 |
| `scores2-12x128-float32` | `9b98496740e225` | 3.346 | 1.170 | 1.139 |
| `scores-1164x769-float16` | `9b98496740e225` | 14.354 | 6.259 | 7.347 |
| `scores2-24x1024-float16` | `9b98496740e225` | 5.825 | 1.428 | 1.729 |
| `scores2-96x1024-bfloat16` | `9b98496740e225` | 6.080 | 1.906 | 2.225 |
| `scores2-96x769-bfloat16` | `9b98496740e225` | 5.619 | 2.035 | 2.705 |
| `scores-384x1024-float16` | `9b98496740e225` | 8.082 | 3.017 | 2.836 |
| `scores2-12x257-float32` | `9b98496740e225` | 3.219 | 1.299 | 1.427 |
| `scores-96x512-float16` | `9b98496740e225` | 4.499 | 1.427 | 1.555 |
| `scores-384x2048-bfloat16` | `9b98496740e225` | 13.571 | 6.386 | 4.500 |
| `scores-384x8192-float16` | `9b98496740e225` | 101.330 | 31.734 | 29.968 |
| `scores-384x257-float32` | `9b98496740e225` | 4.243 | 2.067 | 1.939 |
| `scores-24x97-float32` | `9b98496740e225` | 3.275 | 1.170 | 1.171 |
| `scores-12x128-float16` | `7e06bedbffaf67` | 3.858 | 1.139 | 1.139 |
| `scores-96x769-bfloat16` | `7e06bedbffaf67` | 5.646 | 1.744 | 2.001 |
| `scores-24x97-float16` | `7e06bedbffaf67` | 3.858 | 1.171 | 1.171 |
| `scores2-24x8192-bfloat16` | `f9f6b2632665c3` | 6.290 | 3.474 | 4.882 |
| `scores2-384x128-float32` | `f9f6b2632665c3` | 3.315 | 1.524 | 1.427 |
| `scores2-12x1024-float16` | `f9f6b2632665c3` | 5.779 | 1.395 | 1.683 |
| `scores-96x512-bfloat16` | `f9f6b2632665c3` | 4.467 | 1.427 | 1.554 |
| `scores-12x4096-bfloat16` | `f9f6b2632665c3` | 4.821 | 2.324 | 2.381 |
| `scores2-12x2048-float32` | `351981d945009d` | 5.580 | 3.090 | 2.228 |
| `scores-384x512-float32` | `351981d945009d` | 5.520 | 2.384 | 2.323 |
| `scores2-12x97-float32` | `351981d945009d` | 3.187 | 1.171 | 1.171 |
| `scores-12x97-float16` | `351981d945009d` | 3.859 | 1.170 | 1.268 |
| `scores-24x128-bfloat16` | `351981d945009d` | 3.729 | 1.171 | 1.140 |
| `scores-24x1024-float32` | `351981d945009d` | 4.240 | 1.680 | 1.812 |
| `scores-1536x512-bfloat16` | `351981d945009d` | 10.385 | 3.187 | 3.443 |
| `scores-1536x2048-float32` | `351981d945009d` | 242.105 | 108.692 | 96.065 |
| `scores2-384x2048-bfloat16` | `351981d945009d` | 13.461 | 6.163 | 6.766 |
| `scores-12x1024-bfloat16` | `351981d945009d` | 5.875 | 1.392 | 1.555 |
| `scores2-384x769-float16` | `351981d945009d` | 7.024 | 4.114 | 4.910 |
| `scores2-96x257-float16` | `351981d945009d` | 3.955 | 1.427 | 1.428 |
| `scores-384x97-float32` | `351981d945009d` | 3.721 | 1.780 | 1.426 |
| `scores-1164x2048-bfloat16` | `351981d945009d` | 68.493 | 16.033 | 11.827 |
| `scores-12x769-float16` | `351981d945009d` | 5.520 | 1.395 | 1.683 |
| `scores2-12x769-bfloat16` | `351981d945009d` | 5.427 | 1.459 | 1.843 |
| `scores-1536x769-bfloat16` | `351981d945009d` | 17.793 | 7.826 | 8.816 |
| `scores2-12x512-float16` | `351981d945009d` | 4.369 | 1.203 | 1.427 |
| `scores-24x257-float32` | `351981d945009d` | 3.218 | 1.299 | 1.298 |
| `scores2-384x97-float16` | `351981d945009d` | 3.859 | 1.428 | 1.553 |
| `scores2-24x257-bfloat16` | `351981d945009d` | 3.987 | 1.298 | 1.683 |
| `scores-384x97-bfloat16` | `351981d945009d` | 3.859 | 1.331 | 1.555 |
| `scores2-384x4096-bfloat16` | `351981d945009d` | 41.382 | 24.386 | 27.992 |
| `scores2-384x512-bfloat16` | `351981d945009d` | 5.139 | 2.195 | 2.291 |
| `scores2-1164x1024-bfloat16` | `924612bf2fc197` | 19.968 | 12.018 | 13.823 |
| `scores-24x97-bfloat16` | `d3a2b2ba8d045f` | 3.858 | 1.171 | 1.170 |
| `scores2-384x2048-float16` | `d3a2b2ba8d045f` | 13.836 | 6.160 | 6.847 |
| `scores-12x128-float32` | `d3a2b2ba8d045f` | 3.187 | 1.139 | 1.045 |
| `scores2-1536x2048-bfloat16` | `d3a2b2ba8d045f` | 112.513 | 92.914 | 89.515 |
| `scores-96x1024-bfloat16` | `d3a2b2ba8d045f` | 6.074 | 1.779 | 1.970 |
| `scores-12x4096-float16` | `d3a2b2ba8d045f` | 4.627 | 2.578 | 2.291 |
| `scores2-12x1024-bfloat16` | `d3a2b2ba8d045f` | 5.826 | 1.395 | 1.726 |
| `scores-12x8192-bfloat16` | `d3a2b2ba8d045f` | 5.906 | 3.475 | 4.176 |
| `scores-96x1024-float32` | `d3a2b2ba8d045f` | 4.754 | 2.233 | 2.323 |
| `scores2-384x512-float16` | `d3a2b2ba8d045f` | 5.139 | 2.195 | 2.195 |
| `scores2-24x97-bfloat16` | `d3a2b2ba8d045f` | 3.858 | 1.170 | 1.684 |
| `scores2-1164x769-float32` | `d3a2b2ba8d045f` | 48.163 | 35.597 | 33.464 |
| `scores-1536x2048-bfloat16` | `d3a2b2ba8d045f` | 105.016 | 28.432 | 29.132 |
| `scores2-1536x1024-float16` | `d3a2b2ba8d045f` | 29.760 | 25.504 | 25.153 |
| `scores-96x128-bfloat16` | `d3a2b2ba8d045f` | 3.732 | 1.171 | 1.215 |
| `scores2-384x1024-float16` | `d3a2b2ba8d045f` | 8.082 | 3.443 | 3.888 |
| `scores-24x4096-float16` | `d3a2b2ba8d045f` | 4.851 | 2.707 | 2.612 |
| `scores-24x4096-bfloat16` | `d3a2b2ba8d045f` | 4.813 | 2.451 | 2.451 |
| `scores2-96x4096-float32` | `d3a2b2ba8d045f` | 10.100 | 4.532 | 6.361 |
| `scores-24x2048-float16` | `d3a2b2ba8d045f` | 8.690 | 2.931 | 1.908 |
| `scores-24x8192-bfloat16` | `d3a2b2ba8d045f` | 6.289 | 3.761 | 4.682 |
| `scores-384x128-float32` | `d3a2b2ba8d045f` | 3.346 | 1.395 | 1.298 |
| `scores2-1164x257-float32` | `d3a2b2ba8d045f` | 7.447 | 4.626 | 4.850 |
| `scores-96x4096-float32` | `d3a2b2ba8d045f` | 10.161 | 4.370 | 4.150 |
| `scores2-24x97-float16` | `d3a2b2ba8d045f` | 3.860 | 1.169 | 1.270 |
| `scores2-24x2048-bfloat16` | `d3a2b2ba8d045f` | 8.612 | 3.058 | 2.227 |
| `scores-1164x769-bfloat16` | `d3a2b2ba8d045f` | 14.737 | 6.163 | 7.857 |
| `scores-1536x769-float32` | `d3a2b2ba8d045f` | 60.112 | 14.671 | 14.401 |
| `scores2-24x512-float32` | `7c9241c1e568bc` | 3.444 | 1.425 | 1.457 |
| `scores2-96x1024-float32` | `7c9241c1e568bc` | 4.754 | 2.578 | 2.899 |
| `scores2-1164x97-float16` | `7c9241c1e568bc` | 4.339 | 2.547 | 2.707 |
| `scores-384x769-float32` | `7c9241c1e568bc` | 7.786 | 3.859 | 4.240 |
| `scores2-1536x1024-float32` | `7c9241c1e568bc` | 109.330 | 88.161 | 75.940 |
| `scores-1164x769-float32` | `7c9241c1e568bc` | 31.936 | 9.109 | 8.993 |
| `scores-96x769-float16` | `7c9241c1e568bc` | 5.618 | 1.714 | 1.941 |
| `scores2-384x257-float16` | `7c9241c1e568bc` | 4.339 | 2.195 | 2.451 |
| `scores-96x257-bfloat16` | `7c9241c1e568bc` | 3.956 | 1.395 | 1.396 |
| `scores-1164x97-float16` | `7c9241c1e568bc` | 4.370 | 2.291 | 2.163 |
| `scores2-96x8192-float16` | `7c9241c1e568bc` | 13.507 | 5.521 | 6.781 |
| `scores-96x257-float16` | `7c9241c1e568bc` | 3.892 | 1.395 | 1.350 |
| `scores-96x8192-bfloat16` | `7c9241c1e568bc` | 13.631 | 5.999 | 7.194 |
| `scores-384x769-float16` | `02bba4166db6de` | 6.979 | 3.443 | 3.347 |
| `scores-1536x97-bfloat16` | `02bba4166db6de` | 4.595 | 2.579 | 1.905 |
| `scores2-384x4096-float32` | `02bba4166db6de` | 109.743 | 86.882 | 87.070 |
| `scores-12x4096-float32` | `02bba4166db6de` | 4.372 | 2.707 | 2.707 |
| `scores-12x2048-float32` | `02bba4166db6de` | 5.555 | 2.674 | 1.938 |
| `scores2-1164x512-float32` | `02bba4166db6de` | 16.769 | 11.742 | 13.246 |
| `scores-384x257-float16` | `02bba4166db6de` | 4.340 | 1.908 | 1.972 |
| `scores-96x257-float32` | `02bba4166db6de` | 3.346 | 1.427 | 1.332 |
| `scores2-384x769-bfloat16` | `02bba4166db6de` | 6.931 | 4.083 | 5.615 |
| `scores2-96x769-float16` | `02bba4166db6de` | 5.617 | 2.067 | 2.252 |
| `scores2-24x769-float16` | `02bba4166db6de` | 5.449 | 1.552 | 1.811 |
| `scores2-384x2048-float32` | `02bba4166db6de` | 30.850 | 24.609 | 26.300 |
| `scores-12x257-bfloat16` | `02bba4166db6de` | 3.891 | 1.267 | 1.395 |
| `scores2-1536x512-bfloat16` | `02bba4166db6de` | 10.397 | 5.267 | 5.136 |
| `scores2-96x97-float32` | `02bba4166db6de` | 3.090 | 1.266 | 1.205 |
| `scores-1536x512-float16` | `02bba4166db6de` | 10.484 | 3.092 | 3.637 |
| `scores-384x1024-float32` | `02bba4166db6de` | 8.594 | 4.083 | 4.114 |
| `scores2-24x1024-float32` | `02bba4166db6de` | 4.243 | 1.794 | 2.032 |
| `scores2-24x2048-float32` | `02bba4166db6de` | 5.687 | 3.090 | 2.291 |
| `scores2-12x128-float16` | `02bba4166db6de` | 4.031 | 1.165 | 1.137 |
| `scores-1536x257-bfloat16` | `02bba4166db6de` | 7.635 | 3.343 | 3.125 |
| `scores-96x1024-float16` | `02bba4166db6de` | 6.064 | 1.779 | 1.938 |
| `scores-1536x97-float16` | `02bba4166db6de` | 4.627 | 2.674 | 1.811 |
| `scores2-384x4096-float16` | `02bba4166db6de` | 40.707 | 23.533 | 24.307 |
| `scores-24x1024-float16` | `02bba4166db6de` | 5.875 | 1.427 | 1.651 |
| `scores-384x2048-float32` | `02bba4166db6de` | 24.548 | 6.899 | 6.385 |
| `scores-1164x512-float16` | `02bba4166db6de` | 8.979 | 2.804 | 3.091 |
| `scores2-96x512-float16` | `02bba4166db6de` | 4.498 | 1.428 | 1.554 |
| `scores2-24x4096-float32` | `02bba4166db6de` | 4.754 | 2.995 | 3.346 |
| `scores2-1536x257-bfloat16` | `02bba4166db6de` | 7.571 | 5.011 | 4.849 |
| `scores-12x512-float16` | `02bba4166db6de` | 4.371 | 1.203 | 1.428 |
| `scores-1164x1024-float32` | `02bba4166db6de` | 58.465 | 12.098 | 13.844 |
| `scores2-1536x97-float32` | `02bba4166db6de` | 5.010 | 4.754 | 3.162 |
| `scores-96x97-float32` | `02bba4166db6de` | 3.091 | 1.681 | 1.425 |
| `scores-96x8192-float16` | `02bba4166db6de` | 13.476 | 5.776 | 6.801 |
| `scores2-1164x257-bfloat16` | `02bba4166db6de` | 6.644 | 4.241 | 5.009 |
| `scores2-384x97-float32` | `02bba4166db6de` | 3.643 | 1.683 | 1.554 |
| `scores-1164x2048-float16` | `02bba4166db6de` | 67.106 | 15.987 | 17.282 |
| `scores2-384x1024-float32` | `02bba4166db6de` | 9.104 | 5.715 | 5.790 |
| `scores2-12x769-float32` | `02bba4166db6de` | 3.985 | 1.556 | 1.933 |
| `scores-12x128-bfloat16` | `4aa3fee495d695` | 3.859 | 1.139 | 1.139 |
| `scores2-384x769-float32` | `4aa3fee495d695` | 7.828 | 4.881 | 5.009 |
| `scores-24x128-float16` | `4aa3fee495d695` | 3.955 | 1.267 | 1.139 |
| `scores2-96x128-float16` | `4aa3fee495d695` | 3.731 | 1.170 | 1.169 |
| `scores2-384x97-bfloat16` | `4aa3fee495d695` | 3.954 | 1.427 | 1.555 |
| `scores-12x769-float32` | `4aa3fee495d695` | 3.987 | 1.428 | 1.779 |
| `scores2-12x512-bfloat16` | `4aa3fee495d695` | 4.370 | 1.203 | 1.524 |
| `scores-1536x128-bfloat16` | `4aa3fee495d695` | 4.627 | 2.354 | 1.684 |
| `scores2-1164x1024-float32` | `4aa3fee495d695` | 62.097 | 57.537 | 51.410 |
| `scores-96x2048-bfloat16` | `4aa3fee495d695` | 9.156 | 3.202 | 2.548 |
| `scores2-24x512-bfloat16` | `4aa3fee495d695` | 4.370 | 1.267 | 1.554 |
| `scores2-12x257-bfloat16` | `4aa3fee495d695` | 3.955 | 1.298 | 1.555 |
| `scores2-24x8192-float32` | `4aa3fee495d695` | 7.060 | 4.339 | 5.011 |
| `scores2-1536x97-float16` | `4aa3fee495d695` | 4.625 | 3.000 | 2.963 |
| `scores-24x257-bfloat16` | `4aa3fee495d695` | 4.042 | 1.299 | 1.425 |
| `scores2-1536x512-float32` | `4aa3fee495d695` | 30.243 | 25.410 | 24.164 |
| `scores-24x512-float16` | `4aa3fee495d695` | 4.370 | 1.267 | 1.427 |
| `scores-96x4096-float16` | `4aa3fee495d695` | 8.594 | 3.730 | 3.533 |
| `scores2-12x4096-float16` | `4aa3fee495d695` | 4.626 | 2.323 | 2.579 |
| `scores-24x128-float32` | `4aa3fee495d695` | 3.091 | 1.652 | 1.105 |
| `scores-24x512-float32` | `4aa3fee495d695` | 3.472 | 1.395 | 1.492 |
| `scores2-12x4096-bfloat16` | `4aa3fee495d695` | 4.632 | 2.578 | 2.707 |
| `scores-1164x512-bfloat16` | `4aa3fee495d695` | 8.848 | 2.803 | 3.187 |
| `scores-96x97-float16` | `4aa3fee495d695` | 3.858 | 1.202 | 1.170 |
| `scores2-1536x128-bfloat16` | `4aa3fee495d695` | 4.626 | 2.674 | 1.938 |
| `scores-384x4096-float16` | `4aa3fee495d695` | 32.488 | 10.776 | 8.051 |
| `scores-24x257-float16` | `4aa3fee495d695` | 3.984 | 1.296 | 1.460 |
| `scores2-24x8192-float16` | `4aa3fee495d695` | 6.291 | 3.474 | 4.626 |
| `scores2-24x512-float16` | `4aa3fee495d695` | 4.371 | 1.267 | 1.555 |
| `scores2-24x97-float32` | `4aa3fee495d695` | 3.187 | 1.171 | 1.169 |
| `scores-1536x128-float16` | `4aa3fee495d695` | 4.672 | 2.420 | 1.811 |
| `scores-384x769-bfloat16` | `4aa3fee495d695` | 6.941 | 3.443 | 3.731 |
| `scores-96x97-bfloat16` | `4aa3fee495d695` | 3.827 | 1.703 | 1.171 |
| `scores2-24x257-float16` | `4aa3fee495d695` | 3.955 | 1.298 | 1.683 |
| `scores-384x4096-float32` | `4aa3fee495d695` | 98.126 | 25.762 | 26.191 |
| `scores2-96x2048-float16` | `4aa3fee495d695` | 9.150 | 3.218 | 3.128 |
| `scores2-1164x97-bfloat16` | `4aa3fee495d695` | 4.368 | 2.547 | 2.322 |
| `scores-384x512-bfloat16` | `4aa3fee495d695` | 5.192 | 1.810 | 1.938 |
| `scores-12x1024-float32` | `4aa3fee495d695` | 4.211 | 1.523 | 1.683 |
| `scores2-96x4096-bfloat16` | `4aa3fee495d695` | 9.116 | 3.502 | 4.498 |
| `scores2-96x4096-float16` | `4aa3fee495d695` | 8.564 | 3.860 | 4.165 |
| `scores2-12x769-float16` | `4aa3fee495d695` | 5.395 | 1.459 | 1.779 |
| `scores2-12x512-float32` | `4aa3fee495d695` | 3.441 | 1.298 | 1.555 |
| `scores-12x257-float16` | `4aa3fee495d695` | 3.955 | 1.269 | 1.427 |
| `scores2-96x2048-bfloat16` | `4aa3fee495d695` | 9.107 | 3.315 | 2.962 |
| `scores-96x8192-float32` | `4aa3fee495d695` | 27.090 | 7.570 | 7.411 |
| `scores-1164x2048-float32` | `4aa3fee495d695` | 174.225 | 69.213 | 58.294 |
| `scores2-1164x128-bfloat16` | `4aa3fee495d695` | 4.498 | 2.123 | 1.792 |
| `scores-96x128-float32` | `4aa3fee495d695` | 3.091 | 1.200 | 1.140 |
| `scores2-96x8192-bfloat16` | `4aa3fee495d695` | 13.637 | 7.046 | 7.316 |
| `scores2-1164x512-float16` | `4aa3fee495d695` | 8.850 | 4.403 | 4.498 |
| `scores2-384x512-float32` | `4aa3fee495d695` | 5.520 | 3.090 | 2.961 |
| `scores-1536x1024-bfloat16` | `4aa3fee495d695` | 23.444 | 6.162 | 9.008 |
| `scores-384x512-float16` | `4aa3fee495d695` | 5.139 | 1.844 | 1.936 |
| `scores-12x257-float32` | `4aa3fee495d695` | 3.219 | 1.299 | 1.298 |
| `scores2-12x97-float16` | `4aa3fee495d695` | 3.858 | 1.170 | 1.425 |
| `scores-1164x128-float16` | `4aa3fee495d695` | 4.498 | 2.195 | 1.556 |
| `scores2-1164x512-bfloat16` | `4aa3fee495d695` | 8.852 | 4.323 | 4.399 |
| `scores-12x8192-float32` | `4aa3fee495d695` | 5.939 | 3.987 | 3.859 |
| `scores-1164x1024-float16` | `4aa3fee495d695` | 17.504 | 5.333 | 6.528 |
| `scores2-384x8192-float32` | `4aa3fee495d695` | 258.054 | 179.089 | 184.259 |
| `scores-12x97-bfloat16` | `4aa3fee495d695` | 3.859 | 1.169 | 1.396 |
| `scores2-96x128-float32` | `4aa3fee495d695` | 3.091 | 1.172 | 1.172 |
| `scores2-1536x257-float32` | `4aa3fee495d695` | 8.595 | 5.265 | 6.408 |
| `scores-24x8192-float32` | `4aa3fee495d695` | 7.041 | 4.240 | 4.113 |
| `scores-1164x128-float32` | `4aa3fee495d695` | 4.752 | 2.067 | 1.683 |
| `scores-1536x97-float32` | `4aa3fee495d695` | 5.019 | 2.707 | 2.110 |
| `scores-384x1024-bfloat16` | `4aa3fee495d695` | 8.083 | 2.963 | 3.475 |
| `scores-384x2048-float16` | `4aa3fee495d695` | 13.503 | 6.292 | 4.690 |
| `scores-1164x257-float32` | `4aa3fee495d695` | 7.411 | 3.344 | 2.962 |
| `scores2-96x2048-float32` | `4aa3fee495d695` | 7.219 | 3.731 | 3.727 |
| `scores-12x769-bfloat16` | `4aa3fee495d695` | 5.491 | 1.395 | 1.715 |
| `scores-24x4096-float32` | `5e2df4ae6b63e8` | 4.762 | 2.707 | 3.139 |
| `scores-384x8192-bfloat16` | `5e2df4ae6b63e8` | 99.854 | 30.376 | 32.433 |
| `scores2-96x769-float32` | `5e2df4ae6b63e8` | 4.370 | 2.572 | 2.355 |
| `scores-1164x128-bfloat16` | `5e2df4ae6b63e8` | 4.498 | 2.066 | 1.650 |
| `scores-1536x1024-float32` | `5e2df4ae6b63e8` | 99.924 | 26.721 | 26.175 |
| `scores-1536x128-float32` | `5e2df4ae6b63e8` | 5.267 | 2.803 | 1.906 |
| `scores2-12x2048-float16` | `5e2df4ae6b63e8` | 8.595 | 2.931 | 2.067 |
| `scores2-1164x128-float32` | `5e2df4ae6b63e8` | 4.755 | 2.487 | 2.324 |
| `scores-1536x257-float16` | `5e2df4ae6b63e8` | 7.575 | 3.380 | 3.380 |
| `scores-24x2048-float32` | `5e2df4ae6b63e8` | 5.747 | 2.675 | 2.065 |
| `scores2-384x8192-bfloat16` | `b9df74c8b7c340` | 116.849 | 89.025 | 82.444 |
| `scores-24x1024-bfloat16` | `b9df74c8b7c340` | 5.875 | 1.426 | 1.683 |
| `scores2-384x128-bfloat16` | `b9df74c8b7c340` | 3.826 | 1.352 | 1.299 |
| `scores2-12x97-bfloat16` | `b9df74c8b7c340` | 3.859 | 1.609 | 1.267 |
| `scores2-96x257-float32` | `b9df74c8b7c340` | 3.346 | 1.555 | 1.427 |
| `scores2-24x128-float16` | `b9df74c8b7c340` | 3.763 | 1.170 | 1.169 |
| `scores2-24x257-float32` | `b9df74c8b7c340` | 3.216 | 1.393 | 1.425 |
