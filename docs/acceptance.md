# Reproduction and evidence contract

1. **Frontend**: export both residual-visibility variants. Kernel sources differ in their output list and stores; graph input, IR, lowering and backend are otherwise shared. Matmul is rejected by standalone compilation. In mixed graphs, keep an operator on the PyTorch path when there is no legal insertion point or an unsupported alias might escape.
2. **Numerics**: compare matching dtype, device and shape, including every returned tensor, with explicit per-dtype tolerances. A returned residual must match eager bit for bit on the sampled finite inputs. Cover odd widths, offset-but-contiguous inputs, multiple outputs, parameters, pytree outputs and a non-default CUDA stream.
3. **Performance**: compare against eager and static/fullgraph TorchInductor on the same device, tensors and dtype. The device series uses 9 groups of 4 replays of a 16-invocation graph; it excludes host dispatch and can favor cache residency. The synchronized host series includes fresh output allocation, Python dispatch and completion. Profiling, when enabled, is outside timing; peak allocated bytes include returned outputs and allocator rounding. IR global intermediate bytes are a design property, not a hardware counter.
4. **Search**: training, schedule confirmation and three-way comparison use different seeds. A candidate replaces the default only if it wins at least 4 of 5 paired holdout rounds and has at least 3% median gain. This is a heuristic, not a confidence interval. Never drop losing or failed workload cases from the denominator. Inductor compilation/first call and our source build/load are different operations; caches can be warm.
5. **Real module check**: with `transformers==4.51.3` installed, run `python tools/qwen_integration.py --output /tmp/qwen.json`. Hidden size 1536, intermediate size 8960. The test uses the package's actual classes; no checkpoint is downloaded. See `reports/qwen-integration.json` for both before/after graphs.

## Overnight campaign

From the repository root, set `SPARKFUSE_PYTHON` to an existing CUDA-enabled Python interpreter. Run `tools/start-campaign.sh --run RUN-ID --until 2026-11-26T09:00:00+08:00`. It serializes GPU workload cases, validates candidates, stores a SQLite dispatcher, benches the selected candidate with new inputs, and writes each successful full JSON atomically. The supervisor bounds each worker and enforces the absolute deadline, including terminating an unfinished worker. Five consecutive worker failures or a low-disk guard stop the search early with a recorded reason.

- `.runtime/RUN-ID/index.jsonl`: append-only case index, including failures. Preserve this, `evidence/` and `dispatch.sqlite`; they are results, not disposable compiler caches.
- `reports/RUN-ID-status.json`: atomic progress with the actual stop reason. Rerunning the same ID resumes at unrecorded cases.
- `python tools/report.py .runtime/RUN-ID /tmp/report.md`: rebuild the table from the index, including slower cases. Raw times and schedules are in `evidence/`.
- `.cache/`: generated, safe to clean **after** all compilation and benchmarking processes exit.
- `sparkfuse.tuning.compile_tuned(ep, db_path)`: uses a matching measured entry, otherwise compiles the default schedule. `kernel.prepare(*args)` reuses the same outputs on subsequent calls; copy results if they must outlive a replay. The normal `kernel(*args)` allocates fresh outputs.

### Bounded token batches

`sparkfuse.dynamic.DynamicRows(ep)` accepts an export with one bounded `torch.export.Dim` shared by the leading axis of rank-2 activations. Widths and weights are static. It verifies the symbol identity in exported tensor metadata, guards every runtime shape and passes actual rows to the existing CUDA launch. This is not symbolic expression compilation. Four graph families are tested at 1, 2, 31, 32, 47, 48, 97 and 129 rows after disabling the compiler entry point; row 130 and inconsistent input rows must raise. Expanded and strided inputs are not silently copied.

### Diagnostics and failures

`tools/verify.sh` serializes the test suite against search workers. `tools/sanitize.sh` runs memcheck, racecheck and synccheck on four generated kernels (filtered by their mangled entry name), with odd widths, warp tails and deliberately offset pointers. Initial output is preserved in `reports/sanitizer-gb10.txt`. These tools test specific paths; a clean run does not prove arbitrary code free of races.

A three-way comparison can fail even when SparkFuse matches eager. In one BF16 decomposed RMSNorm case (`512 × 1536`, comparison seed 2261), SparkFuse matched every element; TorchInductor exceeded the predeclared tolerance at three indices. `reports/decomp-bf16-parity-investigation.json` contains the indices and tolerances. It remains a failed three-way comparison, not a speed win. New workers retain a `.rejected.json` report and identify the failing backend. A selected SparkFuse schedule which fails the third input is removed from dispatch.

With `SPARKFUSE_FINAL_CHECK=1`, the overnight supervisor reserves a near-deadline acceptance pass (`pytest`, and optionally the real Qwen integration when `SPARKFUSE_QWEN_PY` is set), resumes benchmarks with the remaining time, and generates a final comparison table. A validation report contains individual exit codes; a completed job is not automatically a passed release.

### Optional ATen host dispatch and tail coverage

`from sparkfuse.native import native_dispatch; call = native_dispatch(compile_export(ep))` builds the launch adapter against the running Torch ABI, CUDA libraries and Python development headers; a working C++17 compiler is required. A build failure is explicit. Use `compile_export(ep)` directly if the optional host adapter is unavailable. Both paths allocate fresh outputs on normal calls. For mixed graphs pass `host_backend="native"` to `partition_export`; `tools/qwen_integration.py --host-backend native --output /tmp/qwen-native.json` exercises that path. Kernel, stream, borrowed input lifetime, non-contiguous/shape guards and pytree behavior are covered by the tests.

The expanded sanitizer fixture includes seven paths on deliberately offset allocations, odd-width reductions and cross-row packed pointwise access. See `reports/sanitizer-tail-gb10.txt`. [Performance evidence](performance.md) separates host dispatch, device replay with different address footprints, and odd-width scheduling.
