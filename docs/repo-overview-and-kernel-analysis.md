# Kernel Orbit: Repo Overview and CUDA/Triton Kernel Analysis

## What this repo does

`kernel-orbit` is a VS Code extension that lets users run GPU workloads remotely on Modal, with a focus on:

- Running CUDA (`.cu`, `.cuh`) kernels on selected NVIDIA GPUs.
- Running Triton (`.py`) kernels on selected NVIDIA GPUs.
- Running Jupyter notebook cells on persistent remote GPU sessions.
- Showing execution and profiling results inside VS Code.

In practice, this repo is a bridge between:

1. **VS Code UX** (commands, status bar, tree views, notebook controller, webviews), and
2. **Remote GPU execution** (Modal Python scripts that compile/run/profile code).


## High-level architecture

### 1) Extension host (TypeScript)

Main files:

- `src/extension.ts`: command registration, activation flow, status bar, UI wiring.
- `src/modalRunner.ts`: kernel run orchestration (`uv run modal run ...`), Modal auth checks, env loading.
- `src/profilingRunner.ts`: profiling orchestration (`kernel_profiler.py`).
- `src/notebookController.ts`: persistent notebook sessions, per-cell execution, file sync, interrupts.
- `src/resultsPanel.ts` and `src/profilingPanel.ts`: webview rendering for run/profiling results.
- `src/kernelDecorationProvider.ts`: inline decorations on kernel definitions using profiling metrics.
- `src/gpuPickerProvider.ts` and `src/sessionTreeProvider.ts`: side-panel trees for GPU selection and active sessions.
- `src/types.ts`: shared data model for run/profiling/session payloads.

### 2) Remote execution layer (Python + Modal)

Main files:

- `modal_scripts/kernel_runner.py`: unified CUDA/Triton runner for benchmarking + metrics.
- `modal_scripts/kernel_profiler.py`: CUDA profiling (`ncu` with fallback) and Triton profiling (`torch.profiler`).
- `modal_scripts/notebook_runner.py`: long-lived notebook session class + interactive JSON protocol.
- `modal_scripts/run_cuda_kernel.py` and `modal_scripts/run_triton_kernel.py`: standalone scripts (currently not used by the extension command path).

### 3) Packaging and release

- `package.json`: VS Code extension metadata, commands, settings, keybindings.
- `esbuild.js`: extension bundling.
- `.github/workflows/release.yml`: CI builds VSIX and publishes a release on `main`.


## End-to-end execution flows

### A) Kernel run flow (`Run Kernel on Modal`)

1. User triggers command in `src/extension.ts`.
2. `ModalRunner` checks Modal install/auth and resolves credentials from `.env` or shell.
3. `ModalRunner` maps extension to kernel type: `.cu/.cuh -> cuda`, `.py -> triton`.
4. Extension spawns:
   - `uv run modal run modal_scripts/kernel_runner.py --kernel-file ... --kernel-type ... --gpu ...`.
5. `kernel_runner.py` dispatches to per-GPU Modal function and runs either:
   - `_run_cuda_kernel_impl(...)`, or
   - `_run_triton_kernel_impl(...)`.
6. Result JSON is written to a temp file, read back by the extension, normalized to camelCase, shown in `ResultsPanel`.

### B) Kernel profiling flow (`Profile Kernel on GPU`)

1. User triggers profile command.
2. `ProfilingRunner` launches:
   - `uv run modal run modal_scripts/kernel_profiler.py --kernel-file ...`.
3. `kernel_profiler.py` routes to CUDA or Triton profiler:
   - CUDA: `ncu --set basic --csv`, fallback to `nvcc --resource-usage` + timing + NVML sampling when `ncu` is not usable.
   - Triton: `torch.profiler`.
4. Results are rendered in `ProfilingPanel` and inline decorations are applied in source.

### C) Notebook flow (`Modal GPU (Python)`)

1. User selects notebook kernel in VS Code.
2. Extension starts `notebook_runner.py --interactive --gpu ...`.
3. Script sends JSON-line protocol events (`ready`, `stream`, `display`, `result`, `sync_complete`, etc.).
4. For each cell:
   - workspace files are synced (incremental hash-based),
   - cell code executes in persistent namespace inside Modal class session,
   - outputs stream progressively,
   - session stays warm for 15 minutes (`scaledown_window=900`).


## CUDA/Triton kernel runtime deep dive

This section answers the six requested questions for the kernel execution path.

---

## 1) What does it do?

The CUDA/Triton path provides a **remote execution harness** that:

- Accepts a local kernel file (`.cu` or Triton `.py`).
- Runs it on a selected Modal GPU (T4 through B200).
- Measures warmup and benchmark timings.
- Captures some GPU telemetry (memory/utilization/temp/power).
- Returns structured JSON to VS Code for visualization and export.

Profiling path additionally returns:

- Per-kernel timing/efficiency/occupancy-style metrics.
- Raw profiler output.
- Optional timeline samples (fallback CUDA path).

---

## 2) How does it do it?

### Dispatch and transport

- Extension side (`src/modalRunner.ts`, `src/profilingRunner.ts`) shells out to `uv run modal run ...`.
- Kernel source is read locally and passed to Modal script.
- Modal script selects per-GPU function (`@app.function(gpu="...")`) and returns JSON.

### CUDA execution (`modal_scripts/kernel_runner.py`)

- Writes source into temp `.cu`.
- Compiles with `nvcc -O3 -lineinfo`.
- Runs compiled executable for warmup and benchmark loops.
- Measures run times using host-side `time.perf_counter()`.
- Captures stdout/stderr from executable and basic telemetry via NVML / `nvidia-smi`.

### Triton execution (`modal_scripts/kernel_runner.py`)

- Executes user Python source with `exec(...)` in a prepared globals dict (`torch`, `triton`, `tl`, `numpy`).
- Tries to find callable entrypoint in order: `benchmark`, `benchmark_kernel`, `main`, `run`, `test`.
- Runs warmup + benchmark loops and synchronizes CUDA before/after timing.
- Captures stdout and torch CUDA memory counters.

### CUDA profiling (`modal_scripts/kernel_profiler.py`)

- Compiles with `nvcc --resource-usage`.
- Tries `ncu --set basic --csv`.
- Parses CSV metrics into normalized schema.
- If `ncu` fails or yields no usable metrics, falls back to:
  - parsed resource usage,
  - sampled NVML telemetry during runs,
  - estimated occupancy/throughput heuristics.

### Triton profiling (`modal_scripts/kernel_profiler.py`)

- Executes source with `exec(...)`.
- Finds callable entrypoint and profiles single invocation with `torch.profiler`.
- Produces per-event table and maps CUDA events to kernel metrics skeleton.

---

## 3) What existing thing can be improved?

Key improvement opportunities found in current implementation:

1. **`gpuCount` setting is not effectively wired for kernel runs**
   - Exposed in settings (`package.json`) but not propagated through the main runner path.
   - Effect: user-facing setting can be misleading.

2. **A100-40GB Triton mapping inconsistency**
   - In `kernel_runner.py`, `run_triton_on_a100_40gb` passes `"A100"` into result metadata instead of `"A100-40GB"`.
   - Effect: inconsistent reporting and potential downstream confusion.

3. **`.py` is always treated as Triton**
   - `ModalRunner.detectKernelType()` maps every `.py` to Triton.
   - Effect: plain Python scripts can be misclassified and fail in non-obvious ways.

4. **Cancellation is incomplete for non-notebook kernel runs**
   - Progress UI is marked cancellable, but spawned process cancellation is not wired.
   - Effect: user clicks cancel but remote run may continue.

5. **`shell: true` process spawning for command execution**
   - Used in multiple spawns, increasing quoting fragility and command-injection surface.
   - Effect: path edge-cases and robustness/security risk.

6. **CUDA benchmark timing includes process startup overhead**
   - Timing around repeated executable launches may measure wrapper/process overhead, not just kernel runtime.
   - Effect: noisy or inflated benchmark numbers.

7. **Notebook file sync hash state can become stale on sync failure**
   - Hash map updates before remote write success is guaranteed.
   - Effect: false "up to date" state can mask unsynced changes.

8. **No handling of deletions in notebook sync**
   - Sync pushes changed/new files but does not remove deleted remote files.
   - Effect: stale files may affect execution determinism.

9. **Low automated test coverage for critical flows**
   - Existing tests mainly cover `parse_ncu_csv`; extension tests are placeholder.
   - Effect: regressions are likely in run/session/error paths.

10. **Duplicate/legacy scripts create maintenance drift risk**
   - `run_cuda_kernel.py` and `run_triton_kernel.py` are not in the extension's primary path.
   - Effect: parallel logic can diverge.

---

## 4) What is missing?

From a production robustness standpoint, missing pieces include:

- **A formal kernel contract** (what functions/signatures are expected for Triton scripts).
- **Deterministic cancellation semantics** (what gets cancelled locally vs remotely, and when).
- **Structured error taxonomy** (compile error vs runtime error vs transport error vs infra error).
- **Retry policy with backoff** for transient Modal/network failures.
- **Session health and heartbeat model** for long notebook sessions.
- **Sync integrity protocol** (manifest, acknowledgements, delete propagation).
- **End-to-end integration tests** for run/profile/notebook happy paths and failure modes.
- **Result schema versioning** to avoid silent breakage between Python and TS models.

---

## 5) How can that be implemented?

Recommended phased implementation:

### Phase 1: Correctness fixes (quick, high impact)

1. Fix Triton A100-40GB metadata mismatch in `kernel_runner.py`.
2. Wire `gpuCount` setting through extension -> runner args -> Modal function selection (or remove setting until supported).
3. Remove `shell: true` and pass args safely via process spawn arrays.
4. Add explicit return code checks for CUDA warmup/benchmark subprocess runs.
5. Clarify `.py` handling:
   - either require explicit command choice (`Run Triton Kernel`),
   - or add lightweight source detection (`@triton.jit`, `import triton`) before classifying.

### Phase 2: Execution model hardening

1. Add cancellation plumbing:
   - keep child process handle in `ModalRunner` / `ProfilingRunner`,
   - on cancellation token, terminate process and surface cancellation state.
2. Improve CUDA timing fidelity:
   - prefer parsing in-program CUDA event timings when present,
   - or run benchmark loop inside one process invocation and return per-run timings from program output.
3. Strengthen notebook sync integrity:
   - update local hash cache only after `sync_complete`,
   - support remote delete list for removed files.
4. Add structured run IDs and correlation IDs across TS + Python logs.

### Phase 3: Robust operations and testability

1. Introduce shared JSON schema module and version field in payloads.
2. Add retry wrapper for transient Modal transport failures.
3. Build integration test matrix:
   - CUDA compile failure,
   - Triton missing entrypoint,
   - ncu unavailable fallback path,
   - interrupted notebook execution,
   - sync failure/recovery.
4. Consolidate unused standalone scripts or clearly label them as deprecated/examples.

---

## 6) Most important: how can it be made robust?

Robustness should be designed as a system, not a single patch. The highest-value blueprint:

1. **Define explicit contracts**
   - Contract for kernel inputs/outputs, profiling payloads, and notebook protocol messages.
   - Add schema validation on both Python producer and TS consumer.

2. **Make state transitions explicit and observable**
   - Standard state machine per run/session: `queued -> starting -> running -> syncing -> finishing -> completed/failed/cancelled`.
   - Emit structured events with run IDs for every transition and failure.

3. **Treat cancellation and timeouts as first-class**
   - Cancellation must terminate local process and remote execution path deterministically.
   - Distinguish `timeout`, `cancelled`, and `failed` in UI and stored history.

4. **Harden transport and sync layers**
   - Retry transient failures with bounded backoff.
   - Use integrity checks (manifest hash + ack).
   - Propagate file deletions and detect sync drift.

5. **Separate infra failures from user-code failures**
   - Compile/runtime errors from kernel code should not look like platform outages.
   - Improve error classification and display actionable remediation per class.

6. **Test resilience paths continuously**
   - Add CI integration tests for failure-in-the-loop scenarios.
   - Keep fallback profiler behavior tested since it is expected in constrained environments.

7. **Reduce divergence and hidden behavior**
   - Minimize duplicate runner logic.
   - Keep one canonical execution/profiling path used by extension and tests.

If these seven areas are implemented, Kernel Orbit will move from a good prototype to a more production-grade remote GPU execution system.


## Notes on current kernel runner behavior

- Primary extension-run paths currently use:
  - `modal_scripts/kernel_runner.py`
  - `modal_scripts/kernel_profiler.py`
  - `modal_scripts/notebook_runner.py`
- `modal_scripts/run_cuda_kernel.py` and `modal_scripts/run_triton_kernel.py` exist but are not referenced by extension command flow.
