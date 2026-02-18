# Kernel Orbit

Run CUDA/Triton kernels and Jupyter notebooks on [Modal](https://modal.com) GPUs from VS Code.

## What it does

- **Kernel files:** Run `.cu` (CUDA) and `.py` (Triton) files on remote GPUs with benchmarking, profiling, and detailed metrics.
- **Jupyter notebooks:** Select "Modal GPU (Python)" as your notebook kernel and run cells on a remote GPU. State persists across cells; the container stays warm for 15 minutes between runs.
- **GPU picker:** Choose from T4, L4, A10G, A100, L40S, H100, H200, B200.
- **Sessions sidebar:** See active GPU sessions and kill them from the Kernel Orbit panel.

## How to use

### Setup

1. Install [uv](https://docs.astral.sh/uv/)
2. `uv add modal`
3. Get API tokens from [modal.com/settings](https://modal.com/settings)
4. Create a `.env` in your project root:
   ```
   MODAL_TOKEN_ID=ak-...
   MODAL_TOKEN_SECRET=as-...
   ```

### Kernels (.cu / .py)

1. Open a `.cu` or `.py` file
2. `Cmd+Shift+R` (Mac) / `Ctrl+Shift+R` (Win/Linux)
3. Results appear in the Results Panel

### Notebooks (.ipynb)

1. Open a `.ipynb` file
2. Select **Modal GPU (Python)** from the kernel picker — the container starts warming immediately
3. Run cells normally — they execute on the remote GPU
4. State persists across cells; container stays warm for 15 min of inactivity

### GPU Selection

Click the GPU name in the status bar or run **Kernel Orbit: Select GPU Type** from the command palette.

## Configuration

Open VS Code Settings and search for "Modal Kernel" to customize:

- **Timeout** (default: 3600s = 1 hour): Maximum time a single notebook cell can run
  - Increase for long training runs (max: 86400s = 24 hours)
  - Decrease for quick experiments to catch runaway cells
- **Default GPU** (default: T4): GPU type to use by default
- **GPU Count** (default: 1): Number of GPUs to attach (1-8)
- **Warmup/Benchmark Runs**: Configure profiling behavior for kernel files
- **Sync Files** (default: true): Sync local workspace files to the remote GPU container
- **Max Sync File Size MB** (default: 100): Files larger than this are skipped with a warning
- **Sync Exclude Patterns**: Additional glob patterns to exclude from sync

## Notebook Features

### File sync

Local workspace files are synced to the remote container before each cell execution. Files ≤ 100 MB are synced automatically; larger files trigger a warning with alternatives (`!wget` on GPU or Modal Volume).

### Package installation

`!pip install`, `%pip install`, and `!pip3 install` are automatically proxied through `uv pip install --system` for faster installs. You can also use `%uv pip install` directly.

### Supported magics

`%time`, `%timeit`, `%%time`, `%%timeit`, `%matplotlib`, `%pip`, `%uv`, `!` shell commands.

## Roadmap

- [ ] `input()` support (`input_request` / `input_reply`)
- [ ] Top-level `await` (async cell execution)
- [ ] More IPython magics (`%cd`, `%pwd`, `%env`, `%who`)
- [ ] `tqdm` progress bar support
- [ ] `ipywidgets` support
- [ ] `IPython.display.clear_output()`
- [ ] `%%bash`, `%%html`, `%%javascript` cell magics
<!-- - [ ] Tab completion / intellisense for remote objects -->
<!-- - [ ] Variable inspector (browse remote namespace) -->
<!-- - [ ] `?` / `??` introspection (docstrings) -->
<!-- - [ ] Kernel info request (language, version metadata) -->

## Development

```bash
git clone https://github.com/kashifulhaque/kernel-orbit
cd kernel-orbit
npm install
# Press F5 in VS Code to launch the extension development host
```
