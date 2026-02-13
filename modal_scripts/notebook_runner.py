"""
Session-based Modal runner for Jupyter notebook cells.

Architecture:
  - Uses @app.cls with scaledown_window=900 (15 min) so the GPU container
    stays alive between cell executions — no cold starts after the first cell.
  - State (Python namespace) persists across cells on the container, keyed by
    a unique session_id so multiple notebooks can safely share one container.
  - The local entrypoint runs an interactive JSON-line protocol over stdin/stdout
    so the VS Code extension spawns ONE process per notebook session instead of
    one per cell.
  - A worker thread handles remote execution while the main thread reads stdin,
    enabling soft interrupt support without killing the session.

Protocol (stdin -> stdout):
  -> {"action":"execute","code":"print(1)"}
  <- {"type":"stream","stream":"stdout","text":"1\\n"}
  <- {"type":"result","successful":true,"execution_time_ms":1.2,...}

  -> {"action":"interrupt"}
  <- {"type":"interrupted"}

  -> {"action":"reset"}
  <- {"type":"reset","successful":true}

  -> {"action":"terminate"}
  <- {"type":"terminated"}

  (On startup, before the interactive loop)
  <- {"type":"ready","gpu":"T4","gpu_name":"NVIDIA T4","session_id":"..."}
"""

import io
import os
import ast
import sys
import json
import time
import modal
import base64
import queue
import traceback
import threading
import contextlib
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

# Force UTF-8 for stdout/stderr on Windows to avoid charmap encoding errors
if sys.platform == "win32":
  os.environ.setdefault("PYTHONUTF8", "1")
  if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
  if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# Build the image and include the current module for serialization support
notebook_image = (
  modal.Image.debian_slim(python_version="3.12")
  .uv_pip_install(
    "torch",
    "triton",
    "numpy",
    "matplotlib",
    "pandas",
    "Pillow",
    "nvidia-ml-py",
  )
  .add_local_file(
    Path(__file__).resolve(),
    remote_path="/root/notebook_runner.py",
    copy=True
  )
  .env({"PYTHONPATH": "/root"})  # Make notebook_runner importable
)

app = modal.App("kernel-orbit-notebook")


def get_gpu_info() -> Dict[str, Any]:
  """Get detailed GPU information using nvidia-ml-py."""
  try:
    from pynvml import (
      nvmlInit,
      nvmlShutdown,
      nvmlDeviceGetHandleByIndex,
      nvmlDeviceGetName,
      nvmlDeviceGetMemoryInfo,
      nvmlDeviceGetTemperature,
      NVML_TEMPERATURE_GPU,
      nvmlDeviceGetPowerUsage,
      nvmlDeviceGetCudaComputeCapability,
      nvmlDeviceGetUtilizationRates,
    )

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    name = nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
      name = name.decode("utf-8")

    info = {
      "name": name,
      "memory_total_mb": nvmlDeviceGetMemoryInfo(handle).total / (1024**2),
      "memory_used_mb": nvmlDeviceGetMemoryInfo(handle).used / (1024**2),
      "memory_free_mb": nvmlDeviceGetMemoryInfo(handle).free / (1024**2),
      "temperature_c": nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU),
      "power_draw_w": nvmlDeviceGetPowerUsage(handle) / 1000.0,
      "compute_capability": ".".join(
        map(str, nvmlDeviceGetCudaComputeCapability(handle))
      ),
      "utilization_percent": nvmlDeviceGetUtilizationRates(handle).gpu,
    }
    nvmlShutdown()
    return info
  except Exception as e:
    return {"error": str(e), "name": "Unknown"}


def _process_magics(code: str) -> str:
  """Transform IPython magics into executable Python.

  Cell magics (%%): transform the entire cell.
  Line magics (%) and shell (!) commands: transform individual lines.
  Returns valid Python source ready for ast.parse().
  """
  lines = code.split("\n")
  first_stripped = lines[0].strip() if lines else ""

  # ── Cell magics ──────────────────────────────────────────────────────
  if first_stripped.startswith("%%"):
    body = "\n".join(lines[1:])
    body_repr = repr(body)

    if first_stripped == "%%time":
      return (
        "import time as __tt\n"
        "__tt_start = __tt.perf_counter()\n"
        f"{body}\n"
        "__tt_elapsed = __tt.perf_counter() - __tt_start\n"
        "if __tt_elapsed >= 1:\n"
        "    print(f'Wall time: {__tt_elapsed:.2f} s')\n"
        "elif __tt_elapsed >= 1e-3:\n"
        "    print(f'Wall time: {__tt_elapsed*1e3:.2f} ms')\n"
        "else:\n"
        "    print(f'Wall time: {__tt_elapsed*1e6:.0f} µs')\n"
      )

    if first_stripped.startswith("%%timeit"):
      n_loops, n_repeat = 7, 3
      parts = first_stripped.split()
      for i, p in enumerate(parts):
        if p == "-n" and i + 1 < len(parts):
          try: n_loops = int(parts[i + 1])
          except ValueError: pass
        if p == "-r" and i + 1 < len(parts):
          try: n_repeat = int(parts[i + 1])
          except ValueError: pass
      return (
        "import time as __tt\n"
        f"__tt_code = {body_repr}\n"
        "__tt_times = []\n"
        f"for __tt_rep in range({n_repeat}):\n"
        "    __tt_batch = []\n"
        f"    for __tt_i in range({n_loops}):\n"
        "        __tt_s = __tt.perf_counter()\n"
        "        exec(__tt_code, globals())\n"
        "        __tt_batch.append(__tt.perf_counter() - __tt_s)\n"
        "    __tt_times.append(min(__tt_batch))\n"
        "__tt_mean = sum(__tt_times) / len(__tt_times)\n"
        "__tt_std = (sum((__t - __tt_mean)**2 for __t in __tt_times) / max(len(__tt_times)-1, 1))**0.5 if len(__tt_times) > 1 else 0\n"
        "if __tt_mean >= 1:\n"
        f"    print(f'{{__tt_mean:.2f}} s ± {{__tt_std:.2f}} s per loop (mean ± std. dev. of {{len(__tt_times)}} runs, {n_loops} loops each)')\n"
        "elif __tt_mean >= 1e-3:\n"
        f"    print(f'{{__tt_mean*1e3:.2f}} ms ± {{__tt_std*1e3:.2f}} ms per loop (mean ± std. dev. of {{len(__tt_times)}} runs, {n_loops} loops each)')\n"
        "else:\n"
        f"    print(f'{{__tt_mean*1e6:.0f}} µs ± {{__tt_std*1e6:.0f}} µs per loop (mean ± std. dev. of {{len(__tt_times)}} runs, {n_loops} loops each)')\n"
      )

    return f"# [unsupported cell magic] {first_stripped}\n{body}"

  # ── Line magics and shell commands ───────────────────────────────────
  transformed: list[str] = []
  for line in lines:
    stripped = line.lstrip()
    indent = line[: len(line) - len(stripped)]

    if stripped.startswith("!"):
      cmd = stripped[1:]
      transformed.append(f"{indent}__shell_exec__({repr(cmd)})")
    elif stripped.startswith("%matplotlib"):
      transformed.append(f"{indent}pass  # matplotlib backend configured (Agg)")
    elif stripped.startswith("%uv "):
      args = stripped[4:].strip()
      transformed.append(f"{indent}__uv_run__({repr(args)})")
    elif stripped.startswith("%timeit "):
      stmt = stripped[8:].strip()
      transformed.append(f"{indent}__magic_timeit__({repr(stmt)}, globals())")
    elif stripped.startswith("%time "):
      stmt = stripped[6:].strip()
      transformed.append(f"{indent}__magic_time__({repr(stmt)}, globals())")
    elif stripped.startswith("%"):
      transformed.append(f"{indent}# [unsupported magic] {stripped}")
    else:
      transformed.append(line)

  return "\n".join(transformed)


def _create_namespace() -> Dict[str, Any]:
  """Create a fresh Python namespace with helpers for magics and display."""
  import matplotlib

  matplotlib.use("Agg")

  # Display output collector
  display_outputs: list[Dict[str, str]] = []

  def _display_func(*objs, **kwargs):
    """IPython-compatible display() function."""
    for obj in objs:
      _capture_rich_display(obj, display_outputs)

  def _capture_rich_display(obj, outputs):
    """Capture rich repr from an object into the outputs list."""
    for attr, mime in [
      ("_repr_html_", "text/html"),
      ("_repr_svg_", "image/svg+xml"),
      ("_repr_latex_", "text/latex"),
      ("_repr_markdown_", "text/markdown"),
      ("_repr_json_", "application/json"),
      ("_repr_png_", "image/png"),
    ]:
      if hasattr(obj, attr):
        try:
          data = getattr(obj, attr)()
          if data:
            if mime == "application/json" and not isinstance(data, str):
              data = json.dumps(data)
            outputs.append({"mime": mime, "data": str(data)})
            return
        except Exception:
          pass
    # Check for PIL Image
    try:
      from PIL import Image as PILImage

      if isinstance(obj, PILImage.Image):
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
        buf.seek(0)
        outputs.append(
          {"mime": "image/png", "data": base64.b64encode(buf.read()).decode()}
        )
        return
    except (ImportError, Exception):
      pass
    # Fallback to text/plain
    outputs.append({"mime": "text/plain", "data": repr(obj)})

  # Shell execution helper
  def _shell_exec(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
      print(result.stdout, end="")
    if result.stderr:
      import sys as _sys

      print(result.stderr, end="", file=_sys.stderr)
    if result.returncode != 0 and not result.stderr:
      import sys as _sys

      print(f"Process exited with code {result.returncode}", file=_sys.stderr)

  # uv run helper
  def _uv_run(args_str):
    import shlex

    cmd = ["uv"] + shlex.split(args_str)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
      print(result.stdout, end="")
    if result.stderr:
      import sys as _sys

      print(result.stderr, end="", file=_sys.stderr)

  # %time helper
  def _magic_time(stmt, ns):
    start = time.perf_counter()
    exec(compile(stmt, "<magic>", "exec"), ns)
    elapsed = time.perf_counter() - start
    if elapsed >= 1:
      print(f"Wall time: {elapsed:.2f} s")
    elif elapsed >= 1e-3:
      print(f"Wall time: {elapsed*1e3:.2f} ms")
    else:
      print(f"Wall time: {elapsed*1e6:.0f} µs")

  # %timeit helper
  def _magic_timeit(stmt, ns, n_loops=7, n_repeat=3):
    compiled = compile(stmt, "<magic>", "exec")
    times = []
    for _ in range(n_repeat):
      batch = []
      for _ in range(n_loops):
        s = time.perf_counter()
        exec(compiled, ns)
        batch.append(time.perf_counter() - s)
      times.append(min(batch))
    mean = sum(times) / len(times)
    std = (
      (sum((t - mean) ** 2 for t in times) / max(len(times) - 1, 1)) ** 0.5
      if len(times) > 1
      else 0
    )
    if mean >= 1:
      print(
        f"{mean:.2f} s ± {std:.2f} s per loop "
        f"(mean ± std. dev. of {len(times)} runs, {n_loops} loops each)"
      )
    elif mean >= 1e-3:
      print(
        f"{mean*1e3:.2f} ms ± {std*1e3:.2f} ms per loop "
        f"(mean ± std. dev. of {len(times)} runs, {n_loops} loops each)"
      )
    else:
      print(
        f"{mean*1e6:.0f} µs ± {std*1e6:.0f} µs per loop "
        f"(mean ± std. dev. of {len(times)} runs, {n_loops} loops each)"
      )

  ns: Dict[str, Any] = {
    "__name__": "__main__",
    "__builtins__": __builtins__,
    "_display_outputs": display_outputs,
    "display": _display_func,
    "__shell_exec__": _shell_exec,
    "__uv_run__": _uv_run,
    "__magic_time__": _magic_time,
    "__magic_timeit__": _magic_timeit,
  }

  # Install mock IPython.display module
  _install_ipython_display_mock(ns)

  return ns


def _install_ipython_display_mock(namespace: Dict[str, Any]) -> None:
  """Install a lightweight IPython.display mock into sys.modules."""
  import types

  class _HTML:
    def __init__(self, data=""):
      self._data = data

    def _repr_html_(self):
      return self._data

  class _Markdown:
    def __init__(self, data=""):
      self._data = data

    def _repr_markdown_(self):
      return self._data

  class _SVG:
    def __init__(self, data=""):
      self._data = data

    def _repr_svg_(self):
      return self._data

  class _Latex:
    def __init__(self, data=""):
      self._data = data

    def _repr_latex_(self):
      return self._data

  class _JSON:
    def __init__(self, data=None):
      self._data = data

    def _repr_json_(self):
      return self._data

  class _Image:
    def __init__(
      self, data=None, url=None, filename=None, format=None, width=None, height=None
    ):
      self._data = data
      self._format = format or "png"

    def _repr_png_(self):
      if isinstance(self._data, bytes):
        return base64.b64encode(self._data).decode()
      return None

  # Create mock modules
  ipython_mod = types.ModuleType("IPython")
  display_mod = types.ModuleType("IPython.display")

  display_mod.display = namespace["display"]
  display_mod.HTML = _HTML
  display_mod.Markdown = _Markdown
  display_mod.SVG = _SVG
  display_mod.Latex = _Latex
  display_mod.JSON = _JSON
  display_mod.Image = _Image

  ipython_mod.display = display_mod

  sys.modules.setdefault("IPython", ipython_mod)
  sys.modules.setdefault("IPython.display", display_mod)


def _execute_cell_in_namespace(
  namespace: Dict[str, Any], cell_code: str
) -> Dict[str, Any]:
  """Execute a single cell inside an existing namespace, capturing all output."""

  result: Dict[str, Any] = {
    "successful": False,
    "stdout": "",
    "stderr": "",
    "error": None,
    "error_traceback": None,
    "images": [],
    "html": [],
    "svg": [],
    "latex": [],
    "markdown": [],
    "json_outputs": [],
    "display_outputs": [],
    "result_repr": None,
    "execution_time_ms": 0.0,
  }

  # Clear display outputs from previous cell
  if "_display_outputs" in namespace:
    namespace["_display_outputs"].clear()
  else:
    namespace["_display_outputs"] = []

  start_time = time.perf_counter()
  stdout_capture = io.StringIO()
  stderr_capture = io.StringIO()

  try:
    processed_code = _process_magics(cell_code)
    tree = ast.parse(processed_code)
    last_expr_node: Optional[ast.Expr] = None

    if tree.body and isinstance(tree.body[-1], ast.Expr):
      last_expr_node = tree.body.pop()

    with (
      contextlib.redirect_stdout(stdout_capture),
      contextlib.redirect_stderr(stderr_capture),
    ):
      if tree.body:
        exec(compile(tree, "<cell>", "exec"), namespace)

      if last_expr_node is not None:
        expr = ast.Expression(last_expr_node.value)
        ast.fix_missing_locations(expr)
        last_value = eval(compile(expr, "<cell>", "eval"), namespace)

        if last_value is not None:
          _format_last_value(last_value, result)

    result["execution_time_ms"] = (time.perf_counter() - start_time) * 1000
    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()

    # Collect matplotlib figures
    try:
      import matplotlib.pyplot as plt

      for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        result["images"].append(base64.b64encode(buf.read()).decode("utf-8"))
        plt.close(fig)
    except Exception:
      pass

    # Collect mid-cell display() outputs
    result["display_outputs"] = list(namespace.get("_display_outputs", []))
    if "_display_outputs" in namespace:
      namespace["_display_outputs"].clear()

    result["successful"] = True

  except Exception as e:
    result["execution_time_ms"] = (time.perf_counter() - start_time) * 1000
    result["error"] = str(e)
    result["error_traceback"] = traceback.format_exc()
    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()

  return result


def _format_last_value(value: Any, result: Dict[str, Any]) -> None:
  """Format the last expression value using the richest available representation."""
  # Check rich repr methods in priority order
  for attr, key in [
    ("_repr_html_", "html"),
    ("_repr_svg_", "svg"),
    ("_repr_latex_", "latex"),
    ("_repr_markdown_", "markdown"),
    ("_repr_json_", "json_outputs"),
    ("_repr_png_", "images"),
  ]:
    if hasattr(value, attr):
      try:
        data = getattr(value, attr)()
        if data:
          if key == "json_outputs" and not isinstance(data, str):
            data = json.dumps(data)
          result[key].append(str(data))
          return
      except Exception:
        pass

  # Pandas check
  try:
    import pandas as pd

    if isinstance(value, (pd.DataFrame, pd.Series)):
      result["html"].append(value.to_html())
      return
  except ImportError:
    pass

  # Fallback
  result["result_repr"] = repr(value)


# Session class defined at global scope with serialized=True
# GPU type and timeout are configured at instantiation time
# The module is baked into the image via add_local_file()
@app.cls(image=notebook_image, scaledown_window=900, serialized=True)
class Session:
  """
  Modal session class for notebook execution.
  
  Each class maintains per-session namespaces (keyed by session_id) so multiple
  notebooks can share one container without leaking state.
  
  GPU type and timeout are specified when spawning via app.cls parameters.
  """
  
  @modal.enter()
  def setup(self):
    self.namespaces: Dict[str, Dict] = {}

  @modal.method()
  def execute_cell(self, session_id: str, cell_code: str) -> Dict:
    if session_id not in self.namespaces:
      self.namespaces[session_id] = _create_namespace()
    return _execute_cell_in_namespace(self.namespaces[session_id], cell_code)

  @modal.method()
  def reset(self, session_id: str) -> Dict:
    self.namespaces[session_id] = _create_namespace()
    return {"successful": True}

  @modal.method()
  def cleanup(self, session_id: str) -> Dict:
    self.namespaces.pop(session_id, None)
    return {"successful": True}

  @modal.method()
  def get_info(self) -> Dict:
    return get_gpu_info()


def get_session_with_gpu(gpu_type: str, timeout_seconds: int):
  """Get a Session instance configured with the specified GPU and timeout."""
  return Session.with_options(gpu=gpu_type, timeout=timeout_seconds)()


SUPPORTED_GPUS = ["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "L40S", "H100", "H200", "B200"]


@app.function(gpu="T4", image=notebook_image, timeout=60)
def warmup_notebook_t4() -> str:
  """Warmup function to pre-build the notebook image."""
  return "Notebook image ready on T4"




_send_lock = threading.Lock()
_interrupt_event = threading.Event()
_command_queue: queue.Queue = queue.Queue()


def _send(msg: Dict[str, Any]) -> None:
  """Write a JSON message to stdout (thread-safe)."""
  with _send_lock:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _send_progressive_outputs(result: Dict[str, Any]) -> None:
  """Send cell outputs as individual messages for progressive display."""
  # stdout
  if result.get("stdout"):
    _send({"type": "stream", "stream": "stdout", "text": result["stdout"]})

  # stderr (only for successful execution; errors are sent with the result)
  if result.get("stderr") and result.get("successful"):
    _send({"type": "stream", "stream": "stderr", "text": result["stderr"]})

  # Mid-cell display() outputs
  for item in result.get("display_outputs", []):
    _send({"type": "display", "mime": item["mime"], "data": item["data"]})

  # Matplotlib images
  for img in result.get("images", []):
    _send({"type": "display", "mime": "image/png", "data": img})

  # HTML
  for h in result.get("html", []):
    _send({"type": "display", "mime": "text/html", "data": h})

  # SVG
  for s in result.get("svg", []):
    _send({"type": "display", "mime": "image/svg+xml", "data": s})

  # LaTeX
  for tex in result.get("latex", []):
    _send({"type": "display", "mime": "text/latex", "data": tex})

  # Markdown
  for md in result.get("markdown", []):
    _send({"type": "display", "mime": "text/markdown", "data": md})

  # JSON
  for j in result.get("json_outputs", []):
    _send({"type": "display", "mime": "application/json", "data": j})

  # Final metadata-only result
  meta: Dict[str, Any] = {
    "type": "result",
    "successful": result.get("successful", False),
    "execution_time_ms": result.get("execution_time_ms", 0),
    "result_repr": result.get("result_repr"),
    "error": result.get("error"),
    "error_traceback": result.get("error_traceback"),
  }
  # Include stderr with error on failure
  if result.get("stderr") and not result.get("successful"):
    meta["stderr"] = result["stderr"]
  _send(meta)


def _worker(session, session_id: str) -> None:
  """Worker thread: processes commands from the queue sequentially."""
  while True:
    cmd = _command_queue.get()
    if cmd is None:  # Poison pill
      break

    action = cmd.get("action")

    if action == "execute":
      _interrupt_event.clear()
      try:
        result = session.execute_cell.remote(session_id, cmd["code"])
      except Exception as e:
        if not _interrupt_event.is_set():
          _send({"type": "error", "message": str(e)})
        continue

      if _interrupt_event.is_set():
        # Discard result — interrupt was already acknowledged
        continue

      _send_progressive_outputs(result)

    elif action == "reset":
      try:
        session.reset.remote(session_id)
        _send({"type": "reset", "successful": True})
      except Exception as e:
        _send({"type": "error", "message": str(e)})

    elif action == "terminate":
      try:
        session.cleanup.remote(session_id)
      except Exception:
        pass
      _send({"type": "terminated"})
      break


@app.local_entrypoint()
def main(
  gpu: str = "T4",
  interactive: bool = False,
  warmup_images: bool = False,
  timeout: int = 3600,
):
  """
  Notebook session manager.

  In --interactive mode (the default for the extension), reads JSON commands
  from stdin and writes JSON responses to stdout.  The first message is
  always ``{"type":"ready", ...}`` once the GPU container is warm.

  A worker thread handles remote execution so the main thread can read stdin
  concurrently, enabling soft interrupt without killing the session.

  Args:
    gpu: GPU type (T4, L4, A10G, A100, etc.)
    interactive: Run in interactive mode for VS Code extension
    warmup_images: Pre-build container images
    timeout: Maximum execution time per cell in seconds (default: 3600 = 1 hour)
  """
  if warmup_images:
    print("Pre-building notebook image …")
    result = warmup_notebook_t4.remote()
    print(f"  {result}")
    print("Notebook image is now cached!")
    return

  if not interactive:
    print("Error: pass --interactive for session mode", file=sys.stderr)
    sys.exit(1)

  if gpu not in SUPPORTED_GPUS:
    _send({"type": "error", "message": f"Unsupported GPU: {gpu}. Supported: {', '.join(SUPPORTED_GPUS)}"})
    sys.exit(1)

  # Create session with user-specified GPU and timeout
  import uuid

  session = get_session_with_gpu(gpu, timeout)
  session_id = str(uuid.uuid4())

  try:
    gpu_info = session.get_info.remote()
    _send(
      {
        "type": "ready",
        "gpu": gpu,
        "gpu_name": gpu_info.get("name", "Unknown"),
        "session_id": session_id,
      }
    )
  except Exception as e:
    _send({"type": "error", "message": f"Failed to start session: {e}"})
    sys.exit(1)

  # Start worker thread for executing remote calls
  worker = threading.Thread(
    target=_worker, args=(session, session_id), daemon=True
  )
  worker.start()

  # Main thread: read stdin commands (non-blocking w.r.t. execution)
  while True:
    line = sys.stdin.readline()
    if not line:  # EOF — extension killed the process
      break
    line = line.strip()
    if not line:
      continue

    try:
      cmd = json.loads(line)
      action = cmd.get("action")

      if action == "interrupt":
        _interrupt_event.set()
        _send({"type": "interrupted"})
        continue

      if action == "terminate":
        _command_queue.put(cmd)
        break

      _command_queue.put(cmd)

    except Exception as e:
      _send({"type": "error", "message": str(e)})

  # Signal worker to exit and wait
  _command_queue.put(None)
  worker.join(timeout=5)
