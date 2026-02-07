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

Protocol (stdin -> stdout):
  -> {"action":"execute","code":"print(1)"}
  <- {"type":"result","successful":true,"stdout":"1\\n",...}

  -> {"action":"reset"}
  <- {"type":"reset","successful":true}

  -> {"action":"terminate"}
  <- {"type":"terminated"}

  (On startup, before the interactive loop)
  <- {"type":"ready","gpu":"T4","gpu_name":"NVIDIA T4","session_id":"..."}
"""

import io
import ast
import sys
import json
import time
import modal
import base64
import traceback
import contextlib
from typing import Dict, Any, Optional
from pathlib import Path


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


def preprocess_cell_code(code: str) -> str:
  """Strip IPython magic commands (%, %%, !) that aren't valid Python."""
  lines = code.split("\n")
  cleaned: list[str] = []
  for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("%") or stripped.startswith("!"):
      cleaned.append(f"# [stripped] {line}")
      continue
    cleaned.append(line)
  return "\n".join(cleaned)


def _create_namespace() -> Dict[str, Any]:
  """Create a fresh Python namespace for exec/eval."""
  import matplotlib

  matplotlib.use("Agg")
  return {"__name__": "__main__", "__builtins__": __builtins__}


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
    "result_repr": None,
    "execution_time_ms": 0.0,
  }

  start_time = time.perf_counter()
  stdout_capture = io.StringIO()
  stderr_capture = io.StringIO()

  try:
    processed_code = preprocess_cell_code(cell_code)
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
          try:
            import pandas as pd

            if isinstance(last_value, (pd.DataFrame, pd.Series)):
              result["html"].append(last_value.to_html())
            else:
              result["result_repr"] = repr(last_value)
          except ImportError:
            result["result_repr"] = repr(last_value)

    result["execution_time_ms"] = (time.perf_counter() - start_time) * 1000
    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()

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

    result["successful"] = True

  except Exception as e:
    result["execution_time_ms"] = (time.perf_counter() - start_time) * 1000
    result["error"] = str(e)
    result["error_traceback"] = traceback.format_exc()
    result["stdout"] = stdout_capture.getvalue()
    result["stderr"] = stderr_capture.getvalue()

  return result


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


SUPPORTED_GPUS = ["T4", "L4", "A10G", "A100-40GB", "A100-80GB", "L40S", "H100"]


@app.function(gpu="T4", image=notebook_image, timeout=60)
def warmup_notebook_t4() -> str:
  """Warmup function to pre-build the notebook image."""
  return "Notebook image ready on T4"




def _send(msg: Dict[str, Any]) -> None:
  """Write a JSON message to stdout (the extension reads from here)."""
  sys.stdout.write(json.dumps(msg) + "\n")
  sys.stdout.flush()


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

      if action == "execute":
        result = session.execute_cell.remote(session_id, cmd["code"])
        result["type"] = "result"
        _send(result)

      elif action == "reset":
        session.reset.remote(session_id)
        _send({"type": "reset", "successful": True})

      elif action == "terminate":
        try:
          session.cleanup.remote(session_id)
        except Exception:
          pass
        _send({"type": "terminated"})
        break

      else:
        _send({"type": "error", "message": f"Unknown action: {action}"})

    except Exception as e:
      _send({"type": "error", "message": str(e)})
