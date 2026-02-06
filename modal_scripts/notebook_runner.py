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


notebook_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
  "torch",
  "triton",
  "numpy",
  "matplotlib",
  "pandas",
  "Pillow",
  "nvidia-ml-py",
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


# Per-GPU session classes
#
# Each class is identical except for the gpu= parameter in @app.cls.
# The scaledown_window=900 keeps it alive for 15 minutes of inactivity.
# Namespaces are keyed by session_id so multiple notebooks can share a
# container without leaking state.


@app.cls(gpu="T4", image=notebook_image, scaledown_window=900, timeout=900)
class SessionT4:
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


@app.cls(gpu="L4", image=notebook_image, scaledown_window=900, timeout=900)
class SessionL4:
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


@app.cls(gpu="A10G", image=notebook_image, scaledown_window=900, timeout=900)
class SessionA10G:
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


@app.cls(gpu="A100", image=notebook_image, scaledown_window=900, timeout=900)
class SessionA100:
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


@app.cls(gpu="A100-80GB", image=notebook_image, scaledown_window=900, timeout=900)
class SessionA100_80GB:
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


@app.cls(gpu="L40S", image=notebook_image, scaledown_window=900, timeout=900)
class SessionL40S:
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


@app.cls(gpu="H100", image=notebook_image, scaledown_window=900, timeout=900)
class SessionH100:
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


@app.function(gpu="T4", image=notebook_image, timeout=60)
def warmup_notebook_t4() -> str:
  return "Notebook image ready on T4"


SESSION_CLASSES: Dict[str, Any] = {
  "T4": SessionT4,
  "L4": SessionL4,
  "A10G": SessionA10G,
  "A100": SessionA100,
  "A100-40GB": SessionA100,
  "A100-80GB": SessionA100_80GB,
  "L40S": SessionL40S,
  "H100": SessionH100,
}


def _send(msg: Dict[str, Any]) -> None:
  """Write a JSON message to stdout (the extension reads from here)."""
  sys.stdout.write(json.dumps(msg) + "\n")
  sys.stdout.flush()


@app.local_entrypoint()
def main(
  gpu: str = "T4",
  interactive: bool = False,
  warmup_images: bool = False,
):
  """
  Notebook session manager.

  In --interactive mode (the default for the extension), reads JSON commands
  from stdin and writes JSON responses to stdout.  The first message is
  always ``{"type":"ready", ...}`` once the GPU container is warm.
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

  SessionClass = SESSION_CLASSES.get(gpu)
  if not SessionClass:
    _send({"type": "error", "message": f"Unsupported GPU: {gpu}"})
    sys.exit(1)

  import uuid

  session = SessionClass()
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
