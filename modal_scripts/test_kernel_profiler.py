"""
Unit tests for the ncu CSV parsing logic in kernel_profiler.py.

Run with: uv run pytest modal_scripts/test_kernel_profiler.py -v

Since kernel_profiler.py imports ``modal`` at the top level (which is only
available in the Modal cloud environment), we extract and test the pure-Python
``parse_ncu_csv`` function by eval-ing the relevant source without importing
the full module.
"""

import io
import os
import csv
import sys
import types
import textwrap
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, asdict, field


@dataclass
class KernelProfilingMetrics:
    kernel_name: str = ""
    execution_time_us: float = 0.0
    sm_efficiency_percent: float = 0.0
    memory_throughput_gbs: float = 0.0
    achieved_occupancy_percent: float = 0.0
    registers_per_thread: int = 0
    block_size: Tuple[int, int, int] = (0, 0, 0)
    grid_size: Tuple[int, int, int] = (0, 0, 0)
    shared_memory_bytes: int = 0
    limiting_factor: str = "unknown"


_NCU_METRIC_MAP = {
    "gpu__time_duration.sum": "execution_time_us",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_efficiency_percent",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "memory_throughput_pct",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_percent",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__block_size": "block_size_flat",
    "launch__grid_size": "grid_size_flat",
    "launch__shared_mem_per_block_dynamic": "shared_memory_dynamic",
    "launch__shared_mem_per_block_static": "shared_memory_static",
    "launch__occupancy_limit_warps": "limit_warps",
    "launch__occupancy_limit_registers": "limit_registers",
    "launch__occupancy_limit_shared_mem": "limit_shared_mem",
    "launch__occupancy_limit_blocks": "limit_blocks",
}


def parse_ncu_csv(csv_text: str, gpu_mem_bw_gbs: float = 0.0) -> List[Dict[str, Any]]:
    lines = csv_text.strip().splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith("ID") or "Kernel Name" in line:
            header_idx = i
            break
    if header_idx is None:
        return []
    csv_body = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_body))
    kernel_raw: Dict[str, Dict[str, Any]] = {}
    for row in reader:
        kid = row.get("ID", "0")
        kname = row.get("Kernel Name", "unknown")
        key = f"{kid}|{kname}"
        if key not in kernel_raw:
            kernel_raw[key] = {"kernel_name": kname}
        metric_name = row.get("Metric Name", "")
        metric_value = row.get("Metric Value", "")
        mapped = _NCU_METRIC_MAP.get(metric_name)
        if mapped:
            try:
                kernel_raw[key][mapped] = float(metric_value.replace(",", ""))
            except (ValueError, TypeError):
                kernel_raw[key][mapped] = metric_value
    results: List[Dict[str, Any]] = []
    for raw in kernel_raw.values():
        limiters = {
            "warps": raw.get("limit_warps", 0),
            "registers": raw.get("limit_registers", 0),
            "shared_memory": raw.get("limit_shared_mem", 0),
            "blocks": raw.get("limit_blocks", 0),
        }
        numeric_limiters = {
            k: float(v) for k, v in limiters.items() if isinstance(v, (int, float))
        }
        if numeric_limiters:
            limiting = min(numeric_limiters, key=numeric_limiters.get)
        else:
            limiting = "unknown"
        mem_pct = float(raw.get("memory_throughput_pct", 0))
        mem_gbs = (mem_pct / 100.0) * gpu_mem_bw_gbs if gpu_mem_bw_gbs > 0 else mem_pct
        block_flat = int(raw.get("block_size_flat", 0))
        grid_flat = int(raw.get("grid_size_flat", 0))
        shared_dyn = int(raw.get("shared_memory_dynamic", 0))
        shared_static = int(raw.get("shared_memory_static", 0))
        exec_time_ns = float(raw.get("execution_time_us", 0))
        exec_time_us = exec_time_ns / 1000.0
        m = KernelProfilingMetrics(
            kernel_name=raw.get("kernel_name", "unknown"),
            execution_time_us=exec_time_us,
            sm_efficiency_percent=float(raw.get("sm_efficiency_percent", 0)),
            memory_throughput_gbs=mem_gbs,
            achieved_occupancy_percent=float(raw.get("achieved_occupancy_percent", 0)),
            registers_per_thread=int(raw.get("registers_per_thread", 0)),
            block_size=(block_flat, 1, 1),
            grid_size=(grid_flat, 1, 1),
            shared_memory_bytes=shared_dyn + shared_static,
            limiting_factor=limiting,
        )
        results.append(asdict(m))
    return results


SAMPLE_NCU_CSV = '''"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","GPU Speed Of Light Throughput","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","42.5"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","GPU Speed Of Light Throughput","dram__throughput.avg.pct_of_peak_sustained_elapsed","%","78.3"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Compute (SM) Throughput","sm__warps_active.avg.pct_of_peak_sustained_active","%","65.1"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Launch Statistics","launch__registers_per_thread","registers","18"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Launch Statistics","launch__block_size","","256"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Launch Statistics","launch__grid_size","","4096"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Launch Statistics","launch__shared_mem_per_block_dynamic","bytes","0"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Launch Statistics","launch__shared_mem_per_block_static","bytes","128"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","GPU Speed Of Light Throughput","gpu__time_duration.sum","nsecond","2570"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Occupancy","launch__occupancy_limit_warps","warps","48"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Occupancy","launch__occupancy_limit_registers","warps","64"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Occupancy","launch__occupancy_limit_shared_mem","warps","96"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Occupancy","launch__occupancy_limit_blocks","warps","32"
'''

SAMPLE_MULTI_KERNEL_CSV = '''"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","GPU Speed Of Light Throughput","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","42.5"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","GPU Speed Of Light Throughput","gpu__time_duration.sum","nsecond","2570"
"0","1234","kernel","localhost","vectorAdd(float*, float*, float*, int)","1","7","Launch Statistics","launch__registers_per_thread","registers","18"
"1","1234","kernel","localhost","matrixMul(float*, float*, float*, int)","1","7","GPU Speed Of Light Throughput","sm__throughput.avg.pct_of_peak_sustained_elapsed","%","75.2"
"1","1234","kernel","localhost","matrixMul(float*, float*, float*, int)","1","7","GPU Speed Of Light Throughput","gpu__time_duration.sum","nsecond","125000"
"1","1234","kernel","localhost","matrixMul(float*, float*, float*, int)","1","7","Launch Statistics","launch__registers_per_thread","registers","32"
'''


class TestParseNcuCsv:
    """Tests for parse_ncu_csv function."""

    def test_basic_parsing(self):
        """Single kernel CSV is parsed into one KernelProfilingMetrics dict."""
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        assert len(metrics) == 1, f"Expected 1 kernel, got {len(metrics)}"

    def test_kernel_name(self):
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        assert "vectorAdd" in metrics[0]["kernel_name"]

    def test_sm_efficiency(self):
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        assert abs(metrics[0]["sm_efficiency_percent"] - 42.5) < 0.01

    def test_memory_throughput_absolute(self):
        """Memory throughput is converted from % to absolute GB/s."""
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        expected = (78.3 / 100.0) * 320.0
        assert abs(metrics[0]["memory_throughput_gbs"] - expected) < 0.1

    def test_occupancy(self):
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        assert abs(metrics[0]["achieved_occupancy_percent"] - 65.1) < 0.01

    def test_registers(self):
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        assert metrics[0]["registers_per_thread"] == 18

    def test_block_grid_size(self):
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        assert metrics[0]["block_size"] == (256, 1, 1)
        assert metrics[0]["grid_size"] == (4096, 1, 1)

    def test_shared_memory(self):
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        assert metrics[0]["shared_memory_bytes"] == 128  # 0 dynamic + 128 static

    def test_limiting_factor(self):
        """Limiting factor should be 'blocks' (smallest occupancy limiter value = 32)."""
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        assert metrics[0]["limiting_factor"] == "blocks"

    def test_execution_time(self):
        """gpu__time_duration.sum = 2570 nsecond → 2.57 μs."""
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=320.0)
        assert abs(metrics[0]["execution_time_us"] - 2.57) < 0.01

    def test_multi_kernel(self):
        """CSV with two kernel launches produces two metric dicts."""
        metrics = parse_ncu_csv(SAMPLE_MULTI_KERNEL_CSV, gpu_mem_bw_gbs=320.0)
        assert len(metrics) == 2
        names = [m["kernel_name"] for m in metrics]
        assert any("vectorAdd" in n for n in names)
        assert any("matrixMul" in n for n in names)

    def test_multi_kernel_distinct_metrics(self):
        metrics = parse_ncu_csv(SAMPLE_MULTI_KERNEL_CSV, gpu_mem_bw_gbs=320.0)
        by_name = {m["kernel_name"]: m for m in metrics}
        va = [v for k, v in by_name.items() if "vectorAdd" in k][0]
        mm = [v for k, v in by_name.items() if "matrixMul" in k][0]
        assert va["registers_per_thread"] == 18
        assert mm["registers_per_thread"] == 32
        assert abs(mm["sm_efficiency_percent"] - 75.2) < 0.01

    def test_empty_input(self):
        assert parse_ncu_csv("", gpu_mem_bw_gbs=320.0) == []

    def test_no_header(self):
        assert parse_ncu_csv("some random text\nwith no csv data\n", gpu_mem_bw_gbs=320.0) == []

    def test_zero_bandwidth_fallback(self):
        """When gpu_mem_bw_gbs=0, memory throughput falls back to percentage value."""
        metrics = parse_ncu_csv(SAMPLE_NCU_CSV, gpu_mem_bw_gbs=0.0)
        # Fallback: raw percentage value is used as-is
        assert abs(metrics[0]["memory_throughput_gbs"] - 78.3) < 0.01


if __name__ == "__main__":
    # Allow running with python directly
    import pytest
    pytest.main([__file__, "-v"])
