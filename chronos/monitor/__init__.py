"""Read-only training monitor shared by the Gradio console tab and the
standalone API web page.

Everything here reads files a run writes (``run_spec.json``, ``status.json``,
``metrics.jsonl``, ``parameters.jsonl``, ``samples.jsonl``, ``train.log``,
``inspections/*.json``, ``checkpoints/``).  It never imports MLX and creates
no Torch tensors (``chronos/__init__`` imports torch for HF registration, but
no device is touched), so a monitor process cannot contend for the GPU, slow a
training process down or change its numerics.
Older runs without ``status.json`` / ``parameters.jsonl`` are still shown:
their status is derived from the metrics stream and the live process table.
"""
from chronos.monitor.jsonl import JsonLines
from chronos.monitor.process import PROCESS_MEMORY, ProcessMemoryReader, find_run_pid, pid_alive
from chronos.monitor.snapshot import RunMonitor

__all__ = ["JsonLines", "PROCESS_MEMORY", "ProcessMemoryReader", "RunMonitor", "find_run_pid", "pid_alive"]
