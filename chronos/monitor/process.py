"""Process liveness and OS-level memory of a training process.

CPU-only system calls; never imports MLX/Torch.  On macOS the physical
footprint (``proc_pid_rusage`` RUSAGE_INFO_V0) is what Activity Monitor
reports as "Memory" and includes Metal buffers in unified memory; on Linux
``/proc/<pid>/status`` gives VmRSS/VmHWM.
"""
from __future__ import annotations

import ctypes
import os
import shlex
import subprocess
import sys
import threading
import time


def pid_alive(pid) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:          # exists, owned by another user
        return True
    except OSError:
        return False
    return True


class _RUsageInfoV0(ctypes.Structure):
    # xnu bsd/sys/resource.h: struct rusage_info_v0 (RUSAGE_INFO_V0 = 0)
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime")]


def _mach_tick_ns() -> float:
    """``ri_user_time``/``ri_system_time`` are Mach absolute-time ticks, not ns
    (Apple silicon: 125/3 ns per tick)."""
    if sys.platform != "darwin":
        return 1.0

    class _Timebase(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]
    try:
        tb = _Timebase()
        if ctypes.CDLL("/usr/lib/libSystem.B.dylib").mach_timebase_info(ctypes.byref(tb)) == 0 and tb.denom:
            return tb.numer / tb.denom
    except (OSError, AttributeError):
        pass
    return 1.0


class ProcessMemoryReader:
    """Rate-limited per-pid memory sampler (cached for ``interval_s``)."""

    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self._cache: dict[int, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self._libproc = None
        if sys.platform == "darwin":
            try:
                lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
                lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
                lib.proc_pid_rusage.restype = ctypes.c_int
                self._libproc = lib
            except OSError:
                self._libproc = None
        self._tick_ns = _mach_tick_ns()

    def _sample(self, pid: int) -> dict:
        row = {"pid": pid, "sampled_at": time.time()}
        if self._libproc is not None:
            usage = _RUsageInfoV0()
            if self._libproc.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
                row.update(state="unavailable", reason=os.strerror(ctypes.get_errno()))
                return row
            row.update(state="measured", source="macOS proc_pid_rusage (phys_footprint)",
                       footprint_gb=usage.ri_phys_footprint / 1e9, resident_gb=usage.ri_resident_size / 1e9,
                       wired_gb=usage.ri_wired_size / 1e9,
                       cpu_s=(usage.ri_user_time + usage.ri_system_time) * self._tick_ns / 1e9)
            return row
        status = f"/proc/{pid}/status"
        if os.path.exists(status):
            fields = {}
            with open(status, encoding="utf-8") as fh:
                for line in fh:
                    key, _, value = line.partition(":")
                    if key in ("VmRSS", "VmHWM") and value.strip().endswith("kB"):
                        fields[key] = int(value.split()[0]) * 1024 / 1e9
            row.update(state="measured", source="/proc status", resident_gb=fields.get("VmRSS"),
                       peak_resident_gb=fields.get("VmHWM"))
            return row
        row.update(state="unavailable", reason="no process accounting on this platform")
        return row

    def read(self, pid) -> dict:
        if not pid_alive(pid):
            return {"pid": pid, "state": "unavailable", "reason": "process not running"}
        with self._lock:
            hit = self._cache.get(pid)
            if hit and time.monotonic() - hit[0] < self.interval_s:
                return hit[1]
            row = self._sample(pid)
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[pid] = (time.monotonic(), row)
            return row


def process_state(pid) -> str | None:
    """``ps`` state letters (``R``/``S`` running or sleeping, ``T`` stopped by
    SIGSTOP, ``Z`` zombie) or None when the process is gone."""
    if not pid_alive(pid):
        return None
    try:
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(int(pid))], capture_output=True, text=True,
                             timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return out or None


PROCESS_MEMORY = ProcessMemoryReader()

_PS_CACHE: tuple[float, list[tuple[int, str]]] = (0.0, [])
_PS_LOCK = threading.Lock()


def _ps_rows(ttl_s: float = 10.0) -> list[tuple[int, str]]:
    global _PS_CACHE
    with _PS_LOCK:
        if time.monotonic() - _PS_CACHE[0] < ttl_s:
            return _PS_CACHE[1]
        rows = []
        try:
            out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True,
                                 timeout=5, check=False).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        for line in out.splitlines():
            pid, _, cmd = line.strip().partition(" ")
            if pid.isdigit() and "longrun" in cmd:
                rows.append((int(pid), cmd.strip()))
        _PS_CACHE = (time.monotonic(), rows)
        return rows


def find_run_pid(run_dir: str, repo_root: str | None = None) -> int | None:
    """PID of a ``chronos.mlx.training.longrun --spec X`` process whose spec's
    ``run_dir`` is ``run_dir`` (for runs started before ``status.json``)."""
    import json
    target = os.path.realpath(run_dir)
    roots = [r for r in (repo_root, os.getcwd()) if r]
    for pid, cmd in _ps_rows():
        try:
            argv = shlex.split(cmd)
        except ValueError:
            continue
        if not argv or "caffeinate" in os.path.basename(argv[0]) or "--spec" not in argv:
            continue
        spec = argv[argv.index("--spec") + 1] if argv.index("--spec") + 1 < len(argv) else ""
        for root in roots:
            path = spec if os.path.isabs(spec) else os.path.join(root, spec)
            try:
                with open(path, encoding="utf-8") as fh:
                    spec_dir = json.load(fh).get("run_dir", "")
            except (OSError, ValueError):
                continue
            spec_dir = spec_dir if os.path.isabs(spec_dir) else os.path.join(root, spec_dir)
            if os.path.realpath(spec_dir) == target:
                return pid
    return None
