"""Background jobs for the Web UI: one registry, two kinds of work.

* ``SubprocessJob`` – training, benchmark: a child process whose stdout is
  kept in a bounded buffer; ``stop`` sends SIGTERM (the train worker turns
  that into a clean stage stop), then SIGKILL after a grace period.
* ``ThreadJob`` – inference, export, data scan, planning: a function run on a
  thread so the server stays responsive; the result is kept for polling.

Only one job of each *exclusive* kind runs at a time (two trainings would
fight for the same accelerator); finished jobs are kept for the session.
"""
from __future__ import annotations

import itertools
import os
import signal
import subprocess
import threading
import time
from collections import deque

EXCLUSIVE_KINDS = frozenset({"train", "benchmark", "export", "inference"})


class JobBusy(RuntimeError):
    """A job of an exclusive kind is already running."""


class _Job:
    def __init__(self, job_id: str, kind: str, title: str = ""):
        self.id, self.kind, self.title = job_id, kind, title
        self.state = "starting"
        self.started = time.time()
        self.ended: float | None = None
        self.lines: deque[str] = deque(maxlen=2000)
        self.result = None
        self.error = ""
        self.meta: dict = {}
        self._lock = threading.Lock()

    def log(self, text: str) -> None:
        with self._lock:
            for line in str(text).splitlines() or [""]:
                self.lines.append(line)

    def running(self) -> bool:
        return self.state in ("starting", "running", "stopping")

    def view(self, *, log_lines: int = 200, with_result: bool = True) -> dict:
        with self._lock:
            tail = list(self.lines)[-log_lines:] if log_lines else []
        out = {"id": self.id, "kind": self.kind, "title": self.title, "state": self.state,
               "started": self.started, "ended": self.ended, "error": self.error or None,
               "log": tail, "meta": self.meta}
        if with_result:
            out["result"] = self.result
        return out


class SubprocessJob(_Job):
    def __init__(self, job_id: str, kind: str, argv: list[str], *, cwd: str, env: dict | None = None,
                 title: str = "", on_exit=None):
        super().__init__(job_id, kind, title)
        self.argv, self.on_exit = argv, on_exit
        full_env = dict(os.environ)
        full_env.update(env or {})
        full_env.setdefault("PYTHONUNBUFFERED", "1")
        self.proc = subprocess.Popen(argv, cwd=cwd, env=full_env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1,
                                     start_new_session=True)
        self.state = "running"
        self.meta["pid"] = self.proc.pid
        threading.Thread(target=self._pump, name=f"job-{job_id}", daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.log(line.rstrip("\n"))
        code = self.proc.wait()
        self.meta["returncode"] = code
        if self.state == "stopping":
            self.state = "stopped"
        else:
            self.state = "finished" if code == 0 else "failed"
            if code != 0:
                self.error = f"exit code {code}"
        self.ended = time.time()
        if self.on_exit is not None:
            try:
                self.result = self.on_exit(self)
            except Exception as exc:              # a broken result file must not kill the pump
                self.error = self.error or f"{type(exc).__name__}: {exc}"

    def stop(self, grace_s: float = 20.0) -> None:
        if self.proc.poll() is not None:
            return
        self.state = "stopping"
        try:
            self.proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            return

        def _escalate():
            try:
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self.proc.kill()

        threading.Thread(target=_escalate, daemon=True).start()


class ThreadJob(_Job):
    def __init__(self, job_id: str, kind: str, fn, *, title: str = ""):
        super().__init__(job_id, kind, title)
        self.cancel = threading.Event()
        self.state = "running"
        threading.Thread(target=self._run, args=(fn,), name=f"job-{job_id}", daemon=True).start()

    def _run(self, fn) -> None:
        try:
            self.result = fn(self)
            self.state = "stopped" if self.cancel.is_set() else "finished"
        except Exception as exc:
            import traceback
            self.error = f"{type(exc).__name__}: {exc}"
            self.log(traceback.format_exc())
            self.state = "failed"
        self.ended = time.time()

    def stop(self, grace_s: float = 0.0) -> None:
        if self.running():
            self.cancel.set()
            self.state = "stopping"


class JobRegistry:
    """Creates, finds and stops jobs; enforces one running job per exclusive kind."""

    def __init__(self, keep: int = 64):
        self._jobs: dict[str, _Job] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self.keep = keep

    def _new_id(self, kind: str) -> str:
        return f"{kind}-{next(self._ids)}-{int(time.time())}"

    def _check(self, kind: str) -> None:
        if kind in EXCLUSIVE_KINDS and any(j.kind == kind and j.running() for j in self._jobs.values()):
            raise JobBusy(kind)

    def _add(self, job: _Job) -> _Job:
        self._jobs[job.id] = job
        done = [j for j in self._jobs.values() if not j.running()]
        for old in sorted(done, key=lambda j: j.started)[:-self.keep] if len(done) > self.keep else []:
            self._jobs.pop(old.id, None)
        return job

    def spawn(self, kind: str, argv: list[str], *, cwd: str, env: dict | None = None, title: str = "",
              on_exit=None, meta: dict | None = None) -> _Job:
        with self._lock:
            self._check(kind)
            job = SubprocessJob(self._new_id(kind), kind, argv, cwd=cwd, env=env, title=title, on_exit=on_exit)
            job.meta.update(meta or {})
            return self._add(job)

    def run(self, kind: str, fn, *, title: str = "", meta: dict | None = None) -> _Job:
        with self._lock:
            self._check(kind)
            job = ThreadJob(self._new_id(kind), kind, fn, title=title)
            job.meta.update(meta or {})
            return self._add(job)

    def get(self, job_id: str) -> _Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def list(self, kind: str | None = None) -> list[dict]:
        jobs = [j for j in self._jobs.values() if kind is None or j.kind == kind]
        return [j.view(log_lines=0, with_result=False) for j in sorted(jobs, key=lambda j: j.started, reverse=True)]

    def stop(self, job_id: str) -> dict:
        job = self.get(job_id)
        job.stop()
        return job.view(log_lines=0, with_result=False)

    def stop_all(self) -> None:
        for job in list(self._jobs.values()):
            if job.running():
                job.stop(grace_s=5.0)
