"""Sandboxed Python execution for tool-using auditors.

Runs model-written code in a separate interpreter (.venv-sandbox: numpy,
pandas, scipy) inside a fresh temporary directory holding the given data
files, with a time limit, memory and process limits, a minimal environment,
and sockets disabled via sitecustomize (this host does not permit network
namespaces). Returns combined stdout/stderr, truncated.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SANDBOX_PYTHON = REPO / ".venv-sandbox" / "bin" / "python"
TIMEOUT_S = 30
MAX_OUTPUT_CHARS = 4000

_SITECUSTOMIZE = '''
import socket
def _blocked(*a, **k):
    raise OSError("network access is disabled in this sandbox")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
'''


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S + 5, TIMEOUT_S + 5))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


def run_python(code: str, files: dict[str, str] | None = None) -> str:
    python = SANDBOX_PYTHON if SANDBOX_PYTHON.exists() else Path(sys.executable)
    with tempfile.TemporaryDirectory(prefix="auditor_sandbox_") as tmp:
        tmp = Path(tmp)
        site = tmp / "_site"
        site.mkdir()
        (site / "sitecustomize.py").write_text(_SITECUSTOMIZE)
        for name, content in (files or {}).items():
            (tmp / name).write_text(content)
        (tmp / "main.py").write_text(code)
        env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site), "HOME": str(tmp),
               "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"}
        try:
            p = subprocess.run([str(python), "main.py"], cwd=tmp, env=env, capture_output=True,
                               text=True, timeout=TIMEOUT_S, preexec_fn=_limits)
            out = p.stdout + (("\n[stderr]\n" + p.stderr) if p.stderr.strip() else "")
            if p.returncode < 0:
                out += f"\n[error] process was killed (signal {-p.returncode})"
        except subprocess.TimeoutExpired:
            out = f"[error] code ran longer than {TIMEOUT_S}s and was stopped"
    out = out.strip() or "[no output — use print() to see results]"
    if len(out) > MAX_OUTPUT_CHARS:
        out = out[:MAX_OUTPUT_CHARS] + "\n[output truncated]"
    return out
