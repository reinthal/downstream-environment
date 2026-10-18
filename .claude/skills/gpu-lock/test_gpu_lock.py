"""Tests with a fake nvidia-smi: a GPU's memory is held by whichever pid is
written in its holder file, for as long as that pid is alive.

    environments/twenty_questions/.venv/bin/python -m pytest -q .claude/skills/gpu-lock/test_gpu_lock.py
"""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "gpu_lock.py"

FAKE_SMI = textwrap.dedent('''\
    #!{python}
    import json, os, sys
    st = json.load(open(os.environ["FAKE_SMI_STATE"]))
    def held(g):
        f = st["holders"].get(str(g))
        if not f or not os.path.exists(f):
            return None
        pid = int(open(f).read().strip() or 0)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        return pid
    q = sys.argv[1]
    for i, g in enumerate(st["gpus"]):
        pid = held(i)
        if q.startswith("--query-gpu="):
            print(f"{{i}}, {{g['uuid']}}, {{g['base'] + (5000 if pid else 0)}}, {{g['total']}}")
        elif pid:
            print(f"{{pid}}, {{g['uuid']}}, 5000")
        for p in g.get("foreign", []):
            if q.startswith("--query-compute-apps="):
                print(f"{{p}}, {{g['uuid']}}, 40000")
''')


@pytest.fixture
def box(tmp_path, monkeypatch):
    smi = tmp_path / "nvidia-smi"
    smi.write_text(FAKE_SMI.format(python=sys.executable))
    smi.chmod(0o755)
    state = {"gpus": [{"uuid": "GPU-a", "base": 10, "total": 49140},
                      {"uuid": "GPU-b", "base": 10, "total": 49140},
                      {"uuid": "GPU-c", "base": 45000, "total": 49140, "foreign": [999999]}],
             "holders": {str(i): str(tmp_path / f"holder{i}") for i in range(3)}}
    (tmp_path / "state.json").write_text(json.dumps(state))
    env = os.environ | {"GPU_LOCK_DIR": str(tmp_path / "locks"), "GPU_LOCK_NVIDIA_SMI": str(smi),
                        "FAKE_SMI_STATE": str(tmp_path / "state.json")}
    def run(*args, timeout=60):
        return subprocess.run([sys.executable, str(SCRIPT), *args], env=env, capture_output=True, text=True,
                              timeout=timeout)
    run.tmp = tmp_path
    run.env = env
    return run


def job(tmp, gpu, seconds, exit_code=0, child=False):
    """A command that 'holds' a GPU (writes its pid to the holder file) and,
    optionally, leaves a child holding it after it exits."""
    holder = tmp / f"holder{gpu}"
    code = (f"import os,sys,time,subprocess\n"
            f"open({str(holder)!r},'w').write(str(os.getpid()))\n"
            f"print('working', flush=True)\n")
    if child:
        child_code = f"import os,time\nopen({str(holder)!r},'w').write(str(os.getpid()))\ntime.sleep(300)\n"
        code += f"c=subprocess.Popen([sys.executable,'-c',{child_code!r}])\ntime.sleep(1)\n"
    code += f"time.sleep({seconds})\nprint('finished', flush=True)\nsys.exit({exit_code})\n"
    return [sys.executable, "-c", code]


def test_run_locks_logs_releases(box):
    log = box.tmp / "exp" / "run.log"
    p = subprocess.Popen([sys.executable, str(SCRIPT), "run", "--exp", "demo", "--log", str(log), "--gpus", "0", "--",
                          *job(box.tmp, 0, 4)], env=box.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    import time
    time.sleep(2)
    locks = box.tmp / "locks"
    lock = json.loads((locks / "gpu0.json").read_text())
    assert lock["experiment"] == "demo" and lock["gpus"] == [0] and lock["pid"]
    user = lock["user"]
    assert (locks / f"{user}-demo.log").resolve() == log.resolve()
    st = box("status").stdout
    assert "demo" in st and "alive" in st and "UNLOCKED but occupied" in st      # GPU 2: foreign process
    assert box("free").stdout.strip() == "1"
    assert box("run", "--exp", "other", "--log", str(log), "--gpus", "0", "--", "true").returncode != 0
    out, err = p.communicate(timeout=60)
    assert p.returncode == 0, err
    text = log.read_text()
    assert "[gpu-lock] START" in text and "finished" in text
    assert "[gpu-lock] DONE" in text and "exit_code=0" in text and "memory freed" in text
    assert not list(locks.glob("gpu*.json")) and not (locks / f"{user}-demo.log").exists()
    hist = (locks / "history.log").read_text()
    assert "ACQUIRE" in hist and "RELEASE" in hist


def test_leftover_child_is_killed_and_memory_verified(box):
    log = box.tmp / "child.log"
    r = box("run", "--exp", "leaky", "--log", str(log), "--gpus", "0", "--", *job(box.tmp, 0, 1, child=True))
    assert r.returncode == 0, r.stderr
    assert "memory freed" in log.read_text()
    assert not list((box.tmp / "locks").glob("gpu*.json"))


def test_failing_job_keeps_exit_code_and_releases(box):
    log = box.tmp / "fail.log"
    r = box("run", "--exp", "fails", "--log", str(log), "--gpus", "1", "--", *job(box.tmp, 1, 1, exit_code=7))
    assert r.returncode == 7
    assert "exit_code=7" in log.read_text() and not list((box.tmp / "locks").glob("gpu*.json"))


def test_refuses_occupied_unlocked_gpu_and_all_free(box):
    log = box.tmp / "x.log"
    r = box("run", "--exp", "x", "--log", str(log), "--gpus", "2", "--", "true")
    assert r.returncode != 0 and "unlocked" in r.stderr
    r = box("run", "--exp", "y", "--log", str(log), "--gpus", "all-free", "--",
            sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])")
    assert r.returncode == 0 and "0,1" in log.read_text()


def test_stale_lock_cleaned(box):
    log = box.tmp / "stale.log"
    log.write_text("")
    locks = box.tmp / "locks"
    locks.mkdir()
    import getpass
    rec = {"experiment": "crashed", "user": getpass.getuser(), "host": "h", "gpus": [1], "pid": 2 ** 22 + 7,
           "pgid": None, "command": ["x"], "log": str(log), "started": "t", "baseline_mib": {"1": 10}}
    for name in ("gpu1.json", f"{rec['user']}-crashed.json"):
        (locks / name).write_text(json.dumps(rec))
    assert "STALE" in box("status").stdout
    r = box("clean-stale")
    assert "releasing" in r.stdout and not (locks / "gpu1.json").exists()
    assert "stale (process gone)" in log.read_text()
