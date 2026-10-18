"""GPU locks for a shared box: who holds which GPU, for which experiment,
with a link to its log — and automatic cleanup when the experiment ends.

Stdlib only; run with any repo venv's python (the box has no system python3):

    .venv/bin/python .claude/skills/gpu-lock/gpu_lock.py status
    .venv/bin/python .claude/skills/gpu-lock/gpu_lock.py free
    .venv/bin/python .claude/skills/gpu-lock/gpu_lock.py run --exp NAME --log PATH --gpus 0,1|N|all-free -- CMD ...
    .venv/bin/python .claude/skills/gpu-lock/gpu_lock.py release --exp NAME [--kill]
    .venv/bin/python .claude/skills/gpu-lock/gpu_lock.py clean-stale

Layout (LOCK_DIR, default /tmp/gpu-locks, mode 1777 so every user can add
locks but only the owner can remove them):

    gpu<i>.json            one per locked GPU (created with O_EXCL: no double booking)
    <user>-<exp>.json      the experiment: gpus, pid/pgid, command, log, started
    <user>-<exp>.log  ->   symlink to the experiment's log
    history.log            one line per acquire / release (kept)

`run` acquires, starts CMD in its own process group with CUDA_VISIBLE_DEVICES
set and stdout/stderr appended to the log, and on exit (normal, error or
signal) kills whatever is left of the group, polls nvidia-smi until none of
its processes hold memory on the locked GPUs and their memory is back near
the pre-run level, writes a `[gpu-lock] DONE ...` line (exit code, memory
check) to the log, removes its locks and symlink and appends to history.log.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

LOCK_DIR = Path(os.environ.get("GPU_LOCK_DIR", "/tmp/gpu-locks"))
NVIDIA_SMI = os.environ.get("GPU_LOCK_NVIDIA_SMI", "nvidia-smi")
BUSY_MIB = 1024          # a GPU using more than this is "occupied"
FREED_MARGIN_MIB = 1024  # released when used <= pre-run baseline + this
RELEASE_TIMEOUT_S = 180


# ── nvidia-smi ───────────────────────────────────────────────────────────


def _smi(query: str, kind: str) -> list[list[str]]:
    out = subprocess.run([NVIDIA_SMI, f"--query-{kind}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def gpus() -> dict[int, dict]:
    """index -> {uuid, used_mib, total_mib}"""
    return {int(i): {"uuid": u, "used_mib": int(float(m)), "total_mib": int(float(t))}
            for i, u, m, t in _smi("index,uuid,memory.used,memory.total", "gpu")}


def compute_apps() -> list[dict]:
    """[{pid, gpu (index), used_mib}] — every process holding GPU memory."""
    by_uuid = {g["uuid"]: i for i, g in gpus().items()}
    return [{"pid": int(p), "gpu": by_uuid.get(u), "used_mib": int(float(m)) if m not in ("", "[N/A]") else None}
            for p, u, m in _smi("pid,gpu_uuid,used_memory", "compute-apps")]


def pid_user(pid: int) -> str | None:
    try:
        uid = os.stat(f"/proc/{pid}").st_uid
    except FileNotFoundError:
        return None
    import pwd
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return str(uid)


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def group_pids(pgid: int) -> set[int]:
    pids = set()
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if os.getpgid(int(d.name)) == pgid:
                    pids.add(int(d.name))
            except (ProcessLookupError, PermissionError):
                pass
    return pids


# ── lock files ───────────────────────────────────────────────────────────


def ensure_dir() -> None:
    if not LOCK_DIR.exists():
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        os.chmod(LOCK_DIR, 0o1777)


def exp_key(user: str, exp: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in exp)
    return f"{user}-{safe}"


def read_locks() -> dict[int, dict]:
    out = {}
    for p in LOCK_DIR.glob("gpu*.json") if LOCK_DIR.exists() else []:
        try:
            out[int(p.stem[3:])] = json.loads(p.read_text())
        except (ValueError, json.JSONDecodeError):
            out[int(p.stem[3:])] = {"corrupt": str(p)}
    return out


def history(line: str) -> None:
    ensure_dir()
    with open(LOCK_DIR / "history.log", "a") as f:
        f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {line}\n")
    try:
        os.chmod(LOCK_DIR / "history.log", 0o666)
    except PermissionError:
        pass


def acquire(exp: str, log: Path, wanted: list[int], command: list[str], force: bool) -> dict:
    ensure_dir()
    user = getpass.getuser()
    locks, info = read_locks(), gpus()
    for g in wanted:
        if g not in info:
            raise SystemExit(f"GPU {g} does not exist (have {sorted(info)})")
        if g in locks:
            raise SystemExit(f"GPU {g} is locked by {locks[g].get('user')} for {locks[g].get('experiment')!r} "
                             f"(pid {locks[g].get('pid')}); see `status`")
        if info[g]["used_mib"] > BUSY_MIB and not force:
            raise SystemExit(f"GPU {g} has {info[g]['used_mib']} MiB in use by unlocked processes; "
                             "pick another GPU (`free`) or pass --force if you know they are yours to share")
    record = {"experiment": exp, "user": user, "host": socket.gethostname(), "gpus": wanted,
              "pid": None, "pgid": None, "command": command, "log": str(log.resolve()),
              "started": dt.datetime.now().isoformat(timespec="seconds"),
              "baseline_mib": {str(g): info[g]["used_mib"] for g in wanted}}
    made = []
    try:
        for g in wanted:
            fd = os.open(LOCK_DIR / f"gpu{g}.json", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            with os.fdopen(fd, "w") as f:
                json.dump(record, f, indent=1)
            made.append(g)
    except FileExistsError:
        for g in made:
            (LOCK_DIR / f"gpu{g}.json").unlink()
        raise SystemExit("lost a race for a GPU lock; run `status` and retry")
    key = exp_key(user, exp)
    (LOCK_DIR / f"{key}.json").write_text(json.dumps(record, indent=1))
    link = LOCK_DIR / f"{key}.log"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(log.resolve())
    history(f"ACQUIRE user={user} exp={exp!r} gpus={wanted} log={log.resolve()}")
    return record


def update(record: dict) -> None:
    key = exp_key(record["user"], record["experiment"])
    for g in record["gpus"]:
        (LOCK_DIR / f"gpu{g}.json").write_text(json.dumps(record, indent=1))
    (LOCK_DIR / f"{key}.json").write_text(json.dumps(record, indent=1))


def verify_freed(record: dict, pids: set[int], timeout: float = RELEASE_TIMEOUT_S) -> tuple[bool, str]:
    """Poll nvidia-smi until none of `pids` holds memory on the locked GPUs and
    each GPU's used memory is back to its pre-run baseline + margin."""
    deadline = time.time() + timeout
    while True:
        apps = compute_apps()
        info = gpus()
        ours = [a for a in apps if a["pid"] in pids and a["gpu"] in record["gpus"]]
        high = {g: info[g]["used_mib"] for g in record["gpus"]
                if info[g]["used_mib"] > record["baseline_mib"][str(g)] + FREED_MARGIN_MIB}
        if not ours and not high:
            used = {g: info[g]["used_mib"] for g in record["gpus"]}
            return True, f"memory freed (MiB used now {used}, before {record['baseline_mib']})"
        if time.time() > deadline:
            return False, f"NOT freed after {timeout:.0f}s: our processes {ours}, above baseline {high}"
        time.sleep(3)


def remove(record: dict) -> None:
    key = exp_key(record["user"], record["experiment"])
    for g in record["gpus"]:
        p = LOCK_DIR / f"gpu{g}.json"
        if p.exists() and json.loads(p.read_text()).get("experiment") == record["experiment"]:
            p.unlink()
    for p in (LOCK_DIR / f"{key}.json", LOCK_DIR / f"{key}.log"):
        if p.is_symlink() or p.exists():
            p.unlink()


def kill_group(pgid: int, grace: float = 20) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        end = time.time() + grace
        while time.time() < end and group_pids(pgid):
            time.sleep(0.5)
        if not group_pids(pgid):
            return


def conclude(record: dict, exit_code, log: Path) -> int:
    """Kill leftovers, verify GPU memory, write evidence, drop the locks."""
    seen = set(record.get("seen_pids") or [])
    if record.get("pgid"):
        seen |= group_pids(record["pgid"])
        kill_group(record["pgid"])
    ok, msg = verify_freed(record, seen | ({record["pid"]} if record.get("pid") else set()))
    with open(log, "a") as f:
        f.write(f"\n[gpu-lock] DONE {dt.datetime.now().isoformat(timespec='seconds')} exp={record['experiment']!r} "
                f"exit_code={exit_code} gpus={record['gpus']} {msg}\n")
    if ok:
        remove(record)
        history(f"RELEASE user={record['user']} exp={record['experiment']!r} gpus={record['gpus']} exit={exit_code} {msg}")
        return 0
    history(f"RELEASE-FAILED user={record['user']} exp={record['experiment']!r} gpus={record['gpus']} {msg} (locks kept)")
    print(f"[gpu-lock] {msg}; locks kept — inspect `status`", file=sys.stderr)
    return 3


# ── commands ─────────────────────────────────────────────────────────────


def pick_gpus(spec: str, force: bool) -> list[int]:
    locks, info = read_locks(), gpus()
    free = [g for g in sorted(info) if g not in locks and info[g]["used_mib"] <= BUSY_MIB]
    if spec == "all-free":
        if not free:
            raise SystemExit("no free GPUs; see `status`")
        return free
    if spec.startswith("n="):
        n = int(spec[2:])
        if len(free) < n:
            raise SystemExit(f"need {n} free GPUs, have {free}")
        return free[:n]
    return [int(x) for x in spec.split(",")]


def cmd_run(a) -> int:
    log = Path(a.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    wanted = pick_gpus(a.gpus, a.force)
    record = acquire(a.exp, log, wanted, a.command, a.force)
    env = os.environ | {"CUDA_VISIBLE_DEVICES": ",".join(map(str, wanted))}
    with open(log, "a") as f:
        f.write(f"[gpu-lock] START {record['started']} exp={a.exp!r} user={record['user']} host={record['host']} "
                f"gpus={wanted} cmd={' '.join(a.command)}\n")
    logf = open(log, "a")
    proc = subprocess.Popen(a.command, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    record |= {"pid": proc.pid, "pgid": os.getpgid(proc.pid)}
    update(record)
    print(f"[gpu-lock] {a.exp}: GPUs {wanted}, pid {proc.pid}, log {log.resolve()}", flush=True)
    seen: set[int] = {proc.pid}

    def on_signal(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(s, on_signal)
    exit_code = None
    try:
        while proc.poll() is None:
            seen |= group_pids(record["pgid"])
            time.sleep(2)
        exit_code = proc.returncode
    except KeyboardInterrupt as e:
        exit_code = f"interrupted ({e})"
    finally:
        logf.close()
        record["seen_pids"] = sorted(seen)
        rc = conclude(record, exit_code, log)
    return exit_code if isinstance(exit_code, int) and exit_code != 0 else rc


def cmd_status(a) -> int:
    info, locks = gpus(), read_locks()
    apps = compute_apps()
    print(f"{'gpu':>3} {'used/total MiB':>15}  {'lock':<44} {'holders (pid user)'}")
    for g in sorted(info):
        holders = [(x["pid"], pid_user(x["pid"]) or "?") for x in apps if x["gpu"] == g]
        lk = locks.get(g)
        if lk and "corrupt" not in lk:
            state = "alive" if lk.get("pid") and alive(lk["pid"]) else ("STALE" if lk.get("pid") else "starting")
            desc = f"{lk['user']}:{lk['experiment']} pid {lk.get('pid')} {state}"
        elif lk:
            desc = f"corrupt lock {lk['corrupt']}"
        else:
            desc = "UNLOCKED but occupied" if info[g]["used_mib"] > BUSY_MIB else "free"
        print(f"{g:>3} {info[g]['used_mib']:>7}/{info[g]['total_mib']:<7}  {desc:<44} "
              f"{' '.join(f'{p} {u}' for p, u in holders)}")
    for rec in sorted(LOCK_DIR.glob("*-*.json")) if LOCK_DIR.exists() else []:
        r = json.loads(rec.read_text())
        link = LOCK_DIR / f"{rec.stem}.log"
        tail = ""
        if link.exists():
            lines = link.read_text(errors="replace").splitlines()
            done = [ln for ln in lines if ln.startswith("[gpu-lock] DONE")]
            tail = done[-1] if done else (lines[-1][:120] if lines else "")
        print(f"\n{rec.stem}: gpus {r['gpus']}, since {r['started']}, log {link} -> {r['log']}\n  last: {tail}")
    return 0


def cmd_free(a) -> int:
    info, locks = gpus(), read_locks()
    print(",".join(str(g) for g in sorted(info) if g not in locks and info[g]["used_mib"] <= BUSY_MIB))
    return 0


def own_record(exp: str) -> tuple[dict, Path]:
    key = exp_key(getpass.getuser(), exp)
    p = LOCK_DIR / f"{key}.json"
    if not p.exists():
        raise SystemExit(f"no lock record {p}")
    r = json.loads(p.read_text())
    return r, Path(r["log"])


def cmd_release(a) -> int:
    r, log = own_record(a.exp)
    if r.get("pid") and alive(r["pid"]) and not a.kill:
        raise SystemExit(f"pid {r['pid']} is still running; pass --kill to stop it and release")
    return conclude(r, "released manually", log)


def cmd_clean_stale(a) -> int:
    me = getpass.getuser()
    for rec in sorted(LOCK_DIR.glob("*-*.json")) if LOCK_DIR.exists() else []:
        r = json.loads(rec.read_text())
        if r.get("pid") and not alive(r["pid"]):
            if r["user"] == me:
                print(f"stale: {rec.stem} (pid {r['pid']} gone) -> releasing")
                conclude(r, "stale (process gone)", Path(r["log"]))
            else:
                print(f"stale: {rec.stem} belongs to {r['user']}; only they can remove it (sticky /tmp dir)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--exp", required=True)
    r.add_argument("--log", required=True)
    r.add_argument("--gpus", required=True, help="0,1 | n=4 | all-free")
    r.add_argument("--force", action="store_true", help="take GPUs that hold memory without a lock")
    r.add_argument("command", nargs=argparse.REMAINDER)
    sub.add_parser("status")
    sub.add_parser("free")
    rel = sub.add_parser("release")
    rel.add_argument("--exp", required=True)
    rel.add_argument("--kill", action="store_true")
    sub.add_parser("clean-stale")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if a.command[:1] == ["--"]:
            a.command = a.command[1:]
        if not a.command:
            ap.error("run needs a command after --")
    return {"run": cmd_run, "status": cmd_status, "free": cmd_free, "release": cmd_release,
            "clean-stale": cmd_clean_stale}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
