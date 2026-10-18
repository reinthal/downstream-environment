---
name: gpu-lock
description: Hold GPUs on a shared box with visible locks and automatic cleanup. Use before launching any GPU job on the shared GPU box (gpaulo-ord-0, the 8x A40 machine), when choosing which GPUs to use, when another user's job may be on the box, or when an experiment that held GPUs has ended. Writes /tmp/gpu-locks/ (which GPUs, which process, which experiment, a symlink to its log), runs the job, and on exit kills leftovers, verifies with nvidia-smi that the GPU memory is freed, writes a DONE line to the log and removes the locks.
---

# GPU lock

The box is shared: other users' jobs hold GPUs too, and an experiment that
crashed can leave memory allocated. Every GPU job of ours goes through
`gpu_lock.py run`, so anyone on the machine can see what holds which GPU and
read the log to see whether it is finished.

The script is stdlib-only. The box has no system `python3`: run it with any
repo venv's python, from the repo root.

```bash
GL=".venv/bin/python .claude/skills/gpu-lock/gpu_lock.py"
$GL status                      # every GPU: memory, lock owner/experiment/pid alive|STALE, log link; unlocked-but-occupied flagged
$GL free                        # comma list of GPUs with no lock and < 1 GiB used
$GL run --exp <experiment> --log experiments/<date>/logs/<name>.log --gpus all-free -- <command ...>
$GL release --exp <experiment> [--kill]    # only for a run whose wrapper died (normal runs release themselves)
$GL clean-stale                 # release our locks whose process is gone; reports other users' stale locks
```

`--gpus` takes `0,1`, `n=4` (the first 4 free GPUs) or `all-free`.
`CUDA_VISIBLE_DEVICES` is set for the command; a job that parallelises over
GPUs (one engine per GPU, tensor parallelism) sees exactly the locked ones.

## What `run` does

1. Refuses a GPU that is locked by anyone, or that holds > 1 GiB without a
   lock (another user's job). `--force` only when that memory is known to be
   shareable — never on someone else's job without asking the researcher.
2. Creates `/tmp/gpu-locks/gpu<i>.json` atomically for each GPU (no double
   booking), plus `<user>-<exp>.json` and the symlink `<user>-<exp>.log` →
   the experiment's log; appends ACQUIRE to `history.log`.
3. Starts the command in its own process group, stdout/stderr appended to the
   log after a `[gpu-lock] START` line.
4. When the command ends — success, failure, Ctrl-C, SIGTERM/SIGHUP — kills
   whatever is left of the process group, then polls `nvidia-smi` (up to
   180 s) until none of the job's processes hold memory on the locked GPUs and
   each GPU is back within 1 GiB of its pre-run memory.
5. Writes `[gpu-lock] DONE <time> exp=... exit_code=... memory freed (...)`
   to the log, removes the locks and the symlink, appends RELEASE to
   `history.log`. The log's last line is the evidence that the experiment is
   done.
6. If the memory does not come back, the locks stay, the log and history say
   `NOT freed` / `RELEASE-FAILED`, and the exit code is 3: inspect with
   `status` and `nvidia-smi` before releasing by hand.

## Rules

- Before any GPU job on the box: `$GL status`, then `run` on free GPUs. Check
  `AGENTS.md` for limits (e.g. the Qwen3.8 server's fixed GPUs) and never
  widen a shared server's GPUs without asking.
- Long jobs: start `run` under `setsid nohup ... &` so the wrapper survives
  the SSH session — the cleanup lives in the wrapper.
- Never delete or edit another user's lock; `/tmp` is sticky, so you cannot
  anyway. Report their stale locks to the researcher.
- When an experiment is concluded, confirm `$GL status` shows its GPUs free
  and no lock of ours remains; run `clean-stale` if one does.

## Tests

`environments/twenty_questions/.venv/bin/python -m pytest -q .claude/skills/gpu-lock/test_gpu_lock.py`
(fake `nvidia-smi`: covers lock/log/release, a leftover child killed and
memory verified, exit codes, occupied-GPU refusal, `all-free`, stale locks).
