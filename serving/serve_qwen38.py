#!/usr/bin/env python3
"""Serve Qwen/Qwen3.8-27B with vLLM on this box (OpenAI-compatible API).

    .venv-serve/bin/python serving/serve_qwen38.py            # serve on :8001
    .venv-serve/bin/python serving/serve_qwen38.py print      # show the command
    .venv-serve/bin/python serving/serve_qwen38.py bench      # load-test a running server

Stdlib only: this script just assembles the ``vllm serve`` command line from
:class:`ServeConfig` and ``exec``s it inside ``.venv-serve`` (vLLM 0.31.0, see
``requirements/setup.sh serve``). Every tunable is a dataclass field; override
any of them with ``--<field> <value>`` (``--gpus 0,3,4,5``), or
``--dump-config`` to see the effective values as JSON.

Layout, each choice measured or researched on 2026-10-10 (serving/README.md,
"Qwen3.8-27B generation server"):
  - 4 GPUs (the box's allowance), as 2 data-parallel replicas x tensor
    parallel 2 behind ONE port: the 55.6 GB bf16 checkpoint needs two 46 GB
    A40s, and the GPUs are PCIe-only (no NVLink), so wider tensor parallelism
    spends time in all-reduces that two independent replicas do not.
  - GPUs 1,2 and 6,7: the only two PCIe-sibling pairs ("PHB" in
    `nvidia-smi topo -m`); every other pair crosses the inter-socket link.
    vLLM assigns consecutive visible devices to a replica, so the order of
    ``gpus`` matters: replica 0 = first ``tensor_parallel_size`` entries.
  - text-only (``--language-model-only``): skips the vision encoder and its
    startup profiling; the repo's workloads are text.
  - prefix caching: agent loops re-send the whole conversation. Hybrid
    models cache in 784-token blocks here (attention block = GDN page), so
    only prefixes that long hit.
  - MTP speculative decoding (the checkpoint ships one MTP layer), depth 3:
    +42% output tok/s on real prompts at 64 concurrent (README); ``mtp_tokens
    0`` turns it off (slightly faster on random-token prompts only).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV = REPO / ".venv-serve"


@dataclass
class ServeConfig:
    model: str = "Qwen/Qwen3.8-27B"
    revision: str = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"  # HF main, 2026-10-09
    served_model_name: str = "Qwen/Qwen3.8-27B"
    host: str = "0.0.0.0"
    port: int = 8001

    # placement
    gpus: list[int] = field(default_factory=lambda: [1, 2, 6, 7])
    tensor_parallel_size: int = 2
    data_parallel_size: int = 2

    # capacity (per replica unless noted)
    max_model_len: int = 262144
    max_num_seqs: int = 128              # per replica; 256 total in flight
    max_num_batched_tokens: int = 8192
    gpu_memory_utilization: float = 0.88   # 0.92 OOMed under load with MTP (README)
    max_num_queued_reqs: int = 2048       # whole server (all replicas)
    dtype: str = "bfloat16"

    # features
    language_model_only: bool = True
    enable_prefix_caching: bool = True
    mamba_cache_mode: str = "align"       # "align" | "all"; 0.31.0 has no "all" for qwen3_5 (README)
    mtp_tokens: int = 3                   # MTP speculative depth; 0 = off
    reasoning_parser: str = "qwen3"
    tool_call_parser: str = "qwen3_xml"
    async_scheduling: bool = True
    enforce_eager: bool = False
    api_server_count: int = 2
    extra_args: list[str] = field(default_factory=list)  # passed through verbatim

    # process env
    log_dir: str = "serving/logs"          # repo-relative; "" = no file log
    hf_hub_offline: bool = False          # set once the weights are cached


def vllm_argv(cfg: ServeConfig) -> list[str]:
    argv = [
        str(VENV / "bin" / "vllm"), "serve", cfg.model,
        "--revision", cfg.revision,
        "--served-model-name", cfg.served_model_name,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--tensor-parallel-size", str(cfg.tensor_parallel_size),
        "--data-parallel-size", str(cfg.data_parallel_size),
        "--dtype", cfg.dtype,
        "--max-model-len", str(cfg.max_model_len),
        "--max-num-seqs", str(cfg.max_num_seqs),
        "--max-num-batched-tokens", str(cfg.max_num_batched_tokens),
        "--max-num-queued-reqs", str(cfg.max_num_queued_reqs),
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--reasoning-parser", cfg.reasoning_parser,
        "--enable-auto-tool-choice",
        "--tool-call-parser", cfg.tool_call_parser,
        "--api-server-count", str(cfg.api_server_count),
    ]
    if cfg.language_model_only:
        argv.append("--language-model-only")
    argv.append("--enable-prefix-caching" if cfg.enable_prefix_caching
                else "--no-enable-prefix-caching")
    if cfg.enable_prefix_caching:
        argv += ["--mamba-cache-mode", cfg.mamba_cache_mode]
    if cfg.mtp_tokens > 0:
        argv += ["--speculative-config",
                 json.dumps({"method": "mtp", "num_speculative_tokens": cfg.mtp_tokens})]
    if cfg.async_scheduling:
        argv.append("--async-scheduling")
    if cfg.enforce_eager:
        argv.append("--enforce-eager")
    argv += cfg.extra_args
    return argv


def vllm_env(cfg: ServeConfig) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, cfg.gpus))
    # spawned workers JIT-compile with ninja, which lives in the venv's bin
    env["PATH"] = f"{VENV / 'bin'}:{env.get('PATH', '')}"
    # no nvcc on this box: FlashInfer cannot JIT its sampler (vLLM then uses
    # its own); attention/GDN kernels ship pre-built in the wheel.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if cfg.hf_hub_offline:
        env["HF_HUB_OFFLINE"] = "1"
    return env


def check(cfg: ServeConfig) -> None:
    n = cfg.tensor_parallel_size * cfg.data_parallel_size
    if len(cfg.gpus) != n:
        raise SystemExit(f"gpus={cfg.gpus} but tensor_parallel_size x data_parallel_size = {n}")
    if not (VENV / "bin" / "vllm").exists():
        raise SystemExit(f"{VENV} missing: run requirements/setup.sh serve")
    if cfg.mamba_cache_mode not in ("all", "align"):
        raise SystemExit(f"mamba_cache_mode must be 'all' or 'align', got {cfg.mamba_cache_mode!r}")


def serve(cfg: ServeConfig) -> None:
    check(cfg)
    argv, env = vllm_argv(cfg), vllm_env(cfg)
    print("CUDA_VISIBLE_DEVICES=" + env["CUDA_VISIBLE_DEVICES"], " ".join(argv), flush=True)
    if not cfg.log_dir:
        os.execve(argv[0], argv, env)
    log_dir = REPO / cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"serve_{cfg.port}.log"
    print(f"log: {log}", flush=True)
    # tee: stdout/stderr -> terminal and the log file, then exec vllm
    tee = subprocess.Popen(["tee", "-a", str(log)], stdin=subprocess.PIPE)
    os.dup2(tee.stdin.fileno(), 1)
    os.dup2(tee.stdin.fileno(), 2)
    os.execve(argv[0], argv, env)


def bench(cfg: ServeConfig, args: argparse.Namespace) -> None:
    """`vllm bench serve` against a running server at fixed concurrency, with
    random token prompts of the given lengths or real ShareGPT prompts (the
    latter for anything speculative: MTP acceptance on random text is
    meaningless). Reports output tok/s, TTFT and TPOT."""
    argv = [
        str(VENV / "bin" / "vllm"), "bench", "serve",
        "--host", "127.0.0.1", "--port", str(cfg.port),
        "--model", cfg.served_model_name,
        "--tokenizer", cfg.model,
        "--dataset-name", args.dataset,
        "--num-prompts", str(args.num_prompts),
        "--max-concurrency", str(args.concurrency),
        "--request-rate", "inf",
        "--ignore-eos",
        "--percentile-metrics", "ttft,tpot,itl,e2el",
    ]
    if args.dataset == "random":
        argv += ["--random-input-len", str(args.input_len),
                 "--random-output-len", str(args.output_len)]
    else:
        argv += ["--dataset-path", args.dataset_path]
    if args.save:
        argv += ["--save-result", "--result-dir", str(REPO / cfg.log_dir)]
    print(" ".join(argv), flush=True)
    env = dict(os.environ)
    env["PATH"] = f"{VENV / 'bin'}:{env.get('PATH', '')}"
    sys.exit(subprocess.call(argv, env=env))


def parse(argv: list[str]) -> tuple[str, ServeConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="serve", choices=["serve", "print", "bench"])
    p.add_argument("--dump-config", action="store_true", help="print the effective ServeConfig as JSON")
    p.add_argument("--config", help="JSON file of ServeConfig fields (CLI overrides win)")
    for f in fields(ServeConfig):
        if f.type == "bool":
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"), metavar="BOOL")
        elif f.type == "list[int]":
            p.add_argument(f"--{f.name}", type=lambda s: [int(x) for x in s.split(",")], metavar="I,J,..")
        elif f.type == "list[str]":
            p.add_argument(f"--{f.name}", nargs="+", metavar="ARG")
        else:
            p.add_argument(f"--{f.name}", type=eval(f.type), metavar=f.type.upper())
    b = p.add_argument_group("bench")
    b.add_argument("--dataset", default="random", choices=["random", "sharegpt"],
                   help="random: fixed input/output lengths; sharegpt: real prompts (needs --dataset-path)")
    b.add_argument("--dataset-path", help="ShareGPT_V3_unfiltered_cleaned_split.json for --dataset sharegpt")
    b.add_argument("--concurrency", type=int, default=64)
    b.add_argument("--num-prompts", type=int, default=256)
    b.add_argument("--input-len", type=int, default=2048)
    b.add_argument("--output-len", type=int, default=512)
    b.add_argument("--save", action="store_true", help="write the bench JSON into log_dir")
    args = p.parse_args(argv)

    values: dict = {}
    if args.config:
        values.update(json.loads(Path(args.config).read_text()))
    for f in fields(ServeConfig):
        v = getattr(args, f.name)
        if v is not None:
            values[f.name] = v
    cfg = ServeConfig(**values)
    return args.command, cfg, args


def main() -> None:
    command, cfg, args = parse(sys.argv[1:])
    if args.dump_config:
        print(json.dumps(asdict(cfg), indent=2))
    if command == "print":
        print("CUDA_VISIBLE_DEVICES=" + ",".join(map(str, cfg.gpus)), " ".join(vllm_argv(cfg)))
    elif command == "bench":
        bench(cfg, args)
    else:
        serve(cfg)


if __name__ == "__main__":
    main()
