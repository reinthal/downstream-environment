"""Template run script: config.json in, primitives out. Copy and adapt.

Stages live in one script but run in different venvs (see README.md):
``generate`` needs `.venv-vllm`, ``score`` needs `.venv`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from linear_probes import ExperimentConfig  # noqa: E402


def stage_generate(cfg: ExperimentConfig):
    """Bulk rollouts (vLLM env). Use LMSteering instead of make_backend when
    the arm is steered; both share the generate() contract."""
    from rollouts.generation import make_backend
    from rollouts.steering import LMSteering

    if cfg.steer_coef:
        backend = LMSteering.from_config(cfg)
        coefs = [cfg.steer_coef]  # per conversation, sign chosen by the caller
    else:
        backend = make_backend(cfg.gen_backend, cfg.model_id,
                               tensor_parallel_size=cfg.tensor_parallel_size,
                               max_model_len=cfg.max_model_len)
        coefs = None
    convs = [[{"role": "user", "content": "Say hello in one short sentence."}]]
    raws = backend.generate(convs, cfg.max_new_tokens, seed=cfg.seed, coefs=coefs)
    print(raws[0][-200:])


def stage_score(cfg: ExperimentConfig):
    """Probe scores (HF env). One LMProbe per read-out; with_probe shares the
    truncated model."""
    from linear_probes import LMProbe

    lr = LMProbe.from_config(cfg, probe="lr")
    dom = lr.with_probe(cfg.probe_paths["dom"])
    convs = [[{"role": "user", "content": "Say hello."},
              {"role": "assistant", "content": "Hello!", "reasoning_content": "Be brief."}]]
    print("lr:", lr.score(convs), "dom:", dom.score(convs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["generate", "score"], required=True)
    args = ap.parse_args()
    cfg = ExperimentConfig.load(OUT / "config.json")
    cfg.save()          # stamp the config into cfg.out_dir on run start
    {"generate": stage_generate, "score": stage_score}[args.stage](cfg)


if __name__ == "__main__":    # vLLM workers re-import this module
    main()
