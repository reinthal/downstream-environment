"""Play Turnip games on the Modal endpoints until the arm has
cfg.n_valid_games valid games (excluded games are replaced, seeds continue).

Every answer to the Turnip is scored live by the Modal Detector
(modal_probe.py, deployed first). interrog arm: the score is recorded, the
Turnip never sees it. probe arm: the Turnip privately sees the flag
contrast > cutpoint, the cutpoint read from cutpoint.json (calibrate.py on
the interrog arm; the pilot uses cutpoint_pilot.json).

Run (repo root; do NOT source .env — common.proxy_token reads it):
  R="uv run --no-project --with-requirements environments/carrot_parsnip/requirements.txt --with modal --with numpy"
  $R python experiments/2026-10-15-turnip/play_games.py --arm interrog [--pilot 3]
"""
from __future__ import annotations

import argparse
import json
import shutil

from common import OUT, REPO, load_config, set_provider_env, tag

import extract  # noqa: E402

from Carrot_Parsnip_Agents import run_games  # noqa: E402  (engine, via common's sys.path)
from game_config import GameConfig  # noqa: E402

RECORD_KEYS = ("input_messages", "tools", "acting_text", "reasoning", "tool_call", "tool_result", "answer")


def make_detector(cfg, cutpoint: float | None):
    import modal
    from common import modal_client
    score = modal.Cls.from_name(cfg.modal_app, "Detector", client=modal_client())().score

    async def detector(record: dict) -> dict:
        s = await score.remote.aio({k: record[k] for k in RECORD_KEYS})
        if cutpoint is None:                 # interrog arm: recorded only, never shown
            return {**s, "cutpoint": None}
        return {**s, "cutpoint": cutpoint, "flag": bool(s["contrast"] > cutpoint)}
    return detector


def wake_endpoints(cfg, game_cfg: GameConfig, timeout_s: int = 1200) -> None:
    """The dedicated Qwen endpoint scales to zero after ~2 idle minutes and
    answers "no upstreams available" while cold-starting: poll each model with
    a one-line request until it answers, or fail after ``timeout_s``."""
    import time
    import requests
    from common import proxy_token
    token = proxy_token()
    models = {game_cfg.carrot.model, game_cfg.parsnip.model, game_cfg.turnip.model}
    for m in sorted(models):
        _, provider, name = m.split("/", 2)
        url = cfg.providers[provider.upper()] + "/chat/completions"
        t0 = time.time()
        while True:
            r = requests.post(url, headers={"Authorization": f"Bearer {token}"}, timeout=300,
                              json={"model": name, "max_tokens": 64, "messages": [{"role": "user", "content": "Say OK."}]})
            if r.status_code == 200 and "choices" in r.json():
                print(f"{name}: up after {time.time() - t0:.0f}s", flush=True)
                break
            if time.time() - t0 > timeout_s:
                raise SystemExit(f"{name} not up after {timeout_s}s: {r.status_code} {r.text[:200]}")
            time.sleep(20)


def play_batch(game_cfg: GameConfig, detector, n: int, base_seed: int, log_dir, first_index: int) -> dict:
    summary = run_games(num_games=n, base_seed=base_seed, config=game_cfg, detector=detector)
    logs = summary.pop("eval_logs")
    if len(logs) != n:
        raise RuntimeError(f"engine returned {len(logs)} logs for {n} games")
    for i, log in enumerate(logs):
        if log.eval.task != f"carrot_parsnip_game_{i}":
            raise RuntimeError(f"log {i} is task {log.eval.task!r}")
        shutil.copy(log.location, log_dir / f"game_{first_index + i:03d}.eval")
        for sample in log.samples or []:
            if sample.error is not None and "ReasoningNotExposedError" in sample.error.message:
                raise SystemExit(f"game {first_index + i}: {sample.error.message.splitlines()[0][:300]}")
    summary["base_seed"], summary["first_index"], summary["n"] = base_seed, first_index, n
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pilot", type=int, default=None, metavar="N", help="play N pilot games and stop")
    args = ap.parse_args()
    cfg = load_config()
    pilot = args.pilot is not None
    t = tag(args.arm, pilot)
    set_provider_env(cfg)
    game_cfg = GameConfig.from_yaml(REPO / cfg.game_configs[args.arm])
    expect = {"interrog": "silent", "probe": "shown"}[args.arm]
    if game_cfg.interrogation.detector != expect:
        raise ValueError(f"{args.arm}: interrogation.detector must be {expect!r}")
    cutpoint = None
    if args.arm == "probe":
        path = OUT / ("cutpoint_pilot.json" if pilot else "cutpoint.json")
        cutpoint = json.loads(path.read_text())["cutpoint"]
        print(f"cutpoint {cutpoint:.4f} from {path.name}")
    detector = make_detector(cfg, cutpoint)
    wake_endpoints(cfg, game_cfg)
    seed = cfg.pilot_seed if pilot else cfg.arm_seeds[args.arm]
    log_dir = OUT / f"inspect_logs_{t}"
    log_dir.mkdir(exist_ok=True)
    summary_path = OUT / f"summary_{t}.json"
    batches = json.loads(summary_path.read_text()) if summary_path.exists() else []

    if pilot:
        if extract.game_logs(log_dir):
            raise SystemExit(f"{log_dir} already has logs — move it away first")
        batches.append(play_batch(game_cfg, detector, args.pilot, seed, log_dir, 0))
        summary_path.write_text(json.dumps(batches, indent=1, default=str) + "\n")
        print(f"pilot: {args.pilot} games in {log_dir}")
        return

    while True:
        played, valid = extract.count(batches)
        print(f"{t}: {played} played, {valid} valid (target {cfg.n_valid_games})", flush=True)
        if valid >= cfg.n_valid_games:
            break
        if played - valid > cfg.max_replacements:
            raise SystemExit(f"{played - valid} replacements exceed max_replacements="
                             f"{cfg.max_replacements}: stop and debug")
        deficit = cfg.n_valid_games - valid
        batches.append(play_batch(game_cfg, detector, deficit, seed + played, log_dir, played))
        summary_path.write_text(json.dumps(batches, indent=1, default=str) + "\n")
    print(f"done: {len(extract.game_logs(log_dir))} logs in {log_dir}")


if __name__ == "__main__":
    main()
