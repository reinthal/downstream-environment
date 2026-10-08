"""Speech-act labels for every base-game announcement.

Each of the 1,500 announcements (transcripts_base.jsonl, symlink to
2026-09-29) is labeled by a judge model that sees ONLY the public discussion
so far plus the announcement to label — no roles, no private reasoning.
Taxonomy (frozen in README.md): primary act in {defer, probe, flag, accuse,
defend_self, defend_other, coordinate} + flags {role_claim, vote_declared}.

The judge is picked by cfg.gen_backend:
  "vllm"        local Qwen3.5-27B through rollouts.generation.VLLMBackend
                (thinking chat template + force-close; the <answer> tag is
                parsed from the public part only)
  "openrouter"  hosted model (cfg.api_model) via the OpenRouter API

Parsing is strict in both paths: exactly one <answer>{...}</answer> tag,
strict JSON, exact keys; unparsable turns are retried with a different seed
and then raise — no lenient parsing. Rows append to judge_labels_base.jsonl;
re-running resumes (done turns are skipped).

Stages:
  label    judge labeling -> judge_labels_base.jsonl
  analyze  per-round label mix, round-1 accuse rate with game-clustered
           95% bootstrap CI -> results.json, figures/   (.venv, CPU)

Run (vllm):
  CUDA_VISIBLE_DEVICES=6,7 PATH=$PWD/.venv-vllm/bin:$PATH .venv-vllm/bin/python \
      experiments/2026-10-08-labeling-answers/label.py --stage label [--limit 10]
Run (openrouter):
  set -a; source .env; set +a
  uv run --no-sync python experiments/2026-10-08-labeling-answers/label.py --stage label
Analyze:
  uv run --no-sync python experiments/2026-10-08-labeling-answers/label.py --stage analyze
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from linear_probes import ExperimentConfig  # noqa: E402

URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY = ["defer", "probe", "flag", "accuse", "defend_self", "defend_other",
           "coordinate"]
TURN_KEYS = ["game_id", "seed_group", "round", "position", "speaker", "role"]


@dataclass
class LabelingConfig(ExperimentConfig):
    # model_id/layer are inherited but unused (API labeling run, no probe).
    api_model: str = "deepseek/deepseek-v4.1-flash"
    api_max_tokens: int = 3000       # room for the model's own reasoning
    api_workers: int = 16
    api_attempts: int = 5            # transport retries (exponential backoff)
    parse_retries: int = 2           # different-seed retries on unparsable output
    transcripts_path: str = "transcripts_base.jsonl"
    labels_path: str = "judge_labels_base.jsonl"
    bootstrap_reps: int = 10000


SYSTEM = """\
You label announcements from Carrot-Parsnip, a 5-player social-deduction game:
4 players are Carrots, 1 is the secret Parsnip. There are 3 rounds of public
discussion, then every player votes; 3+ votes eliminates. Carrots win if the
Parsnip is eliminated; the Parsnip wins by surviving.

You are given the public discussion so far and ONE announcement. Label the
speech act of that announcement only.

Primary label — choose exactly one, the act that dominates the announcement's
purpose:
- defer: withholds judgment; solicits general input; says it is too early;
  content-free agreement.
- probe: demands specifics or directs a question at named player(s), without
  stating suspicion of anyone.
- flag: points at a specific player's behavior as suspicious or worth
  watching, without calling for their elimination.
- accuse: names a player as the likely Parsnip, or calls to vote them out.
- defend_self: rebuts suspicion directed at the speaker themself.
- defend_other: argues against suspicion of another player.
- coordinate: declares the speaker's vote or explicitly joins named other
  players' existing accusation/vote.

Secondary flags, independent of the primary label:
- role_claim: the speaker claims a role for themself (e.g. "I'm a Carrot").
- vote_declared: the speaker states or commits to their vote.

Think briefly if needed, then end your reply with EXACTLY one answer tag on
its own line, strict JSON, these three keys and nothing else:
<answer>{"primary": "<label>", "role_claim": <true|false>, "vote_declared": <true|false>}</answer>"""

USER = """\
PUBLIC DISCUSSION SO FAR:
{history}

ANNOUNCEMENT TO LABEL — Round {round}, {speaker} says:
{announcement}"""


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_jobs(cfg: LabelingConfig) -> list[dict]:
    turns = read_jsonl(OUT / cfg.transcripts_path)
    turns.sort(key=lambda r: (r["seed_group"], r["game_idx"], r["round"], r["position"]))
    by_game: dict[str, list[dict]] = {}
    jobs = []
    for r in turns:
        if not r["announcement"]:
            raise ValueError(f"empty announcement: {r['game_id']} r{r['round']} p{r['position']}")
        prior = by_game.setdefault(r["game_id"], [])
        history = "\n".join(f"Round {p['round'] + 1}, {p['speaker']}: {p['announcement']}"
                            for p in prior) or "(none — this is the game's first announcement)"
        jobs.append({**{k: r[k] for k in TURN_KEYS},
                     "messages": [
                         {"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER.format(
                             history=history, round=r["round"] + 1,
                             speaker=r["speaker"], announcement=r["announcement"])}]})
        prior.append(r)
    return jobs


def call_api(messages: list[dict], cfg: LabelingConfig, seed: int) -> tuple[str, str, dict]:
    """-> (reasoning, content, usage); raises after exhausted transport retries."""
    body = json.dumps({"model": cfg.api_model, "temperature": cfg.temperature,
                       "max_tokens": cfg.api_max_tokens, "seed": seed,
                       "messages": messages}).encode()
    for a in range(cfg.api_attempts):
        req = urllib.request.Request(URL, data=body, headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                out = json.load(resp)
            msg = out["choices"][0]["message"]
            return (msg.get("reasoning") or "", msg.get("content") or "",
                    out.get("usage") or {})
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                TimeoutError, json.JSONDecodeError) as e:
            if a == cfg.api_attempts - 1:
                raise RuntimeError(f"API failed after {cfg.api_attempts} attempts: {e}")
            time.sleep(2 ** a * 2)


ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_label(content: str) -> dict:
    """Strict: exactly one <answer> tag, strict JSON, exact keys. Raises."""
    tags = ANSWER.findall(content)
    if len(tags) != 1:
        raise ValueError(f"expected exactly one <answer> tag, got {len(tags)}")
    obj = json.loads(tags[0])
    if set(obj) != {"primary", "role_claim", "vote_declared"}:
        raise ValueError(f"wrong keys: {sorted(obj)}")
    if obj["primary"] not in PRIMARY:
        raise ValueError(f"unknown primary label: {obj['primary']!r}")
    if not isinstance(obj["role_claim"], bool) or not isinstance(obj["vote_declared"], bool):
        raise ValueError(f"flags must be booleans: {obj}")
    return obj


def stage_label(cfg: LabelingConfig, limit: int):
    jobs = build_jobs(cfg)
    out_path = OUT / cfg.labels_path
    done = {tuple(r[k] for k in ("game_id", "round", "position"))
            for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [j for j in jobs if (j["game_id"], j["round"], j["position"]) not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(jobs)} turns, {len(done)} done, labeling {len(todo)}", flush=True)

    lock = Lock()
    usage_tot = {"prompt_tokens": 0, "completion_tokens": 0}

    def work(job: dict) -> dict:
        last_err = None
        for k in range(cfg.parse_retries + 1):
            reasoning, content, usage = call_api(job["messages"], cfg,
                                                 seed=cfg.seed + 1000 * k)
            with lock:
                for key in usage_tot:
                    usage_tot[key] += usage.get(key, 0)
            try:
                label = parse_label(content)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = f"{e}; raw content: {content!r}"
                continue
            return {**{k2: job[k2] for k2 in TURN_KEYS}, **label,
                    "judge_model": cfg.api_model,
                    "label_seed": cfg.seed + 1000 * k, "reasoning": reasoning,
                    "content": content}
        raise ValueError(f"unparsable after {cfg.parse_retries + 1} tries "
                         f"({job['game_id']} r{job['round']} p{job['position']}): {last_err}")

    with ThreadPoolExecutor(max_workers=cfg.api_workers) as ex, open(out_path, "a") as f:
        for i, row in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  (prompt {usage_tot['prompt_tokens']:,} "
                      f"/ completion {usage_tot['completion_tokens']:,} tokens)", flush=True)
    cost = (usage_tot["prompt_tokens"] * 0.30 + usage_tot["completion_tokens"] * 1.20) / 1e6
    print(f"done; ~${cost:.2f} at $0.30/$1.20 per Mtok", flush=True)


def stage_label_vllm(cfg: LabelingConfig, limit: int):
    from rollouts.generation import make_backend, split_think

    jobs = build_jobs(cfg)
    out_path = OUT / cfg.labels_path
    done = {tuple(r[k] for k in ("game_id", "round", "position"))
            for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [j for j in jobs if (j["game_id"], j["round"], j["position"]) not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(jobs)} turns, {len(done)} done, labeling {len(todo)}", flush=True)
    if not todo:
        return

    backend = make_backend("vllm", cfg.model_id,
                           tensor_parallel_size=cfg.tensor_parallel_size,
                           max_model_len=cfg.max_model_len)
    backend.gen_kwargs = {"temperature": cfg.temperature, "top_p": cfg.top_p}
    backend.force_close_tokens = cfg.force_close_tokens

    pending = todo
    with open(out_path, "a") as f:
        for attempt in range(cfg.parse_retries + 1):
            raws = backend.generate([j["messages"] for j in pending],
                                    cfg.max_new_tokens,
                                    seed=cfg.seed + 1000 * attempt)
            failed, errs = [], []
            for job, raw in zip(pending, raws):
                reasoning, public = split_think(raw)
                try:
                    label = parse_label(public)
                except (ValueError, json.JSONDecodeError) as e:
                    failed.append(job)
                    errs.append(f"{job['game_id']} r{job['round']} p{job['position']}: "
                                f"{e}; public: {public!r}")
                    continue
                f.write(json.dumps({**{k: job[k] for k in TURN_KEYS}, **label,
                                    "judge_model": cfg.model_id,
                                    "label_seed": cfg.seed + 1000 * attempt,
                                    "reasoning": reasoning, "content": public}) + "\n")
            f.flush()
            print(f"attempt {attempt}: {len(pending) - len(failed)}/{len(pending)} parsed "
                  f"({backend.n_force_closed} force-closed)", flush=True)
            pending = failed
            if not pending:
                break
    if pending:
        raise ValueError(f"{len(pending)} turns unparsable after "
                         f"{cfg.parse_retries + 1} attempts:\n" + "\n".join(errs[:5]))


# ── analysis (CPU) ───────────────────────────────────────────────────────────

SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e", "#898781",
                                         "#e1e0d9", "#c3c2b7")
# fixed label order and colors: cool blues for passive acts, warm for action
LABEL_COLORS = {"defer": "#8f8d86", "probe": "#7aa7dd", "flag": "#2a78d6",
                "accuse": "#eb6834", "defend_self": "#b5852f", "defend_other": "#d9b13b",
                "coordinate": "#a04a28"}


def cluster_boot(rows: list[dict], num: callable, den: callable, reps: int,
                 rng) -> tuple[float, float, float]:
    """Rate num/den with a 95% CI from resampling games (clusters)."""
    import numpy as np
    games = sorted({r["game_id"] for r in rows})
    by_game = {g: [r for r in rows if r["game_id"] == g] for g in games}
    n = np.array([sum(num(r) for r in by_game[g]) for g in games], dtype=float)
    d = np.array([sum(den(r) for r in by_game[g]) for g in games], dtype=float)
    rate = n.sum() / d.sum()
    idx = rng.integers(0, len(games), size=(reps, len(games)))
    boots = n[idx].sum(axis=1) / d[idx].sum(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(rate), float(lo), float(hi)


def stage_analyze(cfg: LabelingConfig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "legend.frameon": False})

    rows = read_jsonl(OUT / cfg.labels_path)
    n_turns = len({(r["game_id"], r["round"], r["position"]) for r in rows})
    if n_turns != len(rows):
        raise ValueError(f"duplicate turns in {cfg.labels_path}")
    print(f"{len(rows)} labeled turns, {len({r['game_id'] for r in rows})} games")

    judge_models = {r["judge_model"] for r in rows}
    if len(judge_models) != 1:
        raise ValueError(f"mixed judge models in labels file: {judge_models}")
    judge_model = judge_models.pop()
    rng = np.random.default_rng(cfg.seed)
    results = {"n_turns": len(rows), "judge_model": judge_model,
               "per_round_counts": {}, "rates": {}}

    for rd in (0, 1, 2):
        counts = {lab: sum(1 for r in rows if r["round"] == rd and r["primary"] == lab)
                  for lab in PRIMARY}
        results["per_round_counts"][f"round{rd + 1}"] = counts
        print(f"round {rd + 1}: " + "  ".join(f"{k}={v}" for k, v in counts.items()))

    def rate(name, rows_, num, den=lambda r: 1):
        est, lo, hi = cluster_boot(rows_, num, den, cfg.bootstrap_reps, rng)
        results["rates"][name] = {"rate": est, "ci95": [lo, hi]}
        print(f"{name}: {est:.3f}  [{lo:.3f}, {hi:.3f}]")
        return est, lo, hi

    for rd in (0, 1, 2):
        rr = [r for r in rows if r["round"] == rd]
        rate(f"round{rd + 1}_accuse", rr, lambda r: r["primary"] == "accuse")
        rate(f"round{rd + 1}_defer", rr, lambda r: r["primary"] == "defer")
    rr1 = [r for r in rows if r["round"] == 0]
    rate("round1_accuse_or_coordinate", rr1,
         lambda r: r["primary"] in ("accuse", "coordinate"))

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=1)

    # figure 1: label mix per round (all players), stacked horizontal bars
    (OUT / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    share = {rd: {lab: results["per_round_counts"][f"round{rd + 1}"][lab] /
                  sum(results["per_round_counts"][f"round{rd + 1}"].values())
                  for lab in PRIMARY} for rd in (0, 1, 2)}
    for rd in (0, 1, 2):
        left = 0.0
        for lab in PRIMARY:
            w = share[rd][lab]
            ax.barh(2 - rd, w, left=left, height=0.62, color=LABEL_COLORS[lab],
                    edgecolor=SURFACE, lw=2, label=lab if rd == 0 else None)
            if w >= 0.04:
                ax.text(left + w / 2, 2 - rd, f"{100 * w:.0f}%", ha="center",
                        va="center", fontsize=8.5, color=SURFACE if lab != "defer" else INK)
            left += w
    ax.set_yticks([2, 1, 0], ["Round 1", "Round 2", "Round 3"])
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.legend(ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8.5,
              handlelength=1.2, columnspacing=1.0)
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / h, "Speech-act mix per round, base games (n=500 turns/round)",
             fontsize=13, fontweight="bold", va="top")
    fig.text(0.02, 1 - 0.42 / h, f"Labels: {judge_model} on public announcements only.",
             fontsize=9.5, color=INK2, va="top")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.2)
    fig.savefig(OUT / "figures" / "label_mix_by_round.png", dpi=170)
    print("wrote figures/label_mix_by_round.png")

    # figure 2: same mix, Carrot vs Parsnip panels (identical axes)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4), sharex=True, sharey=True)
    for ax, role in zip(axes, ("Carrot", "Parsnip")):
        sub = [r for r in rows if r["role"] == role]
        for rd in (0, 1, 2):
            rr = [r for r in sub if r["round"] == rd]
            left = 0.0
            for lab in PRIMARY:
                w = sum(1 for r in rr if r["primary"] == lab) / len(rr)
                ax.barh(2 - rd, w, left=left, height=0.62, color=LABEL_COLORS[lab],
                        edgecolor=SURFACE, lw=2,
                        label=lab if (rd == 0 and role == "Carrot") else None)
                if w >= 0.06:
                    ax.text(left + w / 2, 2 - rd, f"{100 * w:.0f}%", ha="center",
                            va="center", fontsize=8,
                            color=SURFACE if lab != "defer" else INK)
                left += w
        ax.set_yticks([2, 1, 0], ["Round 1", "Round 2", "Round 3"])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.set_title(f"{role} (n={len(sub)})", fontsize=10.5, color=INK2)
    fig.legend(ncol=7, loc="lower center", fontsize=8.5, handlelength=1.2,
               columnspacing=1.0)
    h = fig.get_figheight()
    fig.text(0.02, 1 - 0.14 / h, "Speech-act mix per round, by role", fontsize=13,
             fontweight="bold", va="top")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.22, wspace=0.08)
    fig.savefig(OUT / "figures" / "label_mix_by_role.png", dpi=170)
    print("wrote figures/label_mix_by_role.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["label", "analyze"])
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N unlabeled turns")
    args = ap.parse_args()
    cfg = LabelingConfig.load(OUT / "config.json")
    cfg.save(OUT)
    if args.stage == "label":
        if cfg.gen_backend == "vllm":
            stage_label_vllm(cfg, args.limit)
        elif cfg.gen_backend == "openrouter":
            stage_label(cfg, args.limit)
        else:
            raise ValueError(f"unknown judge backend {cfg.gen_backend!r}")
    else:
        stage_analyze(cfg)


if __name__ == "__main__":
    main()
