"""Raw follow-up-probe scores for every announcement, from the RunPod probe
endpoint (no GPU here).

The served model is Qwen3.5-27B truncated at layer 18 plus the follow-up
("did you lie") LR probe, pooled on the LAST token. For each announcement the
client renders the dialogue the speaker saw (system, reasoning prompt, its
reasoning, announcement prompt), the announcement as the assistant's reply,
the follow-up question, and the fixed "No" — the probe reads that "No".
Dialogues longer than cfg.max_len tokens are left-trimmed (the "No" is at the
end), as linear_probes.dyl_probe.encode_no_span does.

  raw_fu_{arm}.json   {game_id: [{round, position, player, score}]}, 1-based
                      round / position, score = log-odds (higher = more likely
                      deceptive) — the 2026-10-04 format auditor.run reads.
                      Silent turns have no row.
  scores_{arm}.csv    one row per turn with p(deceptive) and prompt length

Run:  set -a; source .env; set +a
      uv run --no-project --with requests --with transformers --with jinja2 \
          python experiments/2026-10-09-3p-game/score_probe.py --arm base
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import requests
from transformers import AutoTokenizer

from common import OUT, GameRunConfig, read_jsonl

from linear_probes.dyl_probe import FOLLOWUP_ANSWER, FOLLOWUP_QUESTION  # noqa: E402

API = "https://api.runpod.ai/v2"


def render(tok, rec: dict, max_len: int) -> tuple[str, int, bool]:
    """(text ending at the follow-up "No", its token count, whether trimmed)."""
    conv = [*rec["input_messages"],
            {"role": "assistant", "content": rec["announcement"]},
            {"role": "user", "content": FOLLOWUP_QUESTION}]
    text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False) + FOLLOWUP_ANSWER
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    n = len(enc["input_ids"])
    if n <= max_len:
        return text, n, False
    return text[enc["offset_mapping"][n - max_len][0]:], max_len, True


def classify(cfg: GameRunConfig, texts: list[str]) -> list[float]:
    r = requests.post(f"{API}/{cfg.probe_endpoint_id}/openai/classify",
                      headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"},
                      json={"input": texts}, timeout=cfg.probe_timeout_s)
    r.raise_for_status()
    data = r.json()["data"]
    if [d["index"] for d in data] != list(range(len(texts))):
        raise RuntimeError(f"classify returned indices {[d['index'] for d in data]}")
    return [d["probs"][0] for d in data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["base", "pilot"], required=True)
    args = ap.parse_args()
    cfg = GameRunConfig.load(OUT / "config.json")
    tok = AutoTokenizer.from_pretrained(cfg.probe_hf_repo)

    recs = [r for r in read_jsonl(OUT / f"transcripts_{args.arm}.jsonl") if r["announcement"]]
    rendered = [render(tok, r, cfg.max_len) for r in recs]
    print(f"{len(recs)} announcements; {sum(t for _, _, t in rendered)} left-trimmed to "
          f"{cfg.max_len} tokens; longest {max(n for _, n, _ in rendered)}", flush=True)

    probs: list[float] = []
    for i in range(0, len(recs), cfg.probe_batch):
        probs += classify(cfg, [text for text, _, _ in rendered[i:i + cfg.probe_batch]])
        print(f"scored {len(probs)}/{len(recs)}", flush=True)

    games = defaultdict(list)
    with open(OUT / f"scores_{args.arm}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game_id", "seed_group", "round", "position", "speaker", "role",
                    "followup_p", "followup_logodds", "prompt_tokens", "trimmed"])
        for r, (_, n, trimmed), p in zip(recs, rendered, probs):
            score = math.log(p / (1 - p))
            w.writerow([r["game_id"], r["seed_group"], r["round"], r["position"], r["speaker"],
                        r["role"], f"{p:.6f}", f"{score:.4f}", n, int(trimmed)])
            games[r["game_id"]].append({"round": r["round"] + 1, "position": r["position"] + 1,
                                        "player": r["speaker"], "score": round(score, 2)})
    for rows in games.values():
        rows.sort(key=lambda x: (x["round"], x["position"]))
    (OUT / f"raw_fu_{args.arm}.json").write_text(json.dumps(games, indent=1) + "\n")
    print(f"wrote raw_fu_{args.arm}.json ({len(games)} games) and scores_{args.arm}.csv")


if __name__ == "__main__":
    main()
