"""Token-count distributions of the games (README appendix; exploratory, not
preregistered): reasoning tokens and tool-call tokens per model call, read
from the Inspect logs so that discarded attempts, excluded games and vote
turns are included (the transcripts hold only the accepted call of each
discussion turn of the valid games). CPU only: tokenizer, no weights.

Definitions (repeated in the README appendix):
  reasoning tokens   provider-reported ``usage.reasoning_tokens`` of one call
  tool-call tokens   Qwen3.5-27B tokenizer count of the call's tool call as
                     the chat template renders it:
                     <tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n
                     </parameter>\\n</function>\\n</tool_call>
  announcement tokens  tokenizer count of the ``message`` argument alone
  text tokens        tokenizer count of the plain text the call wrote between
                     its thinking and its tool call (``acting_text``)
  provider residual  ``usage.output_tokens - usage.reasoning_tokens``
  accepted call      the call of a turn that carries the executed tool call
  cap hit            a call with stop reason ``max_tokens``

  figures/tokens_*.svg, token_counts.json (every number of the appendix)

Run:  HF_HUB_OFFLINE=1 uv run --no-sync python experiments/2026-10-12-fibs-probes/token_counts.py
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import extract  # noqa: E402
from analyze import boot_auroc  # noqa: E402
from common import ARMS, OUT, REPO, load_config, read_jsonl  # noqa: E402

from game_config import GameConfig  # noqa: E402  (engine, via common's sys.path)

MAIN = ("3p", "5p")
TOOL = {"disc": ("MakeAnnouncementTool", extract.ANNOUNCED), "vote": ("CastEliminationVoteTool", extract.VOTED)}
ROLE_COL = {"Carrot": "#1f77b4", "Parsnip": "#d62728"}          # as figures.py
ROUND_COL = ("#9ecae1", "#4292c6", "#084594")                   # rounds 1-3, one hue light -> dark
C_ACC, C_DISC, C_EXCL = "#555555", "#e6820e", "#7b4fa3"


def render_tool_call(name: str, arguments: dict) -> str:
    """The tool call as the Qwen3.5 chat template writes it (and as the model emitted it)."""
    params = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in arguments.items())
    return f"<tool_call>\n<function={name}>\n{params}</function>\n</tool_call>"


def collect(arm: str, game_cfg: GameConfig, cap: int) -> list[dict]:
    """One record per model call of every played game of the arm."""
    valid = {g["game_id"] for g in json.loads((OUT / f"games_{arm}.json").read_text())["games"]}
    excluded = {x["game_id"] for x in json.loads((OUT / f"excluded_{arm}.json").read_text())}
    names, calls, seen = game_cfg.names(), [], set()
    for path in extract.game_logs(OUT / f"inspect_logs_{arm}"):
        gid = f"g{int(path.stem.removeprefix('game_')):03d}"
        if (gid in valid) == (gid in excluded):
            raise ValueError(f"{arm} {gid}: must be in exactly one of games_/excluded_")
        seen.add(gid)
        (sample,) = read_eval_log(str(path), resolve_attachments=True).samples
        roles = {name: role for _, name, role, _ in extract.ROLE_RE.findall(sample.messages[-1].text)}
        if sorted(roles) != sorted(names):
            raise ValueError(f"{arm} {gid}: roles in the summary {roles} != players {names}")
        for ti, t in enumerate(extract.turns(sample)):
            system, prompt = t["first"].input[0].text, t["first"].input[1].text
            speaker = extract.NAME_RE.match(system).group(1)
            kind = "vote" if extract.VOTE_MARK in prompt else "disc"
            rnd = pos = None
            if kind == "disc":
                rnd = int(extract.ROUND_RE.search(prompt).group(1)) - 1
                (pos,) = [i for i, (_, _, s) in enumerate(extract.ORDER_RE.findall(prompt)) if s.startswith("YOUR TURN")]
            tool_name, ok = TOOL[kind]
            done = [x for x in t["tools"] if x.function == tool_name and str(x.result).startswith(ok)]
            if len(done) != len(t["tools"]) or len(done) > 1:
                raise ValueError(f"{arm} {gid} turn {ti}: unexpected tool events {[x.function for x in t['tools']]}")
            for ci, c in enumerate(t["calls"]):
                if len(c.input) != 2 or c.config.max_tokens != cap:
                    raise ValueError(f"{arm} {gid} turn {ti} call {ci}: {len(c.input)} input messages, "
                                     f"max_tokens {c.config.max_tokens}")
                tcs = c.output.message.tool_calls or []
                if len(tcs) > 1:
                    raise ValueError(f"{arm} {gid} turn {ti} call {ci}: {len(tcs)} tool calls")
                if tcs and (ci != len(t["calls"]) - 1 or not done or tcs[0].id != done[0].id
                            or tcs[0].function != tool_name):
                    raise ValueError(f"{arm} {gid} turn {ti} call {ci}: tool call is not the turn's executed one")
                if (c.output.stop_reason == "max_tokens") != (c.output.usage.output_tokens == cap):
                    raise ValueError(f"{arm} {gid} turn {ti} call {ci}: stop {c.output.stop_reason} "
                                     f"with {c.output.usage.output_tokens} output tokens")
                calls.append({
                    "arm": arm, "game_id": gid, "valid": gid in valid, "kind": kind, "turn": ti,
                    "round": rnd, "position": pos, "speaker": speaker, "role": roles[speaker],
                    "attempt": ci, "n_calls": len(t["calls"]), "accepted": bool(tcs),
                    "stop": c.output.stop_reason,
                    "output_tokens": c.output.usage.output_tokens,
                    "reasoning_tokens": c.output.usage.reasoning_tokens,
                    "reasoning": extract.reasoning_blocks(c.output.message),
                    "text": c.output.message.text,
                    "tool_call": render_tool_call(tcs[0].function, tcs[0].arguments) if tcs else None,
                    "announcement": tcs[0].arguments["message"] if tcs and kind == "disc" else None,
                })
            if done and not calls[-1]["accepted"]:
                raise ValueError(f"{arm} {gid} turn {ti}: executed tool call not found in the last call")
    if seen != valid | excluded:
        raise ValueError(f"{arm}: logs {len(seen)} != valid {len(valid)} + excluded {len(excluded)}")
    return calls


def check_against_transcripts(arm: str, calls: list[dict]) -> None:
    """The accepted discussion calls of the valid games ARE the transcript rows."""
    acc = {(c["game_id"], c["round"], c["position"]): c for c in calls
           if c["kind"] == "disc" and c["valid"] and c["accepted"]}
    rows = read_jsonl(OUT / f"transcripts_{arm}.jsonl")
    if len(rows) != len(acc):
        raise ValueError(f"{arm}: {len(rows)} transcript rows, {len(acc)} accepted calls in the logs")
    for r in rows:
        c = acc[(r["game_id"], r["round"], r["position"])]
        got = (c["reasoning_tokens"], c["n_calls"], c["stop"], c["announcement"], c["text"], c["role"], c["reasoning"])
        want = (r["reasoning_tokens"], r["n_calls"], r["reasoning_stop"], r["announcement"], r["acting_text"],
                r["role"], r["reasoning"])
        if got != want:
            raise ValueError(f"{arm} {r['game_id']} r{r['round']} p{r['position']}: log and transcript disagree")


def add_token_counts(calls: list[dict], tok) -> None:
    def count(texts: list[str]) -> list[int]:
        return [len(x) for x in tok(texts, add_special_tokens=False).input_ids]
    for key, out in (("tool_call", "tool_call_tokens"), ("announcement", "announcement_tokens"),
                     ("text", "text_tokens"), ("reasoning", "reasoning_retok")):
        idx = [i for i, c in enumerate(calls) if c[key]]
        for c in calls:
            c[out] = 0 if c[key] == "" else None
        for i, n in zip(idx, count([calls[i][key] for i in idx])):
            calls[i][out] = n


def dist(v) -> dict:
    a = np.asarray(v, dtype=float)
    return {"n": int(len(a)), "min": float(a.min()), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()), "mean": float(a.mean())}


def cohen_d(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp)


def turn_summary(calls: list[dict], kind: str, cap: int, cfg) -> dict:
    """The per-arm table row for one kind of turn."""
    sel = [c for c in calls if c["kind"] == kind]
    acc = [c for c in sel if c["valid"] and c["accepted"]]
    turns = {(c["game_id"], c["turn"]): c["n_calls"] for c in sel if c["valid"]}
    if len(turns) != len(acc):
        raise ValueError(f"{len(turns)} {kind} turns in valid games but {len(acc)} accepted calls")
    hit = [c for c in sel if c["stop"] == "max_tokens"]
    out = {
        "n_turns": len(acc),
        "reasoning_tokens": dist([c["reasoning_tokens"] for c in acc]),
        "tool_call_tokens": dist([c["tool_call_tokens"] for c in acc]),
        "text_tokens_when_present": dist([c["text_tokens"] for c in acc if c["text"]]),
        "n_with_text": sum(bool(c["text"]) for c in acc),
        "frac_accepted_stop_not_tool_calls": float(np.mean([c["stop"] != "tool_calls" for c in acc])),
        "accepted_stop_reasons": dict(Counter(c["stop"] for c in acc)),
        "n_turns_resampled": sum(n > 1 for n in turns.values()),
        "frac_turns_resampled": float(np.mean([n > 1 for n in turns.values()])),
        "n_calls_per_turn": dict(sorted(Counter(turns.values()).items())),
        "cap_hits": {"accepted": sum(c["accepted"] for c in hit),
                     "discarded_valid_games": sum(c["valid"] and not c["accepted"] for c in hit),
                     "excluded_games": sum(not c["valid"] for c in hit)},
        "discarded_valid_games": dict(Counter(c["stop"] for c in sel if c["valid"] and not c["accepted"])),
        "failed_excluded_games": dict(Counter(c["stop"] for c in sel if not c["valid"] and not c["accepted"])),
        "accepted_excluded_games": sum(c["accepted"] and not c["valid"] for c in sel),
        "n_accepted_near_cap": sum(c["output_tokens"] >= cfg.tc_near_cap_frac * cap for c in acc),
        "max_accepted_output_tokens": max(c["output_tokens"] for c in acc),
    }
    if kind == "disc":
        ann = [c["announcement"] for c in acc]
        out["announcement_tokens"] = dist([c["announcement_tokens"] for c in acc])
        out["n_announcement_empty"] = sum(not a.strip() for a in ann)
        out["n_announcement_short"] = sum(c["announcement_tokens"] < cfg.tc_short_announcement_tokens for c in acc)
        cut = [a for a in ann if a.rstrip()[-1] not in cfg.tc_terminal_chars]
        out["n_announcement_no_terminal_char"] = len(cut)
        out["announcement_no_terminal_char_tails"] = [a[-80:] for a in cut]
        out["shortest_announcements"] = sorted(ann, key=len)[:3]
    return out


def crosscheck(calls: list[dict]) -> dict:
    """Provider usage vs the tokenizer, over every accepted call (all games)."""
    acc = [c for c in calls if c["accepted"]]
    resid = np.array([c["output_tokens"] - c["reasoning_tokens"] for c in acc])
    ours = np.array([c["tool_call_tokens"] + c["text_tokens"] for c in acc])
    overhead = resid - ours
    r_diff = np.array([c["reasoning_tokens"] - c["reasoning_retok"] for c in acc])
    return {"n": len(acc),
            "residual_minus_toolcall_minus_text": {str(k): v for k, v in sorted(Counter(overhead.tolist()).items())},
            "overhead_by_has_text": {
                "no_text": {str(k): v for k, v in sorted(Counter(overhead[[not c["text"] for c in acc]].tolist()).items())},
                "text": {str(k): v for k, v in sorted(Counter(overhead[[bool(c["text"]) for c in acc]].tolist()).items())}},
            "pearson_residual_vs_toolcall_plus_text": float(np.corrcoef(resid, ours)[0, 1]),
            "pearson_residual_vs_toolcall_only": float(np.corrcoef(resid, [c["tool_call_tokens"] for c in acc])[0, 1]),
            "reasoning_provider_minus_tokenizer": {str(k): v for k, v in sorted(Counter(r_diff.tolist()).items())},
            "frac_reasoning_exact": float(np.mean(r_diff == 0))}


def role_effects(calls: list[dict], cfg) -> dict:
    """Parsnip vs Carrot, accepted discussion calls of the valid 3p/5p games."""
    out = {}
    for label, arms in (("3p", ("3p",)), ("5p", ("5p",)), ("pooled", MAIN)):
        acc = [c for c in calls if c["arm"] in arms and c["kind"] == "disc" and c["valid"] and c["accepted"]]
        y = np.array([c["role"] == "Parsnip" for c in acc], dtype=int)
        g = np.array([f"{c['arm']}/{c['game_id']}" for c in acc])
        out[label] = {"n_parsnip": int(y.sum()), "n_carrot": int((1 - y).sum())}
        for q in ("reasoning_tokens", "announcement_tokens", "text_tokens"):
            s = np.array([c[q] for c in acc], dtype=float)
            b = boot_auroc(s, y, g, cfg.n_boot, cfg.boot_seed)
            out[label][q] = {
                "parsnip": dist(s[y == 1]), "carrot": dist(s[y == 0]),
                "cohen_d": cohen_d(s[y == 1], s[y == 0]),
                "cohen_d_log10": cohen_d(np.log10(s[y == 1]), np.log10(s[y == 0])) if s.min() > 0 else None,
                "auroc": b["auroc"], "auroc_ci95_game": b["ci95_game"]}
        has = np.array([bool(c["text"]) for c in acc])
        out[label]["frac_with_text"] = {"parsnip": float(has[y == 1].mean()), "carrot": float(has[y == 0].mean()),
                                        "n_parsnip": int(has[y == 1].sum()), "n_carrot": int(has[y == 0].sum())}
        re = np.array([c["n_calls"] > 1 for c in acc])
        out[label]["frac_turns_resampled"] = {"parsnip": float(re[y == 1].mean()), "carrot": float(re[y == 0].mean()),
                                              "n_parsnip": int(re[y == 1].sum()), "n_carrot": int(re[y == 0].sum())}
    return out


def cap_grid(calls: list[dict], n_players: int, rounds: int) -> dict:
    """Discussion turns of ALL played games (valid + excluded) by round x
    speaking position: turns, turns with at least one cap hit, cap-hit calls."""
    turns = np.zeros((rounds, n_players), dtype=int)
    hit_turns, hit_calls = turns.copy(), turns.copy()
    by_turn: dict[tuple, list[dict]] = {}
    for c in calls:
        if c["kind"] == "disc":
            by_turn.setdefault((c["game_id"], c["turn"]), []).append(c)
    for cs in by_turn.values():
        r, p = cs[0]["round"], cs[0]["position"]
        n = sum(c["stop"] == "max_tokens" for c in cs)
        turns[r, p] += 1; hit_turns[r, p] += n > 0; hit_calls[r, p] += n
    return {"turns": turns.tolist(), "turns_with_cap_hit": hit_turns.tolist(), "cap_hit_calls": hit_calls.tolist(),
            "share_of_cap_hit_calls_round1_first_speaker": float(hit_calls[0, 0] / hit_calls.sum())}


def breakdown(calls: list[dict]) -> dict:
    acc = [c for c in calls if c["kind"] == "disc" and c["valid"] and c["accepted"]]
    out = {}
    for name, key in (("role", lambda c: c["role"]), ("round", lambda c: f"r{c['round'] + 1}"),
                      ("position", lambda c: f"p{c['position'] + 1}"),
                      ("round_position", lambda c: f"r{c['round'] + 1}p{c['position'] + 1}")):
        out[name] = {}
        for k in sorted({key(c) for c in acc}):
            g = [c for c in acc if key(c) == k]
            out[name][k] = {"reasoning_tokens": dist([c["reasoning_tokens"] for c in g]),
                            "tool_call_tokens": dist([c["tool_call_tokens"] for c in g])}
    return out


# ── figures ──

def fig_reasoning(calls, kind: str, cap: int, bins, path) -> None:
    """Split view, log x: top row the accepted calls of the valid games, bottom
    row the calls that did not make it. Rows share axis ranges across arms."""
    fig, axes = plt.subplots(2, len(ARMS), figsize=(16, 6.5), sharex=True, sharey="row")
    for j, arm in enumerate(ARMS):
        sel = [c for c in calls if c["arm"] == arm and c["kind"] == kind]
        acc = [c["reasoning_tokens"] for c in sel if c["valid"] and c["accepted"]]
        disc = [c["reasoning_tokens"] for c in sel if c["valid"] and not c["accepted"]]
        excl = [c["reasoning_tokens"] for c in sel if not c["valid"] and not c["accepted"]]
        for v in (acc, disc, excl):
            if v and (min(v) < bins[0] or max(v) > bins[-1]):
                raise ValueError(f"{arm} {kind}: reasoning tokens outside the histogram range")
        axes[0, j].hist(acc, bins=bins, color=C_ACC)
        axes[0, j].set_title(f"{arm}: accepted calls, valid games (n={len(acc)})", fontsize=9)
        axes[1, j].hist([disc, excl], bins=bins, stacked=True, color=[C_DISC, C_EXCL],
                        label=[f"discarded attempt, valid game (n={len(disc)})",
                               f"failed attempt, excluded game (n={len(excl)})"])
        axes[1, j].set_title(f"{arm}: calls with no tool call", fontsize=9)
        axes[1, j].legend(fontsize=7, loc="upper left")
        axes[1, j].set_xlabel("reasoning tokens per call (log scale)")
        for ax in axes[:, j]:
            ax.axvline(cap, ls="--", color="k", lw=0.8)
            ax.set_xscale("log"); ax.set_xlim(bins[0], bins[-1] * 1.15)
    axes[0, 0].annotate(f"cap {cap:,}", (cap, 0.97), xycoords=("data", "axes fraction"), ha="right", va="top",
                        fontsize=7, xytext=(-3, 0), textcoords="offset points")
    axes[0, 0].set_ylabel("calls"); axes[1, 0].set_ylabel("calls")
    fig.suptitle(f"Reasoning tokens per {'discussion' if kind == 'disc' else 'vote'} call", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_toolcall(calls, bins, path) -> None:
    fig, axes = plt.subplots(1, len(ARMS), figsize=(16, 3.6), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS):
        acc = [c for c in calls if c["arm"] == arm and c["kind"] == "disc" and c["valid"] and c["accepted"]]
        groups = [[c["tool_call_tokens"] for c in acc if c["role"] == role] for role in ROLE_COL]
        ax.hist(groups, bins=bins, stacked=True, color=list(ROLE_COL.values()),
                label=[f"{role} (n={len(g)})" for role, g in zip(ROLE_COL, groups)])
        ax.set_title(f"{arm}: announcement tool call (n={len(acc)})", fontsize=9)
        ax.set_xlabel("tool-call tokens per accepted call"); ax.set_xlim(bins[0], bins[-1])
        ax.legend(fontsize=7)
    axes[0].set_ylabel("calls")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_crosscheck(calls, path) -> None:
    acc = [c for c in calls if c["kind"] == "disc" and c["accepted"]]
    resid = np.array([c["output_tokens"] - c["reasoning_tokens"] for c in acc])
    tc = np.array([c["tool_call_tokens"] for c in acc])
    txt = np.array([c["text_tokens"] for c in acc])
    has = txt > 0
    hi = 1.05 * max(resid.max(), (tc + txt).max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, x, label in ((axes[0], tc, "tool-call tokens (tokenizer)"),
                         (axes[1], tc + txt, "tool-call + post-thinking text tokens (tokenizer)")):
        ax.plot([0, hi], [0, hi], color="grey", lw=0.8, ls=":")
        ax.scatter(x[~has], resid[~has], s=8, color=C_ACC, label=f"no text before the tool call (n={(~has).sum()})")
        ax.scatter(x[has], resid[has], s=8, color=C_DISC, label=f"text before the tool call (n={has.sum()})")
        ax.set_xlabel(label); ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.legend(fontsize=7, loc="lower right")
    axes[0].set_ylabel("provider: output tokens − reasoning tokens")
    fig.suptitle("Accepted discussion calls, all arms, valid + excluded games", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _boxes(ax, groups, positions, colors, width):
    bp = ax.boxplot(groups, positions=positions, widths=width, patch_artist=True, manage_ticks=False,
                    flierprops={"marker": ".", "markersize": 2, "alpha": 0.4}, medianprops={"color": "k"})
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)


def _quantity_rows(axes, cap: int, tc_max: float) -> None:
    for ax in axes[0]:
        ax.set_yscale("log"); ax.set_ylim(50, cap * 1.3)
        ax.axhline(cap, ls="--", color="k", lw=0.8)
    for ax in axes[1]:
        ax.set_ylim(0, tc_max)
    axes[0, 0].set_ylabel("reasoning tokens (log scale)\ndashed: cap")
    axes[1, 0].set_ylabel("tool-call tokens")


def fig_by_role(calls, cap: int, tc_max: float, path) -> None:
    fig, axes = plt.subplots(2, len(ARMS), figsize=(13, 6.5), sharex=True, sharey="row")
    for j, arm in enumerate(ARMS):
        acc = [c for c in calls if c["arm"] == arm and c["kind"] == "disc" and c["valid"] and c["accepted"]]
        for i, q in enumerate(("reasoning_tokens", "tool_call_tokens")):
            present = [(k, role) for k, role in enumerate(ROLE_COL) if any(c["role"] == role for c in acc)]
            _boxes(axes[i, j], [[c[q] for c in acc if c["role"] == role] for _, role in present],
                   [k for k, _ in present], [ROLE_COL[role] for _, role in present], 0.6)
            axes[i, j].set_xticks(range(len(ROLE_COL))); axes[i, j].set_xticklabels(list(ROLE_COL))
            axes[i, j].set_xlim(-0.6, len(ROLE_COL) - 0.4)
        axes[0, j].set_title(arm)
    _quantity_rows(axes, cap, tc_max)
    fig.suptitle("Accepted discussion calls of the valid games, by role", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_by_round_position(calls, cap: int, tc_max: float, n_pos: int, rounds: int, path) -> None:
    fig, axes = plt.subplots(2, len(ARMS), figsize=(16, 6.5), sharex=True, sharey="row")
    w = 0.8 / rounds
    for j, arm in enumerate(ARMS):
        acc = [c for c in calls if c["arm"] == arm and c["kind"] == "disc" and c["valid"] and c["accepted"]]
        for i, q in enumerate(("reasoning_tokens", "tool_call_tokens")):
            for rd in range(rounds):
                ps = sorted({c["position"] for c in acc})
                _boxes(axes[i, j], [[c[q] for c in acc if c["round"] == rd and c["position"] == p] for p in ps],
                       [p + (rd - (rounds - 1) / 2) * w for p in ps], [ROUND_COL[rd]] * len(ps), w * 0.9)
            axes[i, j].set_xticks(range(n_pos)); axes[i, j].set_xticklabels(range(1, n_pos + 1))
            axes[i, j].set_xlim(-0.6, n_pos - 0.4)
        axes[0, j].set_title(arm)
        axes[1, j].set_xlabel("speaking position within the round")
    _quantity_rows(axes, cap, tc_max)
    axes[0, 0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=ROUND_COL[rd], alpha=0.75, ec="k")
                               for rd in range(rounds)],
                      labels=[f"round {rd + 1}" for rd in range(rounds)], fontsize=7, loc="lower left")
    fig.suptitle("Accepted discussion calls of the valid games, by round and speaking position", fontsize=11)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_cap_hits(grids: dict, n_pos: int, rounds: int, path) -> None:
    rate = {arm: np.array(g["turns_with_cap_hit"]) / np.array(g["turns"]) for arm, g in grids.items()}
    vmax = max(r.max() for r in rate.values())
    fig, axes = plt.subplots(1, len(ARMS), figsize=(16, 3.4), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS):
        full = np.full((rounds, n_pos), np.nan)
        full[:, :rate[arm].shape[1]] = rate[arm]
        im = ax.imshow(full, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        g = grids[arm]
        for r in range(rounds):
            for p in range(rate[arm].shape[1]):
                ax.text(p, r, f"{g['turns_with_cap_hit'][r][p]}/{g['turns'][r][p]}\n({g['cap_hit_calls'][r][p]} calls)",
                        ha="center", va="center", fontsize=7,
                        color="white" if rate[arm][r, p] > 0.6 * vmax else "black")
        ax.set_xticks(range(n_pos)); ax.set_xticklabels(range(1, n_pos + 1))
        ax.set_yticks(range(rounds)); ax.set_yticklabels([f"round {r + 1}" for r in range(rounds)])
        ax.set_xlabel("speaking position within the round")
        ax.set_title(f"{arm}: turns with ≥1 cap hit / turns", fontsize=9)
    fig.colorbar(im, ax=list(axes), label="fraction of turns with a cap hit", pad=0.01)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def markdown_table(res: dict) -> str:
    rows = [
        ("discussion turns (accepted calls)", lambda s: f"{s['n_turns']:,}"),
        ("reasoning tokens: median / p95 / max",
         lambda s: "{median:,.0f} / {p95:,.0f} / {max:,.0f}".format(**s["reasoning_tokens"])),
        ("tool-call tokens: median / p95 / max",
         lambda s: "{median:.0f} / {p95:.0f} / {max:.0f}".format(**s["tool_call_tokens"])),
        ("tool-call tokens: min", lambda s: f"{s['tool_call_tokens']['min']:.0f}"),
        ("announcement tokens: median / p95 / max",
         lambda s: "{median:.0f} / {p95:.0f} / {max:.0f}".format(**s["announcement_tokens"])),
        ("accepted calls, stop reason ≠ `tool_calls`", lambda s: f"{s['frac_accepted_stop_not_tool_calls']:.1%}"),
        ("turns with `n_calls` > 1",
         lambda s: f"{s['n_turns_resampled']} ({s['frac_turns_resampled']:.1%})"),
        ("cap hits: accepted / discarded in valid games / in excluded games",
         lambda s: "{accepted} / {discarded_valid_games} / {excluded_games}".format(**s["cap_hits"])),
        ("discarded attempts in valid games that stopped without a tool call (`stop`)",
         lambda s: str(s["discarded_valid_games"].get("stop", 0))),
        (f"accepted calls with output ≥ {res['near_cap_frac']:.0%} of the cap",
         lambda s: str(s["n_accepted_near_cap"])),
        ("largest accepted call, output tokens", lambda s: f"{s['max_accepted_output_tokens']:,}"),
        ("accepted calls with text between thinking and tool call",
         lambda s: f"{s['n_with_text']} ({s['n_with_text'] / s['n_turns']:.1%})"),
        (f"announcements empty / under {res['short_announcement_tokens']} tokens / no terminal punctuation",
         lambda s: f"{s['n_announcement_empty']} / {s['n_announcement_short']} / {s['n_announcement_no_terminal_char']}"),
    ]
    lines = ["| | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    lines += [f"| {name} | " + " | ".join(f(res["arms"][arm]["disc"]) for arm in ARMS) + " |" for name, f in rows]
    return "\n".join(lines)


def main():
    cfg = load_config()
    fig_dir = OUT / "figures"
    fig_dir.mkdir(exist_ok=True)
    game_cfgs = {arm: GameConfig.from_yaml(REPO / cfg.game_configs[arm]) for arm in ARMS}
    (cap,) = {g.max_tokens for g in game_cfgs.values()}
    (rounds,) = {g.num_rounds_discussion for g in game_cfgs.values()}
    n_pos = max(g.num_players for g in game_cfgs.values())
    tok = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.model_revision)

    calls = []
    for arm in ARMS:
        arm_calls = collect(arm, game_cfgs[arm], cap)
        check_against_transcripts(arm, arm_calls)
        calls += arm_calls
    add_token_counts(calls, tok)

    res = {"cap": cap, "near_cap_frac": cfg.tc_near_cap_frac,
           "short_announcement_tokens": cfg.tc_short_announcement_tokens,
           "n_calls_total": len(calls), "n_calls_without_usage": 0,   # collect() raises on a call without usage
           "arms": {}, "crosscheck": crosscheck(calls), "role_effects": role_effects(calls, cfg)}
    for arm in ARMS:
        ac = [c for c in calls if c["arm"] == arm]
        res["arms"][arm] = {
            "disc": turn_summary(ac, "disc", cap, cfg), "vote": turn_summary(ac, "vote", cap, cfg),
            "cap_hits_by_round_position": cap_grid(ac, game_cfgs[arm].num_players, rounds),
            "breakdown": breakdown(ac),
            "excluded_games": sorted({c["game_id"] for c in ac if not c["valid"]}),
            "excluded_games_failed_turn_kinds": dict(Counter(
                kind for kind, _ in {(c["kind"], (c["game_id"], c["turn"])) for c in ac
                                     if not c["valid"] and not c["accepted"] and c["attempt"] == c["n_calls"] - 1})),
        }
    (OUT / "token_counts.json").write_text(json.dumps(res, indent=1) + "\n")

    lo = min(c["reasoning_tokens"] for c in calls)
    r_bins = np.logspace(np.floor(np.log10(lo)), np.log10(cap), cfg.tc_reasoning_bins + 1)
    disc_acc = [c for c in calls if c["kind"] == "disc" and c["valid"] and c["accepted"]]
    tc_max = 1.05 * max(c["tool_call_tokens"] for c in disc_acc)
    fig_reasoning(calls, "disc", cap, r_bins, fig_dir / "tokens_reasoning.svg")
    fig_reasoning(calls, "vote", cap, r_bins, fig_dir / "tokens_reasoning_vote.svg")
    fig_toolcall(calls, np.linspace(0, tc_max, cfg.tc_toolcall_bins + 1), fig_dir / "tokens_toolcall.svg")
    fig_crosscheck(calls, fig_dir / "tokens_crosscheck.svg")
    fig_by_role(calls, cap, tc_max, fig_dir / "tokens_by_role.svg")
    fig_by_round_position(calls, cap, tc_max, n_pos, rounds, fig_dir / "tokens_by_round_position.svg")
    fig_cap_hits({arm: res["arms"][arm]["cap_hits_by_round_position"] for arm in ARMS}, n_pos, rounds,
                 fig_dir / "tokens_cap_hits.svg")
    print(markdown_table(res))
    print(f"wrote token_counts.json and figures/tokens_*.svg ({len(calls)} model calls)")


if __name__ == "__main__":
    main()
