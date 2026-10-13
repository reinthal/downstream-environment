"""One real probe input, shortened and labelled section by section, as a
self-contained HTML page (for presentations): what the model has in context
when a fibs probe scores an announcement.

The text is the exact rendering collect_acts.py scores (step() +
acts.render_followup), cut into sections at the template's turn markers;
long sections are elided in the middle, never reworded.

  probe_input_example.html

Run:  HF_HUB_OFFLINE=1 uv run --no-sync python experiments/2026-10-12-fibs-probes/probe_input_example.py
"""
from __future__ import annotations

import html

from transformers import AutoTokenizer

from acts import render_followup
from collect_acts import step
from common import OUT, load_config, model_path, read_jsonl

ARM, GAME, ROUND, SPEAKER = "3p", "g004", 2, "Charlie"   # a Parsnip turn that also carries private text
KEEP = 420       # characters kept at each end of an elided section

CSS = """
/* layout: one column; each section = label column + monospace excerpt */
:root{--bg:#f7f8f6;--fg:#1c2321;--muted:#5d6966;--line:#d5dbd8;--panel:#ffffff;
 --seen:#2f6f8f;--wrote:#9a5b13;--engine:#5f6b7a;--probe:#8a2d5c;--mark:#fde68a;--markfg:#1c2321;
 --sans:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151a19;--fg:#e6ebe9;--muted:#9aa7a3;--line:#2e3836;--panel:#1c2322;
 --seen:#7cc0e0;--wrote:#e6a55a;--engine:#a9b4c2;--probe:#e58fbb;--mark:#6b5a12;--markfg:#fff7d6;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#151a19;--fg:#e6ebe9;--muted:#9aa7a3;--line:#2e3836;--panel:#1c2322;
 --seen:#7cc0e0;--wrote:#e6a55a;--engine:#a9b4c2;--probe:#e58fbb;--mark:#6b5a12;--markfg:#fff7d6;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);font-size:15px;line-height:1.5}
main{max-width:1040px;margin:0 auto;padding-block:28px 56px;padding-inline:20px;display:flex;flex-direction:column;gap:18px}
h1{font-size:26px;line-height:1.2;margin:0;text-wrap:balance}
h2{font-size:17px;margin:14px 0 0}
p{margin:0;max-width:72ch}
.meta{color:var(--muted)}
.key{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px}
.key span::before{content:"";display:inline-block;width:10px;height:10px;margin-right:6px;background:currentColor}
.sec{display:grid;grid-template-columns:230px minmax(0,1fr);gap:14px;border-top:1px solid var(--line);padding-top:14px}
.lab{display:flex;flex-direction:column;gap:4px}
.lab b{font-size:14px}
.lab .who{font-size:12px;text-transform:uppercase;letter-spacing:.06em;font-weight:600}
.lab .note{font-size:13px;color:var(--muted)}
.lab .tok{font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
pre{margin:0;background:var(--panel);border:1px solid var(--line);padding:12px 14px;font-family:var(--mono);font-size:12.5px;line-height:1.5;
 white-space:pre-wrap;overflow-wrap:anywhere;min-width:0}
.cut{color:var(--muted);font-style:italic}
mark{background:var(--mark);color:var(--markfg);padding:0 2px}
.seen{color:var(--seen)}.wrote{color:var(--wrote)}.engine{color:var(--engine)}.probe{color:var(--probe)}
table{border-collapse:collapse;font-size:13.5px;width:100%}
td,th{border-top:1px solid var(--line);padding:7px 10px 7px 0;text-align:left;vertical-align:top}
code{font-family:var(--mono);font-size:12.5px}
.scroll{overflow-x:auto}
@media (max-width:720px){.sec{grid-template-columns:minmax(0,1fr)}}
"""


def cut(text: str, marks: list[str]) -> str:
    """HTML for a section: the middle elided past 2*KEEP characters, each
    string in ``marks`` highlighted where it survives the cut."""
    if len(text) > 2 * KEEP + 200:
        parts = [text[:KEEP], None, text[-KEEP:]]
        n = len(text) - 2 * KEEP
    else:
        parts, n = [text], 0
    out = ""
    for p in parts:
        if p is None:
            out += f'\n<span class="cut">[… {n:,} characters omitted …]</span>\n'
            continue
        h = html.escape(p)
        for m in marks:
            h = h.replace(html.escape(m), f"<mark>{html.escape(m)}</mark>")
        out += h
    return out


def main():
    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(model_path(cfg))
    (r,) = [x for x in read_jsonl(OUT / f"transcripts_{ARM}.jsonl")
            if (x["game_id"], x["round"], x["speaker"]) == (GAME, ROUND, SPEAKER)]
    msgs, tools = step(r)
    rd = {pf: render_followup(tok, msgs, cfg.followup_question, pf, tools=tools) for pf in cfg.prefills}
    s = tok.decode(rd[cfg.lie_prefill].ids)
    other = next(p for p in cfg.prefills if p != cfg.lie_prefill)

    def idx(needle: str, start: int = 0) -> int:
        i = s.index(needle, start)
        if s.count(needle, start) != 1 and needle.startswith("<|im_start|>") is False:
            raise ValueError(f"{needle!r} is not unique after {start}")
        return i

    a = idx(f"You are {r['speaker']}, playing")
    b = s.index("<|im_start|>user\n")
    c = idx("GROUP DISCUSSION", b)
    d = s.index("<|im_start|>assistant\n", b)
    e = idx("<tool_call>", d)
    f = s.index("<|im_start|>user\n<tool_response>", e)
    g = s.index("<|im_start|>user\n", f + 1)
    h = s.index("<|im_start|>assistant\n", g)
    bounds = [0, a, b, c, d, e, f, g, h, len(s)]
    role_line = f"You are the {r['role']} player."
    secs = [
        ("seen", "Player saw", "System: tool definitions", "Inserted by the chat template from the engine's tool schemas.", []),
        ("seen", "Player saw", "System: game rules", "Names the speaker. Does not state the role.", [f"You are {r['speaker']}"]),
        ("seen", "Player saw", "User: game state", "The role is stated here, under YOUR PRIVATE KNOWLEDGE.", [role_line]),
        ("seen", "Player saw", "User: discussion so far + instruction", "Earlier public announcements of all players, then the turn instruction.", []),
        ("wrote", "Player wrote", "Assistant: plain text before the tool call",
         "Private; other players never see it. Present in 202 of 2,400 turns (8%), empty otherwise. "
         "The <think> block the model wrote before this is dropped by the template.", [f"I am the {r['role']}", f"I'm the {r['role']}"]),
        ("wrote", "Player wrote", "Assistant: tool call = the public announcement", "The only part the other players see.", []),
        ("engine", "Engine reply", "Tool result", "Echoes the announcement back.", []),
        ("probe", "Added for the probe", "User: follow-up question", "Appended after the game; the player never saw it during play.", [cfg.followup_question]),
        ("probe", "Added for the probe", "Assistant: prefilled answer",
         f"Empty think block, then “{cfg.lie_prefill}” in one pass and “{other}” in a second. "
         "Linear and MLP read the full stop; EFC and axial read this turn and the follow-up question. "
         f"Score = probe(“{cfg.lie_prefill}”) − probe(“{other}”).", [cfg.lie_prefill]),
    ]
    rows = ""
    for k, (cls, who, title, note, marks) in enumerate(secs):
        text = s[bounds[k]:bounds[k + 1]]
        n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
        rows += (f'<section class="sec"><div class="lab"><span class="who {cls}">{who}</span><b>{k + 1}. {html.escape(title)}</b>'
                 f'<span class="note">{html.escape(note)}</span><span class="tok">{n_tok:,} tokens</span></div>'
                 f"<pre>{cut(text, marks)}</pre></section>\n")
    refs = [
        ("Messages of the announcement step (sections 1-7)", "collect_acts.py:34-48", "step(): system + prompt, the assistant's text and tool call, the tool result"),
        ("Follow-up and prefill appended, rendered, token boundaries (8-9)", "acts.py:36-67", "render_followup(): the model's chat template, empty think block, ends at <|im_end|>"),
        ("Question and prefills", "config.json:63-68", "followup_question, prefills, lie_prefill"),
        ("Chat template (what is kept or dropped)", "Qwen/Qwen3.5-27B@fc05daec chat_template.jinja:90-104", "reasoning is rendered only for the turn after the last user query"),
        ("Which tokens each probe reads", "score_probes.py:29-34; vendor/caught-in-the-act-probes/src/probe_inference/score.py:33", "score_one() and build_read_mask()"),
        ("Contrast score", "score_probes.py:75", "score(lie prefill) − score(the other)"),
        ("Engine prompts", "environments/carrot_parsnip/game_config.py:103, 189; Carrot_Parsnip.py:235; Carrot_Parsnip_Agents.py:170", "system prompt, turn instruction, private knowledge, tool result"),
    ]
    ref_rows = "".join(f"<tr><td>{html.escape(x)}</td><td><code>{html.escape(y)}</code></td><td>{html.escape(z)}</td></tr>" for x, y, z in refs)
    body = f"""<title>Fibs Probe Input</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&family=IBM+Plex+Sans:wght@400;600&display=swap">
<style>{CSS}</style>
<main>
<h1>What the model has in context when a fibs probe scores one announcement</h1>
<p class="meta">Real example, shortened: {ARM} game {GAME}, round {ROUND + 1}, {r['speaker']} ({r['role']}).
{rd[cfg.lie_prefill].n_tokens:,} tokens in total. Text is verbatim from the rendering that was scored; long sections are cut in the middle.</p>
<div class="key"><span class="seen">what the player saw</span><span class="wrote">what the player wrote</span>
<span class="engine">engine reply</span><span class="probe">added afterwards for the probe</span></div>
{rows}
<h2>Where this is built in the code</h2>
<div class="scroll"><table><tr><th>What</th><th>Where</th><th>Note</th></tr>{ref_rows}</table></div>
<p class="meta">Paths are relative to <code>experiments/2026-10-12-fibs-probes/</code> unless they start with another top-level directory.</p>
</main>
"""
    (OUT / "probe_input_example.html").write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1"></head><body style="margin:0">\n' + body + "</body></html>\n")
    print("sections (tokens):", [len(tok(s[bounds[k]:bounds[k + 1]], add_special_tokens=False)["input_ids"]) for k in range(len(secs))])


if __name__ == "__main__":
    main()
