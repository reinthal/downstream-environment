# Labeling Carrot / Parsnip answers

Goal: put speech-act labels on every public announcement in the base games
(deferring, probing, accusing, defending, coordinating, ...) and test how the
mix shifts across rounds — in particular the preregistered hypothesis below.

Data: `transcripts_base.jsonl` (symlink → `experiments/2026-09-29`), 100 games
× 3 rounds × 5 players = 1,500 turns, each with private reasoning
(`reasoning`) and public `announcement`.

## Preregistered hypothesis (researcher, 2026-10-08, before any labeling)

> Agents defer the first 3 rounds and only start acting on round 4. Agents
> are mostly deferring in early rounds. Accusing rarely happens (<5% at 95%
> CI) in round 1.

Design note recorded at preregistration time: **the games have exactly 3
rounds** (then the vote) — there is no round 4 in any dataset, so the first
clause is untestable as stated. The testable reading: deferral dominates
rounds 1–2 and "acting" (accusing / vote coordination) concentrates in the
final round. The round-1 accusation rate gets a 95% CI (game-level
clustering: turns within a game are not independent).

## Exploration (pre-labeling, 2026-10-08)

Read in full with private reasoning: games s0g0, s0g7, s0g13. Announcements
only: s2g4, s4g11 (other seed groups). Plus crude regex counts over all
1,500 announcements. Observations that motivated the label set:

- **Round 1 is ritualized deference.** Openers solicit "initial thoughts",
  everyone agrees it is "too early for accusations". The private reasoning is
  explicit about this being a *policy*, for both roles: Carrots reason "I
  should not accuse anyone yet since there's no evidence"; the Parsnip
  reasons "I shouldn't accuse anyone prematurely... keep it vague to avoid
  standing out" (s0g0 Dave). Role claims ("I'm a Carrot") are common and
  made by both roles. No targeted accusations seen; regex hits for
  suspect-naming in round 1 are false positives (generic "has anyone noticed
  anything suspicious?" solicitations).
- **Round 2 is probing and the first targeted criticism.** Demands for
  "concrete observations", questions directed at named players, and the
  first soft flags ("Dave's redirections", "Charlie's intense pressure").
  Criticized players start defending and counter-flagging. In s4g11 the
  *Parsnip* launches the first targeted accusation of the game in round 2 —
  early aggression is not a Carrot-only behavior.
- **Round 3 is a phase transition, not a gradual ramp.** Explicit vote
  declarations ("I am voting to eliminate X") appear almost exclusively here,
  with heavy bandwagoning ("I stand with Alice and Eve"). Crude regex over
  all 500 turns/round: vote-talk 2 → 9 → 339, defer-markers 155 → 37 → 5
  across rounds 1 → 2 → 3.
- **First-mover disadvantage / accusation cascades.** The round-3 opener who
  accuses first is frequently counter-accused ("sudden aggressive push =
  classic Parsnip tactic") and the cascade flips onto them: in s0g0 and
  s2g4 an honest first accuser was voted out-momentum while the Parsnip
  survived by *joining the majority*. The Parsnip's modal round-3 move when
  not under fire is bandwagoning with the consensus.
- Announcements routinely mix acts (accuse + coordinate; defend +
  counter-accuse), so labeling must be multi-label or primary-act +
  secondary flags.

## Proposed label set (to be frozen before the labeling run)

Per announcement, primary speech act:

| label | definition | example |
|---|---|---|
| `defer` | withholds judgment; solicits input; "too early"; content-free agreement | s0g0 R1 all five turns |
| `probe` | demands specifics or directs a question at named player(s), no suspicion stated | "Dave, what patterns have you noticed?" |
| `flag` | names a player's *behavior* as suspicious without calling for elimination | "Dave's redirections are worth tracking" |
| `accuse` | names a player as the likely Parsnip / calls to vote them out | "I suggest we vote Dave out" |
| `defend_self` | rebuts suspicion directed at the speaker | s0g7 R3 Dave |
| `defend_other` | argues against suspicion of another player | s4g11 R2 Alice on Dave |
| `coordinate` | declares a vote / aligns explicitly with named others | "I stand with Alice and Eve, voting Bob" |

Secondary binary flags: `role_claim` ("I'm a Carrot"), `vote_declared`.

## Labeling run (phase 2)

`label.py --stage label`: every announcement is labeled by a judge that sees
only the public discussion so far + the turn to label (no roles, no private
reasoning), and must answer with exactly one strict-JSON
`<answer>{"primary", "role_claim", "vote_declared"}</answer>` tag;
unparsable output is retried with a different seed, then raises. Judge =
local Qwen3.8-27B via `rollouts.generation.VLLMBackend` (cfg.gen_backend
`"vllm"`; an OpenRouter path exists behind `"openrouter"` but the repo key
was dead). Note the judge (Qwen3.8) is a different model generation than
the players (Qwen3.5) — deliberate, avoids pure self-judging. Kernel check on this box (driver 535, A40): FlashAttention-2 for
full-attention layers, Triton/FLA GDN kernels for the hybrid linear-attn
layers, FlashInfer sampler off. Output: `judge_labels_base.jsonl`.

`label.py --stage analyze`: per-round label mix, round-1 accuse rate with
game-clustered 95% bootstrap CIs (10k reps) → `results.json`, `figures/`.

`violin_explorer.ipynb` (run in `.venv`; needs `plotly`): interactive
plotly violins of DYL-probe scores — alpaca control vs Carrot vs Parsnip
by round, with a dropdown filtering the game violins to one speech-act
label. Joins `scores_{alpaca,base}.csv` (symlinks →
`2026-10-07-dyl-alpaca`) with `judge_labels_base.jsonl`.
`violin_explorer.html` is the exported standalone version.

## Deliberate limitations

- Base arm only; steered/nothink arms are out of scope.
- Labels are on the public announcement only — the private reasoning is held
  out as a sanity check for ambiguous cases, not a label input (the auditor
  sees only public text, and labels should match what an observer can see).
- Exploration above read 5 of 100 games closely; the quantitative claims in
  it are crude regex counts, superseded by the labeling run.
