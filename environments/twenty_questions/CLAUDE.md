# Agent instructions — environments/twenty_questions

Twenty Questions (suspect vs auditor, judged every turn). Design, primitives
and run commands: `README.md`. This directory is its own uv project; run
everything with `uv run --directory environments/twenty_questions ...`
(on this host uv comes from nix: `nix shell nixpkgs#uv --command ...`).
Experiments that use it live in the repo-level `experiments/<date>/` and
select it with `uv run --project environments/twenty_questions ...`.

- Every game parameter and prompt is a `GameConfig` field (YAML). Do not
  hard-code variants in the engine; add a config field.
- Public rules are a user message, identical for every player; hidden
  dynamics go in system prompts only.
- Fail loudly: invalid tool calls get bounded, recorded retries and then
  `GameError`. The only forgiving parses are the researcher's decision
  (2026-10-16): `answer` takes yes/no in any case/whitespace (stored as
  `" yes"`/`" no"`), `judge` takes "true"/"false" strings, and arguments sent
  as one JSON object repeated (`{"a":1}{"a":1}`, a GLM/OpenRouter glitch) are
  read as that object (raw text in `Turn.raw_arguments_text`). The model's
  exact arguments stay in `Turn.raw_arguments`; do not widen this without asking.
- Stored answers are exactly `" yes"` / `" no"` (single tokens).
- Concept sets are built by `scripts/build_concepts.py` (sha256-checked
  THINGS download); do not hand-edit `concepts/*.json`.
