# Agent instructions — environments/twenty_questions

Twenty Questions (suspect vs auditor, judged every turn). Design, primitives
and run commands: `README.md`. This directory is its own uv project; run
everything with `uv run --directory environments/twenty_questions ...`
(on this host uv comes from nix: `nix shell nixpkgs#uv --command ...`).
Experiments that use it live in the repo-level `experiments/<date>/` and
select it with `uv run --project environments/twenty_questions ...`.

- Every game parameter and prompt is a `GameConfig` field (YAML). Do not
  hard-code variants in the engine; add a config field.
- Public rules are a user message, identical for every player; they state
  both players' goals. System prompts are only for dynamics the other player
  must not see (empty by default).
- Model-visible text (prompts, tool descriptions) uses the seat names
  (Player 1 / Player 2), never suspect/auditor, and never mentions
  deception, lying or truthfulness — `test_models_never_see_role_or_deception_words`
  enforces it. Code keeps the suspect/auditor names.
- `configs/default.yaml` and `configs/openrouter_smoke.yaml` carry every
  prompt; regenerate their `prompts:` section with
  `scripts/write_prompt_configs.py` after changing a `PromptConfig` default.
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
