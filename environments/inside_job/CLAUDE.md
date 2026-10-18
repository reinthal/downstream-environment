# environments/inside_job

Inside Job: a cooperative vault game with a possible hidden saboteur ("mole"),
built for testing whether a dynamic auditor with a white-box deception probe
adds value beyond the probe. A new game, not Carrot–Parsnip (whose engine
cannot express private fragments, crews or private entries); see
`experiments/2026-10-17-inside-job/README.md` for the design and why.

- `inside_job/engine.py` — the whole engine: `make_setup(seed, GameSpec)`
  (everything a seed fixes: mole seat, corrupted loyal agent, harmless
  confidential notes, codes, fragment holders, crews), `Mission.opened()`,
  and the ground-truth labels `label_claim` / `label_answer`.
- No LLM, prompts or model code here: runners live in `experiments/`.
- No dependencies; experiments put this directory on `sys.path`.
- Keep it deterministic: the same seed must give the same setup, so
  conditions can be paired by seed.
