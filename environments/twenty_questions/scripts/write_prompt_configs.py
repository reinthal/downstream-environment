"""Rewrite the `prompts:` section of configs/default.yaml and
configs/openrouter_smoke.yaml from the PromptConfig defaults, with a comment
above each group. Everything before `# ── Prompts` in each file is kept.

Run:  uv run --directory environments/twenty_questions python scripts/write_prompt_configs.py
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from twenty_questions.config import GameConfig, PromptConfig, _Block, _ConfigDumper

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
TARGETS = ["default.yaml", "openrouter_smoke.yaml"]
MARK = "# ── Prompts"

HEADER = """# ── Prompts ──────────────────────────────────────────────────────────────
# Every prompt text (a test checks that all are listed; unknown keys raise).
# {placeholders} are filled at render time; in a prompt that has
# placeholders a literal brace must be doubled ({{ }}). Block scalars (|-)
# keep the text exactly, minus the final newline. {player1} (holds the
# secret), {player2} (guesses) and {max_questions} work in every prompt.
# No prompt mentions deception, lying or truthfulness.
prompts:
"""

COMMENTS = {
    "rules": "Rules: one user message, identical for both players. Placeholders:\n"
             "{secret_space} (secret_space_*), {final_answer_rule} (final_answer_rule_*,\n"
             "by final_answer_mode), {clue_answer_rule}, {clue_examples} and\n"
             "{clue_counter_example} (by clues / clue_policy),\n"
             "{clue_request_rule} and {clue_request_example} (by clue_requests),\n"
             "{goals} (goals_*, by the suspect's objective). In bounded modes\n"
             "concept_list is appended as its own paragraph.",
    "goals_competitive": "{goals}, by suspects[0].objective",
    "secret_space_free": "{secret_space}: free_form vs the bounded modes; {n} = concept-set size,\n"
                         "{visibility} = visibility_both (auditor.show_concepts) or visibility_suspect",
    "final_answer_rule_end": "{final_answer_rule}, by final_answer_mode",
    "concept_list": "appended to the rules (bounded modes) and the judge prompts; {n}, {concepts}",
    "clue_answer_rule": "{clue_answer_rule} / {clue_counter_example}: lines of the rules, dropped\n"
                        "with every other clue line when clues is false",
    "clue_rule_helpful": "{clue_rule} (inside clue_answer_rule), by clue_policy",
    "clue_examples_helpful": "{clue_examples}, by clue_policy",
    "clue_request_rule_turn": "{clue_request_rule} / {clue_request_example}: lines of the rules, by\n"
                              "clue_request_uses_turn; both dropped when clue_requests is false",
    "role_suspect": "tells each player which seat it holds",
    "system_cooperative": "System prompts, by suspect objective, and for the guesser ('' -> no\n"
                          "system message). Only for dynamics the other player must not see.",
    "commit_instruction": "Turn messages",
    "question_to_suspect": "to the secret holder; {turn} {question}",
    "clue_request_to_suspect": "clue requests: to the secret holder {turn}; to the guesser {clue} and\n"
                               "{next_turn} (clue_request_uses_turn) or {turn} (free requests)",
    "answer_to_auditor": "to the guesser; {answer} {clue} (no_clue if none) {next_turn}",
    "final_answer_wrong": "final_answer_mode: continue; {next_turn}",
    "judge_prompt_question": "Judge (the guesser's conversation, forked, secret revealed): {secret},\n"
                             "{content} (the question + answer, or the submitted final answer).\n"
                             "Bounded modes append concept_list (judge.show_concepts).",
}


def prompts_section() -> str:
    out = [HEADER]
    for name, value in asdict(PromptConfig()).items():
        if name in COMMENTS:
            if out[-1] != HEADER:
                out.append("\n")
            out.append("".join(f"  # {line}\n" for line in COMMENTS[name].splitlines()))
        dumped = yaml.dump({name: _Block(value) if value else value}, Dumper=_ConfigDumper,
                           default_flow_style=False, sort_keys=False, allow_unicode=True, width=10**6)
        out.append("".join(f"  {line}\n" for line in dumped.splitlines()))
    return "".join(out)


def main():
    section = prompts_section()
    for name in TARGETS:
        path = CONFIGS / name
        text = path.read_text()
        head = text[:text.index(MARK)] if MARK in text else text.split("\nprompts:")[0].rstrip() + "\n\n"
        path.write_text(head + section)
        if GameConfig.from_yaml(path).prompts != PromptConfig():
            raise SystemExit(f"{name}: prompts do not round-trip")
        print(f"wrote configs/{name}")


if __name__ == "__main__":
    main()
