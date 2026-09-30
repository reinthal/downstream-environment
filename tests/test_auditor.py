"""Logic checks for auditor answer parsing (no model, no network)."""
from auditor.prompts import ANSWER_CUE, parse_guess

NAMES = ["Alice", "Bob", "Charlie", "Dave", "Eve"]


def test_parse_guess_reads_answer_tag():
    assert parse_guess("hmm\n</think>\n\nIt is <answer>charlie</answer>.", NAMES) == "Charlie"


def test_parse_guess_has_no_name_mention_fallback():
    # a cut-off answer must count as unanswered, not as the last name mentioned
    raw = "...\n</think>\n\nCharlie is 7/7 throughout, while Dave's accusation"
    assert parse_guess(raw, NAMES) is None


def test_parse_guess_prefers_public_then_last_tag_in_raw():
    raw = "format: <answer>Alice</answer> ... pick <answer>Eve</answer>\n</think>\n\nno tag"
    assert parse_guess(raw, NAMES) == "Eve"
    assert parse_guess("<answer>Bob</answer>\n</think>\n\n<answer>Dave</answer>", NAMES) == "Dave"


def test_answer_cue_completion_parses():
    raw = "t\n</think>\n\nCut off mid-sentence" + ANSWER_CUE + "Bob</answer>"
    assert parse_guess(raw, NAMES) == "Bob"
