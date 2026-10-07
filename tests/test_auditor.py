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


def test_grade_summary_appends_player_means():
    from auditor.prompts import build_transcript
    msgs = [{"game_id": "g", "round": 0, "position": i, "speaker": s, "announcement": "hi"}
            for i, s in enumerate(["Bob", "Alice", "Bob"])]
    grades = {("g", 0, 0): 2, ("g", 0, 1): 5, ("g", 0, 2): 4}
    assert "Average" not in build_transcript(msgs, grades)
    assert build_transcript(msgs, grades, summary=True).endswith("Alice 5.0, Bob 3.0")


def test_parse_tool_call():
    from auditor.tools import parse_tool_call
    out = ("plan the analysis\n</think>\n\nLet me look.\n<tool_call>\n<function=python>\n"
           "<parameter=code>\nprint(1)\n</parameter>\n</function>\n")
    assert parse_tool_call(out) == ("plan the analysis", "Let me look.", "print(1)")
    assert parse_tool_call("no call here\n</think>\n\n<answer>Bob</answer>") is None
    assert parse_tool_call("x</think>\n<tool_call>\n<function=python>\n</function>\n")[2] is None


def test_score_table_and_csv():
    from auditor.prompts import score_rows_csv, score_table
    rows = [{"round": 1, "position": 2, "player": "Bob", "score": -3.456}]
    assert score_rows_csv(rows) == "round,position,player,score\n1,2,Bob,-3.46\n"
    assert score_table(rows).splitlines()[-1].split() == ["1", "2", "Bob", "-3.46"]


def test_sandbox_runs_code_on_files():
    from auditor.sandbox import run_python
    out = run_python("import pandas as pd; print(pd.read_csv('s.csv').x.sum())", {"s.csv": "x\n1\n2\n"})
    assert out.strip() == "3"
