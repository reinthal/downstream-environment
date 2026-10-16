import re
import asyncio
import json

import pytest

from twenty_questions import task as task_mod
from twenty_questions.backends import ScriptedBackend, tool_call_output as out
from twenty_questions.config import GameConfig
from twenty_questions.engine import TwentyQuestionsGame
from twenty_questions.players import GameError
from twenty_questions.probes import read_views


def judge_says(secret_word_seen_in=None):
    """Judge: correct iff the auditor's last action mentions the secret."""
    def policy(msgs, tools):
        prompt = msgs[-1].text
        secret = prompt.split("secret is: ")[1].split("\n")[0]
        last = re.search(r"The auditor's (?:last question|final answer): (.*)", prompt).group(1)
        return out("judge", {"correct": secret.lower() in last.lower()})
    return ScriptedBackend(policy)


def auditor_script(questions, final="nothing"):
    it = iter(questions)
    def policy(msgs, tools):
        if "ask_question" in tools:
            q = next(it, None)
            if q is not None:
                return out("ask_question", {"question": q})
        return out("final_answer", {"answer": final})
    return ScriptedBackend(policy)


def honest_suspect(answer=" no", clue=None):
    args = {"answer": answer} | ({"clue": clue} if clue else {})
    return ScriptedBackend(lambda m, t: out("answer", args))


def play(cfg, seed, suspect, auditor, judge=None):
    game = TwentyQuestionsGame(cfg, seed, {"Suspect": suspect, "Auditor": auditor, "judge": judge or judge_says()})
    return game, asyncio.run(game.play())


def test_bounded_random_injects_seeded_commit():
    cfg = GameConfig(max_questions=3)
    _, r1 = play(cfg, 7, honest_suspect(), auditor_script(["a?", "b?"]))
    _, r2 = play(cfg, 7, honest_suspect(), auditor_script(["a?", "b?"]))
    _, r3 = play(cfg, 8, honest_suspect(), auditor_script(["a?", "b?"]))
    commit = r1.turns[0]
    assert commit.phase == "commit" and commit.injected and commit.reasoning == ""
    assert r1.secret == r2.secret and r1.secret != r3.secret
    # same prompt as bounded_free_form: rules + role + commit instruction
    assert "commit it with the commit_secret tool" in commit.input_messages[-1]["content"]


def test_auditor_never_sees_secret_and_judge_fork_is_discarded():
    cfg = GameConfig(max_questions=4)
    game, r = play(cfg, 1, honest_suspect(clue="it is small"), auditor_script(["q1?", "q2?", "q3?"]))
    auditor_text = json.dumps([m.text for m in game.auditor.messages])
    assert r.secret not in auditor_text.split("POSSIBLE SECRETS")[0]  # rules part
    assert "JUDGE MODE" not in auditor_text
    judges = [t for t in r.turns if t.phase == "judge"]
    assert len(judges) == 4 and all("JUDGE MODE" in t.input_messages[-1]["content"] for t in judges)
    assert r.end_reason == "final_answer_wrong" and r.winner == "suspect" and r.end_turn == 4


def test_question_naming_secret_ends_game():
    cfg = GameConfig(max_questions=20)
    _, miss = play(cfg, 2, honest_suspect(" yes"), auditor_script(["Is it alive?"] * 30))
    assert miss.end_turn == 20 and miss.winner == "suspect" and miss.exchanges[-1].kind == "final_answer"
    _, hit = play(cfg, 2, honest_suspect(" yes"), auditor_script(["Is it alive?", f"Is it a {miss.secret}?"]))
    assert hit.winner == "auditor" and hit.end_turn == 2 and hit.end_reason == "judged_correct"
    assert [e.kind for e in hit.exchanges] == ["question", "question"]


def test_last_turn_offers_only_final_answer():
    cfg = GameConfig(max_questions=2)
    _, r = play(cfg, 0, honest_suspect(), auditor_script(["q?", "q?", "q?"]))
    last = [t for t in r.turns if t.role == "auditor"][-1]
    assert [s["function"]["name"] for s in last.tools] == ["final_answer"]
    assert "last turn" in last.input_messages[-1]["content"]


@pytest.mark.parametrize("sent", ["yes", " yes", "Yes", "YES ", "\tyes\n"])
def test_answer_is_forgiving_and_keeps_raw(sent):
    cfg = GameConfig(max_questions=2)
    _, r = play(cfg, 0, honest_suspect(sent, clue="  "), auditor_script(["q?"]))
    ans = next(t for t in r.turns if t.phase == "answer")
    assert ans.arguments["answer"] == " yes" and not ans.failed_attempts
    assert ans.raw_arguments["answer"] == sent
    assert r.exchanges[0].answer == " yes"


def test_invalid_answer_errors_are_recorded():
    cfg = GameConfig(max_questions=2)
    replies = iter([out("answer", {"answer": "maybe"}), out("answer", {"answer": "no", "clue": "  "})])
    _, r = play(cfg, 0, ScriptedBackend(lambda m, t: next(replies)), auditor_script(["q?"]))
    ans = next(t for t in r.turns if t.phase == "answer")
    assert ans.arguments == {"answer": " no", "clue": None}
    assert len(ans.failed_attempts) == 1 and "'maybe'" in ans.failed_attempts[0]["error"]
    # the error exchange is part of what the model saw on the successful attempt
    assert ans.input_messages[-1]["role"] == "tool" and "'maybe'" in ans.input_messages[-1]["content"]


def test_judge_accepts_string_booleans():
    cfg = GameConfig(max_questions=1)
    _, r = play(cfg, 0, honest_suspect(), auditor_script([], final="x"),
                ScriptedBackend(lambda m, t: out("judge", {"correct": "True"})))
    assert r.winner == "auditor" and r.turns[-1].raw_arguments == {"correct": "True"}


def test_max_attempts_raises():
    cfg = GameConfig(max_questions=2, max_attempts=2)
    with pytest.raises(GameError):
        play(cfg, 0, ScriptedBackend(lambda m, t: out("answer", {"answer": "maybe"})), auditor_script(["q?"]))


def test_no_tool_call_is_resampled_not_appended():
    cfg = GameConfig(max_questions=2)
    from inspect_ai.model import ModelOutput
    replies = iter([ModelOutput.from_content("scripted", "I think..."), out("answer", {"answer": " no"})])
    _, r = play(cfg, 0, ScriptedBackend(lambda m, t: next(replies)), auditor_script(["q?"]))
    ans = next(t for t in r.turns if t.phase == "answer")
    assert len(ans.failed_attempts) == 1 and ans.input_messages[-1]["role"] == "user"


def test_bounded_free_form_rejects_unlisted_secret():
    cfg = GameConfig(secret_mode="bounded_free_form", max_questions=1)
    replies = iter([out("commit_secret", {"secret": "unicorn"}), out("commit_secret", {"secret": "apple"}),
                    ])
    suspect = ScriptedBackend(lambda m, t: next(replies))
    _, r = play(cfg, 0, suspect, auditor_script([], final="apple"))
    assert r.secret == "apple" and not r.turns[0].injected and len(r.turns[0].failed_attempts) == 1
    assert r.winner == "auditor" and r.end_turn == 1


def test_free_form_and_hidden_objective():
    cfg = GameConfig.from_dict({"secret_mode": "free_form", "concept_set": None, "max_questions": 1,
                                "suspects": [{"objective": "adversarial"}]})
    _, r = play(cfg, 0, ScriptedBackend(lambda m, t: out("commit_secret", {"secret": "the Eiffel Tower"})),
                auditor_script([], final="eiffel tower"))
    sys_msg = r.turns[0].input_messages[0]
    assert sys_msg["role"] == "system" and "FAIL" in sys_msg["content"]
    assert "anything at all" in r.turns[0].input_messages[1]["content"]
    assert r.secret == "the Eiffel Tower" and r.n_concepts is None


def test_probe_views_round_trip(tmp_path):
    cfg = GameConfig(max_questions=3)
    _, r = play(cfg, 0, honest_suspect(clue="c"), auditor_script(["q1?", "q2?"]))
    path = tmp_path / "transcripts_x.jsonl"
    path.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in r.turns))
    views = read_views(path)
    assert [v.turn for v in views] == [1, 2]
    msgs = views[0].messages()
    assert msgs[-1]["tool_calls"][0]["function"] == {"name": "answer", "arguments": {"answer": " no", "clue": "c"}}
    # probes re-render what the model sent, not the canonical value
    _, r2 = play(cfg, 0, honest_suspect("No ", clue="c"), auditor_script(["q1?", "q2?"]))
    path.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in r2.turns))
    assert read_views(path)[0].arguments["answer"] == "No "


def test_inspect_task_runs_scripted(monkeypatch, tmp_path):
    from inspect_ai import eval as inspect_eval
    cfg = GameConfig(max_questions=3)
    monkeypatch.setattr(task_mod, "make_backends", lambda c: {
        "Suspect": honest_suspect(), "Auditor": auditor_script(["q?"]), "judge": judge_says()})
    [log] = inspect_eval(task_mod.make_task(cfg, 2), model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert log.status == "success" and len(log.samples) == 2
    g = log.samples[0].store["game"]
    assert g["end_turn"] == 2 and g["winner"] == "suspect"


def test_final_answer_continue_mode():
    cfg = GameConfig(max_questions=4, final_answer_mode="continue")
    guesses = iter(["apple", "pear"])
    def auditor(msgs, tools):
        g = next(guesses, None)
        return out("final_answer", {"answer": g}) if g else out("ask_question", {"question": "q?"}) \
            if "ask_question" in tools else out("final_answer", {"answer": "plum"})
    game, r = play(cfg, 0, honest_suspect(), ScriptedBackend(auditor))
    assert [e.kind for e in r.exchanges] == ["final_answer", "final_answer", "question", "final_answer"]
    assert r.end_reason == "turns_exhausted" and r.end_turn == 4 and r.winner == "suspect"
    wrong = [t for t in r.turns if t.phase == "final_answer"][0]
    assert wrong.tool_result.startswith("That is not the secret")
    judge = [t for t in r.turns if t.phase == "judge"][0]
    # the judge's fork answers the pending call with the neutral ack
    assert judge.input_messages[-2]["role"] == "tool"
    assert judge.input_messages[-2]["content"] == cfg.prompts.final_answer_ack
    assert "does not end the game" in r.turns[1].input_messages[-1]["content"]


def test_concept_list_hidden_from_auditor_shown_to_judge():
    cfg = GameConfig(max_questions=2)
    game, r = play(cfg, 0, honest_suspect(), auditor_script(["q?"]))
    auditor_rules = game.auditor.messages[0].text
    suspect_rules = game.suspect.messages[0].text
    assert "POSSIBLE SECRETS" not in auditor_rules and "POSSIBLE SECRETS" in suspect_rules
    # identical rules text; only the appended list differs
    assert auditor_rules.split("\n\nYou are")[0] == suspect_rules.split("\n\nTHE 512 POSSIBLE SECRETS")[0]
    assert "only the suspect sees the list" in auditor_rules
    judge = next(t for t in r.turns if t.phase == "judge")
    assert "POSSIBLE SECRETS" in judge.input_messages[-1]["content"]
    shown = GameConfig(max_questions=2, auditor={"show_concepts": True})
    g2, _ = play(shown, 0, honest_suspect(), auditor_script(["q?"]))
    assert "POSSIBLE SECRETS" in g2.auditor.messages[0].text and "both players see" in g2.auditor.messages[0].text


def test_judge_prompt_depends_on_action():
    cfg = GameConfig(max_questions=2)
    _, r = play(cfg, 0, honest_suspect(), auditor_script(["q?"], final="maybe a goat"))
    judges = [t for t in r.turns if t.phase == "judge"]
    assert "The auditor's last question: q? (answer: no)" in judges[0].input_messages[-1]["content"]
    final = judges[1].input_messages[-1]["content"]
    assert "The auditor's final answer: maybe a goat" in final and "hedges" in final


def test_duplicated_arguments_are_repaired_and_recorded():
    from inspect_ai.model._call_tools import parse_tool_call
    from inspect_ai.model import ChatMessageAssistant, ModelOutput
    from twenty_questions.players import duplicated_arguments

    raw = '{"answer":"yes"}{"answer":"yes"}'
    broken = parse_tool_call("c9", "answer", raw)        # pins Inspect's error format
    assert broken.parse_error and duplicated_arguments(broken.parse_error) == ({"answer": "yes"}, raw)
    for bad in ('{"answer":"yes"}{"answer":"no"}', '{"answer":"yes"}x', '{"answer":"ye'):
        err = parse_tool_call("c", "answer", bad).parse_error
        assert err and duplicated_arguments(err) is None, bad

    def suspect(msgs, tools):
        return ModelOutput.from_message(ChatMessageAssistant(content="", tool_calls=[broken]),
                                        stop_reason="tool_calls")
    cfg = GameConfig(max_questions=2)
    game, r = play(cfg, 0, ScriptedBackend(suspect), auditor_script(["q?"]))
    ans = next(t for t in r.turns if t.phase == "answer")
    assert ans.arguments["answer"] == " yes" and not ans.failed_attempts
    assert ans.raw_arguments_text == raw and ans.raw_arguments == {"answer": "yes"}
    sent = [m for m in game.suspect.messages if getattr(m, "tool_calls", None)][-1].tool_calls[0]
    assert sent.arguments == {"answer": "yes"} and sent.parse_error is None


def test_other_parse_errors_still_fail():
    from inspect_ai.model._call_tools import parse_tool_call
    from inspect_ai.model import ChatMessageAssistant, ModelOutput
    broken = parse_tool_call("c", "answer", '{"answer":"yes"}{"answer":"no"}')
    out_ = lambda m, t: ModelOutput.from_message(ChatMessageAssistant(content="", tool_calls=[broken]),
                                                 stop_reason="tool_calls")
    with pytest.raises(GameError):
        play(GameConfig(max_questions=2), 0, ScriptedBackend(out_), auditor_script(["q?"]))
