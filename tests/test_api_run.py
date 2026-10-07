"""Logic checks for the API auditor loop with a fake client (no network)."""
from types import SimpleNamespace as NS

from auditor.api_run import Spend, audit
from auditor.config import ApiAuditorConfig

NAMES = ["Alice", "Bob", "Charlie", "Dave", "Eve"]
CONV = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]


def usage(i=100, o=50):
    return NS(input_tokens=i, output_tokens=o, cache_read_input_tokens=0, cache_creation_input_tokens=0)


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self._create)

    def _create(self, messages, **kw):
        self.calls.append({"messages": [dict(m) for m in messages], **kw})
        return self.replies.pop(0)


def cfg(**kw):
    return ApiAuditorConfig(model_id="x", layer=0, api_model="m", max_tool_calls=2, **kw)


def test_tool_loop_runs_code_and_returns_answer():
    tool = NS(type="tool_use", id="t1", input={"code": "print(6*7)"})
    replies = [NS(content=[NS(type="thinking", thinking="plan"), tool], stop_reason="tool_use", usage=usage()),
               NS(content=[NS(type="text", text="It is <answer>dave</answer>")], stop_reason="end_turn",
                  usage=usage())]
    client = FakeClient(replies)
    res = audit(client, cfg(), CONV, {"scores.csv": "x\n1\n"}, NAMES)
    assert res["tool"]["n_tool_calls"] == 1 and res["tool"]["tool_calls"][0]["output"] == "42"
    assert res["answer_source"] == "tag" and "dave" in res["text"]
    # the tool result went back with the matching id, and tools were offered
    sent = client.calls[1]["messages"][-1]["content"][0]
    assert sent == {"type": "tool_result", "tool_use_id": "t1", "content": "42"}
    assert client.calls[0]["tools"][0]["name"] == "python"
    assert res["usage"]["input_tokens"] == 200 and res["thinking"] == "plan"


def test_missing_answer_gets_one_followup_and_no_tools_without_files():
    replies = [NS(content=[NS(type="text", text="Probably Bob.")], stop_reason="end_turn", usage=usage()),
               NS(content=[NS(type="text", text="<answer>Bob</answer>")], stop_reason="end_turn", usage=usage())]
    client = FakeClient(replies)
    res = audit(client, cfg(), CONV, None, NAMES)
    assert res["answer_source"] == "followup" and res["tool"] is None
    assert "tools" not in client.calls[0]


def test_tool_call_limit_is_enforced():
    tool = lambda i: NS(type="tool_use", id=f"t{i}", input={"code": "print(1)"})  # noqa: E731
    replies = [NS(content=[tool(i)], stop_reason="tool_use", usage=usage()) for i in range(3)]
    replies.append(NS(content=[NS(type="text", text="<answer>Eve</answer>")], stop_reason="end_turn", usage=usage()))
    res = audit(FakeClient(replies), cfg(), CONV, {"scores.csv": "x\n"}, NAMES)
    assert res["tool"]["n_tool_calls"] == 2 and res["answer_source"] == "tag"


def test_spend_cap():
    s = Spend(cfg(max_cost_usd=0.001))
    s.add(s.cost({"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 0,
                  "cache_creation_input_tokens": 0}))
    assert round(s.usd, 6) == 0.006 and s.over()


def test_load_key_rejects_malformed_values(monkeypatch):
    import pytest
    from auditor.api_run import load_key
    monkeypatch.setenv("TEST_KEY", "sk-or-v1-abc123\n")
    assert load_key("TEST_KEY", "https://openrouter.ai/api") == "sk-or-v1-abc123"
    for bad in ('"sk-or-v1-abc"', "\x1b[200~sk-or-v1-abc", "Bearer sk-or-v1-abc", "sk-ant-abc"):
        monkeypatch.setenv("TEST_KEY", bad)
        with pytest.raises(SystemExit):
            load_key("TEST_KEY", "https://openrouter.ai/api")
