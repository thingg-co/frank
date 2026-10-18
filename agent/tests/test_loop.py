import json
from pathlib import Path

import pytest

from frank_local.client import ToolCall, Turn
from frank_local.config import Settings
from frank_local.loop import COMPACT_NOTE, Agent, TurnInterrupted
from frank_local.session import Session


class FakeClient:
    """Scripted turns: each item is a Turn to return for successive requests."""

    def __init__(self, turns, usage=None):
        self.turns = list(turns)
        self.usage = usage or {}
        self.model = "fake"
        self.base_url = "http://fake/v1"
        self.requests = []
        self.completions = []

    def stream(self, messages, tools=None, max_tokens=None, stop=None):
        self.requests.append([dict(m) for m in messages])
        turn = self.turns.pop(0)
        turn.usage = dict(self.usage)
        if turn.text:
            yield ("text", turn.text)
        yield ("done", turn)

    def complete(self, messages, max_tokens=None, tools=None):
        self.completions.append(messages)
        return "SUMMARY OF EVERYTHING"

    def probe(self):
        return {"models": ["fake"], "context": 1000}


def make_agent(tmp_path, client, ui, **kw):
    settings = Settings(permission_mode="auto", **kw)
    return Agent(settings, client, ui, tmp_path, context_window=100_000)


def test_tool_round_trip(tmp_path, ui):
    (tmp_path / "hello.txt").write_text("hi there\n")
    client = FakeClient([
        Turn(text="", tool_calls=[ToolCall(id="c1", name="read_file", arguments=json.dumps({"path": "hello.txt"}))], reasoning="let me look"),
        Turn(text="The file says hi."),
    ])
    agent = make_agent(tmp_path, client, ui)
    out = agent.run_turn("what does hello.txt say?")
    assert out == "The file says hi."
    roles = [m["role"] for m in agent.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert agent.messages[2]["tool_calls"][0]["function"]["name"] == "read_file"
    assert agent.messages[2]["reasoning_content"] == "let me look"
    assert agent.messages[3]["tool_call_id"] == "c1" and "hi there" in agent.messages[3]["content"]
    # second request carried the tool result
    assert client.requests[1][3]["role"] == "tool"


def test_interrupt_raises_turn_interrupted_and_repairs(tmp_path, ui):
    client = FakeClient([Turn(text="first answer"), Turn(text="second answer")])
    agent = make_agent(tmp_path, client, ui)
    agent.session = Session.new(tmp_path, tmp_path, "fake")  # the repl always has one; hold() repairs on interrupt
    agent._interrupt("clarification", "wait, use tabs")
    with pytest.raises(TurnInterrupted) as ei:
        agent.run_turn("do the thing")
    e = ei.value
    assert e.mode == "clarification" and e.text == "wait, use tabs"
    # the interrupted turn was repaired, so the clarification can follow it
    assert agent.messages[-1]["role"] == "assistant"
    assert agent.messages[-1]["content"] == "[interrupted]"
    # a fresh turn works normally afterwards
    assert agent.run_turn("continue") == "second answer"


def test_plain_stop_interrupts_without_message(tmp_path, ui):
    client = FakeClient([Turn(text="a")])
    agent = make_agent(tmp_path, client, ui)
    agent._stop()
    with pytest.raises(TurnInterrupted) as ei:
        agent.run_turn("go")
    assert ei.value.mode == ""


def test_queue_later_sends_after_the_turn(tmp_path, ui):
    client = FakeClient([Turn(text="first answer"), Turn(text="second answer")])
    agent = make_agent(tmp_path, client, ui)
    agent._queue_later("now do the other thing")
    with pytest.raises(TurnInterrupted) as ei:
        agent.run_turn("do the thing")
    assert ei.value.mode == "instruction" and ei.value.text == "now do the other thing"
    assert not agent.has_pending()


def test_send_now_sends_the_queued_message(tmp_path, ui):
    client = FakeClient([Turn(text="a"), Turn(text="b")])
    agent = make_agent(tmp_path, client, ui)
    agent._queue_later("send me now")
    agent._send_now()
    with pytest.raises(TurnInterrupted) as ei:
        agent.run_turn("go")
    assert ei.value.text == "send me now"
    assert not agent.has_pending()


def test_plain_stop_keeps_the_queue(tmp_path, ui):
    client = FakeClient([Turn(text="a")])
    agent = make_agent(tmp_path, client, ui)
    agent._queue_later("still wanted")
    agent._stop()
    with pytest.raises(TurnInterrupted) as ei:
        agent.run_turn("go")
    assert ei.value.mode == ""  # the turn stopped
    assert agent.has_pending()  # the queued message survived for the next turn
    assert agent._drain() == ("instruction", "still wanted")


def test_reset_interrupt_clears_pending(tmp_path, ui):
    client = FakeClient([Turn(text="a")])
    agent = make_agent(tmp_path, client, ui)
    agent._interrupt("instruction", "no wait")
    agent._reset_interrupt()
    assert agent.run_turn("go") == "a"
    assert not agent.interrupt.is_set()


def test_interrupt_after_tools_is_drained(tmp_path, ui):
    (tmp_path / "hello.txt").write_text("hi there\n")
    client = FakeClient([
        Turn(text="", tool_calls=[ToolCall(id="c1", name="read_file", arguments=json.dumps({"path": "hello.txt"}))]),
        Turn(text="done"),
    ])
    agent = make_agent(tmp_path, client, ui)
    tool = agent.registry.get("read_file")
    orig_run = tool.run

    def run(args, ctx):
        result = orig_run(args, ctx)
        agent._interrupt("instruction", "stop and do it differently")  # lands in the drain after the tools
        return result

    tool.run = run
    with pytest.raises(TurnInterrupted) as ei:
        agent.run_turn("read the file")
    assert ei.value.mode == "instruction" and ei.value.text == "stop and do it differently"
    # the tool result is still in the history, so the next turn can build on it
    assert agent.messages[-1]["role"] == "tool"


def test_unknown_tool_and_bad_json(tmp_path, ui):
    client = FakeClient([
        Turn(tool_calls=[ToolCall(id="c1", name="nope", arguments="{}"), ToolCall(id="c2", name="bash", arguments="{not json")]),
        Turn(text="done"),
    ])
    agent = make_agent(tmp_path, client, ui)
    agent.run_turn("go")
    tool_msgs = [m for m in agent.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert "unknown tool" in tool_msgs[0]["content"] and "JSON" in tool_msgs[1]["content"]


def test_permission_denied_in_readonly(tmp_path, ui):
    client = FakeClient([
        Turn(tool_calls=[ToolCall(id="c1", name="bash", arguments=json.dumps({"command": "touch x"}))]),
        Turn(text="ok"),
    ])
    settings = Settings(permission_mode="ask")
    agent = Agent(settings, client, ui, tmp_path, context_window=100_000)  # non-interactive ui denies
    agent.run_turn("make a file")
    assert not (tmp_path / "x").exists()
    assert "permission denied" in [m for m in agent.messages if m["role"] == "tool"][0]["content"]


def test_compaction_triggers(tmp_path, ui):
    client = FakeClient([Turn(text="first"), Turn(text="second")], usage={"prompt_tokens": 90_000, "completion_tokens": 10})
    agent = make_agent(tmp_path, client, ui)
    agent.run_turn("one")
    assert agent.tokens_used() == 90_010
    agent.run_turn("two")  # maybe_compact runs before the second request
    assert client.completions, "summary was requested"
    assert agent.messages[1]["content"].startswith(COMPACT_NOTE)
    assert agent.messages[1]["content"].endswith("SUMMARY OF EVERYTHING")
    assert [m["role"] for m in agent.messages] == ["system", "user", "assistant", "user", "assistant"]


def test_repair_after_interrupt(tmp_path, ui):
    client = FakeClient([])
    agent = make_agent(tmp_path, client, ui)
    agent.messages.append({"role": "user", "content": "hi"})
    agent.messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": "c9", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]})
    agent.repair()
    assert agent.messages[-2]["role"] == "tool" and agent.messages[-2]["tool_call_id"] == "c9"
    assert agent.messages[-1] == {"role": "assistant", "content": "[interrupted]"}


def test_subagent(tmp_path, ui):
    (tmp_path / "notes.md").write_text("secret: 42\n")
    client = FakeClient([
        Turn(tool_calls=[ToolCall(id="p1", name="spawn_subagent", arguments=json.dumps({"task": "find the secret in notes.md"}))]),
        # child turns
        Turn(tool_calls=[ToolCall(id="k1", name="read_file", arguments=json.dumps({"path": "notes.md"}))]),
        Turn(text="The secret is 42."),
        # parent final
        Turn(text="It is 42."),
    ])
    agent = make_agent(tmp_path, client, ui)
    out = agent.run_turn("what is the secret?")
    assert out == "It is 42."
    tool_msg = [m for m in agent.messages if m["role"] == "tool"][0]
    assert tool_msg["content"] == "The secret is 42."
    # the child could not spawn further subagents and only had read-only tools
    child_request = client.requests[1]
    assert "Subagent brief" in child_request[0]["content"]


def test_instruction_files_are_read_again_after_compaction_and_clear(tmp_path, ui, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "CLAUDE.md").write_text("Use tabs.")
    agent = make_agent(tmp_path, FakeClient([Turn(text="ok")]), ui)
    assert "Use tabs." in agent.messages[0]["content"]
    agent.run_turn("hi")
    (tmp_path / "CLAUDE.md").write_text("Use tabs.\n\n- The build runs on omega.")
    assert "The build runs on omega." not in agent.messages[0]["content"]  # read when the session starts
    agent.compact()
    assert "The build runs on omega." in agent.messages[0]["content"]
    (tmp_path / "CLAUDE.md").write_text("Use spaces.")
    agent.reset()
    assert "Use spaces." in agent.messages[0]["content"] and "Use tabs." not in agent.messages[0]["content"]
