from types import SimpleNamespace

from conftest import delta, tc

from frank_local.client import StreamAccumulator


def test_text_and_reasoning():
    acc = StreamAccumulator()
    ev = acc.feed(delta(reasoning="think "))
    ev += acc.feed(delta(content="Hello"))
    ev += acc.feed(delta(content=" world"))
    assert acc.text == "Hello world" and acc.reasoning == "think "
    assert ev == [("reasoning", "think "), ("text", "Hello"), ("text", " world")]
    assert acc.tool_calls() == []


def test_tool_call_split_across_chunks():
    acc = StreamAccumulator()
    acc.feed(delta(tool_calls=[tc(0, id="call_1", name="bash", arguments="")]))
    acc.feed(delta(tool_calls=[tc(0, arguments='{"command": ')]))
    acc.feed(delta(tool_calls=[tc(0, arguments='"ls"}')]))
    acc.feed(delta(tool_calls=[tc(1, id="call_2", name="read_file", arguments='{"path":"x"}')]))
    calls = acc.tool_calls()
    assert [c.name for c in calls] == ["bash", "read_file"]
    assert calls[0].id == "call_1" and calls[0].parsed() == {"command": "ls"}
    assert calls[1].parsed() == {"path": "x"}


def test_missing_id_is_synthesised():
    acc = StreamAccumulator()
    acc.feed(delta(tool_calls=[tc(0, name="bash", arguments="{}")]))
    (call,) = acc.tool_calls()
    assert call.id.startswith("call_") and call.parsed() == {}


def test_tool_call_deltas_are_reported_while_they_stream():
    acc = StreamAccumulator()
    ev = acc.feed(delta(tool_calls=[tc(0, id="call_1", name="bash", arguments="")]))
    ev += acc.feed(delta(tool_calls=[tc(0, arguments='{"command": "ls"}')]))
    assert ev == [("tool", "bash"), ("tool", "bash")]


def test_llama_cpp_prompt_progress_is_asked_for_and_passed_on():
    from frank_local.client import Client
    from frank_local.config import Settings

    def chunk(progress=None, content=None, finish=None):
        choice = SimpleNamespace(delta=delta(content=content), finish_reason=finish)
        return SimpleNamespace(usage=None, choices=[choice], model_extra={"prompt_progress": progress} if progress else {})

    class Stream(list):
        def close(self):
            pass

    sent = {}

    def create(**kw):
        sent.update(kw)
        return Stream([chunk({"total": 100, "cache": 40, "processed": 70, "time_ms": 900}), chunk(content="hi", finish="stop")])

    c = Client(Settings(backend="llama.cpp", model="m"))
    c._oa = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    events = list(c.stream([{"role": "user", "content": "x"}]))
    assert sent["extra_body"]["return_progress"] is True
    assert events[0] == ("progress", {"total": 100, "cache": 40, "processed": 70, "time_ms": 900})
    assert ("text", "hi") in events and events[-1][0] == "done"
    assert c.complete([{"role": "user", "content": "x"}]) == "hi"  # complete() rides the same stream
