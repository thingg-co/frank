import json

from conftest import delta, tc
from test_loop import FakeClient, make_agent

from frank_local import thinking
from frank_local.client import Client, StreamAccumulator, ToolCall, Turn
from frank_local.config import Settings
from frank_local.prompt import build_system_prompt


def feed_all(chunks):
    acc = StreamAccumulator()
    events = []
    for c in chunks:
        events += acc.feed(delta(content=c))
    events += acc.finish()
    return acc, events


def test_think_block_split_across_chunks():
    acc, events = feed_all(["\n<th", "ink>plan the", " change</thi", "nk>\n\nDone", " now."])
    assert acc.reasoning == "plan the change"
    assert acc.text == "Done now."
    assert acc.raw == "\n<think>plan the change</think>\n\nDone now."
    assert [k for k, _ in events][0] == "reasoning" and events[-1] == ("text", " now.")


def test_think_tag_later_in_the_answer_is_text():
    acc, _ = feed_all(["Use a ", "<think>", " tag, then </think>."])
    assert acc.reasoning == "" and acc.text == "Use a <think> tag, then </think>."


def test_plain_text_and_short_answers_pass_through():
    assert feed_all(["<"])[0].text == "<"
    assert feed_all(["ok"])[0].text == "ok"
    assert feed_all(["<thin", "g>"])[0].text == "<thing>"


def test_unclosed_think_block_is_all_reasoning():
    acc, _ = feed_all(["<think>ran out of tok", "ens"])
    assert acc.reasoning == "ran out of tokens" and acc.text == ""


def test_think_block_before_tool_call():
    acc = StreamAccumulator()
    acc.feed(delta(content="<think>need the file</think>"))
    acc.feed(delta(tool_calls=[tc(0, id="c1", name="read_file", arguments='{"path":"x"}')]))
    acc.finish()
    assert acc.reasoning == "need the file" and acc.text == ""
    assert acc.tool_calls()[0].parsed() == {"path": "x"}


def test_strip_think():
    assert thinking.strip_think("<think>a\nb</think>\n\nanswer") == "answer"
    assert thinking.strip_think("answer <think>x</think>") == "answer <think>x</think>"
    assert thinking.strip_think("<think>never closed") == ""


def test_mode_by_model_and_setting():
    s = Settings()
    assert thinking.mode(s, "qwen3-coder-next") == "prompted"
    assert thinking.mode(s, "gpt-oss-120b") == "native"
    assert thinking.mode(s, "qwen3-next-80b-thinking") == "native"
    assert thinking.mode(s, "deepseek-v4-flash") == "native" and thinking.mode(s, "deepseek-v4-flash-dspark") == "native"
    assert all(thinking.mode(s, m) == "native" for m in ("mimo-v2.6-flash-reap", "dots3-note", "k2-horizon-mova", "seed-oss-36b"))
    assert thinking.mode(s, "kimi-linear-48b") == "prompted"
    assert thinking.mode(Settings(thinking="prompted"), "gpt-oss-120b") == "prompted"
    assert thinking.mode(Settings(reasoning_effort="off"), "gpt-oss-120b") == "off"


def test_system_prompt_carries_the_protocol_only_when_prompted(tmp_path):
    prompted = build_system_prompt(Settings(reasoning_effort="max"), tmp_path, "qwen3-coder-next")
    assert "# Thinking" in prompted and "<think>" in prompted and "Argue against your first idea" in prompted
    assert "# Thinking" not in build_system_prompt(Settings(), tmp_path, "gpt-oss-120b")
    assert "# Thinking" not in build_system_prompt(Settings(reasoning_effort="off"), tmp_path, "qwen3-coder-next")
    low = build_system_prompt(Settings(reasoning_effort="low"), tmp_path, "qwen3-coder-next")
    assert "a few lines" in low and "Argue against" not in low


def test_request_body_per_level():
    def body(model, **kw):
        return Client(Settings(backend="vllm", **kw), model=model)._extra_body()

    assert body("gpt-oss-120b", reasoning_effort="max") == {"reasoning_effort": "high"}
    assert body("gpt-oss-120b", reasoning_effort="off") == {"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": False}}
    assert body("qwen3-coder-next") == {}                       # prompted: nothing on the server to steer
    assert body("qwen3.8-27b") == {"reasoning_effort": "xhigh"}   # Qwen3.8 takes xhigh | medium | low
    assert body("qwen3.8-flash-next", reasoning_effort="low") == {"reasoning_effort": "low"}
    assert body("glm-5.3-flash", reasoning_effort="medium") == {"reasoning_effort": "high"}
    assert body("nemotron-3-super") == {}                       # on or off only
    assert body("deepseek-v4-flash") == {"reasoning_effort": "max"}     # the default level is max
    assert body("deepseek-v4-flash", reasoning_effort="medium") == {}   # its template has no medium
    llama = Client(Settings(backend="llama.cpp"), model="deepseek-v4-flash")._extra_body()
    assert llama == {"chat_template_kwargs": {"reasoning_effort": "max"}}
    assert body("qwen3.5-122b") == {} and body("mimo-v2.6-flash-reap") == {} and body("seed-oss-36b") == {}
    assert body("k2-horizon-mova") == {"reasoning_effort": "high"}      # high is its deepest
    assert body("nemotron-3-super", reasoning_effort="off") == {"chat_template_kwargs": {"enable_thinking": False}}
    assert body("glm-4.5-air", reasoning_effort="off") == {"chat_template_kwargs": {"enable_thinking": False}}


def test_inline_thinking_stays_in_history_and_out_of_the_answer(tmp_path, ui):
    (tmp_path / "a.txt").write_text("42\n")
    client = FakeClient([
        Turn(text="", raw="<think>read it first</think>", reasoning="read it first",
             tool_calls=[ToolCall(id="c1", name="read_file", arguments=json.dumps({"path": "a.txt"}))]),
        Turn(text="It says 42.", raw="<think>done</think>\n\nIt says 42.", reasoning="done"),
    ])
    agent = make_agent(tmp_path, client, ui)
    assert agent.run_turn("what is in a.txt?") == "It says 42."
    first, last = agent.messages[2], agent.messages[4]
    assert first["content"] == "<think>read it first</think>" and "reasoning_content" not in first
    assert last["content"] == "<think>done</think>\n\nIt says 42."
    assert "<think>" not in agent.transcript(10_000)


def test_thinking_dropped_from_history_when_not_sent_back(tmp_path, ui):
    client = FakeClient([Turn(text="Hi.", raw="<think>greet</think>Hi.", reasoning="greet")])
    agent = make_agent(tmp_path, client, ui, send_reasoning_back=False)
    agent.run_turn("hello")
    assert agent.messages[2]["content"] == "Hi."


def test_set_effort_rebuilds_the_prompt_only_when_it_matters(tmp_path, ui):
    agent = make_agent(tmp_path, FakeClient([]), ui)  # model "fake": prompted
    before = agent.messages[0]["content"]                 # at the default, max
    assert agent.set_effort("max") is False
    assert agent.set_effort("low") is True and agent.messages[0]["content"] != before
    assert agent.set_effort("low") is False
    assert agent.set_effort("off") is True and "# Thinking" not in agent.messages[0]["content"]


class RecordingClient(FakeClient):
    def stream(self, messages, tools=None, max_tokens=None, stop=None):
        self.max_tokens = max_tokens
        yield from super().stream(messages, tools, max_tokens)


def test_max_tokens_is_clamped_to_the_room_left(tmp_path, ui):
    client = RecordingClient([Turn(text="a"), Turn(text="b")], usage={"prompt_tokens": 70_000, "completion_tokens": 0})
    agent = make_agent(tmp_path, client, ui)
    agent.run_turn("one")
    assert client.max_tokens == 32768           # plenty of room in a 100k window
    agent.run_turn("two")
    assert client.max_tokens == 100_000 - 70_000 - 2048


def test_tool_output_limit_scales_with_the_window(tmp_path, ui):
    s = Settings(permission_mode="auto")
    from frank_local.loop import Agent
    assert Agent(s, FakeClient([]), ui, tmp_path, context_window=32_768).tool_output_limit == 30_000
    assert Agent(s, FakeClient([]), ui, tmp_path, context_window=1_010_000).tool_output_limit == 353_500
    assert Agent(Settings(max_tool_output_chars=5000), FakeClient([]), ui, tmp_path, context_window=1_010_000).tool_output_limit == 5000


def test_compaction_asks_at_the_end_of_the_real_conversation(tmp_path, ui):
    client = FakeClient([Turn(text="first")])
    agent = make_agent(tmp_path, client, ui)
    agent.run_turn("one")
    agent.compact()
    (request,) = client.completions
    assert [m["role"] for m in request] == ["system", "user", "assistant", "user"]
    assert request[0]["content"] == agent.system_prompt and "continuation summary" in request[-1]["content"]
    assert "Under 800 words" in request[-1]["content"]      # 100k window: the small budget
