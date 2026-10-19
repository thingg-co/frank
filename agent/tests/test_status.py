import io
import time

from rich.console import Console

from frank_local.cli import LineWriter, describe
from frank_local.render import UI, Activity


class Sink:
    def __init__(self):
        self.out = []

    def write(self, s):
        self.out.append(s)


def test_line_writer_holds_back_the_unfinished_line():
    sink = Sink()
    w = LineWriter(sink)
    w.write("hello ")
    w.flush()  # rich flushes after every print; nothing half-written may reach the terminal
    assert sink.out == [] and w.tail == "hello "
    w.write("world\nnext")
    assert sink.out == ["hello world\n"] and w.tail == "next"
    w.close_line()
    assert sink.out == ["hello world\n", "next\n"] and w.tail == ""


def test_preview_is_plain_and_keeps_the_end():
    w = LineWriter(Sink())
    w.write("\x1b[2mthinking:\x1b[0m " + "x" * 500)
    p = w.preview(width=40, lines=2)
    assert "\x1b" not in p and len(p) == 80 and p.startswith("…") and p.endswith("x")


def test_describe_each_step():
    now = 1000.0
    assert describe(Activity(), now) == ""
    assert describe(Activity(what="waiting for the model", since=now - 3), now) == "waiting for the model · 3s"
    a = Activity(what="reading the prompt", since=now - 10, done=6000, total=20000, cached=2000, took_ms=10000)
    assert describe(a, now) == "reading the prompt · 10s · 4.0k/18.0k tokens (22%), 2.0k cached · about 35s left"
    a = Activity(what="thinking", since=now - 40, turn_since=now - 100, tokens=800)
    assert describe(a, now) == "thinking · 40s · 800 tokens (20/s) · 1m40s in all"
    assert describe(Activity(what="writing", since=now, tokens=1), now) == "writing · 0s · 1 token"


def test_ui_keeps_the_activity_and_says_how_long_it_thought():
    out = io.StringIO()
    ui = UI(Console(file=out, width=100), interactive=False)
    ui.reasoning("let me ")
    ui.reasoning("see")
    assert ui.activity.what == "thinking" and ui.activity.tokens == 2 and ui.activity.snippet == "let me see"
    ui.text("Answer.")
    assert ui.activity.what == "writing" and ui.activity.tokens == 1
    ui.turn_end()
    assert out.getvalue() == "thought for 0s\nAnswer.\n"  # the reasoning itself stays hidden


def test_subagents_share_the_status_line():
    ui = UI(Console(file=io.StringIO(), width=100), interactive=False)
    child = ui.child()
    child.tool_call("grep_files", "TODO")
    assert ui.activity.what == "subagent: grep_files TODO"
    child.tool_progress("bash")
    assert ui.activity.what == "subagent: writing a bash call"
    ui.progress(5, 10)
    assert ui.activity.what == "reading the prompt" and (ui.activity.done, ui.activity.total) == (5, 10)
    assert time.time() - ui.activity.since < 5
