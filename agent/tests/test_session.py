import json
import os
import time

import pytest

from frank_local.client import Turn
from frank_local.config import Settings
from frank_local.loop import Agent
from frank_local.session import Session, SessionBusy, load_titles, save_title

from test_loop import FakeClient


def agent_on(tmp_path, ui, session, turns):
    a = Agent(Settings(permission_mode="auto"), FakeClient(turns), ui, tmp_path, context_window=100_000)
    a.load_history(session.messages())
    a.session = session
    return a


def test_a_turn_holds_the_session(tmp_path, monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)
    s = Session.new(tmp_path, tmp_path, "m")
    other = Session(s.path)
    assert other.holder() is None
    with s.turn("the web page"):
        assert other.holder() == f"the web page (pid {os.getpid()})"
        with pytest.raises(SessionBusy, match="the web page"):
            with other.turn("the terminal"):
                pass
    assert other.holder() is None
    with other.turn("the terminal"):
        pass


def test_each_side_picks_up_what_the_other_added(tmp_path, ui):
    s = Session.new(tmp_path, tmp_path, "m")
    terminal = agent_on(tmp_path, ui, Session(s.path), [Turn(text="one"), Turn(text="three")])
    page = agent_on(tmp_path, ui, Session(s.path), [Turn(text="two")])
    terminal.run_turn("first")
    page.run_turn("second")  # sees the terminal's turn before it asks
    assert [m["content"] for m in page.client.requests[0][1:]] == ["first", "one", "second"]
    terminal.run_turn("third")
    assert [m["content"] for m in terminal.client.requests[1][1:]] == ["first", "one", "second", "two", "third"]
    assert [m["content"] for m in Session(s.path).messages()] == ["first", "one", "second", "two", "third", "three"]


def test_an_interrupted_turn_is_repaired_before_the_lock_goes(tmp_path, ui):
    s = Session.new(tmp_path, tmp_path, "m")

    class Stopping(FakeClient):
        def stream(self, messages, tools=None, max_tokens=None, stop=None):
            raise KeyboardInterrupt
            yield

    a = Agent(Settings(permission_mode="auto"), Stopping([]), ui, tmp_path, context_window=100_000)
    a.session = s
    with pytest.raises(KeyboardInterrupt):
        a.run_turn("hello")
    assert [m["content"] for m in Session(s.path).messages()] == ["hello", "[interrupted]"]
    assert Session(s.path).holder() is None


def test_reasoning_of_an_answer_is_kept_for_the_page_only(tmp_path, ui):
    s = Session.new(tmp_path, tmp_path, "m")
    a = agent_on(tmp_path, ui, s, [Turn(text="42", reasoning="six times seven")])
    a.run_turn("what is it?")
    answer = [r for r in s.records() if r.get("type") == "message"][-1]
    assert answer["reasoning"] == "six times seven"
    assert Session(s.path).messages()[-1] == {"role": "assistant", "content": "42"}


def test_listing_titles_and_trash(tmp_path):
    old = Session.new(tmp_path, tmp_path / "a", "m")
    old.add_message({"role": "user", "content": "fix   the\nbuild please"})
    empty = Session.new(tmp_path, tmp_path / "a", "m")
    new = Session.new(tmp_path, tmp_path / "b", "m")
    new.add_message({"role": "user", "content": "x" * 200})
    now = time.time()
    os.utime(old.path, (now + 10, now + 10))  # continued last, on the page say
    os.utime(new.path, (now, now))
    os.utime(empty.path, (now + 20, now + 20))

    assert [s.id for s in Session.list_all(tmp_path)][:2] == [empty.id, old.id]
    assert Session.latest(tmp_path).id == old.id                  # an empty session does not count
    assert Session.latest(tmp_path, tmp_path / "b").id == new.id
    assert old.summary()["title"] == "fix the build please"
    assert len(new.summary()["title"]) == 80 and new.summary()["title"].endswith("…")
    assert empty.summary()["title"] == ""

    save_title(tmp_path, old.id, "Build fix")
    assert load_titles(tmp_path) == {old.id: "Build fix"}
    assert old.summary(load_titles(tmp_path))["title"] == "Build fix"
    old.trash()
    assert not old.path.exists() and (tmp_path / "sessions" / "trash" / old.path.name).exists()
    assert old.id not in [s.id for s in Session.list_all(tmp_path)]


def test_a_partly_written_line_is_left_for_later(tmp_path):
    s = Session.new(tmp_path, tmp_path, "m")
    s.add_message({"role": "user", "content": "hi"})
    with s.path.open("a") as fh:
        fh.write(json.dumps({"type": "message", "message": {"role": "assistant", "content": "hel"}})[:30])
    assert [m["content"] for m in s.messages()] == ["hi"]
