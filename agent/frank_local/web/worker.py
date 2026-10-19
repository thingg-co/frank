"""One chat of the web page: the agent in a process of its own, driven over stdin and stdout.

The server starts one for each chat in use. Commands arrive as JSON lines on stdin and
events leave as JSON lines on the original stdout. The server stops a turn by sending
SIGINT to the process group, which does what Ctrl-C does in the terminal: the request
or the shell command in progress is cut off and the history repaired.

    python -m frank_local.web.worker '{"session": "...jsonl", "cwd": "...", "overrides": {...}}'
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

from .. import thinking
from ..cli import handle_command
from ..client import Client, ContextOverflow
from ..config import Settings, load_settings
from ..loop import Agent
from ..session import Session, SessionBusy
from ..tools import build_registry, truncate_middle
from ..tools.base import ToolResult

OUT = 1  # main() points this at a copy of the real stdout and sends fd 1 to the log
ASKED = itertools.count(1)  # permission questions, numbered across the agent and its subagents


def emit(event: dict[str, Any]) -> None:
    data = memoryview((json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8"))
    # A stop arriving halfway through would leave half a line for the server.
    old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
    try:
        while data:
            data = data[os.write(OUT, data):]
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


PAGE_RESULT_CHARS = 60_000  # a tool result longer than this reaches the page cut in the middle
TOOLS = build_registry()      # for the one-line summaries only


def for_page(record: dict[str, Any]) -> dict[str, Any]:
    """A session record as the page gets it: each tool call with the one-line summary the
    terminal shows, and very long tool results shortened."""
    msg = record.get("message") or {}
    if record.get("type") != "message":
        return record
    if msg.get("role") == "tool" and isinstance(msg.get("content"), str) and len(msg["content"]) > PAGE_RESULT_CHARS:
        return {**record, "message": {**msg, "content": truncate_middle(msg["content"], PAGE_RESULT_CHARS)}}
    calls = msg.get("tool_calls")
    if not calls:
        return record
    out = []
    for tc in calls:
        fn = tc.get("function") or {}
        tool = TOOLS.get(fn.get("name", ""))
        try:
            args = json.loads(fn.get("arguments") or "{}")
            summary = tool.summary(args) if tool and isinstance(args, dict) else ""
        except (ValueError, TypeError, AttributeError):
            summary = ""
        out.append({**tc, "summary": summary})
    return {**record, "message": {**msg, "tool_calls": out}}


class Inbox:
    """Commands from the server."""

    def get(self) -> dict[str, Any] | None:
        line = sys.stdin.buffer.readline()
        if not line:
            return None  # the server closed the pipe: time to go
        try:
            cmd = json.loads(line)
        except ValueError:
            return {}
        return cmd if isinstance(cmd, dict) else {}


class LiveSession(Session):
    """The session file, with every record also sent to the page as it is written."""

    def append(self, record: dict[str, Any]) -> None:
        super().append(record)
        emit({"type": "record", "record": for_page(record)})


class Console:
    """Where handle_command prints (/help, /tools, /todos, /system)."""

    def __init__(self, ui: "WebUI"):
        self.ui = ui

    def print(self, *objects: Any, **_: Any) -> None:
        self.ui.info(" ".join(str(o) for o in objects))


class WebUI:
    """What the agent loop calls to show its work, as events for the page. A subagent's UI
    reports its tool calls only, like the indented lines in the terminal."""

    def __init__(self, inbox: Inbox, depth: int = 0):
        self.inbox = inbox
        self.depth = depth
        self.show_thinking = True
        self.interactive = True
        self.console = Console(self)

    def child(self) -> "WebUI":
        return WebUI(self.inbox, self.depth + 1)

    def text(self, s: str) -> None:
        if not self.depth:
            emit({"type": "text", "s": s})

    def reasoning(self, s: str) -> None:
        if not self.depth:
            emit({"type": "reasoning", "s": s})

    def turn_end(self) -> None:
        pass

    def phase(self, what: str) -> None:
        pass

    def tool_progress(self, name: str) -> None:
        pass

    def progress(self, done: int, total: int, cached: int = 0, took_ms: int = 0) -> None:
        pass

    def tool_call(self, name: str, summary: str) -> None:
        emit({"type": "tool", "depth": self.depth, "name": name, "summary": summary})

    def tool_result(self, name: str, result: ToolResult, max_lines: int = 8) -> None:
        # The agent's own results arrive as records; a subagent's only come this way.
        shown = result.display if result.display is not None else (result.content if self.depth else None)
        emit({"type": "tool_result", "depth": self.depth, "name": name, "error": result.is_error,
              "display": None if shown is None else shown[:4000]})

    def ask_permission(self, name: str, summary: str, detail: str) -> str:
        qid = next(ASKED)
        emit({"type": "permission", "id": qid, "depth": self.depth, "name": name, "summary": summary, "detail": detail[:20000]})
        while True:
            cmd = self.inbox.get()
            if cmd is None:
                return "n"
            if cmd.get("op") == "answer" and cmd.get("id") == qid:
                emit({"type": "answered", "id": qid})
                return str(cmd.get("answer") or "n")

    def info(self, msg: str) -> None:
        emit({"type": "note", "level": "info", "depth": self.depth, "text": msg})

    def warn(self, msg: str) -> None:
        emit({"type": "note", "level": "warn", "depth": self.depth, "text": msg})

    def error(self, msg: str) -> None:
        emit({"type": "note", "level": "error", "depth": self.depth, "text": msg})

    def status(self, used: int, window: int) -> None:
        emit({"type": "status", "used": used, "window": window})


class Unreachable(Exception):
    pass


class Worker:
    def __init__(self, spec: dict[str, Any]):
        self.overrides = dict(spec.get("overrides") or {})
        self.settings: Settings = load_settings(self.overrides)
        self.backend = self.settings.backend  # "auto" is decided again at each probe
        self.session = LiveSession(Path(spec["session"]))
        self.cwd = Path(spec["cwd"])
        self.inbox = Inbox()
        self.ui = WebUI(self.inbox)
        self.agent: Agent | None = None

    # ---- the model server ------------------------------------------------
    def follow_server(self) -> Agent:
        """Start the agent on first use, and keep up with the server: the model it serves can
        change between turns (a profile switch on the control page, a bake-off)."""
        client = self.agent.client if self.agent else Client(self.settings)
        info = client.probe()
        if info.get("error"):
            raise Unreachable(f"cannot reach the model server at {client.base_url}: {info['error']}")
        models = info.get("models") or []
        if not models:
            raise Unreachable(f"the model server at {client.base_url} lists no models")
        if self.backend == "auto":
            self.settings.backend = info.get("server") or "openai"
        window = int(self.settings.context_window or info.get("context") or 32768)
        if self.agent is None:
            self.agent = Agent(self.settings, client, self.ui, self.cwd, context_window=window)
            self.agent.owner = "the web page"
            self.agent.load_history(self.session.messages())  # before the session is attached, as the terminal does
            self.agent.session = self.session
            return self.agent
        agent = self.agent
        if client.model not in models and not self.settings.model:
            client.model = models[0]
            # The model's name and its thinking section are in the system prompt. It is rebuilt
            # only then: the git status in it changes all the time, and a new prompt costs the
            # server its cache of the whole conversation.
            agent.refresh_system_prompt()
            self.ui.info(f"the server now serves {client.model}")
        if window != agent.context_window:
            agent.context_window = window
            agent.tool_output_limit = self.settings.max_tool_output_chars or max(30_000, int(window * 0.35))
        return agent

    def apply(self, cmd: dict[str, Any], agent: Agent) -> None:
        """The thinking level and permission mode picked on the page."""
        effort = cmd.get("effort")
        if effort in thinking.LEVELS and effort != self.settings.reasoning_effort:
            agent.set_effort(effort)
        mode = cmd.get("mode")
        if mode in ("ask", "auto", "readonly") and mode != agent.permissions.mode:
            was_readonly = agent.permissions.mode == "readonly"
            agent.permissions.mode = self.settings.permission_mode = mode
            if was_readonly or mode == "readonly":  # the set of tools changes
                agent.registry = build_registry(read_only=mode == "readonly", allow_subagents=agent.depth < self.settings.max_subagent_depth)
                agent.refresh_system_prompt()

    def state(self) -> dict[str, Any]:
        a = self.agent
        if a is None:
            return {}
        return {"model": a.client.model, "window": a.context_window, "used": a.tokens_used(),
                "effort": self.settings.reasoning_effort, "mode": a.permissions.mode, "cwd": str(a.cwd)}

    # ---- commands ----------------------------------------------------------
    def handle(self, cmd: dict[str, Any]) -> None:
        op = cmd.get("op")
        if op not in ("send", "command"):
            return
        try:
            agent = self.follow_server()
        except Unreachable as e:
            self.ui.error(str(e))
            return
        self.apply(cmd, agent)
        if op == "command":
            line = str(cmd.get("line", "")).strip()
            name = line.split()[0] if line else ""
            if name in ("/quit", "/exit", "/q"):
                self.ui.info("close the tab to leave; the session stays in the list")
            elif name == "/clear":
                self.ui.info("New chat starts a fresh conversation")
            elif name == "/think":
                self.ui.info("the thinking is above each answer; click it to open")
            else:
                handle_command(line, agent, self.ui, self.settings)
            return
        try:
            agent.run_turn(str(cmd.get("text", "")))
        except KeyboardInterrupt:
            if agent.messages[-1].get("content") != "[interrupted]":  # the repair already marked it
                self.ui.warn("stopped")
        except SessionBusy as e:
            emit({"type": "busy", "holder": str(e)})
        except ContextOverflow as e:
            self.ui.error(f"context overflow even after compaction: {e}")
        except Exception as e:  # the chat stays usable
            self.ui.error(f"{type(e).__name__}: {e}")
        self.ui.status(agent.tokens_used(), agent.context_window)

    def run(self) -> int:
        while True:
            try:
                cmd = self.inbox.get()
            except KeyboardInterrupt:
                continue  # a stop with nothing running
            if cmd is None:
                return 0
            if cmd.get("op") == "answer":
                continue  # an answer to a question already settled
            try:
                self.handle(cmd)
            except KeyboardInterrupt:
                self.ui.warn("stopped")
            except Exception as e:
                self.ui.error(f"{type(e).__name__}: {e}")
            finally:
                while True:  # the server waits for this line, so a late stop must not swallow it
                    try:
                        emit({"type": "done", "state": self.state()})
                        break
                    except KeyboardInterrupt:
                        continue


def main() -> int:
    global OUT
    OUT = os.dup(1)
    os.dup2(2, 1)  # print() and anything a library writes go to the log, never into the event stream
    sys.stdout = sys.stderr
    # Stop is SIGINT, which a server started in the background may have been told to ignore.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return Worker(json.loads(sys.argv[1])).run()


if __name__ == "__main__":
    sys.exit(main())
