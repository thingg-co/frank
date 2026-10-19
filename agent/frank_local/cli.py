"""Command line entry point: one-shot, piped, or interactive session."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from . import __version__, thinking
from .client import Client, ContextOverflow
from .config import Settings, load_settings
from .loop import Agent, TurnInterrupted
from .render import UI, Activity, fmt_secs, fmt_tokens
from .session import Session, SessionBusy, load_titles
from .tools.todo import render_todos

COMMANDS = """\
/help              this list
/compact           summarise the conversation to free context
/clear             start a fresh conversation (new session file)
/tokens            context usage
/model NAME        switch model id (must be served)
/effort LEVEL      thinking depth: off, low, medium, high, max
/think             toggle showing the model's reasoning
/yolo              toggle auto-approval of tool calls
/tools             list tools
/todos             show the task list
/cwd [PATH]        show or change the working directory
/system            print the system prompt
/quit              exit (Ctrl-D also works)"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="frank", description="Terminal coding agent for a locally served model.")
    p.add_argument("prompt", nargs="*", help="one-shot prompt; omit for an interactive session")
    p.add_argument("-p", "--print", dest="print_mode", action="store_true", help="non-interactive: answer and exit")
    p.add_argument("-y", "--yes", action="store_true", help="auto-approve every tool call")
    p.add_argument("--readonly", action="store_true", help="only read-only tools")
    p.add_argument("--model", help="model id as served (default: first one the server lists)")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. http://spark:8080/v1")
    p.add_argument("--api-key")
    p.add_argument("--backend", choices=["auto", "llama.cpp", "vllm", "ollama", "openai"], help="default: detected from the server")
    p.add_argument("--effort", choices=list(thinking.LEVELS), help="thinking depth (default high)")
    p.add_argument("--think", action="store_true", help="show the model's reasoning as it streams")
    p.add_argument("-c", "--continue", dest="cont", action="store_true", help="resume the latest session for this directory")
    p.add_argument("-r", "--resume", nargs="?", const="", metavar="ID", help="resume a session by id prefix, or the latest for this directory")
    p.add_argument("--cwd", help="working directory for the agent")
    p.add_argument("--max-turns", type=int, help="tool rounds per user message")
    p.add_argument("--doctor", action="store_true", help="check the server: models, context, speed, tool calling")
    p.add_argument("--sessions", action="store_true", help="list saved sessions")
    p.add_argument("--web", action="store_true", help="serve the chat page, which shares sessions with the terminal")
    p.add_argument("--host", help="with --web: address to listen on (default 127.0.0.1; 0.0.0.0 for the LAN)")
    p.add_argument("--port", type=int, help="with --web: port (default 8089)")
    p.add_argument("--version", action="version", version=f"frank {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "backend": args.backend,
        "reasoning_effort": args.effort,
        "show_thinking": True if args.think else None,
        "permission_mode": "auto" if args.yes else ("readonly" if args.readonly else None),
        "max_iterations": args.max_turns,
        "web_host": args.host,
        "web_port": args.port,
    }
    settings = load_settings(overrides)
    console = Console(highlight=False, soft_wrap=True)
    cwd = Path(args.cwd or os.getcwd()).resolve()

    if args.sessions:
        titles = load_titles(settings.home)
        shown = 0
        for s in Session.list_all(settings.home):
            info = s.summary(titles)
            if info["title"]:  # sessions where nothing was said are left out
                console.print(f"{s.id}  {escape(info['title'])}  [dim]{escape(info['cwd'])}[/dim]")
                shown += 1
                if shown == 30:
                    break
        return 0
    if args.web:
        from .web.server import serve
        # Each chat's worker reads the config again from its project folder, as the terminal
        # does there; the flags given here still win.
        return serve(settings, {k: v for k, v in overrides.items() if v is not None and not k.startswith("web_")})

    client = Client(settings)
    info = client.probe()
    if info.get("error"):
        console.print(f"[bold red]cannot reach {settings.base_url}: {escape(info['error'])}[/bold red]")
        console.print("[dim]is the model server running? check FRANK_BASE_URL or --base-url; the control page shows what is being served[/dim]")
        return 2
    if not client.model:
        console.print("[bold red]the server lists no models[/bold red]")
        return 2
    if settings.backend == "auto":  # decides where the thinking level goes in the request
        settings.backend = info.get("server") or "openai"
    window = settings.context_window or info.get("context") or 32768

    if args.doctor:
        return doctor(client, info, window, console)
    if window < settings.min_context:
        console.print(f"[yellow]context is {fmt_tokens(window)} tokens, below min_context {fmt_tokens(settings.min_context)}; check VLLM_PROFILE and VLLM_LONG on the server[/yellow]")

    interactive = sys.stdin.isatty() and not args.prompt and not args.print_mode
    ui = UI(console, show_thinking=settings.show_thinking, interactive=interactive)
    agent = Agent(settings, client, ui, cwd, context_window=window)

    session = None
    if args.resume is not None:
        if args.resume:
            session = Session.find(settings.home, args.resume)
            if session is None:
                ui.error(f"no session starting with {args.resume}")
                return 2
        else:  # --resume with no id: the latest for this directory, like -c
            session = Session.latest(settings.home, cwd)
            if session is None:
                ui.warn("no previous session for this directory; starting fresh")
    elif args.cont:
        session = Session.latest(settings.home, cwd)
        if session is None:
            ui.warn("no previous session for this directory; starting fresh")
    if session:
        msgs = session.messages()
        agent.load_history(msgs)
        agent.session = session
        ui.info(f"resumed {session.id} ({len(msgs)} messages)")
    else:
        agent.session = Session.new(settings.home, cwd, client.model)

    if not interactive:
        prompt = " ".join(args.prompt).strip()
        if not sys.stdin.isatty():
            piped = sys.stdin.read().strip()
            prompt = f"{prompt}\n\n{piped}".strip() if prompt else piped
        if not prompt:
            console.print("[red]no prompt given[/red]")
            return 2
        if settings.permission_mode == "ask":
            ui.info("non-interactive: tool calls that change things are denied unless you pass -y")
        try:
            agent.run_turn(prompt)
        except KeyboardInterrupt:
            ui.warn("interrupted")
            return 130
        except SessionBusy as e:
            ui.error(f"a turn is running in this session in {e}; wait for it or start a new session")
            return 2
        return 0

    return repl(agent, ui, settings, info)


def thinking_label(settings: Settings, model: str) -> str:
    kind = thinking.mode(settings, model)
    return "off" if kind == "off" else f"{settings.reasoning_effort} ({kind})"


# Escape sequences and carriage returns, dropped from the live preview of a half-written line.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\r")
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class LineWriter:
    """sys.stdout during the interactive session. Whole lines go to the terminal
    above the prompt, through prompt_toolkit's proxy, which redraws the prompt
    under them. A line still being written is held back, so nothing is ever
    drawn over the prompt, and shown live under the conversation instead."""

    def __init__(self, sink, changed=lambda: None) -> None:
        self.sink = sink
        self.changed = changed
        self.tail = ""
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            head, nl, self.tail = (self.tail + data).rpartition("\n")
            if nl:
                self.sink.write(head + nl)
        self.changed()
        return len(data)

    def flush(self) -> None:
        pass  # rich flushes after every print; a half line waits for the rest of it

    def isatty(self) -> bool:
        return True  # rich picked its colours when the console was built

    def close_line(self) -> None:
        """At the end of the session: a line left unfinished goes out too."""
        with self._lock:
            tail, self.tail = self.tail, ""
        if tail:
            self.sink.write(tail + "\n")

    def preview(self, width: int, lines: int = 4) -> str:
        """The end of the unfinished line, plain, at most this many rows of the screen."""
        text = ANSI_ESCAPE.sub("", self.tail)
        keep = max(20, width) * lines
        return text if len(text) <= keep else "…" + text[-keep + 1:]


def describe(a: Activity, now: float) -> str:
    """The status line while the agent works: the step, how long, how far along."""
    if not a.what:
        return ""
    parts = [a.what, fmt_secs(now - a.since)]
    if a.what.endswith("reading the prompt") and a.total:
        new_total = max(1, a.total - a.cached)
        new_done = min(new_total, max(0, a.done - a.cached))
        part = f"{fmt_tokens(new_done)}/{fmt_tokens(new_total)} tokens ({100 * new_done // new_total}%)"
        if a.cached:
            part += f", {fmt_tokens(a.cached)} cached"
        parts.append(part)
        if new_done and a.took_ms:
            left = (new_total - new_done) / (new_done / (a.took_ms / 1000))
            if left >= 1:
                parts.append(f"about {fmt_secs(left)} left")
    elif a.tokens:
        dt = now - a.since
        parts.append(f"{a.tokens} token{'s' if a.tokens != 1 else ''}" + (f" ({a.tokens / dt:.0f}/s)" if dt >= 2 else ""))
    if a.turn_since and a.since - a.turn_since >= 1:
        parts.append(f"{fmt_secs(now - a.turn_since)} in all")
    return " · ".join(parts)


def repl(agent: Agent, ui: UI, settings: Settings, info: dict) -> int:
    import queue

    from prompt_toolkit import Application
    from prompt_toolkit.application import get_app
    from prompt_toolkit.buffer import Buffer
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.patch_stdout import StdoutProxy
    from prompt_toolkit.styles import Style

    settings.home.mkdir(parents=True, exist_ok=True)
    activity = ui.activity

    # The conversation is printed into the terminal itself, so its scrollback,
    # search and copying all work as usual. Under it the app keeps a few live
    # lines: the line being written, a status line, and the input.
    #
    # The app and the proxy are made before sys.stdout is swapped, so both write
    # to the real terminal. raw=True keeps rich's colours (the proxy strips
    # escape codes otherwise).
    lines: queue.Queue[str] = queue.Queue()
    stop = threading.Event()
    turns: dict[str, threading.Thread | None] = {"worker": None}
    working = {"on": False}
    state = {"choosing": None, "footer": ""}  # choosing: a message typed mid-turn, awaiting l/n/c
    send_mode = ["instruction"]

    def running() -> bool:
        w = turns["worker"]
        return w is not None and w.is_alive()

    def refresh_footer() -> None:
        used, window = agent.tokens_used(), agent.context_window
        state["footer"] = f"{fmt_tokens(used)}/{fmt_tokens(window)} tokens ({100 * used / window if window else 0:.0f}%)"

    class PermissionGate:
        """A tool in the turn thread asks for permission; the answer is typed
        at the prompt and handed back here."""

        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.request: tuple[str, queue.Queue[str]] | None = None

        def ask(self, name: str, summary: str, detail: str) -> str:
            reply: queue.Queue[str] = queue.Queue()
            ui.console.print(Panel(escape(detail), title=f"{name} wants to run", border_style="yellow", expand=False))
            with self.lock:
                self.request = (name, reply)
            app.invalidate()
            answer = reply.get()
            with self.lock:
                if self.request is not None and self.request[1] is reply:
                    self.request = None
            return answer

        def pending(self) -> str | None:
            with self.lock:
                return self.request[0] if self.request else None

        def answer(self, line: str) -> bool:
            """Route a submitted line to the waiting tool. True if one was waiting.
            Anything but y/n/a is a no that says what to do instead: it goes to
            the model as the next instruction."""
            with self.lock:
                req, self.request = self.request, None
            if req is None:
                return False
            ans = line.strip().lower()
            if ans not in ("", "y", "yes", "n", "no", "a", "always"):
                agent._interrupt("instruction", line.strip())  # before the tool hears no, so the turn sees it
                ui.info(f"{req[0]} not allowed; your message goes to the model instead")
            req[1].put("y" if ans in ("y", "yes") else "a" if ans in ("a", "always") else "n")
            return True

    gate = PermissionGate()
    ui.permission_router = gate.ask

    def prompt_text() -> str:
        if gate.pending() is not None:
            return "allow [y/n/a]> "
        if state["choosing"] is not None:
            return "send as [l/n/c]> "
        return "> "

    def status_left():
        name = gate.pending()
        if name is not None:
            return [("class:ask", f" {name} is waiting for you: y yes · n no · a always for this tool · or say what to do instead")]
        if state["choosing"] is not None:
            return [("class:ask", " send it as: l later, when this turn ends (Enter) · n the next instruction, now · c a clarification")]
        now = time.time()
        text = describe(activity, now)
        if not text:
            return ""
        line = [("class:spinner", f"{SPINNER[int(now * 8) % len(SPINNER)]} "), ("class:status", text)]
        if activity.what.endswith("thinking") and activity.snippet and not ui.show_thinking:
            line.append(("class:snippet", "  " + " ".join(activity.snippet.split())[-80:]))  # a glimpse of the reasoning
        return line

    def status_right():
        if working["on"]:
            return [("class:hint", " esc to interrupt ")]
        return [("class:hint", f" {state['footer']} ")]

    # Ctrl-C and Esc stop a running turn; Ctrl-D on an empty line exits. Esc is
    # not eager, so Esc+Enter still reads as Alt+Enter (a clarification); Esc
    # alone lands after timeoutlen (see pre_run).
    kb = KeyBindings()

    def interrupt() -> bool:
        if gate.pending() is not None:
            gate.answer("n")  # stop asking: the tool is denied
            return True
        if working["on"]:
            agent._stop()
            activity.step("stopping")
            return True
        return False

    @kb.add("c-c", eager=True)
    def _ctrl_c(event) -> None:
        if not interrupt():
            event.current_buffer.reset()

    @kb.add("escape")
    def _esc(event) -> None:
        interrupt()

    @kb.add("c-d", eager=True, filter=Condition(lambda: not buf.text))
    def _eof(event) -> None:
        event.app.exit()

    # Enter sends a line; Alt+Enter while a turn runs sends it as a
    # clarification at once. The prompt worker reads the flag.
    @kb.add("enter")
    def _enter(event) -> None:
        send_mode[0] = "instruction"
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    @kb.add("escape", "c-j")  # some terminals send LF, not CR, for Enter
    def _alt_enter(event) -> None:
        send_mode[0] = "clarification"
        event.current_buffer.validate_and_handle()

    def accept(b: Buffer) -> bool:
        # Echo what was typed into the conversation, as typed, then hand it over.
        b.append_to_history()
        ui.console.print(f"[bold green]{escape(prompt_text())}[/bold green]{escape(b.text)}")
        lines.put(b.text)
        return False  # clears the input

    buf = Buffer(history=FileHistory(str(settings.home / "history")), multiline=False, accept_handler=accept)

    writer: LineWriter  # set below, once the app exists
    layout = Layout(
        HSplit([
            ConditionalContainer(
                Window(FormattedTextControl(lambda: writer.preview(get_app().output.get_size().columns)), wrap_lines=True),
                filter=Condition(lambda: bool(writer.tail)),
            ),
            VSplit([
                Window(FormattedTextControl(status_left), height=1),
                Window(FormattedTextControl(status_right), height=1, dont_extend_width=True),
            ]),
            VSplit([
                Window(FormattedTextControl(lambda: [("class:prompt", prompt_text())]), dont_extend_width=True),
                Window(BufferControl(buffer=buf), wrap_lines=True),
            ]),
        ]),
        focused_element=buf,
    )
    style = Style.from_dict({
        "prompt": "bold ansigreen",
        "spinner": "ansicyan",
        "status": "",
        "snippet": "italic ansibrightblack",
        "hint": "ansibrightblack",
        "ask": "ansiyellow",
    })

    def pre_run() -> None:
        get_app().timeoutlen = 0.4  # Esc alone lands promptly; Esc+Enter still reads as Alt+Enter

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        erase_when_done=True,     # the live lines go; the conversation stays on the terminal
        refresh_interval=0.25,    # the spinner and the clocks in the status line
        min_redraw_interval=0.05,
    )
    proxy = StdoutProxy(sleep_between_writes=0.05, raw=True)
    writer = LineWriter(proxy, changed=app.invalidate)

    def turn_worker(first: str) -> None:
        """Runs the turns. An interrupt with a message becomes the next turn's
        user message; a clarification is labelled as such for the model."""
        text = first
        working["on"] = True
        activity.begin_turn()
        t0 = time.time()
        ended = "done in"
        try:
            while True:
                try:
                    agent.run_turn(text)
                    break
                except TurnInterrupted as e:
                    if not e.mode:
                        ended = "stopped after"
                        break
                    text = f"Clarification: {e.text}" if e.mode == "clarification" else e.text
                    continue
                except SessionBusy as e:
                    ui.error(f"a turn is running in this session in {e}; try again when it is done")
                    ended = "failed after"
                    break
                except ContextOverflow as e:
                    ui.error(f"context overflow even after compaction: {e}")
                    ended = "failed after"
                    break
                except Exception as e:  # keep the session alive
                    ended = "failed after"
                    msg = f"{type(e).__name__}: {e}"
                    ui.error(msg)
                    # A dead turn is otherwise silent: the session file ends mid-turn and
                    # nothing says why. Record it so the transcript and the web page
                    # (which ignores non-message records) can show what happened.
                    if agent.session:
                        agent.session.append({
                            "type": "error",
                            "what": msg,
                            "traceback": traceback.format_exc(),
                            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        })
                    break
        finally:
            working["on"] = False
            activity.end_turn()
            refresh_footer()
        ui.status(agent.tokens_used(), agent.context_window, lead=f"{ended} {fmt_secs(time.time() - t0)}")

    def start_turn(text: str) -> None:
        turns["worker"] = threading.Thread(target=turn_worker, args=(text,), daemon=True, name="frank-turn")
        turns["worker"].start()

    def prompt_worker() -> None:
        """Owns the prompt logic: answers to the permission gate, the l/n/c
        choice for a message typed mid-turn, commands, and starting turns.
        The app itself only accepts lines and hands them here."""
        while True:
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                if stop.is_set():
                    return
                continue
            line = line.strip()
            if gate.answer(line):
                continue
            if state["choosing"] is not None:
                message, state["choosing"] = state["choosing"], None
                choice = line.lower()
                if not running():  # the turn ended while the choice was open: just send it
                    agent._reset_interrupt()
                    start_turn(message)
                elif not choice or choice.startswith("l"):
                    agent._queue_later(message)
                    ui.info("queued — it goes when this turn finishes; an empty Enter sends it now")
                elif choice.startswith("n"):
                    agent._interrupt("instruction", message)
                    ui.info("sent as the next instruction")
                elif choice.startswith("c"):
                    agent._interrupt("clarification", message)
                    ui.info("sent as a clarification")
                else:
                    ui.warn("pick l (later), n (next instruction) or c (clarification)")
                    state["choosing"] = message  # ask again
                continue
            if running():
                if line.startswith("/"):
                    ui.warn("a turn is running; interrupt it first (Ctrl-C or Esc), then use commands")
                    continue
                if not line:
                    if agent.has_pending():
                        agent._send_now()
                        ui.info("sending the queued message now")
                    continue
                if send_mode[0] == "clarification":
                    agent._interrupt("clarification", line)
                    ui.info("sent as a clarification")
                    continue
                state["choosing"] = line
                continue
            if not line:
                continue
            if line.startswith("/"):
                try:
                    if handle_command(line, agent, ui, settings) is False:
                        app.loop.call_soon_threadsafe(app.exit)
                        return
                finally:
                    activity.end_turn()  # /compact leaves its step behind
                    refresh_footer()
                continue
            pending = agent._drain()  # a message queued as the last turn ended
            if pending is not None:
                mode, text = pending
                start_turn(f"Clarification: {text}" if mode == "clarification" else text)
                agent._queue_later(line)
                continue
            agent._reset_interrupt()
            start_turn(line)

    mode = {"ask": "ask before changes", "auto": "auto-approve", "readonly": "read-only"}[settings.permission_mode]
    refresh_footer()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = writer
    worker_thread = threading.Thread(target=prompt_worker, daemon=True, name="frank-prompt")
    worker_thread.start()
    try:
        ui.console.print(
            f"👽 [bold]frank[/bold] {__version__} · {agent.client.model} @ {agent.client.base_url} · "
            f"ctx {fmt_tokens(agent.context_window)} · thinking {thinking_label(settings, agent.client.model)} · {mode} · session {agent.session.id if agent.session else '-'}"
        )
        ui.console.print("[dim]/help for commands · Esc or Ctrl-C interrupts a turn · Ctrl-D exits[/dim]")
        ui.console.print("[dim]type while a turn runs to queue a message, steer it, or clarify (Alt+Enter)[/dim]")
        try:
            app.run(pre_run=pre_run)
        except (EOFError, KeyboardInterrupt):
            pass
    finally:
        stop.set()
        worker_thread.join(timeout=5)
        if running():
            agent._stop()  # Ctrl-D mid-turn: end the turn too
            turns["worker"].join(timeout=10)
        sys.stdout, sys.stderr = old_out, old_err
        writer.close_line()
        proxy.close()
    return 0


def handle_command(line: str, agent: Agent, ui: UI, settings: Settings) -> bool | None:
    cmd, _, arg = line.partition(" ")
    arg = arg.strip()
    if cmd in ("/quit", "/exit", "/q"):
        return False
    if cmd == "/help":
        ui.console.print(COMMANDS)
    elif cmd == "/compact":
        try:
            with agent.hold():
                agent.compact()
        except SessionBusy as e:
            ui.error(f"a turn is running in this session in {e}")
    elif cmd == "/clear":
        agent.reset()
        agent.session = Session.new(settings.home, agent.cwd, agent.client.model)
        ui.info(f"new session {agent.session.id}")
    elif cmd == "/tokens":
        ui.status(agent.tokens_used(), agent.context_window)
        ui.info(f"{len(agent.messages)} messages, estimate {fmt_tokens(agent.estimate_tokens())}")
    elif cmd == "/model":
        if arg:
            agent.client.model = arg
            ui.info(f"model set to {arg}")
        else:
            ui.info(f"model: {agent.client.model}; served: {', '.join(agent.client.probe().get('models', []))}")
    elif cmd == "/effort":
        if arg in thinking.LEVELS:
            note = "; the system prompt changed, so the next request refills the server's cache" if agent.set_effort(arg) else ""
            ui.info(f"thinking {thinking_label(settings, agent.client.model)}{note}")
        else:
            ui.info(f"thinking is {thinking_label(settings, agent.client.model)}; levels: {', '.join(thinking.LEVELS)}")
    elif cmd == "/think":
        ui.show_thinking = not ui.show_thinking
        ui.info(f"show thinking: {ui.show_thinking}")
    elif cmd == "/yolo":
        new = "ask" if agent.permissions.mode == "auto" else "auto"
        agent.permissions.mode = new
        settings.permission_mode = new
        ui.info(f"permission mode: {new}")
    elif cmd == "/tools":
        ui.console.print(", ".join(agent.registry.names()))
    elif cmd == "/todos":
        todos = agent.ctx.state.get("todos") or []
        ui.console.print(render_todos(todos) if todos else "(no tasks)")
    elif cmd == "/cwd":
        if arg:
            p = Path(arg).expanduser().resolve()
            if p.is_dir():
                agent.cwd = agent.ctx.cwd = p
            else:
                ui.error(f"not a directory: {p}")
        ui.info(f"cwd: {agent.cwd}")
    elif cmd == "/system":
        ui.console.print(agent.messages[0]["content"], markup=False)
    else:
        ui.warn(f"unknown command {cmd}; /help lists them")
    return None


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def doctor(client: Client, info: dict, window: int, console: Console) -> int:
    ok = True
    console.print(f"server    {client.base_url} ({info.get('server') or 'unknown type'}{', ' + info['build'] if info.get('build') else ''})")
    console.print(f"models    {', '.join(info.get('models', [])) or '-'}")
    console.print(f"model     {client.model}")
    console.print(f"context   {fmt_tokens(window)} tokens{'' if info.get('context') else ' (assumed; server did not report it)'}")
    if window < client.settings.min_context:
        ok = False
        console.print(f"context   [red]FAILED[/red] below min_context {fmt_tokens(client.settings.min_context)}; check VLLM_PROFILE and VLLM_LONG on the server")

    t0 = time.time()
    n = 0
    first = None
    for kind, payload in client.stream([{"role": "user", "content": "Count from 1 to 60, separated by spaces."}], None, max_tokens=160):
        if kind in ("text", "reasoning"):
            n += 1
            first = first or time.time()
        elif kind == "done" and payload.usage:
            n = payload.usage.get("completion_tokens", n)
    dt = time.time() - (first or t0)
    console.print(f"speed     {n / dt:.1f} tok/s generation, {((first or t0) - t0) * 1000:.0f} ms to first token")

    # Thinking and a tool call in the same response, the way the agent will ask for them.
    kind_ = thinking.mode(client.settings, client.model)
    system = "You are a helpful assistant. Use the tools when they apply."
    if block := thinking.prompt_block(client.settings, client.model):
        system += "\n\n" + block
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": "What is the weather in Lisbon right now? Use the tool."},
    ]
    turn = None
    for kind, payload in client.stream(msgs, [WEATHER_TOOL], max_tokens=4096):
        if kind == "done":
            turn = payload
    words = len(turn.reasoning.split()) if turn else 0
    if kind_ == "off":
        console.print("thinking  off")
    elif words:
        console.print(f"thinking  [green]ok[/green] {kind_}, effort {client.settings.reasoning_effort}, {words} words before the tool call")
    else:
        console.print(f"thinking  [yellow]none seen[/yellow] ({kind_}, effort {client.settings.reasoning_effort})")
        hint = "the server needs a --reasoning-parser for this model, or set thinking = \"prompted\"" if kind_ == "native" else "the model ignored the <think> instruction; try a thinking profile"
        console.print(f"[dim]{hint}[/dim]")
    if turn and turn.tool_calls and turn.tool_calls[0].name == "get_weather":
        tc = turn.tool_calls[0]
        console.print(f"tool call [green]ok[/green] {tc.name}({tc.arguments.strip()})")
        inline = thinking.strip_think(turn.raw) != turn.raw
        asst = {"role": "assistant", "content": turn.raw or None, "tool_calls": [tc.to_message_part()]}
        if turn.reasoning and not inline:
            asst["reasoning_content"] = turn.reasoning
        msgs += [asst, {"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps({"city": "Lisbon", "temp_c": 21, "sky": "clear"})}]
        final = client.complete(msgs, max_tokens=4096)
        if "21" in final:
            console.print(f"tool loop [green]ok[/green] {final[:100]!r}")
        else:
            console.print(f"tool loop [yellow]odd[/yellow] model did not use the result: {final[:100]!r}")
    else:
        ok = False
        console.print(f"tool call [red]FAILED[/red] no tool call returned; text was {((turn.text if turn else '') or '')[:120]!r}")
        console.print("[dim]llama.cpp needs --jinja; vLLM needs --enable-auto-tool-choice and a --tool-call-parser[/dim]")
        if turn and "<tool_call>" in turn.raw + turn.reasoning:
            console.print("[dim]the call came back as text: the parser does not match this model, or it was written inside the think block[/dim]")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
