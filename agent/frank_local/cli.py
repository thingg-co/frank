"""Command line entry point: one-shot, piped, or interactive session."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from . import __version__, thinking
from .client import Client, ContextOverflow
from .config import Settings, load_settings
from .loop import Agent, TurnInterrupted
from .render import UI, fmt_tokens
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
    p.add_argument("-r", "--resume", metavar="ID", help="resume a session by id prefix")
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
    if args.resume:
        session = Session.find(settings.home, args.resume)
        if session is None:
            ui.error(f"no session starting with {args.resume}")
            return 2
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


def repl(agent: Agent, ui: UI, settings: Settings, info: dict) -> int:
    import queue
    import threading

    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.patch_stdout import StdoutProxy

    settings.home.mkdir(parents=True, exist_ok=True)
    ps: PromptSession = PromptSession(history=FileHistory(str(settings.home / "history")))

    # Esc alone stops a running turn like Ctrl-C (it raises KeyboardInterrupt,
    # same as the key). What to do with a message typed while a turn runs is
    # picked at the prompt after Enter, not by key combinations.
    kb = KeyBindings()

    @kb.add("escape")
    def _esc(event) -> None:
        event.app.exit(exception=KeyboardInterrupt())  # Esc = Ctrl-C

    def pick_send_choice() -> str:
        """What to do with the typed message: later, next instruction, clarification."""
        while True:
            try:
                ans = input("send as: [l]ater (Enter) / [n]ext instruction / [c]larification: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "later"
            if not ans or ans.startswith("l"):
                return "later"
            if ans.startswith("n"):
                return "next"
            if ans.startswith("c"):
                return "clarify"
            ui.warn("pick l (later), n (next instruction) or c (clarification)")

    class PermissionGate:
        """A tool in the turn thread asks for permission; the question is shown
        from the prompt's inputhook and the answer comes back at the prompt,
        so the two never race over stdin."""

        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.request: tuple[str, str, str, queue.Queue[str]] | None = None
            self.shown = False

        def ask(self, name: str, summary: str, detail: str) -> str:
            reply: queue.Queue[str] = queue.Queue()
            with self.lock:
                self.request = (name, summary, detail, reply)
            answer = reply.get()
            with self.lock:
                if self.request is not None and self.request[3] is reply:
                    self.request = None
                    self.shown = False
            return answer

        def pending(self) -> tuple[str, str, str, queue.Queue[str]] | None:
            with self.lock:
                return self.request

        def answer(self, line: str) -> bool:
            """Route a submitted line to the waiting tool. True if one was waiting."""
            with self.lock:
                req, self.request = self.request, None
                self.shown = False
            if req is None:
                return False
            ans = line.strip().lower()
            req[3].put("y" if ans in ("y", "yes") else "a" if ans in ("a", "always") else "n")
            return True

    gate = PermissionGate()
    ui.permission_router = gate.ask

    def inputhook(ctx) -> None:
        """Called whenever the prompt waits for input: put the permission
        question on screen. The answer is typed at the prompt itself."""
        if ctx.input_is_ready():
            return
        with gate.lock:
            req = gate.request
            show = req is not None and not gate.shown
            if show:
                gate.shown = True
        if show:
            name, summary, detail, _ = req
            ui.turn_end()
            ui.console.print(Panel(escape(detail), title=f"{name} wants to run", border_style="yellow", expand=False))
            ui.console.print("[yellow]allow?[/yellow] [y]es / [n]o / [a]lways for this tool (answer at the prompt): ")

    def turn_worker(first: str) -> None:
        """Runs the turns. An interrupt with a message becomes the next turn's
        user message; a clarification is labelled as such for the model."""
        text = first
        while True:
            try:
                agent.run_turn(text)
                break
            except TurnInterrupted as e:
                if not e.mode:
                    break
                text = f"Clarification: {e.text}" if e.mode == "clarification" else e.text
                continue
            except SessionBusy as e:
                ui.error(f"a turn is running in this session in {e}; try again when it is done")
                break
            except ContextOverflow as e:
                ui.error(f"context overflow even after compaction: {e}")
                break
            except Exception as e:  # keep the session alive
                ui.error(f"{type(e).__name__}: {e}")
                break
        ui.status(agent.tokens_used(), agent.context_window)

    def running() -> bool:
        return worker is not None and worker.is_alive()

    mode = {"ask": "ask before changes", "auto": "auto-approve", "readonly": "read-only"}[settings.permission_mode]
    ui.console.print(
        f"👽 [bold]frank[/bold] {__version__} · {agent.client.model} @ {agent.client.base_url} · "
        f"ctx {fmt_tokens(agent.context_window)} · thinking {thinking_label(settings, agent.client.model)} · {mode} · session {agent.session.id if agent.session else '-'}"
    )
    ui.console.print("[dim]/help for commands, Ctrl-C or Esc interrupts a turn, Ctrl-D exits[/dim]")
    ui.console.print("[dim]while a turn runs: type + Enter, then pick later / next instruction / clarification; empty Enter sends a queued message now[/dim]")

    # Output above the live prompt goes through a proxy that redraws the prompt.
    # The proxy is created here so it owns the terminal for the whole session.
    worker: threading.Thread | None = None
    with StdoutProxy(sleep_between_writes=0.0) as proxy:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = proxy
        try:
            while True:
                try:
                    line = ps.prompt(HTML("<b><ansigreen>&gt; </ansigreen></b>"), key_bindings=kb, inputhook=inputhook).strip()
                except EOFError:
                    if running():
                        agent._stop()
                        worker.join(timeout=10)
                    break
                except KeyboardInterrupt:  # Ctrl-C and Esc both land here
                    if gate.pending() is not None:
                        gate.answer("n")  # stop asking: the tool is denied
                    elif running():
                        agent._stop()
                        ui.warn("interrupting; send a message at the prompt or press Enter to stop")
                    continue
                if gate.pending() is not None:
                    gate.answer(line)
                    continue
                if running():
                    if line.startswith("/"):
                        ui.warn("a turn is running; interrupt it first (Ctrl-C or Esc), then use commands")
                        continue
                    if not line:
                        if agent.has_pending():
                            agent._send_now()
                            ui.info("sending queued message now")
                        continue
                    choice = pick_send_choice()
                    if choice == "clarify":
                        agent._interrupt("clarification", line)
                        ui.info("interrupt sent as a clarification")
                    elif choice == "next":
                        agent._interrupt("instruction", line)
                        ui.info("interrupt sent as the next instruction")
                    else:
                        agent._queue_later(line)
                        ui.info("queued — will send when this turn finishes; press Enter again to send now")
                    continue
                if not line:
                    continue
                if line.startswith("/"):
                    if handle_command(line, agent, ui, settings) is False:
                        break
                    continue
                pending = agent._drain()  # a message queued as the last turn ended
                if pending is not None:
                    mode, text = pending
                    if mode == "clarification":
                        text = f"Clarification: {text}"
                    worker = threading.Thread(target=turn_worker, args=(text,), daemon=True, name="frank-turn")
                    worker.start()
                    if line:
                        agent._queue_later(line)
                    continue
                agent._reset_interrupt()
                worker = threading.Thread(target=turn_worker, args=(line,), daemon=True, name="frank-turn")
                worker.start()
        finally:
            sys.stdout, sys.stderr = old_out, old_err
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
