"""Terminal rendering with rich. One UI per agent; subagents get an indented child."""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from .tools.base import ToolResult


def fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def fmt_secs(s: float) -> str:
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


@dataclass
class Activity:
    """What the agent is doing right now. The terminal repl draws it in the
    status line under the conversation; one is shared with subagents' UIs."""

    what: str = ""          # "" while idle
    since: float = 0.0      # when this step started
    turn_since: float = 0.0  # when the user's message went out; 0 outside a turn
    tokens: int = 0         # streamed during this step
    done: int = 0           # prompt tokens read so far, when the server says (llama.cpp)
    total: int = 0
    cached: int = 0
    took_ms: int = 0
    snippet: str = ""       # the latest bit of reasoning, for a glimpse of it

    def step(self, what: str) -> None:
        if what != self.what:
            self.what, self.since = what, time.time()
            self.tokens = self.done = self.total = self.cached = self.took_ms = 0
            self.snippet = ""

    def begin_turn(self) -> None:
        self.turn_since = time.time()

    def end_turn(self) -> None:
        self.what, self.turn_since = "", 0.0


class UI:
    def __init__(self, console: Console | None = None, *, show_thinking: bool = False, interactive: bool = True, prefix: str = "", activity: Activity | None = None):
        self.console = console or Console(highlight=False, soft_wrap=True)
        self.show_thinking = show_thinking
        self.interactive = interactive
        self.prefix = prefix
        self.quiet = bool(prefix)  # subagents do not stream their prose
        self.activity = activity or Activity()
        self._in_text = False
        self._in_reasoning = False
        self._reasoning_since = 0.0
        # the terminal repl sets this so permission questions come back to the
        # prompt thread, which is the only one that reads stdin
        self.permission_router = None

    def child(self) -> "UI":
        c = UI(self.console, show_thinking=self.show_thinking, interactive=self.interactive, prefix=self.prefix + "  > ", activity=self.activity)
        c.permission_router = self.permission_router
        return c

    # --- what is going on, for the status line -----------------------------
    def phase(self, what: str) -> None:
        self.activity.step(("subagent: " if self.prefix else "") + what)

    def progress(self, done: int, total: int, cached: int = 0, took_ms: int = 0) -> None:
        """The server reading the prompt, before the first token."""
        self.phase("reading the prompt")
        a = self.activity
        a.done, a.total, a.cached, a.took_ms = done, total, cached, took_ms

    # --- streaming assistant output ---------------------------------------
    def text(self, s: str) -> None:
        self.phase("writing")
        self.activity.tokens += 1
        if self.quiet:
            return
        self._end_reasoning()
        self._in_text = True
        self.console.print(s, end="", markup=False, highlight=False, soft_wrap=True)

    def reasoning(self, s: str) -> None:
        self.phase("thinking")
        a = self.activity
        a.tokens += 1
        a.snippet = (a.snippet + s)[-300:]
        if self.quiet:
            return
        if self._in_text:  # reasoning again after some text: start it on its own line
            self.console.print()
            self._in_text = False
        if not self._in_reasoning:
            self._in_reasoning = True
            self._reasoning_since = time.time()
            if self.show_thinking:
                self.console.print("[dim italic]thinking:[/dim italic] ", end="")
        if self.show_thinking:
            self.console.print(f"[dim]{escape(s)}[/dim]", end="", soft_wrap=True)

    def tool_progress(self, name: str) -> None:
        """A tool call streaming in; it is shown once complete."""
        self.phase(f"writing a {name or 'tool'} call")
        self.activity.tokens += 1
        if not self.quiet:
            self._end_reasoning()

    def _end_reasoning(self) -> None:
        """Close the reasoning: the text itself when shown, else how long it took."""
        if not self._in_reasoning:
            return
        self._in_reasoning = False
        if self.show_thinking:
            self.console.print()
        else:
            self.console.print(f"[dim italic]thought for {fmt_secs(time.time() - self._reasoning_since)}[/dim italic]")

    def turn_end(self) -> None:
        self._end_reasoning()
        if self._in_text:
            self.console.print()
        self._in_text = False

    # --- tools -------------------------------------------------------------
    def tool_call(self, name: str, summary: str) -> None:
        self.turn_end()
        self.phase(f"{name} {summary}".strip())
        self.console.print(f"{self.prefix}[bold cyan]{name}[/bold cyan] [dim]{escape(summary)}[/dim]")

    def tool_result(self, name: str, result: ToolResult, max_lines: int = 8) -> None:
        shown = result.display if result.display is not None else result.content
        lines = shown.splitlines() or [""]
        style = "red" if result.is_error else "dim"
        limit = max_lines if not self.quiet else 2
        for line in lines[:limit]:
            self.console.print(f"{self.prefix}  [{style}]{escape(line[:200])}[/{style}]")
        if len(lines) > limit:
            self.console.print(f"{self.prefix}  [dim]... +{len(lines) - limit} lines[/dim]")

    def ask_permission(self, name: str, summary: str, detail: str) -> str:
        self.turn_end()
        if not self.interactive:
            return "n"
        if self.permission_router is not None:
            return self.permission_router(name, summary, detail)
        self.console.print(Panel(escape(detail), title=f"{name} wants to run", border_style="yellow", expand=False))
        try:
            return input("allow? [y]es / [n]o / [a]lways for this tool: ")
        except EOFError:
            return "n"

    # --- messages ----------------------------------------------------------
    def info(self, msg: str) -> None:
        self.turn_end()
        self.console.print(f"{self.prefix}[dim]{escape(msg)}[/dim]")

    def warn(self, msg: str) -> None:
        self.turn_end()
        self.console.print(f"{self.prefix}[yellow]{escape(msg)}[/yellow]")

    def error(self, msg: str) -> None:
        self.turn_end()
        self.console.print(f"{self.prefix}[bold red]{escape(msg)}[/bold red]")

    def status(self, used: int, window: int, lead: str = "") -> None:
        pct = 100 * used / window if window else 0
        color = "dim" if pct < 60 else ("yellow" if pct < 85 else "red")
        lead = f"{escape(lead)} · " if lead else ""
        self.console.print(f"[{color}]{lead}{fmt_tokens(used)}/{fmt_tokens(window)} tokens ({pct:.0f}%)[/{color}]")
