"""Terminal rendering with rich. One UI per agent; subagents get an indented child."""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from .tools.base import ToolResult


def fmt_tokens(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


class UI:
    def __init__(self, console: Console | None = None, *, show_thinking: bool = False, interactive: bool = True, prefix: str = ""):
        self.console = console or Console(highlight=False, soft_wrap=True)
        self.show_thinking = show_thinking
        self.interactive = interactive
        self.prefix = prefix
        self.quiet = bool(prefix)  # subagents do not stream their prose
        self._in_text = False
        self._in_reasoning = False
        # the terminal repl sets this so permission questions come back to the
        # prompt thread, which is the only one that reads stdin
        self.permission_router = None

    def child(self) -> "UI":
        c = UI(self.console, show_thinking=self.show_thinking, interactive=self.interactive, prefix=self.prefix + "  > ")
        c.permission_router = self.permission_router
        return c

    # --- streaming assistant output ---------------------------------------
    def text(self, s: str) -> None:
        if self.quiet:
            return
        if self._in_reasoning and not self._in_text:
            self.console.print()
        self._in_text = True
        self.console.print(s, end="", markup=False, highlight=False, soft_wrap=True)

    def reasoning(self, s: str) -> None:
        if self.quiet:
            return
        if self.show_thinking:
            if not self._in_reasoning:
                self.console.print("[dim italic]thinking:[/dim italic] ", end="")
            self.console.print(f"[dim]{escape(s)}[/dim]", end="", soft_wrap=True)
        elif not self._in_reasoning:
            self.console.print("[dim italic]thinking...[/dim italic]", end="")
        self._in_reasoning = True

    def turn_end(self) -> None:
        if self._in_text or self._in_reasoning:
            self.console.print()
        self._in_text = self._in_reasoning = False

    # --- tools -------------------------------------------------------------
    def tool_call(self, name: str, summary: str) -> None:
        self.turn_end()
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

    def status(self, used: int, window: int) -> None:
        pct = 100 * used / window if window else 0
        color = "dim" if pct < 60 else ("yellow" if pct < 85 else "red")
        self.console.print(f"[{color}]{fmt_tokens(used)}/{fmt_tokens(window)} tokens ({pct:.0f}%)[/{color}]")
