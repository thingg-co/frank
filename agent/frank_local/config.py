"""Settings: defaults, then config file, then FRANK_* environment, then CLI flags."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
AGENT_DIR = PACKAGE_DIR.parent          # .../frank/agent when run from the repo
REPO_DIR = AGENT_DIR.parent

Backend = str  # "auto" | "llama.cpp" | "vllm" | "ollama" | "openai"


def default_home() -> Path:
    """Where sessions and history live. Inside the repo when running from it."""
    env = os.environ.get("FRANK_HOME")
    if env:
        return Path(env).expanduser()
    if (AGENT_DIR / "pyproject.toml").exists():
        return AGENT_DIR / ".frank"
    return Path.home() / ".frank"


@dataclass
class Settings:
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "local"
    model: str = ""                       # empty: first model the server lists
    backend: Backend = "auto"            # auto: whatever the server turns out to be when probed
    max_tokens: int = 32768               # per response, thinking included; clamped to the room left in the window
    request_timeout: int = 300            # seconds without a byte from the server; llama.cpp streams progress while it fills a long prompt
    temperature: float | None = None     # None: server default
    context_window: int | None = None    # None: probe the server, else fall back to 32768
    min_context: int = 0                 # warn at startup (and fail --doctor) when the server's window is smaller
    compact_threshold: float = 0.75      # compact when prompt tokens exceed this share of the window
    reasoning_effort: str = "max"        # off | low | medium | high | max; max is the deepest each model has
    thinking: str = "auto"               # auto | native | prompted | off; auto picks by model name (see thinking.py)
    show_thinking: bool = False
    send_reasoning_back: bool = True     # keep the model's reasoning in the history it is sent back
    permission_mode: str = "ask"         # ask | auto | readonly
    allowed_commands: list[str] = field(default_factory=list)  # extra bash prefixes that never prompt
    max_tool_output_chars: int = 0        # 0: scale with the context window, about a tenth of it
    max_iterations: int = 100            # tool-call rounds per user turn
    bash_timeout: int = 120
    max_subagent_depth: int = 1
    subagent_base_url: str = ""          # empty: same server as the main model
    subagent_model: str = ""
    subagent_api_key: str = ""
    web_host: str = ""                   # frank --web listens here; empty: 127.0.0.1, this machine only
    web_port: int = 8089
    projects_dir: str = "~/Development"  # the web page offers the folders in it for a new chat
    name: str = "Frank"                  # how the agent refers to itself in the system prompt
    system_prompt_extra: str = ""        # appended to the system prompt (project rules, style)
    extra_body: dict[str, Any] = field(default_factory=dict)  # merged into every request
    home: Path = field(default_factory=default_home)

    @property
    def server_root(self) -> str:
        """http://host:port without the /v1 suffix (llama.cpp's own endpoints live there)."""
        return self.base_url.rstrip("/").removesuffix("/v1")


def _config_paths() -> list[Path]:
    paths: list[Path] = []
    if env := os.environ.get("FRANK_CONFIG"):
        paths.append(Path(env).expanduser())
    paths.append(Path.cwd() / ".frank.toml")
    paths.append(AGENT_DIR / "config.toml")
    xdg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    paths.append(xdg / "frank" / "config.toml")
    return paths


def _coerce(f_type: Any, value: Any) -> Any:
    if f_type in ("int", int):
        return int(value)
    if f_type in ("float | None", "float"):
        return None if value in ("", None) else float(value)
    if f_type in ("int | None",):
        return None if value in ("", None) else int(value)
    if f_type in ("bool", bool):
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("1", "true", "yes", "on")
    if f_type == "list[str]":
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return list(value)
    if f_type == "Path":
        return Path(value).expanduser()
    if f_type == "dict[str, Any]":
        return dict(value)
    return value


def load_settings(overrides: dict[str, Any] | None = None) -> Settings:
    data: dict[str, Any] = {}
    for p in _config_paths():
        if p.is_file():
            with p.open("rb") as fh:
                data.update(tomllib.load(fh))
            break
    names = {f.name: f.type for f in fields(Settings)}
    for name, f_type in names.items():
        env = os.environ.get("FRANK_" + name.upper())
        if env is not None:
            data[name] = env
    for k, v in (overrides or {}).items():
        if v is not None:
            data[k] = v
    kwargs = {k: _coerce(names[k], v) for k, v in data.items() if k in names}
    return Settings(**kwargs)
