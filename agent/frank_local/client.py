"""Streaming chat client for OpenAI-compatible servers (llama.cpp, vLLM, Ollama)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI

from . import thinking
from .config import Settings


class ContextOverflow(Exception):
    """The server rejected the request because the prompt is too long."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text as produced by the model

    def parsed(self) -> dict[str, Any]:
        if not self.arguments.strip():
            return {}
        data = json.loads(self.arguments)
        if not isinstance(data, dict):
            raise ValueError("tool arguments must be a JSON object")
        return data

    def to_message_part(self) -> dict[str, Any]:
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": self.arguments or "{}"}}


@dataclass
class Turn:
    text: str = ""
    reasoning: str = ""
    raw: str = ""  # content exactly as the model produced it, <think> block included
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None


@dataclass
class StreamAccumulator:
    """Folds streamed deltas into text, reasoning and complete tool calls.

    Works on any object with .content / .reasoning_content / .tool_calls attributes,
    which is what the SDK's ChoiceDelta provides; tests feed SimpleNamespace.
    A <think> block at the start of the content is treated as reasoning too.
    """

    text: str = ""
    reasoning: str = ""
    raw: str = ""
    _split: thinking.ThinkSplitter = field(default_factory=thinking.ThinkSplitter)
    _calls: dict[int, dict[str, str]] = field(default_factory=dict)
    _n: int = 0

    def feed(self, delta: Any) -> list[tuple[str, str]]:
        events: list[tuple[str, str]] = []
        r = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if r:
            self.reasoning += r
            events.append(("reasoning", r))
        c = getattr(delta, "content", None)
        if c:
            self.raw += c
            events += self._take(self._split.feed(c))
        for tc in getattr(delta, "tool_calls", None) or []:
            idx = getattr(tc, "index", None)
            if idx is None:
                idx = self._n
                self._n += 1
            slot = self._calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if getattr(tc, "id", None):
                slot["id"] = tc.id
            fn = getattr(tc, "function", None)
            if fn is not None:
                if getattr(fn, "name", None):
                    slot["name"] += fn.name if not slot["name"] else ""
                if getattr(fn, "arguments", None):
                    slot["arguments"] += fn.arguments
            events.append(("tool", slot["name"]))  # a call being written: something is happening
        return events

    def _take(self, pieces: list[tuple[str, str]]) -> list[tuple[str, str]]:
        for kind, s in pieces:
            if kind == "reasoning":
                self.reasoning += s
            else:
                self.text += s
        return pieces

    def finish(self) -> list[tuple[str, str]]:
        return self._take(self._split.flush())

    def tool_calls(self) -> list[ToolCall]:
        out: list[ToolCall] = []
        for i, slot in sorted(self._calls.items()):
            if not slot["name"]:
                continue
            out.append(ToolCall(id=slot["id"] or f"call_{int(time.time())}_{i}", name=slot["name"], arguments=slot["arguments"]))
        return out


class Client:
    def __init__(self, settings: Settings, *, base_url: str | None = None, api_key: str | None = None, model: str | None = None):
        self.settings = settings
        self.base_url = (base_url or settings.base_url).rstrip("/")
        self.api_key = api_key or settings.api_key or "local"
        self.model = model or settings.model
        self._no_effort: set[str] = set()  # models whose server refused the reasoning_effort we sent
        self._oa = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=httpx.Timeout(float(settings.request_timeout), connect=10.0), max_retries=0)

    @property
    def server_root(self) -> str:
        return self.base_url.removesuffix("/v1")

    # ---- discovery -------------------------------------------------------
    def probe(self) -> dict[str, Any]:
        """Model ids and the context window, as far as the server will tell us."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        info: dict[str, Any] = {"models": [], "context": None, "server": None}
        try:
            r = httpx.get(f"{self.base_url}/models", headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json().get("data", [])
            info["models"] = [m.get("id") for m in data if m.get("id")]
            for m in data:
                if m.get("max_model_len"):           # vLLM
                    info["context"] = int(m["max_model_len"])
                    info["server"] = "vllm"
                elif m.get("owned_by") == "ds4.c" and m.get("context_length"):
                    info["context"] = int(m["context_length"])
                    info["server"] = "ds4"
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
            return info
        try:                                          # llama.cpp
            r = httpx.get(f"{self.server_root}/props", headers=headers, timeout=10)
            if r.status_code == 200:
                props = r.json()
                n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
                if n_ctx:
                    info["context"] = int(n_ctx)
                    info["server"] = "llama.cpp"
                info["build"] = props.get("build_info")
        except Exception:
            pass
        if not self.model and info["models"]:
            self.model = info["models"][0]
        return info

    # ---- requests --------------------------------------------------------
    def _extra_body(self) -> dict[str, Any]:
        body: dict[str, Any] = json.loads(json.dumps(self.settings.extra_body))
        effort = None if self.model in self._no_effort else thinking.server_effort(self.settings, self.model)
        if effort:
            if self.settings.backend == "llama.cpp":
                kw = body.setdefault("chat_template_kwargs", {})
                kw.setdefault("reasoning_effort", effort)
            else:
                body.setdefault("reasoning_effort", effort)
        if thinking.mode(self.settings, self.model) == "off":  # hybrid models (Qwen3, GLM) honour this
            body.setdefault("chat_template_kwargs", {}).setdefault("enable_thinking", False)
        return body

    def _kwargs(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, max_tokens: int | None) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "extra_body": self._extra_body(),
        }
        if tools:
            kw["tools"] = tools
        if self.settings.temperature is not None:
            kw["temperature"] = self.settings.temperature
        return kw

    def _call(self, fn, *args, **kwargs):
        delay = 2.0
        for attempt in range(6):
            try:
                return fn(*args, **kwargs)
            except APIConnectionError:
                if attempt == 5:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 20)
            except APIStatusError as e:
                msg = str(e).lower()
                if e.status_code == 400 and "reasoning" in msg and "effort" in msg and self.model not in self._no_effort:
                    # Levels differ by model family and this one does not take ours: go on with its default.
                    self._no_effort.add(self.model)
                    extra = kwargs.get("extra_body") or {}
                    extra.pop("reasoning_effort", None)
                    (extra.get("chat_template_kwargs") or {}).pop("reasoning_effort", None)
                    continue
                if e.status_code == 400 and any(k in msg for k in ("context", "exceed", "too many tokens", "n_ctx", "too long", "maximum")):
                    raise ContextOverflow(str(e)) from e
                if e.status_code == 503 and attempt < 5:  # model still loading
                    time.sleep(delay)
                    delay = min(delay * 2, 20)
                    continue
                raise
        raise RuntimeError("unreachable")

    def stream(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, max_tokens: int | None = None, stop: Any = None) -> Iterator[tuple[str, Any]]:
        """Yield ('reasoning', str) | ('text', str) | ('tool', name) | ('progress', dict) | ('done', Turn).

        'progress' is llama.cpp reading the prompt: total, cache, processed
        (cache included) and time_ms, a few seconds apart until the first token.

        stop, when given, is called between chunks; if it returns true the
        stream is cut off there and the partial turn is yielded as usual.
        """
        kw = self._kwargs(messages, tools, max_tokens)
        kw["stream"] = True
        kw["stream_options"] = {"include_usage": True}
        if self.settings.backend == "llama.cpp":
            # Progress chunks while the prompt fills keep bytes coming, so a long
            # cold prompt does not look like a dead server to request_timeout.
            kw["extra_body"]["return_progress"] = True
        acc = StreamAccumulator()
        turn = Turn()
        stream = self._call(self._oa.chat.completions.create, **kw)
        try:
            for chunk in stream:
                if stop is not None and stop():
                    break
                if getattr(chunk, "usage", None):
                    u = chunk.usage
                    turn.usage = {"prompt_tokens": u.prompt_tokens or 0, "completion_tokens": u.completion_tokens or 0}
                progress = (getattr(chunk, "model_extra", None) or {}).get("prompt_progress")
                if isinstance(progress, dict):
                    yield ("progress", progress)
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.delta is not None:
                    for ev in acc.feed(choice.delta):
                        yield ev
                if choice.finish_reason:
                    turn.finish_reason = choice.finish_reason
        finally:
            stream.close()
        for ev in acc.finish():
            yield ev
        turn.text = acc.text
        turn.reasoning = acc.reasoning
        turn.raw = acc.raw
        turn.tool_calls = acc.tool_calls()
        yield ("done", turn)

    def complete(self, messages: list[dict[str, Any]], max_tokens: int | None = None, tools: list[dict[str, Any]] | None = None) -> str:
        """Text completion (used for compaction summaries). Streamed underneath,
        so the same silence timeout applies as for a turn."""
        turn = Turn()
        for kind, value in self.stream(messages, tools, max_tokens):
            if kind == "done":
                turn = value
        return thinking.strip_think(turn.text or "").strip()
