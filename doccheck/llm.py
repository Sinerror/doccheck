"""LLM layer for any OpenAI-compatible /v1/chat/completions endpoint (router, LM Studio, Ollama, vLLM).

* Structured output: Pydantic model → JSON Schema → function tool with forced
  `tool_choice`. One streamed call gives live reasoning + strict JSON.
* Streaming: tool-call argument deltas are accumulated; the partial `reasoning`
  string is extracted from the incomplete JSON and emitted as `agent.partial`.
  Native reasoning tokens (`reasoning_content`, e.g. DeepSeek) are emitted as
  `agent.partial` with kind="thinking".
* Retries: tenacity, 3 attempts on 429/5xx/connection errors. Invalid JSON →
  one extra attempt with the validation error in the prompt.
* Usage of every call is pushed to the bus as a cumulative `metrics` event.
* The Analyst can use its own endpoint (DOCCHECK_ANALYST_BASE_URL, e.g. a local model); its calls cost $0.
* Fixtures: mode="replay" reads recorded raw stream chunks from
  tests/fixtures/llm/{key}.json; mode="record" calls the API and rewrites them.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from .config import Settings, price_usd
from .events import EventBus, Role

T = TypeVar("T", bound=BaseModel)
log = logging.getLogger(__name__)
Mode = Literal["live", "record", "replay"]

PARTIAL_EMIT_EVERY_CHARS = 60


# ---- JSON schema helpers -----------------------------------------------------

def inline_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic JSON schema with $refs inlined (some providers reject $defs)."""
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].split("/")[-1]
                merged = {**copy.deepcopy(defs[name]), **{k: v for k, v in node.items() if k != "$ref"}}
                return resolve(merged)
            return {k: resolve(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


def tool_spec(name: str, description: str, model: type[BaseModel] | dict[str, Any]) -> dict[str, Any]:
    params = inline_schema(model) if isinstance(model, type) else model
    return {"type": "function", "function": {"name": name, "description": description, "parameters": params}}


_STR_START = re.compile(r'"reasoning"\s*:\s*"')


def extract_partial_string(buf: str, key: str = "reasoning") -> str | None:
    """Return the (possibly unterminated) string value of `key` in a partial JSON buffer."""
    pat = _STR_START if key == "reasoning" else re.compile(r'"%s"\s*:\s*"' % re.escape(key))
    m = pat.search(buf)
    if not m:
        return None
    i = m.end()
    out = []
    while i < len(buf):
        ch = buf[i]
        if ch == "\\":
            if i + 1 >= len(buf):
                break
            nxt = buf[i + 1]
            if nxt == "u":
                if i + 6 > len(buf):
                    break
                try:
                    out.append(chr(int(buf[i + 2:i + 6], 16)))
                except ValueError:
                    pass
                i += 6
                continue
            out.append({"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}.get(nxt, nxt))
            i += 2
            continue
        if ch == '"':
            break
        out.append(ch)
        i += 1
    return "".join(out)


# ---- results -----------------------------------------------------------------

@dataclass
class ToolCallOut:
    id: str
    name: str
    arguments: str

    def parsed(self) -> dict[str, Any]:
        return json.loads(self.arguments or "{}")


@dataclass
class CallResult:
    content: str = ""
    thinking: str = ""
    tool_calls: list[ToolCallOut] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500 or exc.status_code == 429 or "overload" in str(exc).lower()
    return isinstance(exc, EmptyResponse)


class EmptyResponse(RuntimeError):
    pass


class RunawayOutput(RuntimeError):
    """Client-side guard: the router ignores max_tokens for some models, and a degenerate generation ran to
    128k tokens twice. We stop reading (and close the stream) past a size or time budget."""

    def __init__(self, model: str, chars: int, elapsed: float, prompt_chars: int):
        super().__init__(f"{model}: runaway output aborted after {chars} chars / {elapsed:.0f} s")
        self.model, self.chars, self.elapsed, self.prompt_chars = model, chars, elapsed, prompt_chars


def _messages_chars(messages: list[dict]) -> int:
    return sum(len(str(m.get("content") or "")) + len(json.dumps(m.get("tool_calls") or "")) for m in messages)


class LLM:
    def __init__(self, settings: Settings, bus: EventBus | None = None, mode: Mode | None = None,
                 fixtures_dir: Path | None = None, max_concurrency: int = 8):
        self.settings = settings
        self.bus = bus
        self.mode: Mode = mode or ("record" if settings.record else "live")
        self.fixtures_dir = fixtures_dir
        self._sem = asyncio.Semaphore(max_concurrency)
        self._clients: dict[str, AsyncOpenAI] = {}
        self._fixture_counts: dict[str, int] = {}

    def is_local(self, model: str) -> bool:
        return self.settings.analyst_is_local and model == self.settings.analyst_model

    def client_for(self, model: str) -> AsyncOpenAI:
        if self.is_local(model):
            url, key = self.settings.analyst_base_url, self.settings.analyst_api_key
        else:
            url, key = self.settings.llm_base_url, self.settings.llm_api_key
        if url not in self._clients:
            self._clients[url] = AsyncOpenAI(base_url=url, api_key=key or "none",
                                             timeout=self.settings.llm_timeout_s, max_retries=0)
        return self._clients[url]

    # -- raw transport (live / record / replay) --------------------------------

    def _fixture_path(self, key: str) -> Path:
        assert self.fixtures_dir is not None
        n = self._fixture_counts.get(key, 0)
        self._fixture_counts[key] = n + 1
        name = key if n == 0 else f"{key}__{n}"
        return self.fixtures_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.json"

    async def _raw_stream(self, model: str, messages: list[dict], tools: list[dict] | None,
                          tool_choice: Any, fixture_key: str | None, extra: dict[str, Any] | None = None):
        """Yield raw chunk dicts."""
        if self.mode == "replay":
            path = self._fixture_path(fixture_key or "default")
            for ch in json.loads(path.read_text(encoding="utf-8"))["chunks"]:
                yield ch
            return
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "stream": True,
                                  "max_tokens": self.settings.max_output_tokens,
                                  "stream_options": {"include_usage": True}}
        if tools:
            kwargs["tools"] = tools
            if self.is_local(model) and isinstance(tool_choice, dict):
                # LM Studio / llama.cpp servers accept only "none" | "auto" | "required". With the single tool
                # of a structured call, "required" forces the same function.
                tool_choice = "required" if len(tools) == 1 else "auto"
            kwargs["tool_choice"] = tool_choice
        if extra:
            kwargs["extra_body"] = extra
        recorded: list[dict] = []
        stream = await self.client_for(model).chat.completions.create(**kwargs)
        try:
            async for chunk in stream:
                d = chunk.model_dump(exclude_none=True)
                recorded.append(d)
                yield d
        finally:
            await stream.close()  # an aborted (runaway) call must stop generating on the provider side
        if self.mode == "record" and fixture_key and self.fixtures_dir:
            path = self._fixture_path(fixture_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"model": model, "messages": messages, "chunks": recorded},
                                       ensure_ascii=False, indent=1), encoding="utf-8")

    async def _one_call(self, *, role: Role, model: str, messages: list[dict], tools: list[dict] | None,
                        tool_choice: Any, claim_id: str | None, round: int, stream_tool: str | None,
                        fixture_key: str | None, extra: dict[str, Any] | None = None) -> CallResult:
        res = CallResult(model=model)
        calls: dict[int, dict[str, str]] = {}
        last_emit = {"reasoning": 0, "thinking": 0}

        def maybe_emit(kind: str, text: str, force: bool = False) -> None:
            if self.bus is None or not text:
                return
            if force or len(text) - last_emit[kind] >= PARTIAL_EMIT_EVERY_CHARS:
                if len(text) == last_emit[kind]:
                    return
                last_emit[kind] = len(text)
                self.bus.emit("agent.partial", {"role": role, "claim_id": claim_id, "round": round,
                                                "text": text, "kind": kind}, role=role, claim_id=claim_id)

        started = time.monotonic()
        streamed = 0
        gen = self._raw_stream(model, messages, tools, tool_choice, fixture_key, extra)
        try:
            async for ch in gen:
                streamed += self._consume(ch, res, calls, stream_tool, maybe_emit)
                elapsed = time.monotonic() - started
                if self.mode != "replay" and (streamed > self.settings.max_stream_chars
                                              or elapsed > self.settings.llm_call_max_s):
                    raise RunawayOutput(model, streamed, elapsed, _messages_chars(messages))
        finally:
            await gen.aclose()
        maybe_emit("thinking", res.thinking, force=True)
        res.tool_calls = [ToolCallOut(id=c["id"] or f"call_{i}", name=c["name"], arguments=c["arguments"])
                          for i, c in sorted(calls.items())]
        for c in res.tool_calls:
            if stream_tool and c.name == stream_tool:
                maybe_emit("reasoning", extract_partial_string(c.arguments) or "", force=True)
        if not res.tool_calls and not res.content:
            raise EmptyResponse(f"{model} returned an empty response")
        return res

    async def _guarded_call(self, **kw) -> CallResult:
        """Hard wall-clock budget for one call: a model can also think silently for minutes before the first
        chunk, which the in-stream check cannot see. Cancellation closes the HTTP stream (see _raw_stream)."""
        if self.mode == "replay":
            return await self._one_call(**kw)
        budget = self.settings.llm_call_max_s + 30
        try:
            return await asyncio.wait_for(self._one_call(**kw), timeout=budget)
        except asyncio.TimeoutError:
            raise RunawayOutput(kw["model"], 0, budget, _messages_chars(kw["messages"])) from None

    @staticmethod
    def _consume(ch: dict, res: CallResult, calls: dict[int, dict[str, str]], stream_tool: str | None,
                 maybe_emit) -> int:
        """Fold one raw chunk into the result; return how many visible characters it carried."""
        n = 0
        if ch.get("usage"):
            res.usage = ch["usage"]
        for choice in ch.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                res.thinking += delta["reasoning_content"]
                n += len(delta["reasoning_content"])
                maybe_emit("thinking", res.thinking)
            if delta.get("content"):
                res.content += delta["content"]
                n += len(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
                    n += len(fn["arguments"])
                    if stream_tool and slot["name"] == stream_tool:
                        partial = extract_partial_string(slot["arguments"])
                        if partial:
                            maybe_emit("reasoning", partial)
        return n

    def _account(self, res: CallResult) -> None:
        u = res.usage or {}
        tin = int(u.get("prompt_tokens") or 0)
        tout = int(u.get("completion_tokens") or 0)
        details = u.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0)
        if self.bus is not None:
            cost = 0.0 if self.is_local(res.model) else price_usd(res.model, tin, tout, cached)
            self.bus.add_usage(res.model, tin, tout, cost)

    async def call(self, *, role: Role, model: str, messages: list[dict], tools: list[dict] | None = None,
                   tool_choice: Any = "auto", claim_id: str | None = None, round: int = 0,
                   stream_tool: str | None = None, fixture_key: str | None = None) -> CallResult:
        extra: dict[str, Any] | None = None
        async with self._sem:
            for degrade in (False, True):
                try:
                    async for attempt in AsyncRetrying(stop=stop_after_attempt(3), reraise=True,
                                                       wait=wait_exponential(multiplier=2, min=2, max=20),
                                                       retry=retry_if_exception(_retryable)):
                        with attempt:
                            res = await self._guarded_call(role=role, model=model, messages=messages, tools=tools,
                                                           tool_choice=tool_choice, claim_id=claim_id, round=round,
                                                           stream_tool=stream_tool, fixture_key=fixture_key,
                                                           extra=extra)
                    break
                except RunawayOutput as e:
                    # No usage chunk arrives for an aborted stream: account an estimate (~4 chars per token)
                    # so the cost on screen is not under-reported.
                    if self.bus is not None:
                        tin, tout = e.prompt_chars // 4, e.chars // 4
                        self.bus.add_usage(model, tin, tout,
                                           0.0 if self.is_local(model) else price_usd(model, tin, tout))
                    if degrade or self.mode == "replay":
                        raise
                    log.warning("%s; retrying with reasoning_effort=low", e)
                    if self.bus:
                        self.bus.emit("error", {"message": f"{e}; retrying with reasoning_effort=low "
                                                           "(usage of the aborted call is estimated)",
                                                "where": f"llm.{role}"}, role=role, claim_id=claim_id)
                    extra = {"reasoning_effort": "low"}
                except openai.APIError as e:
                    # The router aborts a stream mid-way when the upstream hits its per-response time limit
                    # ("the same request will fail at the same place"); retry once with lower reasoning effort.
                    if degrade or isinstance(e, openai.APIStatusError) or self.mode == "replay":
                        raise
                    log.warning("%s stream aborted (%s); retrying with reasoning_effort=low", model, str(e)[:200])
                    if self.bus:
                        self.bus.emit("error", {"message": f"{model} stream aborted, retrying with "
                                                           f"reasoning_effort=low: {str(e)[:300]}",
                                                "where": f"llm.{role}"}, role=role, claim_id=claim_id)
                    extra = {"reasoning_effort": "low"}
            self._account(res)
            return res

    # -- structured output -------------------------------------------------------

    async def structured(self, *, role: Role, model: str, system: str, user: str, schema: type[T],
                         tool_name: str, tool_description: str = "Submit your answer.",
                         claim_id: str | None = None, round: int = 0, fixture_key: str | None = None,
                         emit_lifecycle: bool = True) -> T:
        tools = [tool_spec(tool_name, tool_description, schema)]
        choice = {"type": "function", "function": {"name": tool_name}}
        messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if emit_lifecycle and self.bus:
            self.bus.emit("agent.started", {"role": role, "claim_id": claim_id, "round": round, "model": model},
                          role=role, claim_id=claim_id)
        last_err: Exception | None = None
        fallback = self.settings.fallback_model if self.settings.fallback_model != model else self.settings.critic_model
        base_messages = messages
        for attempt in range(3):
            key = fixture_key if attempt == 0 else (f"{fixture_key}.{('fix', 'fallback')[attempt - 1]}"
                                                    if fixture_key else None)
            use_model = model
            if attempt == 2:
                # same model failed twice on the schema: one fresh attempt on a different model, visibly
                use_model, messages = fallback, base_messages
                if self.bus:
                    self.bus.emit("error", {"message": f"{model} returned invalid `{tool_name}` twice; "
                                                       f"retrying once with {fallback}: {str(last_err)[:300]}",
                                            "where": f"llm.{role}"}, role=role, claim_id=claim_id)
            res = await self.call(role=role, model=use_model, messages=messages, tools=tools, tool_choice=choice,
                                  claim_id=claim_id, round=round, stream_tool=tool_name, fixture_key=key)
            raw = next((c.arguments for c in res.tool_calls if c.name == tool_name), None)
            if raw is None:
                raw = _json_from_text(res.content)
            try:
                out = schema.model_validate_json(raw or "")
                if emit_lifecycle and self.bus:
                    self.bus.emit("agent.output", {"role": role, "claim_id": claim_id, "round": round,
                                                   "output": out.model_dump(mode="json")},
                                  role=role, claim_id=claim_id)
                return out
            except (ValidationError, ValueError) as e:
                last_err = e
                messages = messages + [{"role": "user", "content":
                    f"Your previous `{tool_name}` call failed validation:\n{str(e)[:1500]}\n"
                    f"Previous arguments:\n{(raw or '')[:3000]}\n"
                    f"Call `{tool_name}` again with arguments that satisfy the schema exactly."}]
        raise RuntimeError(f"{role}/{tool_name}: invalid structured output after retries: {last_err}")


def _json_from_text(text: str) -> str | None:
    """Fallback when a model answers in plain text instead of a tool call."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        return m.group(1)
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if s != -1 and e > s else None


