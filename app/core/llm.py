"""OpenRouter LLM gateway client with retries, model fallback and a circuit breaker."""
from __future__ import annotations

import json
import logging
import re
import threading
import time

import httpx

from app.config import Settings
from app.core import metrics as m

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    BREAKER_THRESHOLD = 4       # consecutive failures
    BREAKER_COOLDOWN_S = 30.0

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.s = settings
        self._client = httpx.Client(timeout=settings.llm_timeout_s, transport=transport)
        self._fails = 0
        self._open_until = 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.s.llm_enabled

    def _breaker_open(self) -> bool:
        with self._lock:
            is_open = time.time() < self._open_until
        m.LLM_BREAKER_OPEN.set(1 if is_open else 0)
        return is_open

    def _record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._fails = 0
            else:
                self._fails += 1
                if self._fails >= self.BREAKER_THRESHOLD:
                    self._open_until = time.time() + self.BREAKER_COOLDOWN_S
                    self._fails = 0
                    log.error("LLM circuit breaker opened for %ss", self.BREAKER_COOLDOWN_S)

    def _call(self, model: str, messages: list[dict], json_mode: bool) -> str:
        body = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": self.s.llm_max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.s.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/your-org/ticket-resolution-assistant",
            "X-Title": "Ticket Resolution Assistant",
        }
        last: Exception | None = None
        for attempt in range(self.s.llm_max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._client.post(f"{self.s.openrouter_base_url}/chat/completions", json=body, headers=headers)
                m.LLM_LATENCY.observe(time.perf_counter() - t0)
                if r.status_code in (429, 500, 502, 503, 504):
                    m.LLM_CALLS.labels(model, str(r.status_code)).inc()
                    last = RuntimeError(f"HTTP {r.status_code}")
                    time.sleep(min(2 ** attempt * 0.4, 4))
                    continue
                r.raise_for_status()
                m.LLM_CALLS.labels(model, "ok").inc()
                return r.json()["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.TransportError) as e:
                m.LLM_CALLS.labels(model, "network").inc()
                last = e
                time.sleep(min(2 ** attempt * 0.4, 4))
            except (httpx.HTTPStatusError, KeyError, ValueError) as e:
                m.LLM_CALLS.labels(model, "error").inc()
                raise LLMUnavailable(str(e)) from e
        raise LLMUnavailable(f"retries exhausted: {last}")

    def chat(self, messages: list[dict], *, json_mode: bool = True) -> str:
        if not self.enabled:
            raise LLMUnavailable("no OPENROUTER_API_KEY configured")
        if self._breaker_open():
            raise LLMUnavailable("circuit breaker open")
        models = [self.s.llm_model] + ([self.s.llm_fallback_model] if self.s.llm_fallback_model else [])
        err: Exception | None = None
        for model in models:
            try:
                out = self._call(model, messages, json_mode)
                self._record(True)
                return out
            except LLMUnavailable as e:
                err = e
        self._record(False)
        raise LLMUnavailable(str(err))

    def chat_json(self, messages: list[dict]) -> dict:
        raw = self.chat(messages, json_mode=True)
        return parse_json_loose(raw)


def parse_json_loose(raw: str) -> dict:
    """LLMs sometimes wrap JSON in prose or code fences; be tolerant but strict about the result type."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        m_ = re.search(r"\{.*\}", raw, re.S)
        if not m_:
            raise LLMUnavailable("non-JSON LLM output") from None
        try:
            obj = json.loads(m_.group(0))
        except json.JSONDecodeError as e:
            raise LLMUnavailable("malformed JSON from LLM") from e
    if not isinstance(obj, dict):
        raise LLMUnavailable("LLM JSON is not an object")
    return obj
