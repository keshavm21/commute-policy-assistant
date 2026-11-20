"""LLM clients behind one tiny interface, so tests and the offline eval can swap in fakes.

    complete(system, user, timeout) -> str   (raw model text; callers treat it as untrusted)
"""

import json
import re
import time
from typing import Callable, Protocol

import httpx


class LLMError(Exception):
    """The provider failed. `retryable` says whether a second attempt could help."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class LLMBudgetExceeded(Exception):
    """A code path tried to make more LLM calls than the per-question limit allows."""


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str, timeout: float) -> str: ...


class OpenAICompatibleClient:
    """Any OpenAI-style /chat/completions endpoint: Groq, Gemini's compat API, Ollama..."""

    name = "openai_compatible"

    def __init__(self, base_url: str, api_key: str, model: str, max_tokens: int = 300,
                 reasoning_effort: str = ""):
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"), headers={"Authorization": f"Bearer {api_key}"}
        )

    def complete(self, system: str, user: str, timeout: float) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,  # same question -> same answer, as far as the provider allows
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        try:
            resp = self._http.post("/chat/completions", json=payload, timeout=timeout)
        except httpx.TimeoutException:
            raise LLMError("LLM request timed out") from None
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {type(exc).__name__}") from None

        if resp.status_code != 200:
            # 429 (quota) and 4xx won't fix themselves within a 3 s budget; 5xx might.
            retryable = resp.status_code >= 500
            raise LLMError(f"LLM returned HTTP {resp.status_code}", retryable=retryable)
        try:
            return resp.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError("LLM response had an unexpected shape") from None


class BudgetedLLM:
    """Wraps a client for ONE question and refuses to exceed the call limit."""

    def __init__(self, client: LLMClient, max_calls: int):
        self._client = client
        self.max_calls = max_calls
        self.calls = 0

    def complete(self, system: str, user: str, timeout: float) -> str:
        if self.calls >= self.max_calls:
            raise LLMBudgetExceeded(f"limit of {self.max_calls} LLM calls reached")
        self.calls += 1
        return self._client.complete(system, user, timeout)


# ---------------------------------------------------------------------------- fakes

Step = str | Exception | Callable[[str, str], str]


class ScriptedFakeLLM:
    """Test double: returns (or raises) the scripted steps in order, optionally slowly."""

    name = "fake_scripted"

    def __init__(self, *steps: Step, delay_s: float = 0.0):
        self.steps = list(steps)
        self.delay_s = delay_s
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, timeout: float) -> str:
        self.prompts.append((system, user))
        if self.delay_s:
            time.sleep(self.delay_s)
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step(system, user) if callable(step) else step


_EXCERPT_RE = re.compile(r'<policy_excerpt id="(C\d+)"[^>]*>\n(.*?)\n</policy_excerpt>', re.S)


class ExtractiveFakeLLM:
    """Offline stand-in used when no API key is set, and by the eval in `fake` mode.

    It does no language understanding: it "answers" by quoting the top-ranked excerpt
    and citing it. That lets the whole pipeline (auth, retrieval, gating, validation,
    citations) run end to end with no network, while being honest that answer wording
    is only as good as retrieval.
    """

    name = "fake_extractive"

    def complete(self, system: str, user: str, timeout: float) -> str:
        excerpts = _EXCERPT_RE.findall(user)
        if not excerpts:
            return json.dumps({"status": "not_covered", "answer": "", "citations": []})
        label, text = excerpts[0]
        body = text.split("\n", 1)[-1].strip()  # drop the heading line
        return json.dumps({"status": "answered", "answer": body, "citations": [label]})
