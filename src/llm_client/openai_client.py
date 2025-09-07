from __future__ import annotations
from typing import Optional, List
from .base import BaseLLMClient
from .types import Message, ImageInput, GenerationResult
from .errors import UnsupportedFeatureError, APIError
from .utils import backoff_sleep
import httpx
import time
import logging

_logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """A light wrapper around OpenAI ChatCompletions (and text completion for instruct style).
    Uses httpx so dependency is minimal. This will work with OPENAI_API_KEY env var or passed key.
    """

    API_BASE = "https://api.openai.com/v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = httpx.Client(timeout=self.timeout_s)

    def _build_messages(self) -> List[dict]:
        history = self.get_history()
        msgs = []
        for m in history:
            entry = {"role": m.role, "content": m.content or ""}
            msgs.append(entry)
        return msgs

    def generate(
        self, temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> GenerationResult:
        url = f"{self.API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.user_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self._build_messages(),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # check images in history
        for m in self.get_history():
            if m.images:
                # OpenAI Chat API supports content with image references in some multimodal models.
                # For safety, we only allow images if model name suggests multimodal support.
                if (
                    "vision" not in self.model
                    and "multimodal" not in self.model
                    and "gpt-image" not in self.model
                ):
                    raise UnsupportedFeatureError(
                        "Model does not support image inputs via this client"
                    )

        attempt = 0
        while True:
            attempt += 1
            start = time.time()
            try:
                r = self._client.post(url, headers=headers, json=payload)
                latency = time.time() - start
                _logger.info(
                    f"openai request model={self.model} status={r.status_code} time={latency:.2f}s"
                )
                if r.status_code == 200:
                    data = r.json()
                    # take first choice
                    choice = data.get("choices", [])[0]
                    text = (
                        choice.get("message", {}).get("content")
                        or choice.get("text")
                        or ""
                    )
                    usage = data.get("usage")
                    self._update_token_counters(payload.get("messages"), text, usage)
                    res = GenerationResult(
                        text=text,
                        model=self.model,
                        prompt_tokens=usage.get("prompt_tokens") if usage else 0,
                        completion_tokens=usage.get("completion_tokens")
                        if usage
                        else 0,
                        total_tokens=(usage.get("total_tokens") if usage else 0),
                        finish_reason=choice.get("finish_reason"),
                        raw=data,
                    )
                    # append assistant message to history if enabled
                    if self.history_enabled:
                        self.add_assistant(res.text)
                    return res
                if (
                    r.status_code in (429, 500, 502, 503, 504)
                    and attempt <= self.max_retries
                ):
                    _logger.warning(
                        f"openai transient status {r.status_code}, retry {attempt}/{self.max_retries}"
                    )
                    backoff_sleep(attempt)
                    continue
                # non-retryable
                raise APIError(
                    f"OpenAI API error: status={r.status_code} body={r.text}"
                )
            except httpx.RequestError as e:
                if attempt <= self.max_retries:
                    _logger.warning(
                        f"network error, retry {attempt}/{self.max_retries}: {e}"
                    )
                    backoff_sleep(attempt)
                    continue
                raise APIError(f"Network error after retries: {e}")

    async def agenerate(
        self, temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> GenerationResult:
        # Async implementation using httpx.AsyncClient
        url = f"{self.API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.user_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": self._build_messages(),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            attempt = 0
            while True:
                attempt += 1
                try:
                    r = await client.post(url, headers=headers, json=payload)
                    if r.status_code == 200:
                        data = r.json()
                        choice = data.get("choices", [])[0]
                        text = (
                            choice.get("message", {}).get("content")
                            or choice.get("text")
                            or ""
                        )
                        usage = data.get("usage")
                        self._update_token_counters(
                            payload.get("messages"), text, usage
                        )
                        res = GenerationResult(
                            text=text,
                            model=self.model,
                            prompt_tokens=usage.get("prompt_tokens") if usage else 0,
                            completion_tokens=usage.get("completion_tokens")
                            if usage
                            else 0,
                            total_tokens=(usage.get("total_tokens") if usage else 0),
                            finish_reason=choice.get("finish_reason"),
                            raw=data,
                        )
                        if self.history_enabled:
                            self.add_assistant(res.text)
                        return res
                    if (
                        r.status_code in (429, 500, 502, 503, 504)
                        and attempt <= self.max_retries
                    ):
                        backoff_sleep(attempt)
                        continue
                    raise APIError(
                        f"OpenAI API error: status={r.status_code} body={r.text}"
                    )
                except httpx.RequestError as e:
                    if attempt <= self.max_retries:
                        backoff_sleep(attempt)
                        continue
                    raise APIError(f"Network error after retries: {e}")
