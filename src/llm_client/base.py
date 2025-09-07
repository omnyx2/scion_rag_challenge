from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Literal, Optional, List
from dataclasses import dataclass
import logging
import os
import time

from .types import Message, ImageInput, GenerationResult
from .history import HistoryStore, InMemoryHistoryStore
from .errors import UnsupportedFeatureError, AuthError, APIError
from .utils import mask_api_key, simple_token_count, backoff_sleep

_logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    def __init__(
        self,
        model: str,
        user_api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout_s: float = 30.0,
        history_store: Optional[HistoryStore] = None,
        history_enabled: bool = True,
        history_max_messages: Optional[int] = 50,
    ) -> None:
        self.model = model
        self.user_api_key = (
            user_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        )
        if not self.user_api_key:
            raise AuthError("API key not provided. Set env var or pass user_api_key.")
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self.history_enabled = history_enabled
        self.history_max_messages = history_max_messages
        self._history: HistoryStore = history_store or InMemoryHistoryStore()
        self._mode: Literal["instruct", "uninstruct"] = "uninstruct"
        self._system_messages: List[Message] = []
        self._logger = _logger

        # usage counters
        self._prompt_tokens = 0
        self._completion_tokens = 0

        self._log_meta()

    def _log_meta(self) -> None:
        self._logger.debug(
            f"LLM client init model={self.model} key={mask_api_key(self.user_api_key)}"
        )

    def set_mode(self, mode: Literal["instruct", "uninstruct"]) -> None:
        if mode not in ("instruct", "uninstruct"):
            raise ValueError("mode must be 'instruct' or 'uninstruct'")
        self._mode = mode

    def add_system(self, content: str) -> None:
        m = Message(role="system", content=content)
        if self.history_enabled:
            self._history.append(m)
        self._system_messages.append(m)

    def add_user(self, content: str, images: Optional[List[ImageInput]] = None) -> None:
        m = Message(role="user", content=content, images=images)
        if self.history_enabled:
            self._history.append(m)
        return m

    def add_assistant(self, content: str) -> None:
        m = Message(role="assistant", content=content)
        if self.history_enabled:
            self._history.append(m)
        return m

    def clear_history(self) -> None:
        self._history.clear()

    def get_history(self) -> List[Message]:
        return self._history.get()

    @abstractmethod
    def generate(
        self, temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> GenerationResult: ...

    @abstractmethod
    async def agenerate(
        self, temperature: float = 0.7, max_tokens: Optional[int] = None
    ) -> GenerationResult: ...

    # helper used by implementations
    def _update_token_counters(
        self,
        prompt_text: str | None,
        completion_text: str | None,
        usage: dict | None = None,
    ) -> None:
        if usage:
            # prefer authoritative usage if provided
            prompt = usage.get("prompt_tokens") or usage.get("input_tokens")
            comp = usage.get("completion_tokens") or usage.get("output_tokens")
            if prompt is not None:
                self._prompt_tokens += int(prompt)
            if comp is not None:
                self._completion_tokens += int(comp)
            return
        # naive fallback
        self._prompt_tokens += simple_token_count(prompt_text)
        self._completion_tokens += simple_token_count(completion_text)

    def get_token_usage(self) -> dict:
        total = self._prompt_tokens + self._completion_tokens
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": total,
        }
