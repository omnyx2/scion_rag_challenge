from typing import Iterable
import logging
import os
import time

_logger = logging.getLogger(__name__)


def mask_api_key(key: str | None) -> str:
    if not key:
        return "<none>"
    if len(key) <= 8:
        return key[:2] + "*" * max(0, len(key) - 2)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


def simple_token_count(text: str | None) -> int:
    """A naive token estimator: words -> tokens. Replace with model tokenizer for accuracy."""
    if not text:
        return 0
    # count whitespace-separated tokens conservatively
    return max(1, len(text.split()))


def backoff_sleep(attempt: int, base: float = 0.5, cap: float = 30.0) -> None:
    s = min(cap, base * (2 ** (attempt - 1)))
    jitter = s * 0.1
    time.sleep(s + jitter)
