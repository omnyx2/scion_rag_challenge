from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ImageInput:
    url: Optional[str] = None
    bytes: Optional[bytes] = None
    mime_type: Optional[str] = None


@dataclass
class Message:
    role: str  # 'system' | 'user' | 'assistant'
    content: Optional[str] = None
    images: Optional[List[ImageInput]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: Optional[str]
    raw: Dict[str, Any]
