from __future__ import annotations
from typing import Optional, Dict, Any, List
from .base import BaseLLMClient, Message  # Message 타입을 base에서 가져온다고 가정
from .types import GenerationResult
from .errors import UnsupportedFeatureError, APIError
from .utils import backoff_sleep
import httpx
import logging
import base64  # 이미지 처리를 위해 추가

_logger = logging.getLogger(__name__)


class GeminiClient(BaseLLMClient):
    """
    A Gemini client updated to use the correct API contract for authentication,
    payload structure, and response parsing, including basic multimodal support.
    """

    API_BASE = "https://generativeai.googleapis.com/v1beta"  # v1beta2 대신 v1beta 사용

    def __init__(self, *args, google_api_key_env: str = "GOOGLE_API_KEY", **kwargs):
        super().__init__(*args, **kwargs)
        self.google_key = self.user_api_key or __import__("os").environ.get(
            google_api_key_env
        )
        if not self.google_key:
            raise APIError(
                "Google API key not provided; set GOOGLE_API_KEY or pass user_api_key"
            )

    def _supports_images(self) -> bool:
        """More accurately checks if the model supports image inputs."""
        model_name = self.model.lower()
        # gemini-pro-vision, gemini-1.5-pro 등 이미지 지원 모델을 확인
        return "vision" in model_name or "1.5" in model_name

    def _build_payload(
        self, temperature: float, max_tokens: Optional[int]
    ) -> Dict[str, Any]:
        """Builds the request payload according to the Gemini API specification."""

        # History를 Gemini 'contents' 형식으로 변환
        contents: List[Dict[str, Any]] = []
        for msg in self.get_history():
            parts = [{"text": msg.content}]
            # 이미지 데이터가 있다면 parts에 추가 (base64 인코딩된 데이터로 가정)
            if msg.images:
                if not self._supports_images():
                    raise UnsupportedFeatureError(
                        f"Model '{self.model}' does not support images."
                    )
                for img_data in msg.images:
                    # img_data가 base64 문자열이라고 가정. mime-type도 필요.
                    # 실제 구현에서는 Message 객체에 mime_type 필드가 있어야 함.
                    # 여기서는 일반적인 'image/jpeg'로 가정.
                    parts.append(
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_data}}
                    )
            contents.append({"role": msg.role, "parts": parts})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens or 2048,  # 기본값을 넉넉하게 설정
            },
        }
        return payload

    def _parse_response(
        self, data: Dict[str, Any], payload: Dict[str, Any]
    ) -> GenerationResult:
        """Parses the JSON response from the Gemini API."""
        try:
            # 텍스트 추출 경로 수정
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            text = ""  # 응답이 비정상적일 경우 빈 문자열 반환

        # 토큰 사용량 추출 경로 수정 (usageMetadata)
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount", 0)
        completion_tokens = usage.get("candidatesTokenCount", 0)
        total_tokens = usage.get("totalTokenCount", 0)

        # 토큰 카운터 업데이트
        self._update_token_counters(payload, text, usage)  # usage 객체 전체를 전달

        return GenerationResult(
            text=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=data["candidates"][0].get("finishReason"),
            raw=data,
        )

    def generate(
        self, temperature: float = 0.2, max_tokens: Optional[int] = None
    ) -> GenerationResult:
        # URL과 헤더를 올바른 방식으로 수정
        url = (
            f"{self.API_BASE}/models/{self.model}:generateContent?key={self.google_key}"
        )
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.google_key,
        }

        payload = self._build_payload(temperature, max_tokens)

        attempt = 0
        while True:
            attempt += 1
            try:
                r = httpx.post(
                    url, headers=headers, json=payload, timeout=self.timeout_s
                )
                if r.status_code == 200:
                    data = r.json()
                    res = self._parse_response(data, payload)
                    if self.history_enabled:
                        self.add_assistant(res.text)
                    return res

                if (
                    r.status_code in (429, 500, 502, 503, 504)
                    and attempt <= self.max_retries
                ):
                    _logger.warning("Gemini transient error, retrying...")
                    backoff_sleep(attempt)
                    continue

                raise APIError(
                    f"Gemini API error: status={r.status_code} body={r.text}"
                )
            except httpx.RequestError as e:
                if attempt <= self.max_retries:
                    backoff_sleep(attempt)
                    continue
                raise APIError(
                    f"Request failed after {self.max_retries} retries: {str(e)}"
                )

    async def agenerate(
        self, temperature: float = 0.2, max_tokens: Optional[int] = None
    ) -> GenerationResult:
        # URL과 헤더를 올바른 방식으로 수정
        url = (
            f"{self.API_BASE}/models/{self.model}:generateContent?key={self.google_key}"
        )
        headers = {"Content-Type": "application/json"}

        payload = self._build_payload(temperature, max_tokens)

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            attempt = 0
            while True:
                attempt += 1
                try:
                    r = await client.post(url, headers=headers, json=payload)
                    if r.status_code == 200:
                        data = r.json()
                        res = self._parse_response(data, payload)
                        if self.history_enabled:
                            self.add_assistant(res.text)
                        return res

                    if (
                        r.status_code in (429, 500, 502, 503, 504)
                        and attempt <= self.max_retries
                    ):
                        _logger.warning("Gemini transient error, retrying...")
                        backoff_sleep(attempt)
                        continue

                    raise APIError(
                        f"Gemini API error: status={r.status_code} body={r.text}"
                    )
                except httpx.RequestError as e:
                    if attempt <= self.max_retries:
                        backoff_sleep(attempt)
                        continue
                    raise APIError(
                        f"Request failed after {self.max_retries} retries: {str(e)}"
                    )
