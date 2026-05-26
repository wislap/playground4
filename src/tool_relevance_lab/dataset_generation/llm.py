from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class LLMRequest:
    request_id: str
    idempotency_key: str
    messages: list[dict[str, str]]
    temperature: float
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class LLMResponse:
    request_id: str
    text: str
    raw: dict[str, Any]


class LLMClient(ABC):
    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return one completion for an idempotent request."""


class NotConfiguredLLMClient(LLMClient):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError(
            "No LLM client is configured. Provide a relay-backed LLMClient implementation."
        )


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float = 120.0
    max_retries: int = 3


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        if request.response_format is not None:
            payload["response_format"] = request.response_format

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": request.idempotency_key,
        }
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    raw = response.json()
                    text = str(raw["choices"][0]["message"]["content"])
                    return LLMResponse(request_id=request.request_id, text=text, raw=raw)
                except Exception as exc:  # noqa: BLE001 - preserve relay errors for caller context
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        break
        raise RuntimeError(f"LLM request failed after retries: {request.request_id}") from last_error
