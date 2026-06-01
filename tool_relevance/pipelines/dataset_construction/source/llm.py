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
    wire_api: str = "chat_completions"


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config

    async def complete(self, request: LLMRequest) -> LLMResponse:
        if self.config.wire_api == "responses":
            return await self._complete_responses(request)
        if self.config.wire_api != "chat_completions":
            raise ValueError(f"unknown OpenAI-compatible wire_api: {self.config.wire_api}")
        return await self._complete_chat_completions(request)

    async def _complete_chat_completions(self, request: LLMRequest) -> LLMResponse:
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

    async def _complete_responses(self, request: LLMRequest) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": request.messages,
            "temperature": request.temperature,
        }
        if request.response_format is not None:
            payload["text"] = {"format": _responses_text_format(request.response_format)}

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Idempotency-Key": request.idempotency_key,
        }
        url = self.config.base_url.rstrip("/") + "/responses"
        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            for attempt in range(self.config.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    raw = response.json()
                    return LLMResponse(
                        request_id=request.request_id,
                        text=_responses_text(raw),
                        raw=raw,
                    )
                except Exception as exc:  # noqa: BLE001 - preserve relay errors for caller context
                    last_error = exc
                    if attempt >= self.config.max_retries:
                        break
        raise RuntimeError(f"LLM request failed after retries: {request.request_id}") from last_error


def _responses_text(raw: dict[str, Any]) -> str:
    output_text = raw.get("output_text")
    if isinstance(output_text, str):
        return output_text
    chunks: list[str] = []
    for item in raw.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    if chunks:
        return "".join(chunks)
    raise RuntimeError("Responses API response did not contain output text")


def _responses_text_format(response_format: dict[str, Any]) -> dict[str, Any]:
    if response_format.get("type") == "json_object":
        return {"type": "json_object"}
    if response_format.get("type") == "json_schema":
        return response_format
    return response_format
