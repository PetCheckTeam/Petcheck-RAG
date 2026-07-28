from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_CHAT_BASE_URL = "https://clovastudio.stream.ntruss.com"
DEFAULT_CHAT_MODEL = "HCX-DASH-002"
DEFAULT_CHAT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 1.0


class ClovaChatClientError(Exception):
    """외부에 원문 응답이나 인증 정보를 노출하지 않는 기본 예외입니다."""


class ClovaChatConfigurationError(ClovaChatClientError):
    pass


class ClovaChatConnectionError(ClovaChatClientError):
    pass


class ClovaChatTimeoutError(ClovaChatClientError):
    pass


class ClovaChatUpstreamError(ClovaChatClientError):
    pass


class ClovaChatResponseError(ClovaChatClientError):
    pass


@dataclass(frozen=True)
class ClovaChatResult:
    answer: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ClovaChatClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        post: Callable[..., requests.Response] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("CLOVA_STUDIO_API_KEY")
        self.base_url = (
            base_url
            or os.getenv("CLOVA_CHAT_BASE_URL")
            or DEFAULT_CHAT_BASE_URL
        ).rstrip("/")
        self.model = model or os.getenv("CLOVA_CHAT_MODEL") or DEFAULT_CHAT_MODEL
        self.timeout_seconds = self._resolve_timeout(timeout_seconds)
        self.max_retries = max_retries
        self._post = post or requests.post
        self._sleep = sleep or time.sleep

        if not self.api_key:
            raise ClovaChatConfigurationError(
                "HyperCLOVA X API 설정이 올바르지 않습니다."
            )
        if self.max_retries < 0:
            raise ClovaChatConfigurationError(
                "HyperCLOVA X 재시도 설정이 올바르지 않습니다."
            )

    @staticmethod
    def _resolve_timeout(timeout_seconds: float | None) -> float:
        raw_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv(
                "CLOVA_CHAT_TIMEOUT_SECONDS",
                str(DEFAULT_CHAT_TIMEOUT_SECONDS),
            )
        )
        try:
            resolved = float(raw_timeout)
        except (TypeError, ValueError) as error:
            raise ClovaChatConfigurationError(
                "HyperCLOVA X timeout 설정이 올바르지 않습니다."
            ) from error
        if resolved <= 0:
            raise ClovaChatConfigurationError(
                "HyperCLOVA X timeout 설정이 올바르지 않습니다."
            )
        return resolved

    def chat(self, messages: list[dict[str, str]]) -> ClovaChatResult:
        url = f"{self.base_url}/v3/chat-completions/{self.model}"
        payload = {
            "messages": messages,
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 700,
            "temperature": 0.2,
            "repetitionPenalty": 1.1,
            "stop": [],
        }

        for attempt in range(self.max_retries + 1):
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid4()),
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

            try:
                response = self._post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except requests.Timeout as error:
                raise ClovaChatTimeoutError(
                    "HyperCLOVA X 응답 시간이 초과되었습니다."
                ) from error
            except requests.RequestException as error:
                raise ClovaChatConnectionError(
                    "HyperCLOVA X 서비스에 연결할 수 없습니다."
                ) from error

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < self.max_retries:
                    self._sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
                    continue
                raise ClovaChatUpstreamError(
                    "HyperCLOVA X 서비스가 요청을 처리하지 못했습니다."
                )

            if not 200 <= response.status_code < 300:
                raise ClovaChatUpstreamError(
                    "HyperCLOVA X 요청이 거부되었습니다."
                )

            return self._parse_response(response)

        raise ClovaChatUpstreamError(
            "HyperCLOVA X 서비스가 요청을 처리하지 못했습니다."
        )

    def _parse_response(self, response: requests.Response) -> ClovaChatResult:
        try:
            data = response.json()
        except ValueError as error:
            raise ClovaChatResponseError(
                "HyperCLOVA X 응답 형식이 올바르지 않습니다."
            ) from error

        result = data.get("result") if isinstance(data, dict) else None
        message = result.get("message") if isinstance(result, dict) else None
        usage = result.get("usage") if isinstance(result, dict) else None
        finish_reason = (
            result.get("finishReason") if isinstance(result, dict) else None
        )
        content = message.get("content") if isinstance(message, dict) else None

        if (
            not isinstance(content, str)
            or not content.strip()
            or not isinstance(finish_reason, str)
            or not finish_reason.strip()
            or not isinstance(usage, dict)
        ):
            raise ClovaChatResponseError(
                "HyperCLOVA X 응답 형식이 올바르지 않습니다."
            )

        try:
            prompt_tokens = int(usage.get("promptTokens", 0))
            completion_tokens = int(usage.get("completionTokens", 0))
            total_tokens = int(usage.get("totalTokens", 0))
        except (TypeError, ValueError) as error:
            raise ClovaChatResponseError(
                "HyperCLOVA X 사용량 응답 형식이 올바르지 않습니다."
            ) from error

        if min(prompt_tokens, completion_tokens, total_tokens) < 0:
            raise ClovaChatResponseError(
                "HyperCLOVA X 사용량 응답 형식이 올바르지 않습니다."
            )

        return ClovaChatResult(
            answer=content.strip(),
            model=self.model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
