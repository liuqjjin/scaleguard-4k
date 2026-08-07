"""Bounded OpenAI-compatible transport for the audited 4KAgent scheduler.

The canonical runtime uses Alibaba Cloud Model Studio (DashScope).  This
module deliberately carries no image transport: the remote service receives
only the degradation labels and scheduling prompt produced by the local
perception stack.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import requests


class SchedulerError(RuntimeError):
    """Base class for scheduler transport and response failures."""


class SchedulerProtocolError(SchedulerError):
    """Raised when a successful HTTP response violates the chat contract."""


@dataclass(frozen=True, slots=True)
class SchedulerReply:
    content: str
    response_id: str | None
    response_model: str | None
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchedulerProtocolError(f"scheduler response field {field} must be a non-negative int")
    return value


def _official_endpoint(provider: str, base_url: str, region: str) -> tuple[str, str]:
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError as error:
        raise SchedulerError(f"invalid scheduler base URL: {error}") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SchedulerError("scheduler base URL must be credential-free HTTPS without a port")

    normalized = base_url.rstrip("/")
    host = parsed.hostname.lower()
    if provider == "dashscope":
        official = {
            ("dashscope.aliyuncs.com", "cn-beijing"),
            ("dashscope-intl.aliyuncs.com", "ap-southeast-1"),
            ("dashscope-us.aliyuncs.com", "us-east-1"),
        }
        if (host, region) not in official or parsed.path.rstrip("/") != "/compatible-mode/v1":
            raise SchedulerError(
                "scheduler must use the official DashScope endpoint for its region"
            )
    elif provider == "openai":
        if host != "api.openai.com" or region != "global" or parsed.path.rstrip("/") != "/v1":
            raise SchedulerError("OpenAI scheduler must use the official global API endpoint")
    else:
        raise SchedulerError("unsupported scheduler provider")
    return f"{normalized}/chat/completions", host


class SchedulerClient:
    """Small, auditable chat-completions client with bounded retries."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        region: str,
        model: str,
        api_key: str,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        max_transport_retries: int,
        max_completion_tokens: int,
        temperature: float,
        requester: Callable[..., Any] = requests.post,
        sleeper: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        if not api_key:
            raise SchedulerError("scheduler credential is empty")
        if provider == "dashscope" and not model.startswith("qwen"):
            raise SchedulerError("DashScope scheduler requires a Qwen model")
        if provider == "openai" and not model.startswith("gpt-"):
            raise SchedulerError("OpenAI scheduler requires a GPT model")
        if not 0.1 <= connect_timeout_seconds <= 30.0:
            raise SchedulerError("scheduler connect timeout is outside the audited range")
        if not 1.0 <= read_timeout_seconds <= 600.0:
            raise SchedulerError("scheduler read timeout is outside the audited range")
        if not 0 <= max_transport_retries <= 8:
            raise SchedulerError("scheduler transport retry budget is outside the audited range")
        if not 64 <= max_completion_tokens <= 4096:
            raise SchedulerError("scheduler completion budget is outside the audited range")
        if temperature != 0.0:
            raise SchedulerError("scheduler temperature must be zero")
        self.provider = provider
        self.region = region
        self.model = model
        self.endpoint, endpoint_host = _official_endpoint(provider, base_url, region)
        self.endpoint_host_sha256 = hashlib.sha256(endpoint_host.encode("utf-8")).hexdigest()
        self._api_key = api_key
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.max_transport_retries = max_transport_retries
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._requester = requester
        self._sleeper = sleeper
        self._random_value = random_value
        self._attempts: list[dict[str, object]] = []

    def _payload(self, messages: Sequence[Mapping[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        if self.provider == "dashscope":
            payload["enable_thinking"] = False
        return payload

    @staticmethod
    def _retry_after(response: Any) -> float | None:
        value = getattr(response, "headers", {}).get("Retry-After")
        if value is None:
            return None
        try:
            delay = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(delay, 30.0))

    @staticmethod
    def _request_id(response: Any, document: Mapping[str, object] | None = None) -> str | None:
        headers = getattr(response, "headers", {})
        value = headers.get("x-request-id") or headers.get("request-id")
        if value is None and document is not None:
            value = document.get("id")
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _parse_reply(response: Any) -> SchedulerReply:
        try:
            document = response.json()
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise SchedulerProtocolError("scheduler returned invalid JSON") from error
        if not isinstance(document, dict):
            raise SchedulerProtocolError("scheduler response must be a JSON object")
        choices = document.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise SchedulerProtocolError("scheduler response must contain exactly one choice")
        choice = choices[0]
        if choice.get("index") != 0:
            raise SchedulerProtocolError("scheduler choice must have index zero")
        finish_reason = choice.get("finish_reason")
        if finish_reason != "stop":
            raise SchedulerProtocolError("scheduler response did not finish with reason=stop")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise SchedulerProtocolError("scheduler choice is missing its message object")
        if message.get("role") != "assistant":
            raise SchedulerProtocolError("scheduler message must have role=assistant")
        if message.get("tool_calls") is not None or message.get("function_call") is not None:
            raise SchedulerProtocolError("scheduler must not return tool or function calls")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise SchedulerProtocolError("scheduler returned empty or non-text content")
        usage = document.get("usage")
        if not isinstance(usage, dict):
            raise SchedulerProtocolError("scheduler response must contain a usage object")
        if not {"prompt_tokens", "completion_tokens", "total_tokens"}.issubset(usage):
            raise SchedulerProtocolError("scheduler usage is missing required token counts")
        prompt_tokens = _non_negative_int(usage.get("prompt_tokens"), field="usage.prompt_tokens")
        completion_tokens = _non_negative_int(
            usage.get("completion_tokens"), field="usage.completion_tokens"
        )
        total_tokens = _non_negative_int(
            usage.get("total_tokens"),
            field="usage.total_tokens",
        )
        if total_tokens != prompt_tokens + completion_tokens:
            raise SchedulerProtocolError("scheduler total token count is inconsistent")
        response_model = document.get("model")
        if not isinstance(response_model, str) or not response_model:
            raise SchedulerProtocolError("scheduler response must identify its model")
        response_id = document.get("id")
        if not isinstance(response_id, str) or not response_id:
            raise SchedulerProtocolError("scheduler response must contain a request id")
        return SchedulerReply(
            content=content,
            response_id=response_id,
            response_model=response_model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    def complete(self, messages: Sequence[Mapping[str, object]]) -> SchedulerReply:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = self._payload(messages)
        for retry_index in range(self.max_transport_retries + 1):
            try:
                response = self._requester(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(self.connect_timeout_seconds, self.read_timeout_seconds),
                )
            except requests.RequestException as error:
                self._attempts.append(
                    {"outcome": "transport_error", "error_type": type(error).__name__}
                )
                if retry_index == self.max_transport_retries:
                    raise SchedulerError("scheduler transport retry budget exhausted") from error
                delay = min(0.5 * (2**retry_index), 8.0) + 0.25 * self._random_value()
                self._sleeper(delay)
                continue

            status_code = getattr(response, "status_code", None)
            request_id = self._request_id(response)
            if not isinstance(status_code, int):
                raise SchedulerProtocolError("scheduler HTTP response has no integer status")
            if status_code == 429 or 500 <= status_code <= 599:
                self._attempts.append(
                    {
                        "outcome": "retryable_http_error",
                        "status_code": status_code,
                        "request_id": request_id,
                    }
                )
                if retry_index == self.max_transport_retries:
                    raise SchedulerError("scheduler HTTP retry budget exhausted")
                delay = self._retry_after(response)
                if delay is None:
                    delay = min(0.5 * (2**retry_index), 8.0) + 0.25 * self._random_value()
                self._sleeper(delay)
                continue
            if status_code < 200 or status_code >= 300:
                self._attempts.append(
                    {
                        "outcome": "terminal_http_error",
                        "status_code": status_code,
                        "request_id": request_id,
                    }
                )
                raise SchedulerError(f"scheduler rejected the request with HTTP {status_code}")

            try:
                reply = self._parse_reply(response)
                if reply.response_model is not None and reply.response_model != self.model:
                    raise SchedulerProtocolError(
                        "scheduler response model does not match the requested snapshot"
                    )
            except SchedulerProtocolError:
                self._attempts.append(
                    {
                        "outcome": "protocol_error",
                        "status_code": status_code,
                        "request_id": request_id,
                    }
                )
                raise
            self._attempts.append(
                {
                    "outcome": "completed",
                    "status_code": status_code,
                    "request_id": request_id or reply.response_id,
                    "response_model": reply.response_model,
                    "finish_reason": reply.finish_reason,
                    "prompt_tokens": reply.prompt_tokens,
                    "completion_tokens": reply.completion_tokens,
                    "total_tokens": reply.total_tokens,
                }
            )
            return reply
        raise AssertionError("unreachable scheduler retry state")

    def evidence(self) -> dict[str, object]:
        return {
            "provider": self.provider,
            "api_style": "openai-compatible-chat-completions",
            "region": self.region,
            "endpoint_host_sha256": self.endpoint_host_sha256,
            "requested_model": self.model,
            "request_parameters": {
                "max_completion_tokens": self.max_completion_tokens,
                "temperature": self.temperature,
                "response_format": "json_object",
                "enable_thinking": False if self.provider == "dashscope" else None,
                "connect_timeout_seconds": self.connect_timeout_seconds,
                "read_timeout_seconds": self.read_timeout_seconds,
                "max_transport_retries": self.max_transport_retries,
            },
            "attempts": list(self._attempts),
        }
