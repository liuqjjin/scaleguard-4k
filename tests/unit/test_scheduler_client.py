from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import requests

ROOT = Path(__file__).parents[2]


def _load_scheduler() -> ModuleType:
    path = ROOT / "third_party/overlays/4kagent/scheduler_client.py"
    specification = importlib.util.spec_from_file_location("scaleguard_test_scheduler", path)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


SCHEDULER = _load_scheduler()


class Response:
    def __init__(
        self,
        status_code: int,
        document: object,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.document = document
        self.headers = headers or {}

    def json(self) -> object:
        if isinstance(self.document, BaseException):
            raise self.document
        return self.document


def reply_document() -> dict[str, object]:
    return {
        "id": "request-from-body",
        "model": "qwen3.7-flash-2026-07-15",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"thought":"quality first","order":["denoise"]}',
                },
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }


def client(**overrides: Any) -> Any:
    arguments: dict[str, object] = {
        "provider": "dashscope",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "region": "cn-beijing",
        "model": "qwen3.7-flash-2026-07-15",
        "api_key": "private-dashscope-key",
        "connect_timeout_seconds": 10.0,
        "read_timeout_seconds": 120.0,
        "max_transport_retries": 2,
        "max_completion_tokens": 1024,
        "temperature": 0.0,
    }
    arguments.update(overrides)
    return SCHEDULER.SchedulerClient(**arguments)


def test_dashscope_request_is_bounded_structured_and_secret_free_in_evidence() -> None:
    observed: dict[str, object] = {}

    def request(url: str, **kwargs: object) -> Response:
        observed.update({"url": url, **kwargs})
        return Response(200, reply_document(), headers={"x-request-id": "header-request"})

    scheduler = client(requester=request)
    messages = [
        {"role": "system", "content": "Return JSON."},
        {"role": "user", "content": "order the tasks"},
    ]

    reply = scheduler.complete(messages)

    assert reply.content.startswith("{")
    assert reply.prompt_tokens == 20
    assert observed["url"] == ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    assert observed["timeout"] == (10.0, 120.0)
    payload = observed["json"]
    assert isinstance(payload, dict)
    assert payload["enable_thinking"] is False
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["max_completion_tokens"] == 1024
    evidence = scheduler.evidence()
    evidence_text = repr(evidence)
    assert evidence["provider"] == "dashscope"
    assert "header-request" in evidence_text
    assert "private-dashscope-key" not in evidence_text
    assert "order the tasks" not in evidence_text


def test_scheduler_retries_only_retryable_transport_failures_with_a_budget() -> None:
    responses = [
        Response(429, {}, headers={"Retry-After": "0.25"}),
        Response(503, {}),
        Response(200, reply_document()),
    ]
    delays: list[float] = []

    def request(_url: str, **_kwargs: object) -> Response:
        return responses.pop(0)

    scheduler = client(requester=request, sleeper=delays.append, random_value=lambda: 0.0)

    result = scheduler.complete([{"role": "user", "content": "JSON"}])

    assert result.finish_reason == "stop"
    assert delays == [0.25, 1.0]
    assert [attempt["outcome"] for attempt in scheduler.evidence()["attempts"]] == [
        "retryable_http_error",
        "retryable_http_error",
        "completed",
    ]


def test_scheduler_stops_after_the_transport_retry_budget() -> None:
    calls = 0

    def request(_url: str, **_kwargs: object) -> Response:
        nonlocal calls
        calls += 1
        raise requests.Timeout("no response")

    scheduler = client(
        requester=request,
        sleeper=lambda _delay: None,
        random_value=lambda: 0.0,
        max_transport_retries=1,
    )

    with pytest.raises(SCHEDULER.SchedulerError, match="retry budget exhausted"):
        scheduler.complete([{"role": "user", "content": "JSON"}])
    assert calls == 2


@pytest.mark.parametrize(
    "document",
    [
        ValueError("not JSON"),
        {"choices": []},
        {"choices": [{"finish_reason": "length", "message": {"content": '{"order": []}'}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]},
        {
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            "usage": {"prompt_tokens": -1},
        },
        {
            **reply_document(),
            "model": "qwen3.7-flash-moving-alias",
        },
    ],
)
def test_scheduler_rejects_malformed_success_responses(document: object) -> None:
    scheduler = client(requester=lambda _url, **_kwargs: Response(200, document))

    with pytest.raises(SCHEDULER.SchedulerProtocolError):
        scheduler.complete([{"role": "user", "content": "JSON"}])


@pytest.mark.parametrize(
    ("base_url", "region"),
    [
        ("http://dashscope.aliyuncs.com/compatible-mode/v1", "cn-beijing"),
        ("https://example.com/compatible-mode/v1", "cn-beijing"),
        ("https://dashscope.aliyuncs.com/compatible-mode/v1", "ap-southeast-1"),
        ("https://user:secret@dashscope.aliyuncs.com/compatible-mode/v1", "cn-beijing"),
    ],
)
def test_scheduler_rejects_unofficial_or_mismatched_endpoints(
    base_url: str,
    region: str,
) -> None:
    with pytest.raises(SCHEDULER.SchedulerError):
        client(base_url=base_url, region=region)


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "gpt-4-turbo"},
        {"connect_timeout_seconds": 0.0},
        {"read_timeout_seconds": 601.0},
        {"max_transport_retries": 9},
        {"max_completion_tokens": 32},
        {"temperature": 0.1},
    ],
)
def test_scheduler_constructor_revalidates_direct_overlay_arguments(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(SCHEDULER.SchedulerError):
        client(**overrides)
