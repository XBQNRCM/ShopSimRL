"""Model boundary with an OpenAI-compatible HTTP implementation."""

from __future__ import annotations

import json
import hashlib
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import requests

from .schemas import ModelOutput, ToolCall


Message = dict[str, Any]


class ChatModel(Protocol):
    def identity(self) -> dict[str, Any]: ...

    def generate(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ModelOutput: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    model: str
    base_url: str
    api_key_env: str | None = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float | None = None
    timeout: float = 180.0
    max_retries: int = 5
    retry_backoff_seconds: float = 1.0
    trust_env: bool = True
    stream: bool = False
    send_seed: bool = True
    tool_choice: str = "auto"
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.base_url.strip():
            raise ValueError("model and base_url must be non-empty")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be positive")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")
        if not isinstance(self.stream, bool):
            raise ValueError("stream must be true or false")
        if not self.tool_choice.strip():
            raise ValueError("tool_choice must be non-empty")

    def identity(self) -> dict[str, Any]:
        # Streaming changes only HTTP transport, not the requested model behavior.
        # Keep it out of fingerprints so resumable artifacts remain reusable when
        # switching between equivalent streaming and non-streaming delivery.
        return {
            "provider": "openai_compatible_http",
            "model": self.model,
            "base_url": self.base_url.rstrip("/"),
            "transport": {
                "timeout_seconds": self.timeout,
                "max_retries": self.max_retries,
                "retry_backoff_seconds": self.retry_backoff_seconds,
            },
            "sampling": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "send_seed": self.send_seed,
                "tool_choice": self.tool_choice,
                "extra_body": self.extra_body,
            },
        }


class ModelRequestError(RuntimeError):
    pass


class _StreamingResponseError(RuntimeError):
    pass


class OpenAICompatibleChatModel:
    """Minimal client that works with OpenAI, vLLM, SGLang and compatible APIs."""

    RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

    def __init__(self, config: OpenAICompatibleConfig):
        self.config = config
        self.session = requests.Session()
        self.session.trust_env = config.trust_env

    def identity(self) -> dict[str, Any]:
        return self.config.identity()

    def _api_key(self) -> str | None:
        if not self.config.api_key_env:
            return None
        value = os.environ.get(self.config.api_key_env)
        if not value:
            raise ModelRequestError(
                f"model API key environment variable is not set: "
                f"{self.config.api_key_env}"
            )
        return value

    def generate(
        self,
        messages: Sequence[Message],
        *,
        seed: int | None = None,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "stream": self.config.stream,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **self.config.extra_body,
        }
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if seed is not None and self.config.send_seed:
            payload["seed"] = seed
        allowed_tools: set[str] | None = None
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = self.config.tool_choice
            allowed_tools = {
                str(tool.get("function", {}).get("name")) for tool in tools
            }

        headers = {"Content-Type": "application/json"}
        api_key = self._api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        started = time.perf_counter()
        last_error = "unknown model error"
        for attempt in range(self.config.max_retries):
            response = None
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                    stream=self.config.stream,
                )
                if response.status_code in self.RETRYABLE_STATUS:
                    last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                elif not response.ok:
                    raise ModelRequestError(
                        f"non-retryable model HTTP {response.status_code}: "
                        f"{response.text[:500]}"
                    )
                else:
                    if self.config.stream:
                        body = self._read_streaming_response(response)
                    else:
                        try:
                            body = response.json()
                        except ValueError as exc:
                            return ModelOutput(
                                protocol_error={
                                    "code": "invalid_json_response",
                                    "message": f"model response is not valid JSON: {exc}",
                                },
                                raw_response={"text": response.text[:4000]},
                                latency_ms=(time.perf_counter() - started) * 1000,
                            )
                    output = self._parse_response(
                        body,
                        require_tool_call=bool(tools),
                        allowed_tools=allowed_tools,
                    )
                    return ModelOutput(
                        **output,
                        latency_ms=(time.perf_counter() - started) * 1000,
                    )
            except (requests.RequestException, _StreamingResponseError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            finally:
                if response is not None and self.config.stream:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()

            if attempt + 1 < self.config.max_retries:
                base = self.config.retry_backoff_seconds * (2**attempt)
                time.sleep(min(base, 30.0) + random.random() * min(base, 1.0))

        raise ModelRequestError(
            f"model request failed after {self.config.max_retries} attempts: "
            f"{last_error}"
        )

    @staticmethod
    def _read_streaming_response(response: Any) -> dict[str, Any]:
        """Assemble an OpenAI-compatible SSE stream into one chat response."""
        response_id: str | None = None
        usage: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str | None = None
        tool_calls: dict[int, dict[str, Any]] = {}
        saw_data = False
        saw_done = False

        for raw_line in response.iter_lines(decode_unicode=True):
            if isinstance(raw_line, bytes):
                line = raw_line.decode("utf-8")
            else:
                line = str(raw_line or "")
            line = line.strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as exc:
                raise _StreamingResponseError(
                    f"stream returned invalid JSON data: {exc}"
                ) from exc
            if not isinstance(chunk, dict):
                raise _StreamingResponseError("stream data must be a JSON object")
            if isinstance(chunk.get("error"), dict):
                message = str(chunk["error"].get("message") or chunk["error"])
                raise _StreamingResponseError(f"stream returned an error: {message[:500]}")
            saw_data = True
            if response_id is None and isinstance(chunk.get("id"), str):
                response_id = chunk["id"]
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for key in ("created", "system_fingerprint", "service_tier"):
                if key in chunk:
                    metadata[key] = chunk[key]

            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                raise _StreamingResponseError("stream choice must be an object")
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            reasoning = delta.get("reasoning_content", delta.get("reasoning"))
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)

            raw_calls = delta.get("tool_calls")
            if raw_calls is None:
                continue
            if not isinstance(raw_calls, list):
                raise _StreamingResponseError("stream tool_calls must be a list")
            for fallback_index, raw_call in enumerate(raw_calls):
                if not isinstance(raw_call, dict):
                    raise _StreamingResponseError("stream tool call must be an object")
                raw_index = raw_call.get("index", fallback_index)
                if isinstance(raw_index, bool) or not isinstance(raw_index, int):
                    raise _StreamingResponseError("stream tool call index must be an integer")
                assembled = tool_calls.setdefault(
                    raw_index,
                    {
                        "id": None,
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if isinstance(raw_call.get("id"), str):
                    assembled["id"] = raw_call["id"]
                if isinstance(raw_call.get("type"), str):
                    assembled["type"] = raw_call["type"]
                function = raw_call.get("function")
                if function is None:
                    continue
                if not isinstance(function, dict):
                    raise _StreamingResponseError(
                        "stream tool call function must be an object"
                    )
                if isinstance(function.get("name"), str):
                    assembled["function"]["name"] += function["name"]
                if isinstance(function.get("arguments"), str):
                    assembled["function"]["arguments"] += function["arguments"]

        if not saw_data:
            raise _StreamingResponseError("stream ended without any data")
        if not saw_done:
            raise _StreamingResponseError("stream ended before the [DONE] marker")

        for index, assembled in tool_calls.items():
            call_id = assembled.get("id")
            if isinstance(call_id, str) and call_id.strip():
                continue
            function = assembled["function"]
            identity_source = json.dumps(
                {
                    "response_id": response_id,
                    "index": index,
                    "name": function.get("name"),
                    "arguments": function.get("arguments"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            digest = hashlib.sha256(identity_source.encode("utf-8")).hexdigest()[:24]
            assembled["id"] = f"call-stream-{digest}"

        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(content_parts) or None,
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        body: dict[str, Any] = {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
            **metadata,
        }
        if response_id is not None:
            body["id"] = response_id
        return body

    @staticmethod
    def _text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            pieces = []
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    pieces.append(item["text"])
            return "".join(pieces)
        return str(value)

    @classmethod
    def _parse_response(
        cls,
        body: Any,
        *,
        require_tool_call: bool = False,
        allowed_tools: set[str] | None = None,
    ) -> dict[str, Any]:
        response_id = body.get("id") if isinstance(body, dict) else None
        usage = (body.get("usage") or {}) if isinstance(body, dict) else {}
        metadata = (
            {
                key: body[key]
                for key in ("created", "system_fingerprint", "service_tier")
                if key in body
            }
            if isinstance(body, dict)
            else {}
        )
        content: str | None = None
        reasoning: str | None = None
        finish_reason: str | None = None
        tool_calls: list[ToolCall] = []

        def result(
            *,
            protocol_code: str | None = None,
            policy_code: str | None = None,
            message: str | None = None,
        ) -> dict[str, Any]:
            rejected = protocol_code is not None or policy_code is not None
            return {
                "content": content,
                "reasoning": reasoning,
                "tool_calls": tuple(tool_calls),
                "finish_reason": finish_reason,
                "usage": usage if isinstance(usage, dict) else {},
                "response_id": response_id if isinstance(response_id, str) else None,
                "metadata": metadata,
                "protocol_error": (
                    {"code": protocol_code, "message": message}
                    if protocol_code is not None
                    else None
                ),
                "policy_failure": (
                    {"code": policy_code, "message": message}
                    if policy_code is not None
                    else None
                ),
                "raw_response": body if rejected else None,
            }

        if not isinstance(body, dict):
            return result(
                protocol_code="invalid_response_body",
                message="model response body must be a JSON object",
            )
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return result(
                protocol_code="missing_choices",
                message="model response has no choices",
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            return result(
                protocol_code="invalid_choice",
                message="model response choice must be an object",
            )
        finish_reason = choice.get("finish_reason")
        message = choice.get("message")
        if not isinstance(message, dict):
            return result(
                protocol_code="invalid_message",
                message="model response choice has no message object",
            )
        content = cls._text(message.get("content"))
        reasoning = cls._text(
            message.get("reasoning_content", message.get("reasoning"))
        )
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls is None:
            raw_tool_calls = []
        if not isinstance(raw_tool_calls, list):
            return result(
                protocol_code="invalid_tool_calls",
                message="model tool_calls must be a list",
            )
        for raw_call in raw_tool_calls:
            if not isinstance(raw_call, dict) or raw_call.get("type") != "function":
                return result(
                    protocol_code="invalid_tool_call",
                    message="model returned an invalid function tool call",
                )
            function = raw_call.get("function")
            if not isinstance(function, dict):
                return result(
                    protocol_code="missing_function",
                    message="tool call has no function object",
                )
            call_id = raw_call.get("id")
            name = function.get("name")
            raw_arguments = function.get("arguments")
            if not isinstance(call_id, str) or not call_id:
                return result(
                    protocol_code="missing_tool_call_id",
                    message="tool call has no id",
                )
            if not isinstance(name, str) or not name:
                return result(
                    protocol_code="missing_tool_name",
                    message="tool call has no function name",
                )
            if not isinstance(raw_arguments, str):
                return result(
                    protocol_code="invalid_arguments_type",
                    message="tool call arguments must be a JSON string",
                )
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                return result(
                    protocol_code="invalid_arguments_json",
                    message=f"tool call arguments are not valid JSON: {exc}",
                )
            if not isinstance(arguments, dict):
                return result(
                    protocol_code="invalid_arguments_object",
                    message="tool call arguments must decode to an object",
                )
            tool_calls.append(
                ToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )
            if allowed_tools is not None and name not in allowed_tools:
                return result(
                    protocol_code="unavailable_tool",
                    message=f"model called unavailable tool: {name}",
                )
        if require_tool_call and finish_reason == "length" and len(tool_calls) != 1:
            return result(
                policy_code="generation_length",
                message=(
                    "model generation reached max_tokens before returning exactly "
                    "one valid tool call"
                ),
            )
        if require_tool_call and len(tool_calls) != 1:
            return result(
                protocol_code="tool_call_count",
                message=f"model must return exactly one tool call, got {len(tool_calls)}",
            )
        if require_tool_call and content and content.strip():
            return result(
                protocol_code="mixed_content",
                message=(
                    "model must not mix natural-language content with an action "
                    "tool call"
                ),
            )
        if not tool_calls and not (content and content.strip()):
            return result(
                protocol_code="empty_response",
                message=(
                    "model returned neither content nor tool calls "
                    f"(finish_reason={finish_reason!r}, "
                    f"reasoning_chars={len(reasoning or '')})"
                ),
            )
        return result()

    def close(self) -> None:
        self.session.close()
