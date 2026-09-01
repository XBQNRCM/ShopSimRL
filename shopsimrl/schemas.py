"""Small, dependency-free value objects shared by the pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


TRACE_SCHEMA_VERSION = "shopsimrl-episode-v4"
MANIFEST_SCHEMA_VERSION = "shopsimrl-run-manifest-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EpisodeJob:
    task_id: int
    sample_id: int
    seed: int
    split: str

    @property
    def episode_id(self) -> str:
        return f"task-{self.task_id:06d}__sample-{self.sample_id:03d}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_message_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.raw_arguments,
            },
        }


@dataclass(frozen=True)
class ModelOutput:
    content: str | None = None
    reasoning: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    response_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    protocol_error: dict[str, Any] | None = None
    policy_failure: dict[str, Any] | None = None
    raw_response: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tool_calls"] = [call.to_dict() for call in self.tool_calls]
        return payload

    def assistant_message(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
        }
        if self.tool_calls:
            payload["tool_calls"] = [
                call.to_message_dict() for call in self.tool_calls
            ]
        return payload


@dataclass(frozen=True)
class Skill:
    skill_id: str
    content: str
    version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["content_sha256"] = fingerprint(self.content)
        return payload
