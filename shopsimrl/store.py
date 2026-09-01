"""Atomic metadata plus append-only, resumable JSONL trajectory storage."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable

from .schemas import (
    MANIFEST_SCHEMA_VERSION,
    TRACE_SCHEMA_VERSION,
    EpisodeJob,
    fingerprint,
    utc_now,
)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not cleaned:
        raise ValueError(f"invalid empty artifact name from {value!r}")
    return cleaned


class RunStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.traces_path = self.root / "traces.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.summary_path = self.root / "summary.json"
        self._lock = threading.Lock()
        self._latest: dict[str, dict[str, Any]] | None = None

    def initialize(self, semantic_plan: dict[str, Any]) -> dict[str, Any]:
        plan_fingerprint = fingerprint(semantic_plan)
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if manifest.get("plan_fingerprint") != plan_fingerprint:
                raise ValueError(
                    f"existing run manifest at {self.manifest_path} belongs to a "
                    "different semantic plan; use another experiment/model id"
                )
            return manifest
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": utc_now(),
            "plan_fingerprint": plan_fingerprint,
            "plan": semantic_plan,
        }
        atomic_write_json(self.manifest_path, manifest)
        return manifest

    def save_trace(self, trace: dict[str, Any]) -> Path:
        episode_id = trace.get("episode_id")
        if not isinstance(episode_id, str):
            raise ValueError("trace has no episode_id")
        line = (json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.traces_path.open("a+b") as file:
                file.seek(0, os.SEEK_END)
                if file.tell() > 0:
                    file.seek(-1, os.SEEK_END)
                    if file.read(1) != b"\n":
                        file.write(b"\n")
                file.write(line)
                file.flush()
                os.fsync(file.fileno())
            if self._latest is None:
                self._latest = self._load_latest()
            self._latest[episode_id] = trace
        return self.traces_path

    def _load_latest(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        if not self.traces_path.exists():
            return latest
        with self.traces_path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                episode_id = payload.get("episode_id") if isinstance(payload, dict) else None
                if isinstance(episode_id, str):
                    latest[episode_id] = payload
        return latest

    def _records(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            if self._latest is None:
                self._latest = self._load_latest()
            return dict(self._latest)

    def load_trace(self, job: EpisodeJob) -> dict[str, Any] | None:
        with self._lock:
            if self._latest is None:
                self._latest = self._load_latest()
            return self._latest.get(job.episode_id)

    def is_complete(self, job: EpisodeJob) -> bool:
        payload = self.load_trace(job)
        return bool(
            payload
            and payload.get("schema_version") == TRACE_SCHEMA_VERSION
            and payload.get("status") == "completed"
            and isinstance(payload.get("final"), dict)
            and payload["final"].get("done") is True
        )

    def iter_traces(self) -> Iterable[dict[str, Any]]:
        records = self._records()
        for episode_id in sorted(records):
            yield records[episode_id]

    def write_summary(self, summary: dict[str, Any]) -> None:
        atomic_write_json(self.summary_path, summary)
