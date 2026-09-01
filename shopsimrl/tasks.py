"""Stable task split loading without hard-coded corpus boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSplit:
    name: str
    task_ids: tuple[int, ...]
    persona: bool | None
    source_path: Path
    split_version: str | None

    def identity(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_path": str(self.source_path),
            "split_version": self.split_version,
            "persona": self.persona,
            "task_count": len(self.task_ids),
        }


def load_task_split(path: str | Path, name: str) -> TaskSplit:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    try:
        record = payload["splits"][name]
    except KeyError as exc:
        choices = ", ".join(sorted(payload.get("splits", {})))
        raise KeyError(f"unknown task split {name!r}; choose one of: {choices}") from exc

    if "task_ids" in record:
        ids = record["task_ids"]
    else:
        ids = list(range(int(record["start"]), int(record["end"])))
    if (
        not isinstance(ids, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
        or any(value < 0 for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError(f"invalid task IDs in split {name!r}")
    persona = record.get("persona")
    if persona not in {True, False, None}:
        raise ValueError(f"split {name!r} has invalid persona flag")
    return TaskSplit(
        name=name,
        task_ids=tuple(ids),
        persona=persona,
        source_path=source,
        split_version=payload.get("version"),
    )
