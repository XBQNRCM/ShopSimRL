"""Load persisted task-ID manifests."""

from __future__ import annotations

import json
import os
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
CLEANED_SPLITS_PATH = CONFIG_DIR / "task_splits.cleaned.v2.json"


def default_splits_path() -> Path:
    return CLEANED_SPLITS_PATH


def _task_splits_path(path: str | Path | None) -> Path:
    if path is not None:
        return Path(path)
    configured = os.environ.get("SHOP_TASK_SPLITS")
    return Path(configured) if configured else default_splits_path()


def load_task_splits(path: str | Path | None = None) -> dict:
    payload = json.loads(_task_splits_path(path).read_text(encoding="utf-8"))
    for name, split in payload["splits"].items():
        if "task_ids" in split:
            ids = split["task_ids"]
            if (
                not isinstance(ids, list)
                or any(isinstance(value, bool) or not isinstance(value, int) for value in ids)
                or any(value < 0 for value in ids)
                or len(ids) != len(set(ids))
            ):
                raise ValueError(f"invalid explicit task IDs in split {name!r}")
        else:
            corpus_size = int(payload["corpus_size"])
            start, end = int(split["start"]), int(split["end"])
            if start < 0 or end <= start or end > corpus_size:
                raise ValueError(f"invalid task split {name!r}: [{start}, {end})")
    return payload


def task_ids(name: str, path: str | Path | None = None) -> range | tuple[int, ...]:
    payload = load_task_splits(path)
    try:
        split = payload["splits"][name]
    except KeyError as exc:
        choices = ", ".join(sorted(payload["splits"]))
        raise KeyError(f"unknown task split {name!r}; choose one of: {choices}") from exc
    if "task_ids" in split:
        return tuple(split["task_ids"])
    return range(int(split["start"]), int(split["end"]))
