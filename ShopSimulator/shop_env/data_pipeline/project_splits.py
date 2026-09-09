"""Deterministic project train/val/test splits for cleaned persona tasks."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re
import unicodedata
from typing import Iterable, Mapping


SPLIT_SCHEMA_VERSION = "shopsimrl-persona-splits-v1"
SPLIT_POLICY_VERSION = "exact-linked-components-v1"
NORMALIZATION_VERSION = "unicode-nfkc-casefold-whitespace-v1"
GROUP_KEYS = (
    "asin",
    "instruction_simple",
    "instruction",
    "persona_fingerprint",
    "user_id",
)


class ProjectSplitError(RuntimeError):
    """Raised when a reproducible project split cannot be constructed."""


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _public_persona(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _public_persona(child)
            for key, child in value.items()
            if str(key) != "__reasoning__"
        }
    if isinstance(value, list):
        return [_public_persona(child) for child in value]
    return value


def _persona_fingerprint(persona: Mapping[str, object]) -> str:
    encoded = json.dumps(
        _public_persona(dict(persona)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _user_id(persona: Mapping[str, object]) -> str:
    normalized_names = {"userid", "user_id", "用户id", "用户_id"}
    for key, value in persona.items():
        if _normalize(key).replace(" ", "") in normalized_names:
            return _normalize(value)
    return ""


def persona_records(cleaned_products: Iterable[dict]) -> list[dict]:
    """Extract one private grouping record for every cleaned persona task."""

    records = []
    seen_ids = set()
    for product in cleaned_products:
        persona = product.get("user_persona")
        if not persona:
            continue
        if not isinstance(persona, dict):
            raise ProjectSplitError("user_persona must be an object")
        asin = _normalize(product.get("asin"))
        persona_fingerprint = _persona_fingerprint(persona)
        user_id = _user_id(persona)
        for instruction in product.get("instructions") or []:
            if not isinstance(instruction, dict):
                raise ProjectSplitError("instruction must be an object")
            simple = _normalize(instruction.get("instruction_simple"))
            if not simple:
                raise ProjectSplitError(
                    "persona product contains an instruction without instruction_simple"
                )
            try:
                task_id = int(instruction["task_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ProjectSplitError("persona instruction lacks a valid task_id") from exc
            if task_id in seen_ids:
                raise ProjectSplitError(f"duplicate persona task_id: {task_id}")
            seen_ids.add(task_id)
            records.append(
                {
                    "task_id": task_id,
                    "task_uid": str(instruction.get("task_uid") or ""),
                    "group_values": {
                        "asin": asin,
                        "instruction_simple": simple,
                        "instruction": _normalize(instruction.get("instruction")),
                        "persona_fingerprint": persona_fingerprint,
                        "user_id": user_id,
                    },
                }
            )
    if not records:
        raise ProjectSplitError("cleaned corpus contains no persona tasks")
    return records


class _UnionFind:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def linked_groups(records: Iterable[dict]) -> list[dict]:
    """Connect tasks sharing any leakage-sensitive exact key."""

    records = list(records)
    union_find = _UnionFind(record["task_id"] for record in records)
    owners: dict[tuple[str, str], int] = {}
    for record in records:
        task_id = record["task_id"]
        for key in GROUP_KEYS:
            value = record["group_values"].get(key, "")
            if not value:
                continue
            identity = (key, value)
            owner = owners.setdefault(identity, task_id)
            union_find.union(task_id, owner)

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for record in records:
        members_by_root[union_find.find(record["task_id"])].append(record["task_id"])

    groups = []
    for members in members_by_root.values():
        members.sort()
        material = ",".join(str(task_id) for task_id in members).encode("ascii")
        groups.append(
            {
                "group_id": "group-" + hashlib.sha256(material).hexdigest()[:16],
                "task_ids": members,
            }
        )
    return sorted(groups, key=lambda group: group["group_id"])


def _rank(group: dict, *, seed: int, split: str) -> str:
    material = f"{seed}:{split}:{group['group_id']}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _select_exact_groups(
    groups: Iterable[dict],
    target_size: int,
    *,
    seed: int,
    split: str,
    blocked_task_ids: set[int] | None = None,
    max_group_size: int | None = None,
) -> set[str]:
    """Select a deterministic subset of whole groups with an exact task count."""

    if target_size < 0:
        raise ProjectSplitError(f"{split} size cannot be negative")
    blocked = blocked_task_ids or set()
    ranked = sorted(groups, key=lambda group: _rank(group, seed=seed, split=split))
    parents: dict[int, tuple[int, int] | None] = {0: None}
    for index, group in enumerate(ranked):
        members = set(group["task_ids"])
        if members & blocked:
            continue
        size = len(members)
        if max_group_size is not None and size > max_group_size:
            continue
        for total in sorted(tuple(parents), reverse=True):
            new_total = total + size
            if new_total <= target_size and new_total not in parents:
                parents[new_total] = (total, index)
        if target_size in parents:
            break
    if target_size not in parents:
        raise ProjectSplitError(
            f"cannot allocate exactly {target_size} tasks to {split} without "
            "breaking linked groups"
        )

    selected = set()
    total = target_size
    while total:
        parent = parents[total]
        if parent is None:
            raise AssertionError("invalid subset reconstruction")
        previous, index = parent
        selected.add(ranked[index]["group_id"])
        total = previous
    return selected


def _ids_sha256(task_ids: Iterable[int]) -> str:
    payload = json.dumps(
        list(task_ids), separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_project_split_manifest(
    cleaned_products: Iterable[dict],
    source_splits: Mapping[str, object],
    *,
    seed: int,
    validation_size: int,
    test_size: int,
    held_out_max_group_size: int,
    source: Mapping[str, object],
    test_excluded_task_ids: Iterable[int] = (),
) -> dict:
    """Build a frozen project split manifest without duplicating product data."""

    records = persona_records(cleaned_products)
    persona_ids = {record["task_id"] for record in records}
    excluded_from_test = {int(task_id) for task_id in test_excluded_task_ids}
    unknown_exclusions = excluded_from_test - persona_ids
    if unknown_exclusions:
        raise ProjectSplitError(
            f"test exclusions contain non-persona IDs: {sorted(unknown_exclusions)[:10]}"
        )
    if validation_size + test_size >= len(persona_ids):
        raise ProjectSplitError("validation and test must leave at least one train task")
    if held_out_max_group_size < 1:
        raise ProjectSplitError("held_out_max_group_size must be positive")

    source_membership: dict[int, str] = {}
    split_records = source_splits.get("splits", {})
    if not isinstance(split_records, Mapping):
        raise ProjectSplitError("source split manifest lacks splits")
    for name, split_record in split_records.items():
        if not isinstance(split_record, Mapping) or split_record.get("persona") is not True:
            continue
        for raw_task_id in split_record.get("task_ids", []):
            task_id = int(raw_task_id)
            if task_id in source_membership:
                raise ProjectSplitError(f"source persona split overlap at task {task_id}")
            source_membership[task_id] = str(name)
    if set(source_membership) != persona_ids:
        raise ProjectSplitError(
            "source persona split coverage does not match cleaned persona tasks"
        )

    groups = linked_groups(records)
    test_group_ids = _select_exact_groups(
        groups,
        test_size,
        seed=seed,
        split="test",
        blocked_task_ids=excluded_from_test,
        max_group_size=held_out_max_group_size,
    )
    remaining_groups = [
        group for group in groups if group["group_id"] not in test_group_ids
    ]
    validation_group_ids = _select_exact_groups(
        remaining_groups,
        validation_size,
        seed=seed,
        split="val",
        max_group_size=held_out_max_group_size,
    )

    ids_by_split = {"train": [], "val": [], "test": []}
    groups_by_split = {"train": [], "val": [], "test": []}
    group_split = {}
    for group in groups:
        group_id = group["group_id"]
        if group_id in test_group_ids:
            name = "test"
        elif group_id in validation_group_ids:
            name = "val"
        else:
            name = "train"
        group_split[group_id] = name
        groups_by_split[name].append(group)
        ids_by_split[name].extend(group["task_ids"])
    for task_ids in ids_by_split.values():
        task_ids.sort()

    split_payload = {}
    for name in ("train", "val", "test"):
        task_ids = ids_by_split[name]
        source_counts: dict[str, int] = defaultdict(int)
        for task_id in task_ids:
            source_counts[source_membership[task_id]] += 1
        split_payload[name] = {
            "persona": True,
            "role": name,
            "task_ids": task_ids,
            "task_count": len(task_ids),
            "task_ids_sha256": _ids_sha256(task_ids),
            "source_split_counts": dict(sorted(source_counts.items())),
            "group_count": len(groups_by_split[name]),
            "multi_task_group_count": sum(
                len(group["task_ids"]) > 1 for group in groups_by_split[name]
            ),
            "max_group_size": max(
                len(group["task_ids"]) for group in groups_by_split[name]
            ),
        }

    train_ids = set(ids_by_split["train"])
    val_ids = set(ids_by_split["val"])
    test_ids = set(ids_by_split["test"])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise AssertionError("project splits overlap")
    if train_ids | val_ids | test_ids != persona_ids:
        raise AssertionError("project splits do not cover all persona tasks")
    if test_ids & excluded_from_test:
        raise AssertionError("test split contains an explicitly excluded task")

    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "version": SPLIT_SCHEMA_VERSION,
        "source": dict(source),
        "policy": {
            "version": SPLIT_POLICY_VERSION,
            "seed": int(seed),
            "allocation_order": ["test", "val", "train"],
            "validation_size": int(validation_size),
            "test_size": int(test_size),
            "held_out_max_group_size": int(held_out_max_group_size),
            "normalization": NORMALIZATION_VERSION,
            "group_keys": list(GROUP_KEYS),
            "test_excluded_task_ids": sorted(excluded_from_test),
        },
        "grouping": {
            "group_count": len(groups),
            "multi_task_group_count": sum(
                len(group["task_ids"]) > 1 for group in groups
            ),
            "max_group_size": max(len(group["task_ids"]) for group in groups),
        },
        "splits": split_payload,
        "checks": {
            "persona_task_count": len(persona_ids),
            "complete_coverage": True,
            "pairwise_task_disjoint": True,
            "linked_groups_disjoint": len(group_split) == len(groups),
            "test_exclusions_respected": True,
        },
    }
