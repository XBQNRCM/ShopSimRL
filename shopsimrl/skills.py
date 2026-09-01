"""Skill selection boundary and a deliberately small JSON SkillBank adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .schemas import Skill, fingerprint


class SkillProvider(Protocol):
    def identity(self) -> dict[str, Any]: ...

    def select(self, context: dict[str, Any]) -> Sequence[Skill]: ...


class NoSkills:
    def identity(self) -> dict[str, Any]:
        return {"provider": "none"}

    def select(self, context: dict[str, Any]) -> Sequence[Skill]:
        return ()


class JsonSkillBank:
    """Versioned skill storage; final retrieval policy can replace this adapter.

    A skill with no ``task_ids`` is global. A skill with ``task_ids`` is selected
    only for matching tasks. This provides a reproducible baseline without
    freezing the research taxonomy or retrieval algorithm.
    """

    def __init__(self, path: str | Path, *, max_skills: int | None = None):
        self.path = Path(path)
        self.max_skills = max_skills
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload.get("skills"), list):
            raise ValueError("SkillBank must contain a skills list")
        self.schema_version = str(
            payload.get("schema_version", "shopsimrl-skillbank-v1")
        )
        self.bank_version = str(payload.get("bank_version", "1"))
        self.records = tuple(self._validate_record(item) for item in payload["skills"])
        self.bank_sha256 = fingerprint(payload)

    @staticmethod
    def _validate_record(item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("each SkillBank record must be an object")
        skill_id = item.get("skill_id")
        content = item.get("content")
        if not isinstance(skill_id, str) or not skill_id.strip():
            raise ValueError("skill_id must be a non-empty string")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"skill {skill_id!r} has empty content")
        task_ids = item.get("task_ids")
        if task_ids is not None and (
            not isinstance(task_ids, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in task_ids)
        ):
            raise ValueError(f"skill {skill_id!r} has invalid task_ids")
        return item

    def identity(self) -> dict[str, Any]:
        return {
            "provider": "json_skillbank",
            "schema_version": self.schema_version,
            "bank_version": self.bank_version,
            "bank_sha256": self.bank_sha256,
            "max_skills": self.max_skills,
        }

    def select(self, context: dict[str, Any]) -> Sequence[Skill]:
        task_id = int(context["task_id"])
        selected: list[Skill] = []
        for record in self.records:
            if record.get("enabled", True) is False:
                continue
            task_ids = record.get("task_ids")
            if task_ids is not None and task_id not in task_ids:
                continue
            metadata = dict(record.get("metadata") or {})
            metadata["selection_scope"] = "task" if task_ids is not None else "global"
            selected.append(
                Skill(
                    skill_id=record["skill_id"],
                    content=record["content"],
                    version=str(record.get("version", "1")),
                    metadata=metadata,
                )
            )
            if self.max_skills is not None and len(selected) >= self.max_skills:
                break
        return tuple(selected)


class AssignedSkillProvider:
    """Expose an externally assigned skill subset for each evaluation episode.

    Gate A creates all assignments before any rollout. Keeping the randomization
    outside the provider makes the intervention auditable and prevents resume or
    worker scheduling from changing a task's treatment.
    """

    def __init__(
        self,
        bank: JsonSkillBank,
        assignments: Mapping[tuple[str, int, int], Sequence[str]],
        *,
        assignment_identity: dict[str, Any],
    ):
        self.bank = bank
        self.assignment_identity = dict(assignment_identity)
        known_ids = {record["skill_id"] for record in bank.records}
        normalized: dict[tuple[str, int, int], tuple[str, ...]] = {}
        for key, values in assignments.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 3
                or not isinstance(key[0], str)
                or isinstance(key[1], bool)
                or not isinstance(key[1], int)
                or isinstance(key[2], bool)
                or not isinstance(key[2], int)
            ):
                raise ValueError(f"invalid assignment key: {key!r}")
            skill_ids = tuple(values)
            if len(skill_ids) != len(set(skill_ids)):
                raise ValueError(f"duplicate skill IDs in assignment {key!r}")
            unknown = sorted(set(skill_ids) - known_ids)
            if unknown:
                raise ValueError(
                    f"assignment {key!r} contains unknown skill IDs: {unknown}"
                )
            normalized[key] = skill_ids
        self.assignments = normalized

    def identity(self) -> dict[str, Any]:
        return {
            "provider": "assigned_json_skillbank",
            "bank": self.bank.identity(),
            "assignment": self.assignment_identity,
        }

    def select(self, context: dict[str, Any]) -> Sequence[Skill]:
        key = (
            str(context["split"]),
            int(context["task_id"]),
            int(context["sample_id"]),
        )
        if key not in self.assignments:
            raise KeyError(f"no skill assignment for episode context {key!r}")
        selected_ids = set(self.assignments[key])
        candidates = self.bank.select(context)
        selected = tuple(
            skill for skill in candidates if skill.skill_id in selected_ids
        )
        if {skill.skill_id for skill in selected} != selected_ids:
            missing = sorted(selected_ids - {skill.skill_id for skill in selected})
            raise ValueError(
                f"assigned skills are disabled or out of scope for {key!r}: {missing}"
            )
        return selected
