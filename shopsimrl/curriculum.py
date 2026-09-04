"""Group-consistent skill scheduling for ShopSimulator training.

The curriculum is deliberately independent from slime.  A validation round
freezes one JSON state file; rollout workers then derive a treatment from the
group identity, so every response sampled for the same GRPO prompt receives
exactly the same skill context regardless of worker order or resume behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .schemas import Skill, fingerprint, utc_now
from .store import atomic_write_json
from .tasks import load_task_split


CURRICULUM_SCHEMA_VERSION = "shopsimrl-skill-curriculum-v1"
TRAINING_DATA_SCHEMA_VERSION = "shopsimrl-slime-tasks-v1"


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class CurriculumChunk:
    skill: Skill
    contribution: float
    inclusion_probability: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.contribution):
            raise ValueError("chunk contribution must be finite")
        _probability(self.inclusion_probability, "chunk inclusion_probability")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.skill.to_dict(),
            "contribution": self.contribution,
            "inclusion_probability": self.inclusion_probability,
        }


@dataclass(frozen=True)
class SkillAssignment:
    state_id: str
    group_key: str
    mode: str
    skills: tuple[Skill, ...]

    @property
    def skill_ids(self) -> tuple[str, ...]:
        return tuple(skill.skill_id for skill in self.skills)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "group_key": self.group_key,
            "mode": self.mode,
            "skill_ids": list(self.skill_ids),
        }


@dataclass(frozen=True)
class SkillCurriculum:
    state_id: str
    round_id: str
    seed: int
    skill_free_probability: float
    chunks: tuple[CurriculumChunk, ...]
    source: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.state_id or not self.round_id:
            raise ValueError("curriculum state_id and round_id cannot be empty")
        _probability(self.skill_free_probability, "skill_free_probability")
        ids = [chunk.skill.skill_id for chunk in self.chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("curriculum contains duplicate skill IDs")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SkillCurriculum":
        if payload.get("schema_version") != CURRICULUM_SCHEMA_VERSION:
            raise ValueError("unsupported skill curriculum schema")
        raw_chunks = payload.get("chunks")
        if not isinstance(raw_chunks, list):
            raise ValueError("curriculum chunks must be a list")
        chunks: list[CurriculumChunk] = []
        for record in raw_chunks:
            if not isinstance(record, dict):
                raise ValueError("each curriculum chunk must be an object")
            skill_id = record.get("skill_id")
            content = record.get("content")
            if not isinstance(skill_id, str) or not skill_id.strip():
                raise ValueError("curriculum chunk has an invalid skill_id")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"curriculum chunk {skill_id!r} has empty content")
            contribution = record.get("contribution")
            if (
                isinstance(contribution, bool)
                or not isinstance(contribution, (int, float))
            ):
                raise ValueError(f"curriculum chunk {skill_id!r} has no contribution")
            chunks.append(
                CurriculumChunk(
                    skill=Skill(
                        skill_id=skill_id,
                        content=content,
                        version=str(record.get("version", "1")),
                        metadata=dict(record.get("metadata") or {}),
                    ),
                    contribution=float(contribution),
                    inclusion_probability=_probability(
                        record.get("inclusion_probability"),
                        f"chunk {skill_id!r} inclusion_probability",
                    ),
                )
            )
        core = {
            key: payload[key]
            for key in (
                "round_id",
                "seed",
                "skill_free_probability",
                "chunks",
                "source",
            )
        }
        expected_id = fingerprint(core)
        if payload.get("state_id") != expected_id:
            raise ValueError("curriculum state_id does not match its contents")
        return cls(
            state_id=expected_id,
            round_id=str(payload["round_id"]),
            seed=int(payload["seed"]),
            skill_free_probability=_probability(
                payload["skill_free_probability"], "skill_free_probability"
            ),
            chunks=tuple(chunks),
            source=dict(payload.get("source") or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SkillCurriculum":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"skill curriculum not found: {source}")
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": CURRICULUM_SCHEMA_VERSION,
            "state_id": self.state_id,
            "round_id": self.round_id,
            "seed": self.seed,
            "skill_free_probability": self.skill_free_probability,
            "chunk_ids": [chunk.skill.skill_id for chunk in self.chunks],
        }

    def full_skills(self) -> tuple[Skill, ...]:
        return tuple(chunk.skill for chunk in self.chunks)

    def assign(self, group_key: str | int) -> SkillAssignment:
        """Sample q once, then rho per chunk, deterministically per group."""

        key = str(group_key)
        skill_free = self._draw(key, "skill-free") < self.skill_free_probability
        if skill_free:
            return SkillAssignment(self.state_id, key, "skill_free", ())
        selected = tuple(
            chunk.skill
            for chunk in self.chunks
            if self._draw(key, f"chunk:{chunk.skill.skill_id}")
            < chunk.inclusion_probability
        )
        return SkillAssignment(
            self.state_id,
            key,
            "assisted" if selected else "assisted_empty",
            selected,
        )

    def _draw(self, group_key: str, treatment: str) -> float:
        digest = hashlib.sha256(
            f"{self.seed}:{self.state_id}:{group_key}:{treatment}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") / float(1 << 64)


def build_curriculum_state(
    skillbank_path: str | Path,
    *,
    round_id: str,
    seed: int,
    skill_free_probability: float,
    rho_min: float,
    rho_max: float,
    contribution_scale: float | None = None,
) -> dict[str, Any]:
    """Build a frozen training state from a validated active SkillBank.

    The named mapping is linear and clipped.  ``contribution_scale`` is the
    positive effect that maps to ``rho_max``; when omitted, the largest active
    contribution is used and recorded, avoiding an implicit reward-scale
    constant.  The gate, not this function, owns admission and retirement.
    """

    from .skills import JsonSkillBank

    q = _probability(skill_free_probability, "skill_free_probability")
    lower = _probability(rho_min, "rho_min")
    upper = _probability(rho_max, "rho_max")
    if lower > upper:
        raise ValueError("rho_min cannot exceed rho_max")
    bank = JsonSkillBank(skillbank_path, max_skills=None)
    records: list[tuple[dict[str, Any], float]] = []
    for record in bank.records:
        if record.get("enabled", True) is False:
            continue
        metadata = record.get("metadata") or {}
        effect = metadata.get("estimated_effect")
        if isinstance(effect, bool) or not isinstance(effect, (int, float)):
            raise ValueError(
                f"active skill {record['skill_id']!r} has no numeric estimated_effect"
            )
        effect = float(effect)
        if not math.isfinite(effect) or effect <= 0.0:
            raise ValueError(
                f"active skill {record['skill_id']!r} must have positive estimated_effect"
            )
        records.append((record, effect))
    positive_max = max((effect for _, effect in records), default=1.0)
    scale = positive_max if contribution_scale is None else float(contribution_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("contribution_scale must be positive and finite")

    chunks = []
    for record, effect in records:
        rho = min(upper, max(lower, lower + (upper - lower) * effect / scale))
        chunks.append(
            {
                "skill_id": record["skill_id"],
                "version": str(record.get("version", "1")),
                "content": record["content"],
                "metadata": dict(record.get("metadata") or {}),
                "contribution": effect,
                "inclusion_probability": rho,
            }
        )
    core = {
        "round_id": str(round_id),
        "seed": int(seed),
        "skill_free_probability": q,
        "chunks": chunks,
        "source": {
            "skillbank_path": str(Path(skillbank_path).resolve()),
            "skillbank_sha256": bank.bank_sha256,
            "probability_mapping": {
                "name": "clipped_linear_positive_contribution",
                "rho_min": lower,
                "rho_max": upper,
                "contribution_scale": scale,
                "scale_source": (
                    "largest_active_contribution"
                    if contribution_scale is None
                    else "explicit"
                ),
            },
        },
    }
    return {
        "schema_version": CURRICULUM_SCHEMA_VERSION,
        "created_at": utc_now(),
        **core,
        "state_id": fingerprint(core),
    }


def write_curriculum_state(path: str | Path, payload: dict[str, Any]) -> None:
    # Validate before publishing an atomic state transition to rollout workers.
    SkillCurriculum.from_dict(payload)
    atomic_write_json(Path(path), payload)


def write_slime_task_data(
    split_file: str | Path, split: str, output_path: str | Path
) -> dict[str, Any]:
    """Materialize stable task IDs as slime's global prompt dataset."""

    task_split = load_task_split(split_file, split)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for task_id in task_split.task_ids:
        lines.append(
            json.dumps(
                {
                    "prompt": str(task_id),
                    "metadata": {
                        "schema_version": TRAINING_DATA_SCHEMA_VERSION,
                        "task_id": task_id,
                        "split": task_split.name,
                    },
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    contents = "\n".join(lines) + "\n"
    output.write_text(contents, encoding="utf-8")
    return {
        "schema_version": TRAINING_DATA_SCHEMA_VERSION,
        "output_path": str(output.resolve()),
        "split": task_split.identity(),
        "records": len(lines),
        "sha256": hashlib.sha256(contents.encode("utf-8")).hexdigest(),
    }
