"""Configuration for reusable Trace2Skill randomized validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentSpec, load_experiment_config
from .store import safe_name


@dataclass(frozen=True)
class GateASpec:
    name: str
    output_dir: Path
    experiment: ExperimentSpec
    draft_path: Path
    draft_skillbank_path: Path
    mask_seed: int
    mask_probability: float
    active_skill_budget: int | None


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _required_path(record: dict[str, Any], key: str, *, base: Path, name: str) -> Path:
    value = record.get(key)
    if value is None:
        raise ValueError(f"{name}.{key} is required")
    return _resolve(base, value)


def load_trace2skill_evaluation_config(
    path: str | Path,
) -> GateASpec:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, "config")
    base = source.parent

    gate_a_record = _mapping(root.get("gate_a"), "gate_a")
    gate_a_experiment_path = _required_path(
        gate_a_record,
        "experiment_config",
        base=base,
        name="gate_a",
    )
    draft_path = _required_path(
        gate_a_record, "draft_path", base=base, name="gate_a"
    )
    draft_skillbank_raw = gate_a_record.get("draft_skillbank_path")
    draft_skillbank_path = (
        _resolve(base, draft_skillbank_raw)
        if draft_skillbank_raw is not None
        else draft_path.with_name("initial_skillbank.json")
    )
    mask_probability = float(gate_a_record.get("mask_probability", 0.5))
    if mask_probability != 0.5:
        raise ValueError("gate_a.mask_probability must be 0.5 for cold-start Gate A")
    active_budget_raw = gate_a_record.get("active_skill_budget")
    active_skill_budget = (
        int(active_budget_raw) if active_budget_raw is not None else None
    )
    if active_skill_budget is not None and active_skill_budget < 1:
        raise ValueError("gate_a.active_skill_budget must be positive")
    return GateASpec(
        name=safe_name(str(gate_a_record.get("name", "trace2skill-gate-a"))),
        output_dir=_required_path(
            gate_a_record, "output_dir", base=base, name="gate_a"
        ),
        experiment=load_experiment_config(gate_a_experiment_path),
        draft_path=draft_path,
        draft_skillbank_path=draft_skillbank_path,
        mask_seed=int(gate_a_record.get("mask_seed", 20260901)),
        mask_probability=mask_probability,
        active_skill_budget=active_skill_budget,
    )
