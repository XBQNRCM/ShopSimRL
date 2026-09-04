"""ShopSimRL metrics hooks for slime rollout logging.

This module deliberately does not import W&B.  The hook enriches slime's
primary-process metric dictionary; slime then sends the combined dictionary to
W&B and/or TensorBoard exactly once for each rollout batch.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .curriculum import SkillCurriculum
from .slime_runtime import _project_path


def _leaves(values: Iterable[Any]) -> Iterable[Any]:
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from _leaves(value)
        else:
            yield value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float]) -> float | None:
    rows = list(values)
    return sum(rows) / len(rows) if rows else None


def _set(metrics: dict[str, float], name: str, value: float | None) -> None:
    if value is not None and math.isfinite(value):
        metrics[f"rollout/shopsim/{name}"] = float(value)


def compute_shopsim_rollout_metrics(
    samples: Iterable[Any], curriculum: SkillCurriculum
) -> dict[str, float]:
    """Aggregate trajectory and treatment metrics without fan-out bias."""

    rollouts: dict[tuple[int, int], dict[str, Any]] = {}
    group_triage: dict[int, Mapping[str, Any]] = {}
    for sample in _leaves(samples):
        metadata = dict(getattr(sample, "metadata", None) or {})
        group = int(getattr(sample, "group_index", -1))
        raw_rollout_id = getattr(sample, "rollout_id", None)
        rollout_id = int(
            raw_rollout_id if raw_rollout_id is not None else getattr(sample, "index", -1)
        )
        key = (group, rollout_id)
        reward = metadata.get("shopsim_reward")
        if not isinstance(reward, Mapping):
            reward = getattr(sample, "reward", None)
        assignment = metadata.get("shopsim_assignment")
        if key not in rollouts:
            rollouts[key] = {
                "group": group,
                "reward": dict(reward) if isinstance(reward, Mapping) else {},
                "scored": metadata.get("shopsim_scored") is True,
                "assignment": (
                    dict(assignment) if isinstance(assignment, Mapping) else {}
                ),
            }
        triage = metadata.get("shopsim_group_triage")
        if isinstance(triage, Mapping):
            group_triage[group] = triage

    rows = list(rollouts.values())
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)

    metrics: dict[str, float] = {}
    _set(metrics, "trajectory_count", float(len(rows)))
    _set(metrics, "group_count", float(len(groups)))
    _set(
        metrics,
        "scored_rate",
        _mean(float(row["scored"]) for row in rows),
    )
    for reward_key, metric_name in (
        ("reward", "reward_mean"),
        ("r_strict", "strict_reward_mean"),
        ("r_success", "success_rate"),
    ):
        values = [
            value
            for row in rows
            if (value := _number(row["reward"].get(reward_key))) is not None
        ]
        _set(metrics, metric_name, _mean(values))

    assignments = [
        next(
            (
                row["assignment"]
                for row in group
                if isinstance(row.get("assignment"), Mapping) and row["assignment"]
            ),
            {},
        )
        for group in groups.values()
    ]
    if assignments:
        _set(
            metrics,
            "skill_free_group_rate",
            _mean(float(row.get("mode") == "skill_free") for row in assignments),
        )
        _set(
            metrics,
            "assisted_empty_group_rate",
            _mean(float(row.get("mode") == "assisted_empty") for row in assignments),
        )
        assisted = [row for row in assignments if row.get("mode") != "skill_free"]
        _set(
            metrics,
            "skills_per_assisted_group_mean",
            _mean(float(len(row.get("skill_ids") or [])) for row in assisted),
        )
        for chunk in curriculum.chunks:
            included = [
                float(chunk.skill.skill_id in (row.get("skill_ids") or []))
                for row in assisted
            ]
            _set(
                metrics,
                f"chunk/{chunk.skill.skill_id}/inclusion_rate",
                _mean(included),
            )
            _set(
                metrics,
                f"chunk/{chunk.skill.skill_id}/target_rho",
                chunk.inclusion_probability,
            )

    _set(metrics, "q", curriculum.skill_free_probability)
    rhos = [chunk.inclusion_probability for chunk in curriculum.chunks]
    if rhos:
        _set(metrics, "rho_min", min(rhos))
        _set(metrics, "rho_mean", _mean(rhos))
        _set(metrics, "rho_max", max(rhos))

    all_wrong = []
    for group in groups.values():
        if not group or not all(row["scored"] for row in group):
            continue
        successes = [_number(row["reward"].get("r_success")) for row in group]
        if all(value is not None for value in successes):
            all_wrong.append(float(all(value == 0.0 for value in successes)))
    _set(metrics, "all_wrong_group_rate", _mean(all_wrong))

    triages = list(group_triage.values())
    if triages:
        _set(
            metrics,
            "full_skill_retry_rate",
            _mean(float(bool(row.get("full_skill_retry"))) for row in triages),
        )
        retries = [row for row in triages if row.get("full_skill_retry")]
        scored_retries = [row for row in retries if row.get("full_skill_retry_scored")]
        _set(
            metrics,
            "full_skill_retry_success_rate",
            _mean(float(bool(row.get("full_skill_retry_success"))) for row in scored_retries),
        )
        for classification, name in (
            ("model_internalization_deficit", "internalization_deficit_rate"),
            ("full_skill_failure_pending_analysis", "pending_analysis_rate"),
            ("environment_or_runtime_error", "runtime_error_group_rate"),
        ):
            _set(
                metrics,
                name,
                _mean(
                    float(row.get("classification") == classification)
                    for row in triages
                ),
            )
    return metrics


def enrich_rollout_metrics(
    rollout_id: int,
    args: Any,
    samples: list[Any],
    rollout_extra_metrics: dict[str, Any] | None,
    rollout_time: float,
) -> bool:
    """slime ``--custom-rollout-log-function-path`` entry point."""

    del rollout_id, rollout_time
    if rollout_extra_metrics is None:
        return False
    curriculum_path = _project_path(getattr(args, "shopsim_curriculum_path"))
    curriculum = SkillCurriculum.load(Path(curriculum_path))
    metrics = compute_shopsim_rollout_metrics(samples, curriculum)
    dropped = _number(
        rollout_extra_metrics.get(
            "rollout/dynamic_filter/drop_shopsim_unscored_technical_group"
        )
    )
    accepted = metrics.get("rollout/shopsim/group_count", 0.0)
    if dropped is not None:
        _set(metrics, "technical_group_drop_count", dropped)
        _set(
            metrics,
            "technical_group_drop_rate",
            dropped / (accepted + dropped) if accepted + dropped else 0.0,
        )
    rollout_extra_metrics.update(metrics)
    return False
