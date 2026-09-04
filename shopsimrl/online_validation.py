"""Randomized online validation for active chunks and Trace2Skill candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from .config import ExperimentSpec, load_experiment_config
from .evaluation import Evaluator
from .paired_validation import (
    PAIRED_ESTIMATOR,
    PAIRED_FORMULA,
    load_bare_validation,
    outcome_means,
    paired_observations,
    read_validation_traces,
)
from .schemas import EpisodeJob, Skill, fingerprint, utc_now
from .skills import AssignedSkillProvider
from .store import RunStore, atomic_write_json, safe_name
from .trace2skill_evaluation import (
    _atomic_write_text,
    _default_runtime_factory,
    _observed_provenance,
    _prepare_experiment,
    _semantic_plan,
    fit_paired_delta_ols,
)


CANDIDATE_POOL_SCHEMA_VERSION = "shopsimrl-skill-candidate-pool-v1"
ONLINE_ASSIGNMENT_SCHEMA_VERSION = "shopsimrl-online-skill-assignments-v1"
ONLINE_CONTRIBUTION_SCHEMA_VERSION = "shopsimrl-online-chunk-contributions-v2"
ONLINE_GATE_SCHEMA_VERSION = "shopsimrl-online-validation-gate-v2"

Progress = Callable[[int, int, EpisodeJob, dict[str, Any]], None]


@dataclass(frozen=True)
class OnlineGateSpec:
    name: str
    output_dir: Path
    experiment: ExperimentSpec
    bare_run_dir: Path
    candidate_pool_path: Path
    mask_seed: int
    active_skill_budget: int


class InMemorySkillBank:
    """The complete intervention bank, including mutually-exclusive versions."""

    def __init__(self, records: Sequence[Mapping[str, Any]], *, pool_sha256: str):
        self.records = tuple(dict(record) for record in records)
        self.pool_sha256 = pool_sha256
        ids = [record.get("skill_id") for record in self.records]
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("intervention bank contains an invalid skill_id")
        if len(ids) != len(set(ids)):
            raise ValueError("intervention bank contains duplicate skill IDs")

    def identity(self) -> dict[str, Any]:
        return {
            "provider": "online_intervention_bank",
            "pool_sha256": self.pool_sha256,
            "skill_ids": [record["skill_id"] for record in self.records],
        }

    def select(self, context: dict[str, Any]) -> tuple[Skill, ...]:
        return tuple(
            Skill(
                skill_id=record["skill_id"],
                content=record["content"],
                version=str(record.get("version", "1")),
                metadata=dict(record.get("metadata") or {}),
            )
            for record in self.records
            if record.get("enabled", True) is not False
        )


def load_candidate_pool(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"candidate pool not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != CANDIDATE_POOL_SCHEMA_VERSION:
        raise ValueError("unsupported online candidate-pool schema")
    expected_sha256 = fingerprint(
        {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "pool_sha256"}
        }
    )
    if payload.get("pool_sha256") != expected_sha256:
        raise ValueError("candidate pool hash does not match its contents")
    current = payload.get("current_skills")
    candidates = payload.get("candidates")
    if not isinstance(current, list) or not isinstance(candidates, list):
        raise ValueError("candidate pool must contain current_skills and candidates")

    current_ids: list[str] = []
    for record in current:
        if not isinstance(record, dict):
            raise ValueError("current skill records must be objects")
        skill_id = record.get("skill_id")
        content = record.get("content")
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError("current skill has an invalid skill_id")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"current skill {skill_id!r} has empty content")
        current_ids.append(skill_id)
    if len(current_ids) != len(set(current_ids)):
        raise ValueError("candidate pool contains duplicate current skill IDs")

    candidate_ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate records must be objects")
        candidate_id = candidate.get("candidate_id")
        operation = candidate.get("operation")
        content = candidate.get("content")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate has an invalid candidate_id")
        if candidate_id in current_ids:
            raise ValueError(f"candidate ID collides with active chunk: {candidate_id}")
        if operation not in {"ADD", "REWRITE"}:
            raise ValueError(f"candidate {candidate_id!r} has invalid operation")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"candidate {candidate_id!r} has empty content")
        target = candidate.get("target_chunk_id")
        if operation == "ADD" and target not in {None, ""}:
            raise ValueError(f"ADD candidate {candidate_id!r} cannot have a target")
        if operation == "REWRITE" and target not in current_ids:
            raise ValueError(
                f"REWRITE candidate {candidate_id!r} has unknown target {target!r}"
            )
        candidate_ids.append(candidate_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate pool contains duplicate candidate IDs")
    return payload


def build_candidate_pool(
    current_skillbank_path: str | Path,
    candidates: Sequence[Mapping[str, Any]],
    *,
    round_id: str,
    proposal_checkpoint: str,
) -> dict[str, Any]:
    from .skills import JsonSkillBank

    bank = JsonSkillBank(current_skillbank_path, max_skills=None)
    current = [
        {
            "skill_id": record["skill_id"],
            "version": str(record.get("version", "1")),
            "content": record["content"],
            "metadata": dict(record.get("metadata") or {}),
        }
        for record in bank.records
        if record.get("enabled", True) is not False
    ]
    normalized_candidates = [dict(candidate) for candidate in candidates]
    payload = {
        "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
        "created_at": utc_now(),
        "round_id": str(round_id),
        "proposal_checkpoint": str(proposal_checkpoint),
        "current_skillbank_sha256": bank.bank_sha256,
        "current_skills": current,
        "candidates": normalized_candidates,
    }
    # Run the same strict validation used by the gate before returning.
    temporary = {**payload}
    _validate_candidate_pool_payload(temporary)
    payload["pool_sha256"] = fingerprint(
        {key: value for key, value in payload.items() if key != "created_at"}
    )
    return payload


def _validate_candidate_pool_payload(payload: dict[str, Any]) -> None:
    # Avoid a temporary file in build_candidate_pool while sharing validation.
    current = payload.get("current_skills")
    candidates = payload.get("candidates")
    if not isinstance(current, list) or not isinstance(candidates, list):
        raise ValueError("candidate pool must contain list fields")
    ids = [record.get("skill_id") for record in current if isinstance(record, dict)]
    if len(ids) != len(current) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("candidate pool has invalid current skill IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate pool has duplicate current skill IDs")
    for record in current:
        if not isinstance(record.get("content"), str) or not record["content"].strip():
            raise ValueError(f"current skill {record.get('skill_id')!r} has empty content")
    candidate_ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be an object")
        candidate_id = candidate.get("candidate_id")
        operation = candidate.get("operation")
        content = candidate.get("content")
        target = candidate.get("target_chunk_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if candidate_id in ids:
            raise ValueError("candidate ID collides with a current skill")
        if operation not in {"ADD", "REWRITE"}:
            raise ValueError("candidate operation must be ADD or REWRITE")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"candidate {candidate_id!r} has empty content")
        if operation == "ADD" and target not in {None, ""}:
            raise ValueError("ADD candidate cannot have a target")
        if operation == "REWRITE" and target not in ids:
            raise ValueError("REWRITE candidate target must be active")
        candidate_ids.append(candidate_id)
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs must be unique")


def _pool_layout(pool: dict[str, Any]) -> dict[str, Any]:
    """Compile logical slots, intervention columns and version records."""

    current = pool["current_skills"]
    candidates = pool["candidates"]
    rewrites: dict[str, list[dict[str, Any]]] = {}
    additions: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["operation"] == "REWRITE":
            rewrites.setdefault(candidate["target_chunk_id"], []).append(candidate)
        else:
            additions.append(candidate)

    slots = []
    records = []
    version_info: dict[str, dict[str, Any]] = {}
    for order, record in enumerate(current, 1):
        old_id = record["skill_id"]
        versions = [old_id]
        records.append(dict(record))
        version_info[old_id] = {
            "kind": "current",
            "logical_chunk_id": old_id,
            "operation": None,
            "order": order,
        }
        for candidate in sorted(rewrites.get(old_id, []), key=lambda row: row["candidate_id"]):
            candidate_id = candidate["candidate_id"]
            versions.append(candidate_id)
            records.append(
                {
                    "skill_id": candidate_id,
                    "version": str(candidate.get("version", "candidate")),
                    "content": candidate["content"],
                    "metadata": {
                        **dict(candidate.get("metadata") or {}),
                        "operation": "REWRITE",
                        "target_chunk_id": old_id,
                    },
                }
            )
            version_info[candidate_id] = {
                "kind": "candidate",
                "logical_chunk_id": old_id,
                "operation": "REWRITE",
                "order": order,
            }
        slots.append({"slot_id": old_id, "old_version_id": old_id, "versions": versions})

    next_order = len(current) + 1
    for candidate in sorted(additions, key=lambda row: row["candidate_id"]):
        candidate_id = candidate["candidate_id"]
        slots.append(
            {"slot_id": candidate_id, "old_version_id": None, "versions": [candidate_id]}
        )
        records.append(
            {
                "skill_id": candidate_id,
                "version": str(candidate.get("version", "candidate")),
                "content": candidate["content"],
                "metadata": {
                    **dict(candidate.get("metadata") or {}),
                    "operation": "ADD",
                },
            }
        )
        version_info[candidate_id] = {
            "kind": "candidate",
            "logical_chunk_id": candidate_id,
            "operation": "ADD",
            "order": next_order,
        }
        next_order += 1
    columns = [version for slot in slots for version in slot["versions"]]
    return {
        "slots": slots,
        "columns": columns,
        "records": records,
        "version_info": version_info,
    }


def _hash_index(seed: int, job: EpisodeJob, slot_id: str, choices: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{job.split}:{job.task_id}:{job.sample_id}:{slot_id}".encode(
            "utf-8"
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") % choices


def build_online_assignments(
    jobs: Sequence[EpisodeJob], pool: dict[str, Any], *, mask_seed: int
) -> dict[str, Any]:
    if not jobs:
        raise ValueError("online validation jobs cannot be empty")
    layout = _pool_layout(pool)
    columns = layout["columns"]
    assignments = []
    for job in jobs:
        included: list[str] = []
        slot_states: dict[str, str | None] = {}
        for slot in layout["slots"]:
            versions = slot["versions"]
            # Absent plus all versions are symmetric randomized states.  A
            # singleton slot therefore reduces exactly to Bernoulli(0.5).
            choice = _hash_index(mask_seed, job, slot["slot_id"], len(versions) + 1)
            selected = None if choice == 0 else versions[choice - 1]
            slot_states[slot["slot_id"]] = selected
            if selected is not None:
                included.append(selected)
        mask = [int(column in included) for column in columns]
        assignments.append(
            {
                "episode_id": job.episode_id,
                **job.to_dict(),
                "mask": mask,
                "included_chunk_ids": included,
                "slot_states": slot_states,
            }
        )
    core = {
        "mask_seed": int(mask_seed),
        "pool_sha256": pool.get("pool_sha256") or fingerprint(pool),
        "columns": columns,
        "slots": layout["slots"],
        "assignments": assignments,
    }
    result = {
        "schema_version": ONLINE_ASSIGNMENT_SCHEMA_VERSION,
        "created_at": utc_now(),
        **core,
        "assignment_sha256": fingerprint(core),
    }
    # The OLS gate must fail before rollouts if the frozen design is singular.
    fit_paired_delta_ols([row["mask"] for row in assignments], [0.0] * len(assignments))
    return result


def estimate_online_contributions(
    *,
    pool: dict[str, Any],
    assignments: dict[str, Any],
    traces: Iterable[dict[str, Any]],
    bare_traces: Iterable[dict[str, Any]],
    active_skill_budget: int,
) -> dict[str, Any]:
    if active_skill_budget < 1:
        raise ValueError("active_skill_budget must be positive")
    masks, outcomes, observations = paired_observations(assignments, traces, bare_traces)
    columns = tuple(assignments["columns"])
    estimates: dict[str, dict[str, Any]] = {}
    for metric, values in outcomes.items():
        intercept, coefficients = fit_paired_delta_ols(masks, values)
        estimates[metric] = {
            "intercept": intercept,
            "coefficients": dict(zip(columns, coefficients)),
            "observations": len(values),
            **outcome_means(observations, metric),
        }
    primary = estimates["reward"]["coefficients"]
    layout = _pool_layout(pool)
    info = layout["version_info"]

    winners: dict[str, str] = {}
    losers: set[str] = set()
    for slot in layout["slots"]:
        versions = slot["versions"]
        old_id = slot["old_version_id"]
        if len(versions) == 1:
            winners[slot["slot_id"]] = versions[0]
            continue
        # Equality keeps old by placing it first in the stable max tie-break.
        winner = max(
            versions,
            key=lambda version: (
                primary[version],
                int(version == old_id),
                -versions.index(version),
            ),
        )
        winners[slot["slot_id"]] = winner
        losers.update(set(versions) - {winner})

    survivor_versions = list(winners.values())
    ranked = sorted(survivor_versions, key=lambda value: (-primary[value], value))
    positive = [value for value in ranked if primary[value] > 0.0]
    selected_versions = set(positive[:active_skill_budget])
    ranks = {value: index for index, value in enumerate(ranked, 1)}

    rows = []
    for column in columns:
        winner = column not in losers
        if not winner:
            status = "rewrite_loser"
        elif column in selected_versions:
            status = "selected"
        elif primary[column] <= 0.0:
            status = "retired" if info[column]["kind"] == "current" else "non_positive"
        else:
            status = "retired" if info[column]["kind"] == "current" else "budget_excluded"
        rows.append(
            {
                "intervention_id": column,
                **info[column],
                "coefficient": primary[column],
                "rank": ranks.get(column),
                "status": status,
                "auxiliary_coefficients": {
                    metric: estimate["coefficients"][column]
                    for metric, estimate in estimates.items()
                    if metric != "reward"
                },
            }
        )

    selected_by_order = sorted(
        selected_versions,
        key=lambda version: (info[version]["order"], info[version]["logical_chunk_id"]),
    )
    return {
        "schema_version": ONLINE_CONTRIBUTION_SCHEMA_VERSION,
        "created_at": utc_now(),
        "estimator": {
            "name": PAIRED_ESTIMATOR,
            "formula": PAIRED_FORMULA,
            "fit_intercept": False,
            "baseline": "same_task_bare_validation",
            "primary_outcome": "reward",
            "rewrite_rule": "mutually_exclusive_family_argmax_old_on_tie",
            "selection_rule": "rewrite_winners_then_strictly_positive_top_k",
        },
        "pool_sha256": assignments["pool_sha256"],
        "assignment_sha256": assignments["assignment_sha256"],
        "observations": len(observations),
        "active_skill_budget": active_skill_budget,
        "selected_intervention_ids": selected_by_order,
        "selected_count": len(selected_by_order),
        "replacement_winners": winners,
        "estimates": estimates,
        "contributions": rows,
        "observations_table": observations,
    }


def selected_skillbank(
    pool: dict[str, Any], contributions: dict[str, Any]
) -> dict[str, Any]:
    layout = _pool_layout(pool)
    record_by_id = {record["skill_id"]: record for record in layout["records"]}
    result_by_id = {
        row["intervention_id"]: row for row in contributions["contributions"]
    }
    records = []
    for intervention_id in contributions["selected_intervention_ids"]:
        source = record_by_id[intervention_id]
        result = result_by_id[intervention_id]
        logical_id = result["logical_chunk_id"]
        metadata = dict(source.get("metadata") or {})
        metadata.update(
            {
                "validation_result": "selected",
                "estimated_effect": result["coefficient"],
                "contribution_rank": result["rank"],
                "selected_intervention_id": intervention_id,
            }
        )
        records.append(
            {
                "skill_id": logical_id,
                "version": str(source.get("version", "1")),
                "enabled": True,
                "content": source["content"],
                "metadata": metadata,
            }
        )
    return {
        "schema_version": "shopsimrl-skillbank-v1",
        "bank_version": f"online-{pool.get('round_id', 'round')}-selected-v1",
        "metadata": {
            "round_id": pool.get("round_id"),
            "proposal_checkpoint": pool.get("proposal_checkpoint"),
            "pool_sha256": contributions["pool_sha256"],
            "assignment_sha256": contributions["assignment_sha256"],
            "active_skill_budget": contributions["active_skill_budget"],
            "selected_count": contributions["selected_count"],
            "selected_intervention_ids": contributions["selected_intervention_ids"],
            "estimator": contributions["estimator"],
            "bare_baseline": contributions.get("bare_baseline"),
        },
        "skills": records,
    }


def _proposal_ledger_rows(
    pool: dict[str, Any],
    contributions: dict[str, Any],
    history: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Attach frozen-gate outcomes to every proposed ADD/REWRITE candidate."""

    results = {
        row["intervention_id"]: row for row in contributions["contributions"]
    }
    by_id = {
        str(record["candidate_id"]): dict(record)
        for record in history
        if isinstance(record.get("candidate_id"), str)
    }
    order = list(by_id)

    def attach_result(
        record: dict[str, Any],
        result: Mapping[str, Any],
        *,
        intervention_id: str,
        status: str,
    ) -> None:
        event = {
            "proposal_checkpoint": pool.get("proposal_checkpoint"),
            "evaluation_checkpoint": (contributions.get("bare_baseline") or {}).get("checkpoint_id"),
            "validation_estimator": contributions["estimator"]["name"],
            "bare_validation_observations_sha256": (contributions.get("bare_baseline") or {}).get("observations_sha256"),
            "intervention_id": intervention_id,
            "validation_result": status,
            "estimated_effect": result["coefficient"],
            "contribution_rank": result["rank"],
            "pool_sha256": contributions["pool_sha256"],
            "assignment_sha256": contributions["assignment_sha256"],
        }
        history_events = [
            dict(item)
            for item in record.get("validation_history") or []
            if isinstance(item, dict)
            and not (
                item.get("assignment_sha256") == event["assignment_sha256"]
                and item.get("validation_estimator") == event["validation_estimator"]
                and item.get("bare_validation_observations_sha256") == event["bare_validation_observations_sha256"]
            )
        ]
        history_events.append(event)
        record.update(event)
        record["validation_history"] = history_events

    # A previously selected proposal is represented by a current logical
    # chunk in later rounds. Update its lifecycle before processing this
    # round's new candidates. Current-version losses mean retirement, not a
    # second candidate rejection.
    for current in pool["current_skills"]:
        metadata = dict(current.get("metadata") or {})
        origin_id = metadata.get("selected_intervention_id") or current["skill_id"]
        if origin_id not in by_id:
            continue
        result = results[current["skill_id"]]
        record = by_id[origin_id]
        record["active_chunk_id"] = current["skill_id"]
        attach_result(
            record,
            result,
            intervention_id=current["skill_id"],
            status=("selected" if result["status"] == "selected" else "retired"),
        )

    for candidate in pool["candidates"]:
        candidate_id = candidate["candidate_id"]
        result = results[candidate_id]
        metadata = dict(candidate.get("metadata") or {})
        if candidate_id not in by_id:
            order.append(candidate_id)
        record = dict(by_id.get(candidate_id) or {})
        record.update(
            {
                "candidate_id": candidate_id,
                "operation": candidate["operation"],
                "target_chunk_id": candidate.get("target_chunk_id") or None,
                "source_card_ids": list(metadata.get("evidence_card_ids") or []),
                "failure_mechanism": metadata.get("failure_mechanism"),
                "proposal_checkpoint": pool.get("proposal_checkpoint"),
                "stage": "online",
                "consolidation_cluster_id": metadata.get("consolidation_cluster_id"),
                "replacement_family_id": (
                    candidate.get("target_chunk_id")
                    if candidate["operation"] == "REWRITE"
                    else None
                ),
                "content": candidate["content"],
            }
        )
        attach_result(
            record,
            result,
            intervention_id=candidate_id,
            status=result["status"],
        )
        by_id[candidate_id] = record
    return [by_id[candidate_id] for candidate_id in order]


def _read_proposal_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid proposal ledger JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"proposal ledger row must be an object at {path}:{line_number}")
        records.append(record)
    return records


def _assigned_provider(
    bank: InMemorySkillBank, assignments: dict[str, Any]
) -> AssignedSkillProvider:
    mapping = {
        (row["split"], row["task_id"], row["sample_id"]): tuple(
            row["included_chunk_ids"]
        )
        for row in assignments["assignments"]
    }
    return AssignedSkillProvider(
        bank,
        mapping,
        assignment_identity={
            "schema_version": assignments["schema_version"],
            "pool_sha256": assignments["pool_sha256"],
            "assignment_sha256": assignments["assignment_sha256"],
            "mask_seed": assignments["mask_seed"],
        },
    )


def _load_spec(path: str | Path) -> OnlineGateSpec:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("online_gate"), dict):
        raise ValueError("config must contain an online_gate mapping")
    record = payload["online_gate"]
    base = source.parent

    def resolve(key: str) -> Path:
        value = record.get(key)
        if value is None:
            raise ValueError(f"online_gate.{key} is required")
        candidate = Path(value)
        return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    budget = int(record.get("active_skill_budget", 10))
    if budget < 1:
        raise ValueError("online_gate.active_skill_budget must be positive")
    return OnlineGateSpec(
        name=safe_name(str(record.get("name", "trace2skill-online-gate"))),
        output_dir=resolve("output_dir"),
        experiment=load_experiment_config(resolve("experiment_config")),
        bare_run_dir=resolve("bare_run_dir"),
        candidate_pool_path=resolve("candidate_pool_path"),
        mask_seed=int(record.get("mask_seed", 20260901)),
        active_skill_budget=budget,
    )


def load_online_gate_config(path: str | Path) -> OnlineGateSpec:
    return _load_spec(path)


def online_gate_plan(spec: OnlineGateSpec) -> dict[str, Any]:
    pool = load_candidate_pool(spec.candidate_pool_path)
    task_split, jobs, persona, prompt_builder = _prepare_experiment(
        spec.experiment, required_split="val"
    )
    assignments = build_online_assignments(jobs, pool, mask_seed=spec.mask_seed)
    layout = _pool_layout(pool)
    return {
        "gate": "online",
        "estimator": PAIRED_ESTIMATOR,
        "bare_baseline": load_bare_validation(
            spec.bare_run_dir,
            _semantic_plan(
                experiment_name=spec.name, spec=spec.experiment, task_split=task_split,
                jobs=jobs, persona=persona, prompt_builder=prompt_builder,
                skill_provider=InMemorySkillBank(layout["records"], pool_sha256=assignments["pool_sha256"]),
            ),
            require_checkpoint=True,
        )[1],
        "output_dir": str(spec.output_dir),
        "pool_sha256": assignments["pool_sha256"],
        "tasks": len(jobs),
        "slots": len(layout["slots"]),
        "regression_factors": len(layout["columns"]),
        "active_skill_budget": spec.active_skill_budget,
        "assignment_sha256": assignments["assignment_sha256"],
        "split": task_split.identity(),
        "persona": persona,
        "model": spec.experiment.model.config.identity(),
    }


def run_online_gate(
    spec: OnlineGateSpec,
    *,
    runtime_factory: Callable[[Any], Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    pool = load_candidate_pool(spec.candidate_pool_path)
    task_split, jobs, persona, prompt_builder = _prepare_experiment(
        spec.experiment, required_split="val"
    )
    assignments = build_online_assignments(jobs, pool, mask_seed=spec.mask_seed)
    layout = _pool_layout(pool)
    bank = InMemorySkillBank(layout["records"], pool_sha256=assignments["pool_sha256"])
    provider = _assigned_provider(bank, assignments)
    semantic_plan = _semantic_plan(
        experiment_name=spec.name,
        spec=spec.experiment,
        task_split=task_split,
        jobs=jobs,
        persona=persona,
        prompt_builder=prompt_builder,
        skill_provider=provider,
    )
    bare_traces, baseline = load_bare_validation(spec.bare_run_dir, semantic_plan, require_checkpoint=True)
    semantic_plan["online_validation_gate"] = {
        "estimator": PAIRED_ESTIMATOR,
        "bare_baseline": baseline,
        "pool_sha256": assignments["pool_sha256"],
        "assignment_sha256": assignments["assignment_sha256"],
        "active_skill_budget": spec.active_skill_budget,
        "rewrite_treatment": "mutually_exclusive_absent_old_new",
    }
    store = RunStore(spec.output_dir)
    read_validation_traces(store.traces_path, resume=spec.experiment.resume)
    store.initialize(semantic_plan)
    atomic_write_json(spec.output_dir / "mask_assignments.json", assignments)
    factory = (
        _default_runtime_factory(
            spec=spec.experiment,
            persona=persona,
            prompt_builder=prompt_builder,
            skill_provider=provider,
        )
        if runtime_factory is None
        else lambda: runtime_factory(provider)
    )
    summary = Evaluator(
        runtime_factory=factory,
        store=store,
        max_workers=spec.experiment.concurrency,
        progress=progress,
    ).run(jobs, resume=spec.experiment.resume)
    traces = read_validation_traces(store.traces_path)
    observed_provenance = _observed_provenance(traces)
    manifest: dict[str, Any] = {
        "schema_version": ONLINE_GATE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "incomplete",
        "config": {
            "name": spec.name,
            "output_dir": str(spec.output_dir),
            "candidate_pool_path": str(spec.candidate_pool_path),
            "bare_run_dir": str(spec.bare_run_dir),
            "mask_seed": spec.mask_seed,
            "active_skill_budget": spec.active_skill_budget,
        },
        "protocol": {
            "split": "val",
            "masked_rollouts_per_task": 1,
            "reused_bare_rollouts_per_task": 1,
            "masking": "uniform_absent_or_one_version_per_logical_slot",
            "rewrite_treatment": "mutually_exclusive_absent_old_new",
            "estimator": PAIRED_ESTIMATOR,
            "formula": PAIRED_FORMULA,
            "selection": "replacement_winner_then_strictly_positive_top_k",
        },
        "input": {
            "proposal_checkpoint": pool.get("proposal_checkpoint"),
            "pool_sha256": assignments["pool_sha256"],
            "assignment_sha256": assignments["assignment_sha256"],
            "task_ids": [row["task_id"] for row in assignments["assignments"]],
            "model": spec.experiment.model.config.identity(),
            "environment": {
                "base_url": spec.experiment.environment_base_url.rstrip("/"),
                "persona": persona,
            },
            "observed_provenance": observed_provenance,
            "bare_baseline": baseline,
        },
        "summary": summary,
        "output": None,
    }
    if summary["counts"]["failed"] or summary["counts"]["coverage"] != 1.0:
        atomic_write_json(spec.output_dir / "online_gate_manifest.json", manifest)
        return manifest

    contributions = estimate_online_contributions(
        pool=pool,
        assignments=assignments,
        traces=traces,
        bare_traces=bare_traces,
        active_skill_budget=spec.active_skill_budget,
    )
    contributions["bare_baseline"] = baseline
    selected = selected_skillbank(pool, contributions)
    ledger_rows = _proposal_ledger_rows(
        pool,
        contributions,
        _read_proposal_ledger(
            spec.candidate_pool_path.with_name("proposal_ledger.jsonl")
        ),
    )
    atomic_write_json(spec.output_dir / "contributions.json", contributions)
    atomic_write_json(spec.output_dir / "selected_skillbank.json", selected)
    _atomic_write_text(
        spec.output_dir / "proposal_ledger.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in ledger_rows
        ),
    )
    selected_text = "\n\n".join(record["content"] for record in selected["skills"])
    _atomic_write_text(
        spec.output_dir / "selected_skill.md",
        "# Shopping Skill\n\n> Active skill frozen by online randomized validation.\n\n"
        + (selected_text or "_No positive survivor._")
        + "\n",
    )
    manifest.update(
        {
            "status": "complete",
            "output": {
                "contributions_sha256": fingerprint(contributions),
                "selected_skillbank_sha256": fingerprint(selected),
                "proposal_ledger_sha256": fingerprint(ledger_rows),
                "selected_chunk_ids": [
                    record["skill_id"] for record in selected["skills"]
                ],
            },
        }
    )
    atomic_write_json(spec.output_dir / "online_gate_manifest.json", manifest)
    return manifest
