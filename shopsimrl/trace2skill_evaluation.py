"""Reusable randomized skill validation and cold-start Gate A orchestration."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .config import ExperimentSpec
from .environment import ShopSimulatorConfig, ShopSimulatorHTTPEnvironment
from .evaluation import EvaluationPlan, Evaluator, build_jobs
from .model import OpenAICompatibleChatModel
from .prompts import DEFAULT_SYSTEM_PROMPT, ShoppingPromptBuilder
from .runtime import ACTION_PROTOCOL_VERSION, AgentRuntime, RuntimeConfig
from .schemas import TRACE_SCHEMA_VERSION, EpisodeJob, fingerprint, utc_now
from .skills import AssignedSkillProvider, JsonSkillBank, SkillProvider
from .store import RunStore, atomic_write_json
from .tasks import TaskSplit, load_task_split
from .trace2skill_evaluation_config import GateASpec


GATE_A_SCHEMA_VERSION = "shopsimrl-trace2skill-gate-a-v1"
MASK_ASSIGNMENT_SCHEMA_VERSION = "shopsimrl-chunk-mask-assignments-v1"
CONTRIBUTION_SCHEMA_VERSION = "shopsimrl-chunk-contributions-v1"
SELECTED_SKILLBANK_VERSION = "cold-start-gate-a-selected-v1"
NUMERICAL_ZERO_TOLERANCE = 1e-12

Progress = Callable[[int, int, EpisodeJob, dict[str, Any]], None]
RuntimeFactory = Callable[[SkillProvider], AgentRuntime]


class GateEvaluationError(RuntimeError):
    pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_completed(trace: dict[str, Any]) -> bool:
    return bool(
        trace.get("schema_version") == TRACE_SCHEMA_VERSION
        and trace.get("status") == "completed"
        and isinstance(trace.get("final"), dict)
        and trace["final"].get("done") is True
    )


def _observed_provenance(traces: Iterable[dict[str, Any]]) -> dict[str, Any]:
    unique: dict[str, dict[str, Any]] = {
        "model": {},
        "environment": {},
        "prompt": {},
        "runtime": {},
    }
    for trace in traces:
        provenance = trace.get("provenance") or {}
        if not isinstance(provenance, dict):
            continue
        for field in unique:
            value = provenance.get(field)
            if isinstance(value, dict):
                unique[field][fingerprint(value)] = value
    return {
        field: [values[key] for key in sorted(values)]
        for field, values in unique.items()
    }


def _resolve_persona(spec: ExperimentSpec, task_split: TaskSplit) -> bool:
    if spec.environment_persona is None:
        if task_split.persona is None:
            raise ValueError(
                "environment.persona must be set for a mixed/unspecified task split"
            )
        return task_split.persona
    if task_split.persona is not None and spec.environment_persona != task_split.persona:
        raise ValueError(
            f"environment.persona={spec.environment_persona} conflicts with "
            f"split {task_split.name!r} persona={task_split.persona}"
        )
    return spec.environment_persona


def _prepare_experiment(
    spec: ExperimentSpec, *, required_split: str
) -> tuple[TaskSplit, tuple[EpisodeJob, ...], bool, ShoppingPromptBuilder]:
    if spec.split != required_split:
        raise GateEvaluationError(
            f"expected {required_split!r} experiment split, got {spec.split!r}"
        )
    if spec.repeats != 1:
        raise GateEvaluationError("cold-start gates require exactly one rollout per task")
    if spec.sample_size is not None:
        raise GateEvaluationError("cold-start gates require the complete frozen split")
    if spec.skillbank_path is not None:
        raise GateEvaluationError(
            "referenced experiment config must be bare; gates own skill injection"
        )
    task_split = load_task_split(spec.split_file, spec.split)
    persona = _resolve_persona(spec, task_split)
    jobs = build_jobs(
        EvaluationPlan(
            split=task_split.name,
            task_ids=task_split.task_ids,
            seed=spec.seed,
            sample_size=None,
            repeats=1,
        )
    )
    prompt_builder = ShoppingPromptBuilder(
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT
    )
    return task_split, jobs, persona, prompt_builder


def _semantic_plan(
    *,
    experiment_name: str,
    spec: ExperimentSpec,
    task_split: TaskSplit,
    jobs: Sequence[EpisodeJob],
    persona: bool,
    prompt_builder: ShoppingPromptBuilder,
    skill_provider: SkillProvider,
) -> dict[str, Any]:
    environment = ShopSimulatorConfig(
        base_url=spec.environment_base_url,
        persona=persona,
        timeout=spec.environment_timeout,
    )
    return {
        "pipeline_version": "shopsimrl-v0.4",
        "experiment": experiment_name,
        "model_id": spec.model.model_id,
        "model": spec.model.config.identity(),
        "task_split": task_split.identity(),
        "jobs": [job.to_dict() for job in jobs],
        "environment": environment.identity(),
        "runtime": {
            "max_steps": spec.max_steps,
            "concurrency": spec.concurrency,
            "action_protocol": ACTION_PROTOCOL_VERSION,
        },
        "prompt": prompt_builder.identity(),
        "skills": skill_provider.identity(),
    }


def _default_runtime_factory(
    *,
    spec: ExperimentSpec,
    persona: bool,
    prompt_builder: ShoppingPromptBuilder,
    skill_provider: SkillProvider,
) -> Callable[[], AgentRuntime]:
    environment_config = ShopSimulatorConfig(
        base_url=spec.environment_base_url,
        persona=persona,
        timeout=spec.environment_timeout,
    )

    def factory() -> AgentRuntime:
        return AgentRuntime(
            model=OpenAICompatibleChatModel(spec.model.config),
            environment=ShopSimulatorHTTPEnvironment(environment_config),
            prompt_builder=prompt_builder,
            skill_provider=skill_provider,
            config=RuntimeConfig(max_steps=spec.max_steps),
        )

    return factory


def _load_draft_bundle(
    spec: GateASpec,
) -> tuple[dict[str, Any], JsonSkillBank, tuple[str, ...], int, str]:
    if not spec.draft_path.is_file():
        raise FileNotFoundError(f"initial skill draft not found: {spec.draft_path}")
    if not spec.draft_skillbank_path.is_file():
        raise FileNotFoundError(
            f"initial draft SkillBank not found: {spec.draft_skillbank_path}"
        )
    draft = json.loads(spec.draft_path.read_text(encoding="utf-8"))
    if draft.get("schema_version") != "shopsimrl-initial-skill-draft-v2":
        raise GateEvaluationError("Gate A requires an initial v2 canonical skill draft")
    chunks = draft.get("chunks")
    if not isinstance(chunks, list) or not chunks or len(chunks) > 16:
        raise GateEvaluationError("Gate A requires between 1 and 16 canonical chunks")
    chunk_ids = tuple(chunk.get("chunk_id") for chunk in chunks)
    if any(not isinstance(value, str) or not value for value in chunk_ids):
        raise GateEvaluationError("draft contains an invalid chunk_id")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise GateEvaluationError("draft contains duplicate chunk IDs")
    orders = [chunk.get("order") for chunk in chunks]
    if orders != list(range(1, len(chunks) + 1)):
        raise GateEvaluationError("draft chunk order must be contiguous and stable")

    bank = JsonSkillBank(spec.draft_skillbank_path, max_skills=None)
    bank_ids = tuple(record["skill_id"] for record in bank.records)
    if bank_ids != chunk_ids:
        raise GateEvaluationError(
            "draft SkillBank IDs/order do not match initial_skill_draft.json"
        )
    for record in bank.records:
        metadata = record.get("metadata") or {}
        if metadata.get("draft_only") is not True:
            raise GateEvaluationError(
                f"draft SkillBank record {record['skill_id']!r} is not draft_only"
            )

    draft_budget = draft.get("active_skill_budget")
    budget = spec.active_skill_budget
    if budget is None:
        if isinstance(draft_budget, bool) or not isinstance(draft_budget, int):
            raise GateEvaluationError("draft has no valid active_skill_budget")
        budget = draft_budget
    elif draft_budget is not None and budget != draft_budget:
        raise GateEvaluationError(
            "configured Gate A budget differs from the frozen compiler budget"
        )
    if not 1 <= budget <= len(chunk_ids):
        raise GateEvaluationError(
            f"active skill budget must be in [1, {len(chunk_ids)}]"
        )
    return draft, bank, chunk_ids, budget, fingerprint(draft)


def _mask_bit(
    *, mask_seed: int, job: EpisodeJob, chunk_id: str, probability: float
) -> int:
    digest = hashlib.sha256(
        (
            f"{mask_seed}:{job.split}:{job.task_id}:"
            f"{job.sample_id}:{chunk_id}"
        ).encode("utf-8")
    ).digest()
    draw = int.from_bytes(digest[:8], "big")
    return int(draw < int(probability * (1 << 64)))


def build_mask_assignments(
    jobs: Sequence[EpisodeJob],
    chunk_ids: Sequence[str],
    *,
    mask_seed: int,
    probability: float = 0.5,
    draft_sha256: str | None = None,
) -> dict[str, Any]:
    if probability != 0.5:
        raise ValueError("cold-start Gate A mask probability must be 0.5")
    if not jobs or not chunk_ids:
        raise ValueError("jobs and chunk_ids cannot be empty")
    assignments = []
    for job in jobs:
        mask = [
            _mask_bit(
                mask_seed=mask_seed,
                job=job,
                chunk_id=chunk_id,
                probability=probability,
            )
            for chunk_id in chunk_ids
        ]
        assignments.append(
            {
                "episode_id": job.episode_id,
                **job.to_dict(),
                "mask": mask,
                "included_chunk_ids": [
                    chunk_id
                    for chunk_id, included in zip(chunk_ids, mask)
                    if included
                ],
            }
        )
    core = {
        "mask_seed": mask_seed,
        "mask_probability": probability,
        "draft_sha256": draft_sha256,
        "chunk_ids": list(chunk_ids),
        "assignments": assignments,
    }
    return {
        "schema_version": MASK_ASSIGNMENT_SCHEMA_VERSION,
        "created_at": utc_now(),
        **core,
        "assignment_sha256": fingerprint(core),
    }


def _assigned_provider(
    bank: JsonSkillBank, assignment_payload: dict[str, Any]
) -> AssignedSkillProvider:
    mapping = {
        (record["split"], record["task_id"], record["sample_id"]): tuple(
            record["included_chunk_ids"]
        )
        for record in assignment_payload["assignments"]
    }
    identity = {
        "schema_version": assignment_payload["schema_version"],
        "mask_seed": assignment_payload["mask_seed"],
        "mask_probability": assignment_payload["mask_probability"],
        "draft_sha256": assignment_payload["draft_sha256"],
        "assignment_sha256": assignment_payload["assignment_sha256"],
        "episodes": len(assignment_payload["assignments"]),
    }
    return AssignedSkillProvider(bank, mapping, assignment_identity=identity)


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    size = len(rhs)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    scale = max((abs(value) for row in matrix for value in row), default=1.0)
    tolerance = max(1.0, scale) * 1e-12
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= tolerance:
            raise GateEvaluationError(
                "randomized mask design is rank deficient; use a different mask seed"
            )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] /= pivot_value
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            for entry in range(column, size + 1):
                augmented[row][entry] -= factor * augmented[column][entry]
    return [augmented[index][-1] for index in range(size)]


def fit_main_effect_ols(
    masks: Sequence[Sequence[int]], values: Sequence[float]
) -> tuple[float, tuple[float, ...]]:
    """Fit an intercept plus mask main effects without optional dependencies."""

    if not masks or len(masks) != len(values):
        raise ValueError("masks and values must have equal non-zero length")
    width = len(masks[0])
    if width < 1 or any(len(row) != width for row in masks):
        raise ValueError("mask matrix must be rectangular and non-empty")
    if len(masks) <= width:
        raise GateEvaluationError("OLS requires more observations than chunk factors")
    design = []
    for row in masks:
        if any(value not in (0, 1) for value in row):
            raise ValueError("mask values must be binary")
        design.append([1.0, *(float(value) for value in row)])
    numeric_values = []
    for value in values:
        if not _is_number(value) or not math.isfinite(float(value)):
            raise ValueError("OLS outcomes must be finite numbers")
        numeric_values.append(float(value))

    columns = width + 1
    xtx = [[0.0 for _ in range(columns)] for _ in range(columns)]
    xty = [0.0 for _ in range(columns)]
    for row, outcome in zip(design, numeric_values):
        for left in range(columns):
            xty[left] += row[left] * outcome
            for right in range(columns):
                xtx[left][right] += row[left] * row[right]
    coefficients = _solve_linear_system(xtx, xty)
    coefficients = [
        0.0 if abs(value) <= NUMERICAL_ZERO_TOLERANCE else value
        for value in coefficients
    ]
    return coefficients[0], tuple(coefficients[1:])


def _validate_mask_design(assignments: dict[str, Any]) -> None:
    masks = [record["mask"] for record in assignments["assignments"]]
    fit_main_effect_ols(masks, [0.0] * len(masks))


def _assignment_trace_observations(
    assignments: dict[str, Any], traces: Iterable[dict[str, Any]]
) -> tuple[list[list[int]], dict[str, list[float]], list[dict[str, Any]]]:
    trace_map = {trace.get("episode_id"): trace for trace in traces}
    expected_ids = {record["episode_id"] for record in assignments["assignments"]}
    if set(trace_map) != expected_ids:
        missing = sorted(expected_ids - set(trace_map))
        extra = sorted(set(trace_map) - expected_ids)
        raise GateEvaluationError(
            f"Gate A trace coverage mismatch; missing={missing[:5]} extra={extra[:5]}"
        )

    masks: list[list[int]] = []
    rows: list[dict[str, Any]] = []
    reward_details: list[dict[str, Any]] = []
    rewards: list[float] = []
    for assignment in assignments["assignments"]:
        trace = trace_map[assignment["episode_id"]]
        if not _is_completed(trace):
            raise GateEvaluationError(
                f"Gate A trace {assignment['episode_id']} is not completed"
            )
        selected_ids = [item.get("skill_id") for item in trace.get("selected_skills", [])]
        if selected_ids != assignment["included_chunk_ids"]:
            raise GateEvaluationError(
                f"Gate A trace {assignment['episode_id']} treatment does not match "
                "the frozen mask assignment"
            )
        final = trace["final"]
        reward = final.get("reward")
        if not _is_number(reward):
            raise GateEvaluationError(
                f"Gate A trace {assignment['episode_id']} has no numeric reward"
            )
        detail = final.get("reward_detail")
        if not isinstance(detail, dict):
            raise GateEvaluationError(
                f"Gate A trace {assignment['episode_id']} has no reward_detail"
            )
        for required in ("r_strict", "r_success"):
            if not _is_number(detail.get(required)):
                raise GateEvaluationError(
                    f"Gate A trace {assignment['episode_id']} has no numeric {required}"
                )
        if not math.isclose(
            float(reward), float(detail["r_strict"]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise GateEvaluationError(
                f"Gate A trace {assignment['episode_id']} reward != r_strict"
            )
        masks.append(list(assignment["mask"]))
        rewards.append(float(reward))
        reward_details.append(detail)
        rows.append(
            {
                "episode_id": assignment["episode_id"],
                "task_id": assignment["task_id"],
                "sample_id": assignment["sample_id"],
                "mask": list(assignment["mask"]),
                "included_chunk_ids": list(assignment["included_chunk_ids"]),
                "reward": float(reward),
                "reward_detail": {
                    key: float(value)
                    for key, value in sorted(detail.items())
                    if key.startswith("r_") and _is_number(value)
                },
            }
        )

    component_names = sorted(
        {
            key
            for detail in reward_details
            for key, value in detail.items()
            if key.startswith("r_") and _is_number(value)
        }
    )
    outcomes: dict[str, list[float]] = {"reward": rewards}
    for name in component_names:
        if all(_is_number(detail.get(name)) for detail in reward_details):
            outcomes[name] = [float(detail[name]) for detail in reward_details]
    return masks, outcomes, rows


def estimate_gate_a_contributions(
    *,
    draft: dict[str, Any],
    assignments: dict[str, Any],
    traces: Iterable[dict[str, Any]],
    active_skill_budget: int,
) -> dict[str, Any]:
    chunk_ids = tuple(assignments["chunk_ids"])
    masks, outcomes, observation_rows = _assignment_trace_observations(
        assignments, traces
    )
    estimates: dict[str, dict[str, Any]] = {}
    for metric, values in outcomes.items():
        intercept, coefficients = fit_main_effect_ols(masks, values)
        estimates[metric] = {
            "intercept": intercept,
            "coefficients": dict(zip(chunk_ids, coefficients)),
            "observations": len(values),
        }

    primary = estimates["reward"]["coefficients"]
    ranked_ids = sorted(chunk_ids, key=lambda chunk_id: (-primary[chunk_id], chunk_id))
    ranks = {chunk_id: index for index, chunk_id in enumerate(ranked_ids, 1)}
    positive = [chunk_id for chunk_id in ranked_ids if primary[chunk_id] > 0.0]
    selected_ids = set(positive[:active_skill_budget])
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in draft["chunks"]}
    contributions = []
    for chunk_id in chunk_ids:
        coefficient = primary[chunk_id]
        if chunk_id in selected_ids:
            status = "selected"
        elif coefficient <= 0.0:
            status = "non_positive"
        else:
            status = "budget_excluded"
        contributions.append(
            {
                "chunk_id": chunk_id,
                "draft_order": chunk_by_id[chunk_id]["order"],
                "title": chunk_by_id[chunk_id]["title"],
                "coefficient": coefficient,
                "rank": ranks[chunk_id],
                "status": status,
                "auxiliary_coefficients": {
                    metric: record["coefficients"][chunk_id]
                    for metric, record in estimates.items()
                    if metric != "reward"
                },
            }
        )
    selected_in_injection_order = [
        chunk_id for chunk_id in chunk_ids if chunk_id in selected_ids
    ]
    return {
        "schema_version": CONTRIBUTION_SCHEMA_VERSION,
        "created_at": utc_now(),
        "estimator": {
            "name": "naive_ols_main_effects_with_intercept",
            "primary_outcome": "reward",
            "selection_rule": "strictly_positive_top_k",
            "numerical_zero_tolerance": NUMERICAL_ZERO_TOLERANCE,
        },
        "draft_sha256": assignments["draft_sha256"],
        "assignment_sha256": assignments["assignment_sha256"],
        "mask_seed": assignments["mask_seed"],
        "mask_probability": assignments["mask_probability"],
        "observations": len(observation_rows),
        "active_skill_budget": active_skill_budget,
        "selected_count": len(selected_in_injection_order),
        "selected_chunk_ids": selected_in_injection_order,
        "estimates": estimates,
        "contributions": contributions,
        "observations_table": observation_rows,
    }


def _write_contribution_csv(path: Path, payload: dict[str, Any]) -> None:
    metrics = [
        metric for metric in payload["estimates"] if metric != "reward"
    ]
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "chunk_id",
                "draft_order",
                "title",
                "coefficient_reward",
                *(f"coefficient_{metric}" for metric in metrics),
                "rank",
                "status",
            ],
        )
        writer.writeheader()
        for row in payload["contributions"]:
            writer.writerow(
                {
                    "chunk_id": row["chunk_id"],
                    "draft_order": row["draft_order"],
                    "title": row["title"],
                    "coefficient_reward": row["coefficient"],
                    **{
                        f"coefficient_{metric}": row["auxiliary_coefficients"].get(
                            metric
                        )
                        for metric in metrics
                    },
                    "rank": row["rank"],
                    "status": row["status"],
                }
            )
    os.replace(temporary, path)


def _selected_skillbank(
    *,
    draft: dict[str, Any],
    draft_bank: JsonSkillBank,
    contributions: dict[str, Any],
) -> dict[str, Any]:
    contribution_by_id = {
        row["chunk_id"]: row for row in contributions["contributions"]
    }
    selected_ids = set(contributions["selected_chunk_ids"])
    records = []
    for record in draft_bank.records:
        skill_id = record["skill_id"]
        if skill_id not in selected_ids:
            continue
        result = contribution_by_id[skill_id]
        metadata = dict(record.get("metadata") or {})
        metadata.pop("draft_only", None)
        metadata.update(
            {
                "validation_result": "selected",
                "estimated_effect": result["coefficient"],
                "contribution_rank": result["rank"],
                "draft_order": result["draft_order"],
            }
        )
        records.append(
            {
                "skill_id": skill_id,
                "version": "active-1",
                "enabled": True,
                "content": record["content"],
                "metadata": metadata,
            }
        )
    return {
        "schema_version": "shopsimrl-skillbank-v1",
        "bank_version": SELECTED_SKILLBANK_VERSION,
        "metadata": {
            "skill_title": draft.get("skill_title", "Shopping Skill"),
            "gate_status": {"gate_a": "passed", "gate_b": "not_run"},
            "draft_sha256": contributions["draft_sha256"],
            "assignment_sha256": contributions["assignment_sha256"],
            "active_skill_budget": contributions["active_skill_budget"],
            "selected_count": contributions["selected_count"],
            "selected_chunk_ids": list(contributions["selected_chunk_ids"]),
            "injection_order": "canonical_draft_order",
            "estimator": contributions["estimator"],
        },
        "skills": records,
    }


def _render_selected_skill(bank: dict[str, Any]) -> str:
    title = bank.get("metadata", {}).get("skill_title", "Shopping Skill")
    if not bank["skills"]:
        body = "_Gate A found no strictly positive chunks._"
    else:
        body = "\n\n".join(record["content"] for record in bank["skills"])
    return (
        f"# {title}\n\n"
        "> Active cold-start skill frozen by Gate A. Gate B is evaluation-only.\n\n"
        f"{body}\n"
    )


def _write_gate_a_ledger(
    *,
    source_path: Path,
    output_path: Path,
    contributions: dict[str, Any],
) -> None:
    original: dict[str, dict[str, Any]] = {}
    if source_path.is_file():
        for line_number, raw_line in enumerate(
            source_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise GateEvaluationError(
                    f"invalid proposal ledger JSON at line {line_number}"
                ) from exc
            candidate_id = record.get("candidate_id")
            if isinstance(candidate_id, str):
                original[candidate_id] = record
    lines = []
    for result in contributions["contributions"]:
        chunk_id = result["chunk_id"]
        record = dict(original.get(chunk_id) or {})
        record.setdefault("candidate_id", chunk_id)
        record.setdefault("operation", "ADD")
        record.setdefault("stage", "cold_start")
        record.update(
            {
                "validation_result": result["status"],
                "estimated_effect": result["coefficient"],
                "contribution_rank": result["rank"],
                "validation_assignment_sha256": contributions[
                    "assignment_sha256"
                ],
            }
        )
        lines.append(json.dumps(record, ensure_ascii=False, allow_nan=False))
    _atomic_write_text(output_path, "\n".join(lines) + "\n")


def gate_a_plan(spec: GateASpec) -> dict[str, Any]:
    draft, bank, chunk_ids, budget, draft_sha256 = _load_draft_bundle(spec)
    task_split, jobs, persona, prompt_builder = _prepare_experiment(
        spec.experiment, required_split="val"
    )
    assignments = build_mask_assignments(
        jobs,
        chunk_ids,
        mask_seed=spec.mask_seed,
        probability=spec.mask_probability,
        draft_sha256=draft_sha256,
    )
    _validate_mask_design(assignments)
    included_counts = {
        chunk_id: sum(
            record["mask"][index]
            for record in assignments["assignments"]
        )
        for index, chunk_id in enumerate(chunk_ids)
    }
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
    return {
        "gate": "A",
        "output_dir": str(spec.output_dir),
        "draft_path": str(spec.draft_path),
        "draft_sha256": draft_sha256,
        "chunks": len(chunk_ids),
        "active_skill_budget": budget,
        "tasks": len(jobs),
        "split": task_split.identity(),
        "persona": persona,
        "mask_seed": spec.mask_seed,
        "mask_probability": spec.mask_probability,
        "assignment_sha256": assignments["assignment_sha256"],
        "included_counts": included_counts,
        "design_matrix": "full_rank",
        "semantic_plan_fingerprint": fingerprint(semantic_plan),
        "model": spec.experiment.model.config.identity(),
    }


def _gate_a_manifest(
    *,
    spec: GateASpec,
    status: str,
    draft_sha256: str,
    assignments: dict[str, Any],
    summary: dict[str, Any],
    observed_provenance: dict[str, Any],
    contributions: dict[str, Any] | None = None,
    selected_bank: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": GATE_A_SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": status,
        "config": json.loads(json.dumps(asdict(spec), default=str)),
        "protocol": {
            "split": "val",
            "one_rollout_per_task": True,
            "mask_probability": spec.mask_probability,
            "mask_seed": spec.mask_seed,
            "estimator": "naive_ols_main_effects_with_intercept",
            "selection": "strictly_positive_top_k",
        },
        "input": {
            "draft_sha256": draft_sha256,
            "assignment_sha256": assignments["assignment_sha256"],
            "task_ids": [
                record["task_id"] for record in assignments["assignments"]
            ],
            "model": spec.experiment.model.config.identity(),
            "environment": {
                "base_url": spec.experiment.environment_base_url.rstrip("/"),
                "persona": spec.experiment.environment_persona,
            },
            "observed_provenance": observed_provenance,
        },
        "summary": summary,
        "output": {
            "contributions_sha256": (
                fingerprint(contributions) if contributions is not None else None
            ),
            "selected_skillbank_sha256": (
                fingerprint(selected_bank) if selected_bank is not None else None
            ),
            "selected_chunk_ids": (
                list(contributions["selected_chunk_ids"])
                if contributions is not None
                else None
            ),
        },
    }


def run_gate_a(
    spec: GateASpec,
    *,
    runtime_factory: RuntimeFactory | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    draft, bank, chunk_ids, budget, draft_sha256 = _load_draft_bundle(spec)
    task_split, jobs, persona, prompt_builder = _prepare_experiment(
        spec.experiment, required_split="val"
    )
    assignments = build_mask_assignments(
        jobs,
        chunk_ids,
        mask_seed=spec.mask_seed,
        probability=spec.mask_probability,
        draft_sha256=draft_sha256,
    )
    _validate_mask_design(assignments)
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
    semantic_plan["trace2skill_gate"] = {
        "gate": "A",
        "draft_sha256": draft_sha256,
        "assignment_sha256": assignments["assignment_sha256"],
        "mask_seed": spec.mask_seed,
        "mask_probability": spec.mask_probability,
        "active_skill_budget": budget,
    }
    store = RunStore(spec.output_dir)
    store.initialize(semantic_plan)
    atomic_write_json(spec.output_dir / "mask_assignments.json", assignments)

    if runtime_factory is None:
        factory = _default_runtime_factory(
            spec=spec.experiment,
            persona=persona,
            prompt_builder=prompt_builder,
            skill_provider=provider,
        )
    else:
        factory = lambda: runtime_factory(provider)
    summary = Evaluator(
        runtime_factory=factory,
        store=store,
        max_workers=spec.experiment.concurrency,
        progress=progress,
    ).run(jobs, resume=spec.experiment.resume)
    trace_records = list(store.iter_traces())
    observed_provenance = _observed_provenance(trace_records)
    if summary["counts"]["failed"] or summary["counts"]["coverage"] != 1.0:
        manifest = _gate_a_manifest(
            spec=spec,
            status="incomplete",
            draft_sha256=draft_sha256,
            assignments=assignments,
            summary=summary,
            observed_provenance=observed_provenance,
        )
        atomic_write_json(spec.output_dir / "gate_a_manifest.json", manifest)
        return manifest

    contributions = estimate_gate_a_contributions(
        draft=draft,
        assignments=assignments,
        traces=trace_records,
        active_skill_budget=budget,
    )
    selected_bank = _selected_skillbank(
        draft=draft,
        draft_bank=bank,
        contributions=contributions,
    )
    atomic_write_json(spec.output_dir / "contributions.json", contributions)
    _write_contribution_csv(spec.output_dir / "contributions.csv", contributions)
    atomic_write_json(spec.output_dir / "selected_skillbank.json", selected_bank)
    _atomic_write_text(
        spec.output_dir / "selected_skill.md", _render_selected_skill(selected_bank)
    )
    _write_gate_a_ledger(
        source_path=spec.draft_path.with_name("proposal_ledger.jsonl"),
        output_path=spec.output_dir / "proposal_ledger.jsonl",
        contributions=contributions,
    )
    manifest = _gate_a_manifest(
        spec=spec,
        status="complete",
        draft_sha256=draft_sha256,
        assignments=assignments,
        summary=summary,
        observed_provenance=observed_provenance,
        contributions=contributions,
        selected_bank=selected_bank,
    )
    atomic_write_json(spec.output_dir / "gate_a_manifest.json", manifest)
    return manifest
