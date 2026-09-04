"""Shared task-paired validation data contract for cold-start and online gates.

Only the treatment differs: both rollouts use the same frozen checkpoint,
tasks, prompt, environment and sampling distribution. A bare run is read, never
generated here. Failed attempts may be resumed; multiple completed replicates
require a separately designed experiment, not a completion-order tie-break.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .schemas import TRACE_SCHEMA_VERSION, EpisodeJob, fingerprint


PAIRED_ESTIMATOR = "paired_delta_ols_main_effects_no_intercept"
PAIRED_FORMULA = "R_i - B_i = C_i^T beta + epsilon_i"


class GateEvaluationError(RuntimeError):
    pass


def completed(trace: dict[str, Any]) -> bool:
    return (
        trace.get("schema_version") == TRACE_SCHEMA_VERSION
        and trace.get("status") == "completed"
        and isinstance(trace.get("final"), dict)
        and trace["final"].get("done") is True
    )


def unique_traces(traces: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    completed_ids: set[str] = set()
    for trace in traces:
        episode_id = trace.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise GateEvaluationError("validation trace has no episode_id")
        if episode_id in completed_ids and not completed(trace):
            raise GateEvaluationError(f"validation attempt follows a completed trace: {episode_id}")
        if completed(trace):
            if episode_id in completed_ids:
                raise GateEvaluationError(f"duplicate completed validation trace: {episode_id}")
            completed_ids.add(episode_id)
        latest[episode_id] = trace
    return latest


def read_validation_traces(path: Path, *, resume: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                raise GateEvaluationError(f"invalid validation JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise GateEvaluationError(f"invalid validation record at {path}:{line_number}")
            records.append(record)
    latest = unique_traces(records)
    if not resume and any(completed(trace) for trace in latest.values()):
        raise GateEvaluationError("completed validation traces already exist; use resume or a new run directory")
    return [latest[key] for key in sorted(latest)]


def _model_identity(identity: dict[str, Any]) -> dict[str, Any]:
    # Retry/timeout/streaming and concurrency do not define the sampling policy.
    return {key: value for key, value in identity.items() if key != "transport"}


def _job_key(job: dict[str, Any], *, send_seed: bool) -> tuple[Any, ...]:
    fields = ("split", "task_id", "sample_id", "seed") if send_seed else ("split", "task_id", "sample_id")
    if any(key not in job for key in fields):
        raise GateEvaluationError("validation job is missing pairing fields")
    return tuple(job[key] for key in fields)


def _provenance(trace: dict[str, Any]) -> dict[str, Any]:
    provenance = trace.get("provenance")
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(field), dict) or not provenance[field]
        for field in ("model", "environment", "prompt", "runtime", "skills")
    ):
        raise GateEvaluationError(f"validation trace {trace.get('episode_id')} has incomplete provenance")
    return provenance


def _outcomes(trace: dict[str, Any]) -> dict[str, float]:
    if not completed(trace):
        raise GateEvaluationError(f"validation trace {trace.get('episode_id')} is not completed")
    final = trace["final"]
    detail = final.get("reward_detail")
    if not isinstance(detail, dict) or not {"r_strict", "r_success"} <= detail.keys():
        raise GateEvaluationError("validation reward_detail requires r_strict and r_success")
    values = {"reward": final.get("reward"), **{k: v for k, v in detail.items() if k.startswith("r_")}}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise GateEvaluationError(f"validation {name} must be a finite number")
    if not math.isclose(values["reward"], values["r_strict"], rel_tol=0, abs_tol=1e-12):
        raise GateEvaluationError("validation reward != r_strict")
    return {name: float(value) for name, value in sorted(values.items())}


def _require_bare(trace: dict[str, Any]) -> None:
    if trace.get("selected_skills") != [] or _provenance(trace)["skills"] != {"provider": "none"}:
        raise GateEvaluationError("bare validation must use NoSkills, not an equipped or empty masked run")


def _check_trace_plan(trace: dict[str, Any], plan: dict[str, Any]) -> None:
    actual = _provenance(trace)
    for field in ("model", "prompt", "runtime", "environment"):
        expected = dict(plan[field])
        observed = dict(actual[field])
        if field == "model":
            expected, observed = _model_identity(expected), _model_identity(observed)
        elif field == "runtime":
            expected.pop("concurrency", None)
        elif field == "environment":
            # Observed version fields are checked across all paired rollouts below.
            observed = {key: observed.get(key) for key in expected}
        if observed != expected:
            raise GateEvaluationError(f"validation trace/plan {field} mismatch")


def load_bare_validation(
    run_dir: Path, expected_plan: dict[str, Any], *, require_checkpoint: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail before rollouts if the existing baseline cannot be paired safely."""
    checkpoint = expected_plan["model"].get("checkpoint_id")
    if require_checkpoint and not checkpoint:
        raise GateEvaluationError("online validation requires model.checkpoint_id for the frozen evaluation weights")
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise GateEvaluationError(f"bare validation run missing: {run_dir}; complete the bare val run first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan = manifest.get("plan")
    if not isinstance(plan, dict) or manifest.get("plan_fingerprint") != fingerprint(plan):
        raise GateEvaluationError("bare validation manifest fingerprint mismatch")
    if plan.get("skills") != {"provider": "none"}:
        raise GateEvaluationError("bare validation manifest must use NoSkills")
    for field in ("model", "prompt", "runtime", "environment", "task_split"):
        left, right = dict(plan.get(field) or {}), dict(expected_plan[field])
        if field == "model":
            left, right = _model_identity(left), _model_identity(right)
        if field == "runtime":
            left.pop("concurrency", None)
            right.pop("concurrency", None)
        if field == "task_split":
            left.pop("source_path", None)
            right.pop("source_path", None)
        if left != right:
            raise GateEvaluationError(f"bare validation {field} mismatch; use the same checkpoint and evaluation protocol")
    send_seed = expected_plan["model"].get("sampling", {}).get("send_seed", True)
    expected_jobs = {_job_key(job, send_seed=send_seed) for job in expected_plan["jobs"]}
    bare_jobs = [_job_key(job, send_seed=send_seed) for job in plan.get("jobs", [])]
    if len(bare_jobs) != len(set(bare_jobs)) or set(bare_jobs) != expected_jobs:
        raise GateEvaluationError("bare validation job coverage mismatch (split/task/sample/seed)")
    traces = read_validation_traces(run_dir / "traces.jsonl")
    if len(traces) != len(expected_jobs) or {
        _job_key(trace.get("job") or {}, send_seed=send_seed) for trace in traces
    } != expected_jobs:
        raise GateEvaluationError("bare validation trace coverage mismatch; complete the bare val run first")
    metrics: dict[str, list[float]] = {}
    observed_environments = set()
    for trace in traces:
        if EpisodeJob(**trace["job"]).episode_id != trace["episode_id"]:
            raise GateEvaluationError("bare validation episode_id/job mismatch")
        _require_bare(trace)
        _check_trace_plan(trace, plan)
        values = _outcomes(trace)
        if metrics and values.keys() != metrics.keys():
            raise GateEvaluationError("bare validation reward component coverage mismatch")
        for key, value in values.items():
            metrics.setdefault(key, []).append(value)
        observed_environments.add(fingerprint(trace["provenance"]["environment"]))
    if len(observed_environments) != 1:
        raise GateEvaluationError("bare validation mixes environment versions")
    digest = fingerprint([
        {key: trace[key] for key in ("episode_id", "job", "provenance", "selected_skills", "final")}
        for trace in traces
    ])
    return traces, {
        "run_dir": str(run_dir.resolve()),
        "plan_fingerprint": manifest["plan_fingerprint"],
        "observations_sha256": digest,
        "observations": len(traces),
        "checkpoint_id": checkpoint,
        "mean_outcomes": {key: sum(values) / len(values) for key, values in metrics.items()},
    }


def paired_observations(
    assignments: dict[str, Any], traces: Iterable[dict[str, Any]],
    bare_traces: Iterable[dict[str, Any]],
) -> tuple[list[list[int]], dict[str, list[float]], list[dict[str, Any]]]:
    """Align by task/sample/split and subtract each task's own observed baseline."""
    masked, bare = unique_traces(traces), unique_traces(bare_traces)
    expected = {row["episode_id"] for row in assignments["assignments"]}
    if len(expected) != len(assignments["assignments"]) or set(masked) != expected or set(bare) != expected:
        raise GateEvaluationError("paired validation trace coverage mismatch")
    masks, rows, outcomes = [], [], {}
    reference = None
    for assignment in assignments["assignments"]:
        episode_id = assignment["episode_id"]
        treated, baseline = masked[episode_id], bare[episode_id]
        _require_bare(baseline)
        observed, control = _provenance(treated), _provenance(baseline)
        for field in ("model", "environment", "prompt", "runtime"):
            left, right = observed[field], control[field]
            if field == "model":
                left, right = _model_identity(left), _model_identity(right)
            if left != right:
                raise GateEvaluationError(f"paired validation {field} mismatch for {episode_id}")
        signature = fingerprint({k: (_model_identity(control[k]) if k == "model" else control[k])
                                 for k in ("model", "environment", "prompt", "runtime")})
        if reference is not None and signature != reference:
            raise GateEvaluationError("paired validation mixes checkpoint/evaluation protocols")
        reference = signature
        send_seed = control["model"].get("sampling", {}).get("send_seed", True)
        for trace in (treated, baseline):
            job = trace.get("job") or {}
            if job.get("split") != "val" or _job_key(job, send_seed=send_seed) != _job_key(assignment, send_seed=send_seed):
                raise GateEvaluationError(f"paired validation job mismatch for {episode_id}")
            if EpisodeJob(**job).episode_id != episode_id:
                raise GateEvaluationError(f"paired validation episode_id/job mismatch for {episode_id}")
        if [item.get("skill_id") for item in treated.get("selected_skills", [])] != assignment["included_chunk_ids"]:
            raise GateEvaluationError(f"validation treatment does not match frozen mask for {episode_id}")
        reward, base = _outcomes(treated), _outcomes(baseline)
        if reward.keys() != base.keys() or (outcomes and reward.keys() != outcomes.keys()):
            raise GateEvaluationError("paired validation reward component coverage mismatch")
        delta = {key: reward[key] - base[key] for key in reward}
        for key, value in delta.items():
            outcomes.setdefault(key, []).append(value)
        masks.append(list(assignment["mask"]))
        rows.append({
            **{key: assignment[key] for key in ("episode_id", "split", "task_id", "sample_id", "mask", "included_chunk_ids")},
            "reward": reward["reward"], "bare_reward": base["reward"], "delta_reward": delta["reward"],
            "reward_detail": {k: v for k, v in reward.items() if k != "reward"},
            "bare_reward_detail": {k: v for k, v in base.items() if k != "reward"},
            "delta_reward_detail": {k: v for k, v in delta.items() if k != "reward"},
        })
    return masks, outcomes, rows


def outcome_means(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    def value(row: dict[str, Any], prefix: str) -> float:
        return row[f"{prefix}reward"] if metric == "reward" else row[f"{prefix}reward_detail"][metric]
    return {
        name: sum(value(row, prefix) for row in rows) / len(rows)
        for name, prefix in (("masked_mean", ""), ("bare_mean", "bare_"), ("delta_mean", "delta_"))
    }
