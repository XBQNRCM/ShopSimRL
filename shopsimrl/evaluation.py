"""Deterministic sampling, concurrent evaluation and metric aggregation."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Callable, Iterable, Sequence

from .runtime import AgentRuntime
from .schemas import TRACE_SCHEMA_VERSION, EpisodeJob, utc_now
from .store import RunStore

DEFAULT_MAX_EPISODE_RETRIES = 2


@dataclass(frozen=True)
class EvaluationPlan:
    split: str
    task_ids: tuple[int, ...]
    seed: int
    sample_size: int | None = None
    repeats: int = 1

    def __post_init__(self) -> None:
        if not self.task_ids:
            raise ValueError("task_ids cannot be empty")
        if self.repeats < 1:
            raise ValueError("repeats must be positive")
        if self.sample_size is not None and not 0 < self.sample_size <= len(self.task_ids):
            raise ValueError(
                f"sample_size must be in [1, {len(self.task_ids)}]"
            )


def _episode_seed(seed: int, task_id: int, sample_id: int) -> int:
    digest = hashlib.sha256(
        f"{seed}:{task_id}:{sample_id}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:4], "big")


def build_jobs(plan: EvaluationPlan) -> tuple[EpisodeJob, ...]:
    candidates = list(plan.task_ids)
    if plan.sample_size is not None:
        selected = random.Random(plan.seed).sample(candidates, plan.sample_size)
    else:
        selected = candidates
    return tuple(
        EpisodeJob(
            task_id=task_id,
            sample_id=sample_id,
            seed=_episode_seed(plan.seed, task_id, sample_id),
            split=plan.split,
        )
        for task_id in selected
        for sample_id in range(plan.repeats)
    )


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _sem(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = fmean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def summarize_traces(
    traces: Iterable[dict[str, Any]], *, requested: int | None = None
) -> dict[str, Any]:
    records = list(traces)
    def is_completed(trace: dict[str, Any]) -> bool:
        return bool(
            trace.get("schema_version") == TRACE_SCHEMA_VERSION
        and trace.get("status") == "completed"
        and isinstance(trace.get("final"), dict)
        and trace["final"].get("done") is True
        )

    completed = [trace for trace in records if is_completed(trace)]
    failed = [trace for trace in records if not is_completed(trace)]
    rewards: list[float] = []
    step_counts: list[float] = []
    durations: list[float] = []
    component_values: dict[str, list[float]] = defaultdict(list)
    terminations: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    total_actions = 0
    invalid_actions = 0
    model_steps = 0
    protocol_errors = 0
    policy_failures = 0
    token_totals: Counter[str] = Counter()

    for trace in completed:
        final = trace["final"]
        reward = final.get("reward")
        if isinstance(reward, (int, float)) and not isinstance(reward, bool):
            rewards.append(float(reward))
        detail = final.get("reward_detail") or {}
        if isinstance(detail, dict):
            for key, value in detail.items():
                if (
                    key.startswith("r_")
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                ):
                    component_values[key].append(float(value))
        steps = trace.get("steps") or []
        step_counts.append(float(len(steps)))
        duration = trace.get("duration_ms")
        if isinstance(duration, (int, float)):
            durations.append(float(duration))
        reason = str(final.get("termination_reason", "unknown"))
        terminations[reason] += 1
        for step in steps:
            model_steps += 1
            if isinstance(step.get("protocol_error"), dict):
                protocol_errors += 1
            if isinstance(step.get("policy_failure"), dict):
                policy_failures += 1
            environment = step.get("environment")
            if isinstance(environment, dict):
                total_actions += 1
                feedback = environment.get("action_feedback") or {}
                if (
                    isinstance(feedback, dict)
                    and feedback
                    and feedback.get("valid") is False
                ):
                    invalid_actions += 1
            model = step.get("model") or {}
            usage = model.get("usage") or {}
            if isinstance(usage, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        token_totals[key] += value
                completion_details = usage.get("completion_tokens_details") or {}
                if isinstance(completion_details, dict):
                    reasoning_tokens = completion_details.get("reasoning_tokens")
                    if isinstance(reasoning_tokens, int) and not isinstance(
                        reasoning_tokens, bool
                    ):
                        token_totals["reasoning_tokens"] += reasoning_tokens

    for trace in failed:
        error = trace.get("error") or {}
        error_types[str(error.get("type", "unknown"))] += 1

    requested_count = requested if requested is not None else len(records)
    summary = {
        "schema_version": "shopsimrl-eval-summary-v2",
        "created_at": utc_now(),
        "counts": {
            "requested": requested_count,
            "trace_files": len(records),
            "completed": len(completed),
            "failed": len(failed),
            "scored": len(rewards),
            "coverage": len(completed) / requested_count if requested_count else 0.0,
        },
        "primary": {
            "reward_mean": _mean(rewards),
            "reward_sem": _sem(rewards),
            "success_rate": _mean(component_values.get("r_success", [])),
        },
        "reward_components": {
            key: _mean(values) for key, values in sorted(component_values.items())
        },
        "behavior": {
            "steps_mean": _mean(step_counts),
            "model_steps": model_steps,
            "protocol_errors": protocol_errors,
            "protocol_error_rate": (
                protocol_errors / model_steps if model_steps else None
            ),
            "policy_failures": policy_failures,
            "policy_failure_rate": (
                policy_failures / model_steps if model_steps else None
            ),
            "invalid_action_rate": (
                invalid_actions / total_actions if total_actions else None
            ),
            "invalid_actions": invalid_actions,
            "total_actions": total_actions,
            "termination_reasons": dict(sorted(terminations.items())),
        },
        "cost": {
            "duration_ms_mean": _mean(durations),
            "tokens": dict(sorted(token_totals.items())),
        },
        "errors": dict(sorted(error_types.items())),
    }
    return summary


class Evaluator:
    def __init__(
        self,
        *,
        runtime_factory: Callable[[], AgentRuntime],
        store: RunStore,
        max_workers: int = 1,
        progress: Callable[[int, int, EpisodeJob, dict[str, Any]], None] | None = None,
        max_episode_retries: int = DEFAULT_MAX_EPISODE_RETRIES,
    ):
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if isinstance(max_episode_retries, bool) or max_episode_retries < 0:
            raise ValueError("max_episode_retries cannot be negative")
        self.runtime_factory = runtime_factory
        self.store = store
        self.max_workers = max_workers
        self.progress = progress
        self.max_episode_retries = max_episode_retries

    def run(
        self, jobs: Sequence[EpisodeJob], *, resume: bool = True
    ) -> dict[str, Any]:
        del resume
        max_attempts = 1 + self.max_episode_retries
        while True:
            pending = [
                job
                for job in jobs
                if not self.store.is_complete(job)
                and self.store.attempt_count(job.episode_id) < max_attempts
            ]
            if not pending:
                break
            self._run_wave(pending)
        summary = summarize_traces(
            self.store.iter_traces(), requested=len(jobs)
        )
        self.store.write_summary(summary)
        return summary

    def _run_wave(self, pending: Sequence[EpisodeJob]) -> None:
        def run_one(job: EpisodeJob) -> dict[str, Any]:
            return self.runtime_factory().run(job)

        total_pending = len(pending)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_one, job): job for job in pending}
            for completed_count, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                try:
                    trace = future.result()
                except Exception as exc:  # defensive boundary around custom runtimes
                    trace = {
                        "schema_version": TRACE_SCHEMA_VERSION,
                        "episode_id": job.episode_id,
                        "status": "failed",
                        "job": job.to_dict(),
                        "provenance": {},
                        "started_at": None,
                        "ended_at": utc_now(),
                        "duration_ms": None,
                        "reset": None,
                        "selected_skills": [],
                        "conversation": [],
                        "steps": [],
                        "final": None,
                        "error": {
                            "stage": "evaluator_boundary",
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                self.store.save_trace(trace)
                if self.progress:
                    self.progress(completed_count, total_pending, job, trace)
