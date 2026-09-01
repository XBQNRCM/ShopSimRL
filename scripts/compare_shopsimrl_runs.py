"""Compare two completed ShopSimRL runs with task-level paired bootstrap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean
import sys
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shopsimrl.evaluation import summarize_traces
from shopsimrl.schemas import TRACE_SCHEMA_VERSION, fingerprint, utc_now
from shopsimrl.store import RunStore, atomic_write_json


class ComparisonError(RuntimeError):
    pass


def _completed(trace: dict[str, Any]) -> bool:
    return bool(
        trace.get("schema_version") == TRACE_SCHEMA_VERSION
        and trace.get("status") == "completed"
        and isinstance(trace.get("final"), dict)
        and trace["final"].get("done") is True
    )


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"run manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("plan"), dict):
        raise ComparisonError(f"manifest has no semantic plan: {path}")
    return payload


def _alignment_view(plan: dict[str, Any]) -> dict[str, Any]:
    runtime = plan.get("runtime") or {}
    return {
        "pipeline_version": plan.get("pipeline_version"),
        "model_id": plan.get("model_id"),
        "model": plan.get("model"),
        "task_split": plan.get("task_split"),
        "jobs": plan.get("jobs"),
        "environment": plan.get("environment"),
        "runtime": {
            "max_steps": runtime.get("max_steps"),
            "action_protocol": runtime.get("action_protocol"),
        },
        "prompt": plan.get("prompt"),
    }


def validate_plan_alignment(
    bare_manifest: dict[str, Any], equipped_manifest: dict[str, Any]
) -> dict[str, Any]:
    bare_plan = bare_manifest["plan"]
    equipped_plan = equipped_manifest["plan"]
    if bare_plan.get("skills") != {"provider": "none"}:
        raise ComparisonError("the bare run manifest is not a no-skill run")
    if (equipped_plan.get("skills") or {}).get("provider") == "none":
        raise ComparisonError("the equipped run manifest has no skill provider")
    bare = _alignment_view(bare_plan)
    equipped = _alignment_view(equipped_plan)
    mismatched = [key for key in bare if bare[key] != equipped[key]]
    if mismatched:
        details = {
            key: {"bare": bare[key], "equipped": equipped[key]}
            for key in mismatched
        }
        raise ComparisonError(
            "runs differ outside skill injection: "
            + json.dumps(details, ensure_ascii=False, sort_keys=True)
        )
    return {
        "status": "matched",
        "checked_fields": list(bare),
        "bare_plan_fingerprint": bare_manifest.get("plan_fingerprint"),
        "equipped_plan_fingerprint": equipped_manifest.get("plan_fingerprint"),
    }


def _trace_key(trace: dict[str, Any]) -> tuple[str, int, int]:
    job = trace.get("job")
    if not isinstance(job, dict):
        raise ComparisonError(f"trace {trace.get('episode_id')} has no job")
    try:
        return str(job["split"]), int(job["task_id"]), int(job["sample_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError(
            f"trace {trace.get('episode_id')} has an invalid job"
        ) from exc


def _manifest_job_keys(manifest: dict[str, Any]) -> set[tuple[str, int, int]]:
    keys = set()
    for job in manifest["plan"].get("jobs") or []:
        try:
            keys.add((str(job["split"]), int(job["task_id"]), int(job["sample_id"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ComparisonError("manifest contains an invalid job") from exc
    if not keys:
        raise ComparisonError("manifest contains no jobs")
    return keys


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


def _validate_trace_alignment(
    bare_traces: Sequence[dict[str, Any]],
    equipped_traces: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    bare = {_trace_key(trace): trace for trace in bare_traces}
    equipped = {_trace_key(trace): trace for trace in equipped_traces}
    if set(bare) != set(equipped) or not bare:
        raise ComparisonError("bare and equipped task keys do not align")
    fields = ("model", "environment", "prompt", "runtime")
    expected_equipped_ids: list[str] | None = None
    for key in sorted(bare):
        if not _completed(bare[key]) or not _completed(equipped[key]):
            raise ComparisonError(f"task pair {key!r} is not fully completed")
        bare_ids = [
            record.get("skill_id") for record in bare[key].get("selected_skills", [])
        ]
        equipped_ids = [
            record.get("skill_id")
            for record in equipped[key].get("selected_skills", [])
        ]
        if bare_ids:
            raise ComparisonError(f"bare trace {key!r} unexpectedly contains skills")
        if expected_equipped_ids is None:
            expected_equipped_ids = equipped_ids
        elif equipped_ids != expected_equipped_ids:
            raise ComparisonError(
                f"equipped trace {key!r} did not receive the same frozen skill"
            )
        bare_provenance = bare[key].get("provenance") or {}
        equipped_provenance = equipped[key].get("provenance") or {}
        for field in fields:
            if bare_provenance.get(field) != equipped_provenance.get(field):
                raise ComparisonError(
                    f"runtime provenance mismatch for {key!r}: {field}"
                )
    return {
        "status": "matched",
        "pairs": len(bare),
        "checked_fields": list(fields),
        "equipped_chunk_ids": expected_equipped_ids or [],
        "bare": _observed_provenance(bare_traces),
        "equipped": _observed_provenance(equipped_traces),
    }


def _task_metrics(trace: dict[str, Any]) -> dict[str, float]:
    final = trace["final"]
    metrics: dict[str, float] = {}
    reward = final.get("reward")
    if isinstance(reward, (int, float)) and not isinstance(reward, bool):
        metrics["reward"] = float(reward)
    detail = final.get("reward_detail") or {}
    if isinstance(detail, dict):
        for key, value in detail.items():
            if (
                key.startswith("r_")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                metrics[key] = float(value)

    steps = trace.get("steps") or []
    metrics["steps"] = float(len(steps))
    model_steps = len(steps)
    protocol_errors = sum(
        isinstance(step.get("protocol_error"), dict) for step in steps
    )
    policy_failures = sum(
        isinstance(step.get("policy_failure"), dict) for step in steps
    )
    metrics["protocol_error_rate"] = (
        protocol_errors / model_steps if model_steps else 0.0
    )
    metrics["policy_failure_rate"] = (
        policy_failures / model_steps if model_steps else 0.0
    )
    actions = [
        step["environment"]
        for step in steps
        if isinstance(step.get("environment"), dict)
    ]
    invalid_actions = sum(
        isinstance(action.get("action_feedback"), dict)
        and action["action_feedback"].get("valid") is False
        for action in actions
    )
    metrics["invalid_action_rate"] = (
        invalid_actions / len(actions) if actions else 0.0
    )

    token_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }
    for step in steps:
        usage = (step.get("model") or {}).get("usage") or {}
        if not isinstance(usage, dict):
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                token_totals[key] += value
        details = usage.get("completion_tokens_details") or {}
        reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
        if isinstance(reasoning, int) and not isinstance(reasoning, bool):
            token_totals["reasoning_tokens"] += reasoning
    metrics.update({key: float(value) for key, value in token_totals.items()})
    duration = trace.get("duration_ms")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        metrics["duration_ms"] = float(duration)
    return metrics


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    samples: int,
    confidence_level: float,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    size = len(differences)
    means = sorted(
        sum(differences[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(samples)
    )
    tail = (1.0 - confidence_level) / 2.0
    return [_percentile(means, tail), _percentile(means, 1.0 - tail)]


def paired_comparison(
    bare_traces: Sequence[dict[str, Any]],
    equipped_traces: Sequence[dict[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    bare = {_trace_key(trace): trace for trace in bare_traces}
    equipped = {_trace_key(trace): trace for trace in equipped_traces}
    keys = sorted(bare)
    bare_metrics = {key: _task_metrics(bare[key]) for key in keys}
    equipped_metrics = {key: _task_metrics(equipped[key]) for key in keys}
    metric_names = sorted(
        set.intersection(
            *(set(bare_metrics[key]) & set(equipped_metrics[key]) for key in keys)
        )
    )
    for required in ("reward", "r_strict", "r_success"):
        if required not in metric_names:
            raise ComparisonError(f"cannot compare required metric {required}")

    metrics = {}
    for metric in metric_names:
        bare_values = [bare_metrics[key][metric] for key in keys]
        equipped_values = [equipped_metrics[key][metric] for key in keys]
        differences = [
            equipped_value - bare_value
            for bare_value, equipped_value in zip(bare_values, equipped_values)
        ]
        metric_seed = int.from_bytes(
            hashlib.sha256(f"{bootstrap_seed}:{metric}".encode("utf-8")).digest()[:8],
            "big",
        )
        metrics[metric] = {
            "bare_mean": fmean(bare_values),
            "equipped_mean": fmean(equipped_values),
            "mean_difference": fmean(differences),
            "confidence_interval": _bootstrap_mean_ci(
                differences,
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=metric_seed,
            ),
            "pairs": len(keys),
        }
    return {
        "pairs": len(keys),
        "difference_direction": "equipped_minus_bare",
        "metric_aggregation": "mean_of_task_level_values",
        "bootstrap": {
            "method": "task_level_paired_percentile",
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "confidence_level": confidence_level,
        },
        "metrics": metrics,
    }


def compare_runs(
    bare_run: str | Path,
    equipped_run: str | Path,
    *,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260901,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    bare_dir = Path(bare_run).resolve()
    equipped_dir = Path(equipped_run).resolve()
    bare_manifest = _load_manifest(bare_dir)
    equipped_manifest = _load_manifest(equipped_dir)
    plan_alignment = validate_plan_alignment(bare_manifest, equipped_manifest)
    bare_traces = list(RunStore(bare_dir).iter_traces())
    equipped_traces = list(RunStore(equipped_dir).iter_traces())
    expected_keys = _manifest_job_keys(bare_manifest)
    for label, traces in (("bare", bare_traces), ("equipped", equipped_traces)):
        actual_keys = {_trace_key(trace) for trace in traces}
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ComparisonError(
                f"{label} run trace coverage mismatch; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
    trace_alignment = _validate_trace_alignment(bare_traces, equipped_traces)
    paired = paired_comparison(
        bare_traces,
        equipped_traces,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return {
        "schema_version": "shopsimrl-paired-run-comparison-v1",
        "created_at": utc_now(),
        "runs": {"bare": str(bare_dir), "equipped": str(equipped_dir)},
        "alignment": {"plan": plan_alignment, "traces": trace_alignment},
        "absolute": {
            "bare": summarize_traces(bare_traces, requested=len(expected_keys)),
            "equipped": summarize_traces(
                equipped_traces, requested=len(expected_keys)
            ),
        },
        "paired": paired,
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    metrics = comparison["paired"]["metrics"]
    preferred = [
        "reward",
        "r_strict",
        "r_success",
        "steps",
        "invalid_action_rate",
        "protocol_error_rate",
        "total_tokens",
    ]
    lines = [
        "# ShopSimRL paired run comparison",
        "",
        "Difference direction: equipped minus bare. Confidence intervals use "
        "task-level paired percentile bootstrap.",
        "",
        "| Metric | Bare | Equipped | Difference | Confidence interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in preferred:
        if metric not in metrics:
            continue
        row = metrics[metric]
        low, high = row["confidence_interval"]
        lines.append(
            f"| {metric} | {row['bare_mean']:.6f} | "
            f"{row['equipped_mean']:.6f} | {row['mean_difference']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare aligned bare and equipped ShopSimRL runs"
    )
    parser.add_argument("bare_run")
    parser.add_argument("equipped_run")
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    comparison = compare_runs(
        args.bare_run,
        args.equipped_run,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence_level=args.confidence_level,
    )
    output = (
        Path(args.output).resolve()
        if args.output
        else Path(args.equipped_run).resolve() / "comparison.json"
    )
    atomic_write_json(output, comparison)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(render_markdown(comparison), encoding="utf-8")
    primary = comparison["paired"]["metrics"]
    print(f"comparison={output}")
    for metric in ("reward", "r_strict", "r_success"):
        row = primary[metric]
        print(
            f"{metric}: bare={row['bare_mean']:.6f} "
            f"equipped={row['equipped_mean']:.6f} "
            f"diff={row['mean_difference']:+.6f} "
            f"ci={row['confidence_interval']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
