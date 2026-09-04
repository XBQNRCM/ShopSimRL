"""Optional round-end W&B reporting for the slow co-evolution loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _copy_numbers(
    output: dict[str, float], prefix: str, record: Mapping[str, Any] | None
) -> None:
    for key, value in (record or {}).items():
        number = _number(value)
        if number is not None:
            output[f"{prefix}/{key}"] = number


def analysis_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    _copy_numbers(
        metrics,
        "analysis",
        {
            key: summary.get(key)
            for key in (
                "triage_groups",
                "analyzed_cards",
                "eligible_cards",
                "candidates",
                "proposal_history_records",
            )
        },
    )
    evidence_only = summary.get("evidence_only_card_ids")
    if isinstance(evidence_only, list):
        metrics["analysis/evidence_only_cards"] = float(len(evidence_only))
    return metrics


def gate_metrics(
    manifest: Mapping[str, Any], contributions: Mapping[str, Any] | None
) -> dict[str, float]:
    metrics = {"gate/complete": float(manifest.get("status") == "complete")}
    summary = manifest.get("summary")
    if isinstance(summary, Mapping):
        _copy_numbers(metrics, "gate/validation", summary.get("counts"))
        _copy_numbers(metrics, "gate/masked", summary.get("primary"))
        _copy_numbers(metrics, "gate/masked/reward_component", summary.get("reward_components"))
    baseline = (manifest.get("input") or {}).get("bare_baseline")
    if isinstance(baseline, Mapping):
        _copy_numbers(metrics, "gate/bare", baseline.get("mean_outcomes"))
    if contributions:
        selected = contributions.get("selected_intervention_ids")
        if isinstance(selected, list):
            metrics["gate/selected_chunk_count"] = float(len(selected))
        rows = contributions.get("contributions")
        if isinstance(rows, list):
            numeric = [
                float(row["coefficient"])
                for row in rows
                if isinstance(row, Mapping)
                and isinstance(row.get("coefficient"), (int, float))
                and not isinstance(row.get("coefficient"), bool)
            ]
            metrics["gate/intervention_count"] = float(len(rows))
            metrics["gate/positive_chunk_count"] = float(
                sum(value > 0.0 for value in numeric)
            )
            if numeric:
                metrics["gate/coefficient_mean"] = sum(numeric) / len(numeric)
                metrics["gate/coefficient_max"] = max(numeric)
        reward = (contributions.get("estimates") or {}).get("reward")
        if isinstance(reward, Mapping):
            _copy_numbers(
                metrics,
                "gate/reward",
                {
                    key: reward.get(key)
                    for key in ("bare_mean", "masked_mean", "delta_mean")
                },
            )
    return metrics


def _artifact_files(stage: str, output_dir: Path) -> list[Path]:
    names = (
        (
            "analysis_summary.json",
            "candidate_pool.json",
            "failure_cards.jsonl",
            "proposal_ledger.jsonl",
        )
        if stage == "analysis"
        else (
            "online_gate_manifest.json",
            "contributions.json",
            "selected_skillbank.json",
            "selected_skill.md",
            "proposal_ledger.jsonl",
            "mask_assignments.json",
        )
    )
    return [output_dir / name for name in names if (output_dir / name).is_file()]


def report_round_to_wandb(
    *,
    stage: str,
    name: str,
    output_dir: Path,
    payload: Mapping[str, Any],
    project: str,
    entity: str | None = None,
    group: str | None = None,
    mode: str = "online",
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Publish one Analyst or gate result from the CLI primary process."""

    if stage not in {"analysis", "gate"}:
        raise ValueError("W&B stage must be analysis or gate")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("W&B mode must be online, offline, or disabled")
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise RuntimeError(
            "W&B reporting requested but wandb is not installed; install wandb in the CLI environment"
        ) from exc

    output_dir = Path(output_dir).resolve()
    wandb_dir = Path(directory).resolve() if directory else output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    contributions = None
    contributions_path = output_dir / "contributions.json"
    if stage == "gate" and contributions_path.is_file():
        contributions = json.loads(contributions_path.read_text(encoding="utf-8"))
    metrics = (
        analysis_metrics(payload)
        if stage == "analysis"
        else gate_metrics(payload, contributions)
    )

    run = wandb.init(
        project=project,
        entity=entity,
        group=group or name,
        name=f"{name}-{stage}",
        job_type=stage,
        mode=mode,
        dir=str(wandb_dir),
        config={
            "shopsimrl_stage": stage,
            "shopsimrl_round": name,
            "output_dir": str(output_dir),
        },
    )
    try:
        log_payload: dict[str, Any] = dict(metrics)
        if stage == "gate" and contributions:
            rows = contributions.get("contributions") or []
            columns = [
                "intervention_id",
                "logical_chunk_id",
                "operation",
                "coefficient",
                "rank",
                "status",
            ]
            log_payload["gate/chunk_contributions"] = wandb.Table(
                columns=columns,
                data=[
                    [row.get(column) for column in columns]
                    for row in rows
                    if isinstance(row, Mapping)
                ],
            )
        run.log(log_payload)
        artifact = wandb.Artifact(
            name=f"shopsimrl-{name}-{stage}", type=f"shopsimrl-{stage}"
        )
        files = _artifact_files(stage, output_dir)
        for path in files:
            artifact.add_file(str(path), name=path.name)
        if files:
            run.log_artifact(artifact)
    finally:
        run.finish()
    return {
        "stage": stage,
        "metrics": metrics,
        "artifact_files": [str(path) for path in _artifact_files(stage, output_dir)],
    }
