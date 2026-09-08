"""Run Trace2Skill Gate A to completion, then launch the equipped test run."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "run_shopsimrl.py"
DEFAULT_GATE_A_CONFIG = ROOT / "configs" / "archive" / "trace2skill_gate_a.example.yaml"
DEFAULT_EQUIPPED_CONFIG = (
    ROOT / "configs" / "qwen35_4b_test_trace2skill_equipped.yaml"
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shopsimrl.config import load_experiment_config
from shopsimrl.trace2skill_evaluation_config import (
    load_trace2skill_evaluation_config,
)


class OvernightRunError(RuntimeError):
    pass


def _log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _run_cli(arguments: Sequence[str]) -> int:
    command = [sys.executable, str(RUNNER), *arguments]
    _log("running: " + subprocess.list2cmdline(command))
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _gate_a_state(manifest_path: Path) -> tuple[str, dict[str, Any] | None]:
    manifest = _load_json(manifest_path)
    if manifest is None:
        return "missing", None
    status = manifest.get("status")
    summary = manifest.get("summary") or {}
    counts = summary.get("counts") or {}
    requested = counts.get("requested")
    completed = counts.get("completed")
    failed = counts.get("failed")
    coverage = counts.get("coverage")
    if (
        status == "complete"
        and isinstance(requested, int)
        and completed == requested
        and failed == 0
        and coverage == 1.0
    ):
        return "complete", manifest
    if status == "incomplete":
        return "incomplete", manifest
    return "invalid", manifest


def _format_gate_a_counts(manifest: dict[str, Any] | None) -> str:
    counts = ((manifest or {}).get("summary") or {}).get("counts") or {}
    return (
        f"requested={counts.get('requested')} completed={counts.get('completed')} "
        f"failed={counts.get('failed')} coverage={counts.get('coverage')}"
    )


def run_gate_a_until_complete(
    config_path: Path,
    *,
    max_retries: int,
    retry_delay_seconds: float,
) -> Path:
    spec = load_trace2skill_evaluation_config(config_path)
    manifest_path = spec.output_dir / "gate_a_manifest.json"
    selected_skillbank_path = spec.output_dir / "selected_skillbank.json"
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        _log(
            f"Gate A attempt {attempt}/{total_attempts} "
            f"(initial run + at most {max_retries} retries)"
        )
        returncode = _run_cli(["trace2skill", "gate-a", str(config_path)])
        state, manifest = _gate_a_state(manifest_path)
        _log(
            f"Gate A exit={returncode} state={state} "
            f"{_format_gate_a_counts(manifest)}"
        )
        if state == "complete":
            if returncode != 0:
                raise OvernightRunError(
                    "Gate A exited with an error even though an older complete "
                    "manifest exists"
                )
            if not selected_skillbank_path.is_file():
                raise OvernightRunError(
                    "Gate A is complete but selected_skillbank.json is missing"
                )
            _log(f"Gate A complete: {selected_skillbank_path}")
            return selected_skillbank_path
        if state != "incomplete":
            raise OvernightRunError(
                "Gate A did not produce a valid incomplete manifest; "
                "treating this as a configuration/protocol error instead of retrying"
            )
        if attempt == total_attempts:
            raise OvernightRunError(
                f"Gate A is still incomplete after {total_attempts} attempts"
            )
        _log(
            "Gate A has failed episodes; resume will rerun only unfinished tasks "
            f"after {retry_delay_seconds:g}s"
        )
        if retry_delay_seconds:
            time.sleep(retry_delay_seconds)

    raise AssertionError("unreachable")


def run_equipped_test(config_path: Path) -> Path:
    spec = load_experiment_config(config_path)
    run_dir = spec.output_dir / spec.name
    _log(f"validating equipped test plan: {config_path}")
    plan_returncode = _run_cli(["plan", str(config_path)])
    if plan_returncode != 0:
        raise OvernightRunError(
            f"equipped test plan failed with exit code {plan_returncode}"
        )

    _log(f"starting equipped test run: {run_dir}")
    run_returncode = _run_cli(["run", str(config_path)])
    summary = _load_json(run_dir / "summary.json")
    counts = (summary or {}).get("counts") or {}
    _log(
        f"equipped test exit={run_returncode} "
        f"requested={counts.get('requested')} completed={counts.get('completed')} "
        f"failed={counts.get('failed')} coverage={counts.get('coverage')}"
    )
    if run_returncode != 0:
        raise OvernightRunError(
            "equipped test did not fully complete; rerun this overnight script or "
            "the same generic run command to resume unfinished tasks"
        )
    if not summary or counts.get("failed") != 0 or counts.get("coverage") != 1.0:
        raise OvernightRunError("equipped test returned success without full coverage")
    _log(f"equipped test complete: {run_dir}")
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gate A with resumable failed-task retries, then run the equipped test"
        )
    )
    parser.add_argument(
        "--gate-a-config",
        type=Path,
        default=DEFAULT_GATE_A_CONFIG,
    )
    parser.add_argument(
        "--equipped-config",
        type=Path,
        default=DEFAULT_EQUIPPED_CONFIG,
    )
    parser.add_argument(
        "--max-gate-a-retries",
        type=int,
        default=5,
        help="retries after the initial Gate A run (default: 5)",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=10.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_gate_a_retries < 0:
        raise ValueError("--max-gate-a-retries cannot be negative")
    if args.retry_delay_seconds < 0:
        raise ValueError("--retry-delay-seconds cannot be negative")
    gate_a_config = args.gate_a_config.resolve()
    equipped_config = args.equipped_config.resolve()
    _log("overnight Trace2Skill evaluation started")
    try:
        run_gate_a_until_complete(
            gate_a_config,
            max_retries=args.max_gate_a_retries,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        run_dir = run_equipped_test(equipped_config)
    except (OSError, ValueError, OvernightRunError) as exc:
        _log(f"FAILED: {type(exc).__name__}: {exc}")
        return 1
    _log(f"SUCCESS: Gate A and equipped test completed; run={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
