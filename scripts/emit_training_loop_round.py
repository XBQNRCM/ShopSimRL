#!/usr/bin/env python3
"""Write per-round slime / analysis / val / gate configs for the 20-step outer loop."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--skillbank", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--history-ledger")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    round_id = args.round_id
    out = root / "runs" / "loop" / round_id
    out.mkdir(parents=True, exist_ok=True)

    slime = {
        "shopsim_skillbank_path": str(Path(args.skillbank).resolve()),
        "shopsim_curriculum_path": str(Path(args.curriculum).resolve()),
        "shopsim_training_output_dir": str((root / "runs" / "slime-training").resolve()),
        "shopsim_system_prompt_file": "configs/prompts/persona_single_turn.txt",
        "shopsim_environment_base_url": "http://127.0.0.1:5700",
        "shopsim_environment_timeout": 30,
        "shopsim_environment_persona": True,
        "shopsim_environment_trust_env": False,
        "shopsim_episode_concurrency": 32,
        "shopsim_max_steps": 30,
        "shopsim_full_skill_retry": True,
        "shopsim_max_consecutive_failed_groups": 8,
        "shopsim_tool_parser": "qwen3_coder",
        "shopsim_reasoning_parser": "qwen3",
    }
    (out / "slime.yaml").write_text(yaml.safe_dump(slime, sort_keys=False), encoding="utf-8")

    analysis = {
        "online_analysis": {
            "name": round_id,
            "training_round_dir": str((root / "runs" / "slime-training" / round_id).resolve()),
            "current_skillbank_path": str(Path(args.skillbank).resolve()),
            "output_dir": str((root / "runs" / f"{round_id}-online-analysis").resolve()),
            "proposal_checkpoint": f"qwen35-4b-{round_id}",
            "max_candidates": 6,
            "resume": True,
            "trace2skill_config": str((root / "configs" / "trace2skill_cold_start.yaml").resolve()),
        }
    }
    if args.history_ledger:
        analysis["online_analysis"]["proposal_ledger_history_path"] = str(
            Path(args.history_ledger).resolve()
        )
    (out / "analysis.yaml").write_text(
        yaml.safe_dump(analysis, sort_keys=False), encoding="utf-8"
    )

    val = {
        "experiment": {
            "name": f"{round_id}-bare-val",
            "output_dir": str((root / "runs").resolve()),
            "dotenv": str((root / ".env").resolve()),
            "seed": 20260830,
            "concurrency": 32,
            "resume": True,
        },
        "tasks": {
            "split_file": str(
                (root / "ShopSimulator" / "shop_env" / "configs" / "persona_splits.v1.json").resolve()
            ),
            "split": "val",
            "sample_size": None,
            "repeats": 1,
        },
        "environment": {
            "base_url": "http://127.0.0.1:5700",
            "timeout": 30,
            "persona": None,
        },
        "runtime": {"max_steps": 30},
        "prompt": {"system_prompt_file": str((root / "configs" / "prompts" / "persona_single_turn.txt").resolve())},
        "skills": {"path": None, "max_skills": None},
        "model": {
            "id": "qwen35-4b-current-checkpoint",
            "checkpoint_id": f"{round_id}-checkpoint",
            "name": "Qwen/Qwen3.5-4B",
            "base_url": "http://127.0.0.1:30000/v1",
            "api_key_env": None,
            "temperature": 0.6,
            "top_p": 1.0,
            "max_tokens": int(os.environ.get("ROLLOUT_MAX_RESPONSE_LEN", 4096)),
            "timeout": 180,
            "max_retries": 3,
            "trust_env": False,
            "send_seed": False,
            "tool_choice": "auto",
            "extra_body": {"enable_thinking": True},
        },
    }
    (out / "val.yaml").write_text(yaml.safe_dump(val, sort_keys=False), encoding="utf-8")

    gate = {
        "online_gate": {
            "name": f"{round_id}-online-gate",
            "output_dir": str((root / "runs" / f"{round_id}-online-gate").resolve()),
            "experiment_config": str((out / "val.yaml").resolve()),
            "bare_run_dir": str((root / "runs" / f"{round_id}-bare-val").resolve()),
            "candidate_pool_path": str(
                (root / "runs" / f"{round_id}-online-analysis" / "candidate_pool.json").resolve()
            ),
            "mask_seed": 20260901,
            "active_skill_budget": 10,
        }
    }
    (out / "gate.yaml").write_text(yaml.safe_dump(gate, sort_keys=False), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
