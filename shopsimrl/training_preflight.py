"""CPU-only input checks before allocating the slime training workers."""

import json
import os
from pathlib import Path

import yaml

from .curriculum import SkillCurriculum, TRAINING_DATA_SCHEMA_VERSION
from .skills import JsonSkillBank
from .tasks import load_task_split


def check_training_inputs(config_path, task_data, split_file=None) -> dict:
    root = Path(os.environ.get("SHOPSIMRL_PROJECT_ROOT", Path.cwd())).resolve()

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else root / path

    config = yaml.safe_load(resolve(config_path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("training custom config must be a mapping")
    curriculum = SkillCurriculum.load(resolve(config["shopsim_curriculum_path"]))
    bank = JsonSkillBank(resolve(config["shopsim_skillbank_path"]))
    if bank.bank_sha256 != curriculum.source["skillbank_sha256"]:
        raise ValueError("curriculum source SkillBank differs from shopsim_skillbank_path")
    records = [row for row in bank.records if row.get("enabled", True) is not False]
    expected = [(row["skill_id"], str(row.get("version", "1")), row["content"],
                 row["metadata"]["estimated_effect"]) for row in records]
    actual = [(chunk.skill.skill_id, chunk.skill.version, chunk.skill.content, chunk.contribution)
              for chunk in curriculum.chunks]
    if actual != expected:
        raise ValueError("curriculum chunks differ from its source SkillBank")
    if any(chunk.contribution <= 0 for chunk in curriculum.chunks):
        raise ValueError("active curriculum chunks must have positive contributions")
    for key in ("shopsim_episode_concurrency", "shopsim_max_steps", "shopsim_max_consecutive_failed_groups"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    resolve(config["shopsim_system_prompt_file"]).read_text(encoding="utf-8")
    split = load_task_split(resolve(split_file or "ShopSimulator/shop_env/configs/persona_splits.v1.json"), "train")
    if split.persona is not True or config.get("shopsim_environment_persona") is not True:
        raise ValueError("training requires the persona train split and persona environment")
    ids = []
    with resolve(task_data).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata") or {}
            task_id = metadata.get("task_id")
            if (metadata.get("schema_version") != TRAINING_DATA_SCHEMA_VERSION
                    or metadata.get("split") != "train"
                    or isinstance(task_id, bool) or not isinstance(task_id, int)
                    or row.get("prompt") != str(task_id)):
                raise ValueError(f"invalid training task metadata at line {line_number}")
            ids.append(task_id)
    if len(ids) != len(set(ids)) or set(ids) != set(split.task_ids):
        raise ValueError("task data must contain every frozen train ID exactly once")
    return {
        "status": "ready", "scope": "offline_inputs_only",
        "round_id": curriculum.round_id, "curriculum_state_id": curriculum.state_id,
        "skillbank_sha256": bank.bank_sha256, "active_chunks": len(curriculum.chunks),
        "training_tasks": len(ids), "skill_free_probability": curriculum.skill_free_probability,
        "episode_concurrency": config["shopsim_episode_concurrency"],
    }
