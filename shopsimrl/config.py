"""YAML configuration parsing for repeatable experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml

from .model import OpenAICompatibleConfig
from .store import safe_name


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    config: OpenAICompatibleConfig


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    output_dir: Path
    dotenv_path: Path | None
    seed: int
    concurrency: int
    resume: bool
    split_file: Path
    split: str
    sample_size: int | None
    repeats: int
    environment_base_url: str
    environment_timeout: float
    environment_persona: bool | None
    max_steps: int
    system_prompt: str | None
    skillbank_path: Path | None
    max_skills: int | None
    model: ModelSpec


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"dotenv file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.removeprefix("export ").strip()
        if name:
            os.environ.setdefault(name, value.strip().strip('"').strip("'"))


def load_experiment_config(path: str | Path) -> ExperimentSpec:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, "config")
    base = source.parent

    experiment = _mapping(root.get("experiment"), "experiment")
    name = safe_name(str(experiment.get("name", source.stem)))
    output_dir = _resolve(base, experiment.get("output_dir", "../runs"))
    dotenv_raw = experiment.get("dotenv")
    dotenv_path = _resolve(base, dotenv_raw) if dotenv_raw is not None else None
    if dotenv_path is not None:
        _load_dotenv(dotenv_path)
    seed = int(experiment.get("seed", 20260828))
    concurrency = int(experiment.get("concurrency", 1))
    resume_raw = experiment.get("resume", True)
    if not isinstance(resume_raw, bool):
        raise ValueError("experiment.resume must be true or false")
    resume = resume_raw
    if concurrency < 1:
        raise ValueError("experiment.concurrency must be positive")

    tasks = _mapping(root.get("tasks"), "tasks")
    if not tasks.get("split_file") or not tasks.get("split"):
        raise ValueError("tasks.split_file and tasks.split are required")
    split_file = _resolve(base, tasks["split_file"])
    split = str(tasks["split"])
    sample_size_raw = tasks.get("sample_size")
    sample_size = int(sample_size_raw) if sample_size_raw is not None else None
    repeats = int(tasks.get("repeats", 1))

    environment = _mapping(root.get("environment"), "environment")
    environment_base_url = str(
        environment.get("base_url", "http://127.0.0.1:5700")
    )
    environment_timeout = float(environment.get("timeout", 30.0))
    persona_raw = environment.get("persona")
    if persona_raw not in {True, False, None}:
        raise ValueError("environment.persona must be true, false, or null")

    runtime = _mapping(root.get("runtime"), "runtime")
    max_steps = int(runtime.get("max_steps", 30))

    prompt = _mapping(root.get("prompt"), "prompt")
    system_prompt = prompt.get("system_prompt")
    system_prompt_file = prompt.get("system_prompt_file")
    if system_prompt is not None and system_prompt_file is not None:
        raise ValueError("set only one of prompt.system_prompt/system_prompt_file")
    if system_prompt_file is not None:
        system_prompt = _resolve(base, system_prompt_file).read_text(encoding="utf-8")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise ValueError("prompt.system_prompt must be a string")

    skills = _mapping(root.get("skills"), "skills")
    skillbank_path = (
        _resolve(base, skills["path"]) if skills.get("path") is not None else None
    )
    max_skills_raw = skills.get("max_skills")
    max_skills = int(max_skills_raw) if max_skills_raw is not None else None
    if max_skills is not None and max_skills < 1:
        raise ValueError("skills.max_skills must be positive")

    if "models" in root:
        raise ValueError("use one model mapping; multi-model configs are not supported")
    record = _mapping(root.get("model"), "model")
    if not record.get("name") or not record.get("base_url"):
        raise ValueError("model.name and model.base_url are required")
    model_id = safe_name(str(record.get("id", record["name"])))
    extra_body = _mapping(record.get("extra_body"), "model.extra_body")
    api_key_env = record.get("api_key_env", "OPENAI_API_KEY")
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not api_key_env.strip()
    ):
        raise ValueError("model.api_key_env must be a non-empty string or null")
    trust_env = record.get("trust_env", True)
    if not isinstance(trust_env, bool):
        raise ValueError("model.trust_env must be true or false")
    stream = record.get("stream", False)
    if not isinstance(stream, bool):
        raise ValueError("model.stream must be true or false")
    send_seed = record.get("send_seed", True)
    if not isinstance(send_seed, bool):
        raise ValueError("model.send_seed must be true or false")
    model = ModelSpec(
        model_id=model_id,
        config=OpenAICompatibleConfig(
            model=str(record["name"]),
            base_url=str(record["base_url"]),
            api_key_env=api_key_env,
            temperature=float(record.get("temperature", 0.0)),
            max_tokens=int(record.get("max_tokens", 512)),
            top_p=(
                float(record["top_p"])
                if record.get("top_p") is not None
                else None
            ),
            timeout=float(record.get("timeout", 180.0)),
            max_retries=int(record.get("max_retries", 5)),
            retry_backoff_seconds=float(record.get("retry_backoff_seconds", 1.0)),
            trust_env=trust_env,
            stream=stream,
            send_seed=send_seed,
            tool_choice=str(record.get("tool_choice", "auto")),
            extra_body=dict(extra_body),
            checkpoint_id=record.get("checkpoint_id"),
        ),
    )

    return ExperimentSpec(
        name=name,
        output_dir=output_dir,
        dotenv_path=dotenv_path,
        seed=seed,
        concurrency=concurrency,
        resume=resume,
        split_file=split_file,
        split=split,
        sample_size=sample_size,
        repeats=repeats,
        environment_base_url=environment_base_url,
        environment_timeout=environment_timeout,
        environment_persona=persona_raw,
        max_steps=max_steps,
        system_prompt=system_prompt,
        skillbank_path=skillbank_path,
        max_skills=max_skills,
        model=model,
    )
