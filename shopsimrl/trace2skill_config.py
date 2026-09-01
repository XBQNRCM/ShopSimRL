"""Configuration for the train-only Trace2Skill cold-start pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import _load_dotenv, _mapping, _resolve
from .model import OpenAICompatibleConfig


@dataclass(frozen=True)
class Trace2SkillSpec:
    source_path: Path
    traces_path: Path
    output_dir: Path
    expected_trajectories: int
    seed: int
    concurrency: int
    resume: bool
    environment_base_url: str
    environment_timeout: float
    environment_persona: bool
    max_failure_analysis_steps: int
    max_cards_per_trajectory: int
    consolidation_batch_size: int
    validation_dimension_cap: int
    active_skill_budget: int
    analyst_model: OpenAICompatibleConfig
    compiler_model: OpenAICompatibleConfig


def _model_config(record: dict[str, Any], name: str) -> OpenAICompatibleConfig:
    if not record.get("name") or not record.get("base_url"):
        raise ValueError(f"{name}.name and {name}.base_url are required")
    api_key_env = record.get("api_key_env", "OPENAI_API_KEY")
    if api_key_env is not None and (
        not isinstance(api_key_env, str) or not api_key_env.strip()
    ):
        raise ValueError(f"{name}.api_key_env must be a non-empty string or null")
    extra_body = _mapping(record.get("extra_body"), f"{name}.extra_body")
    trust_env = record.get("trust_env", True)
    stream = record.get("stream", False)
    send_seed = record.get("send_seed", True)
    if (
        not isinstance(trust_env, bool)
        or not isinstance(stream, bool)
        or not isinstance(send_seed, bool)
    ):
        raise ValueError(f"{name}.trust_env/stream/send_seed must be true or false")
    return OpenAICompatibleConfig(
        model=str(record["name"]),
        base_url=str(record["base_url"]),
        api_key_env=api_key_env,
        temperature=float(record.get("temperature", 0.0)),
        max_tokens=int(record.get("max_tokens", 4096)),
        top_p=(
            float(record["top_p"])
            if record.get("top_p") is not None
            else None
        ),
        timeout=float(record.get("timeout", 300.0)),
        max_retries=int(record.get("max_retries", 5)),
        retry_backoff_seconds=float(record.get("retry_backoff_seconds", 1.0)),
        trust_env=trust_env,
        stream=stream,
        send_seed=send_seed,
        tool_choice=str(record.get("tool_choice", "required")),
        extra_body=dict(extra_body),
    )


def load_trace2skill_config(path: str | Path) -> Trace2SkillSpec:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _mapping(payload, "config")
    base = source.parent

    experiment = _mapping(root.get("experiment"), "experiment")
    dotenv_raw = experiment.get("dotenv")
    if dotenv_raw is not None:
        _load_dotenv(_resolve(base, dotenv_raw))
    traces_raw = experiment.get("traces")
    output_raw = experiment.get("output_dir")
    if traces_raw is None or output_raw is None:
        raise ValueError("experiment.traces and experiment.output_dir are required")
    expected = int(experiment.get("expected_trajectories", 400))
    concurrency = int(experiment.get("concurrency", 4))
    resume = experiment.get("resume", True)
    if expected < 1 or concurrency < 1:
        raise ValueError("expected_trajectories and concurrency must be positive")
    if not isinstance(resume, bool):
        raise ValueError("experiment.resume must be true or false")

    environment = _mapping(root.get("environment"), "environment")
    persona = environment.get("persona", True)
    if not isinstance(persona, bool):
        raise ValueError("environment.persona must be true or false")

    analysis = _mapping(root.get("analysis"), "analysis")
    max_failure_steps = int(analysis.get("max_failure_steps", 12))
    max_cards = int(analysis.get("max_cards_per_trajectory", 2))
    if max_failure_steps < 1 or max_cards < 1:
        raise ValueError("analysis step/card limits must be positive")

    compilation = _mapping(root.get("compilation"), "compilation")
    batch_size = int(compilation.get("batch_size", 40))
    factor_cap = int(compilation.get("validation_dimension_cap", 16))
    active_budget = int(compilation.get("active_skill_budget", 10))
    if batch_size < 1:
        raise ValueError("compilation.batch_size must be positive")
    if not 1 <= factor_cap <= 16:
        raise ValueError("validation_dimension_cap must be in [1, 16]")
    if not 1 <= active_budget <= factor_cap:
        raise ValueError(
            "active_skill_budget must be positive and no larger than the factor cap"
        )

    analyst_record = _mapping(root.get("analyst_model"), "analyst_model")
    compiler_record = _mapping(
        root.get("compiler_model", analyst_record), "compiler_model"
    )
    return Trace2SkillSpec(
        source_path=source,
        traces_path=_resolve(base, traces_raw),
        output_dir=_resolve(base, output_raw),
        expected_trajectories=expected,
        seed=int(experiment.get("seed", 20260831)),
        concurrency=concurrency,
        resume=resume,
        environment_base_url=str(
            environment.get("base_url", "http://127.0.0.1:5700")
        ),
        environment_timeout=float(environment.get("timeout", 30.0)),
        environment_persona=persona,
        max_failure_analysis_steps=max_failure_steps,
        max_cards_per_trajectory=max_cards,
        consolidation_batch_size=batch_size,
        validation_dimension_cap=factor_cap,
        active_skill_budget=active_budget,
        analyst_model=_model_config(analyst_record, "analyst_model"),
        compiler_model=_model_config(compiler_record, "compiler_model"),
    )
