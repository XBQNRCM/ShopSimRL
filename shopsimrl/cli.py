"""Command-line entry point for sampling and checkpoint evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .config import ExperimentSpec, ModelSpec, load_experiment_config
from .environment import ShopSimulatorConfig, ShopSimulatorHTTPEnvironment
from .evaluation import EvaluationPlan, Evaluator, build_jobs, summarize_traces
from .model import OpenAICompatibleChatModel
from .prompts import DEFAULT_SYSTEM_PROMPT, ShoppingPromptBuilder
from .runtime import ACTION_PROTOCOL_VERSION, AgentRuntime, RuntimeConfig
from .skills import JsonSkillBank, NoSkills, SkillProvider
from .store import RunStore
from .tasks import TaskSplit, load_task_split
from .trace2skill import (
    analyze_trajectories,
    compile_initial_skill,
    run_cold_start,
    trace2skill_plan,
)
from .trace2skill_config import load_trace2skill_config
from .trace2skill_evaluation import (
    gate_a_plan,
    run_gate_a,
)
from .trace2skill_evaluation_config import load_trace2skill_evaluation_config


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


def _skill_provider(spec: ExperimentSpec) -> SkillProvider:
    if spec.skillbank_path is None:
        return NoSkills()
    return JsonSkillBank(spec.skillbank_path, max_skills=spec.max_skills)


def _plan(spec: ExperimentSpec) -> tuple[TaskSplit, tuple, bool]:
    task_split = load_task_split(spec.split_file, spec.split)
    persona = _resolve_persona(spec, task_split)
    plan = EvaluationPlan(
        split=task_split.name,
        task_ids=task_split.task_ids,
        seed=spec.seed,
        sample_size=spec.sample_size,
        repeats=spec.repeats,
    )
    return task_split, build_jobs(plan), persona


def _prompt_builder(spec: ExperimentSpec) -> ShoppingPromptBuilder:
    return ShoppingPromptBuilder(
        system_prompt=spec.system_prompt or DEFAULT_SYSTEM_PROMPT
    )


def _semantic_plan(
    *,
    spec: ExperimentSpec,
    model: ModelSpec,
    task_split: TaskSplit,
    jobs: Sequence,
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
        "experiment": spec.name,
        "model_id": model.model_id,
        "model": model.config.identity(),
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


def _progress(done: int, total: int, job, trace: dict[str, Any]) -> None:
    final = trace.get("final") or {}
    reward = final.get("reward")
    print(
        f"[{done}/{total}] {job.episode_id} status={trace.get('status')} "
        f"reward={reward}"
    )


def command_plan(config_path: str) -> int:
    spec = load_experiment_config(config_path)
    task_split, jobs, persona = _plan(spec)
    skill_provider = _skill_provider(spec)
    prompt_builder = _prompt_builder(spec)
    payload = {
        "experiment": spec.name,
        "output_dir": str(spec.output_dir),
        "split": task_split.identity(),
        "selected_tasks": len({job.task_id for job in jobs}),
        "episodes": len(jobs),
        "persona": persona,
        "concurrency": spec.concurrency,
        "resume": spec.resume,
        "model": {
            "id": spec.model.model_id,
            **spec.model.config.identity(),
        },
        "runtime": {
            "max_steps": spec.max_steps,
            "action_protocol": ACTION_PROTOCOL_VERSION,
        },
        "prompt": prompt_builder.identity(),
        "skills": skill_provider.identity(),
        "first_jobs": [job.to_dict() for job in jobs[:5]],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def command_run(config_path: str) -> int:
    spec = load_experiment_config(config_path)
    task_split, jobs, persona = _plan(spec)
    skill_provider = _skill_provider(spec)
    prompt_builder = _prompt_builder(spec)
    model_spec = spec.model
    run_dir = spec.output_dir / spec.name
    store = RunStore(run_dir)
    semantic_plan = _semantic_plan(
        spec=spec,
        model=model_spec,
        task_split=task_split,
        jobs=jobs,
        persona=persona,
        prompt_builder=prompt_builder,
        skill_provider=skill_provider,
    )
    store.initialize(semantic_plan)
    env_config = ShopSimulatorConfig(
        base_url=spec.environment_base_url,
        persona=persona,
        timeout=spec.environment_timeout,
    )

    def runtime_factory() -> AgentRuntime:
        return AgentRuntime(
            model=OpenAICompatibleChatModel(model_spec.config),
            environment=ShopSimulatorHTTPEnvironment(env_config),
            prompt_builder=prompt_builder,
            skill_provider=skill_provider,
            config=RuntimeConfig(max_steps=spec.max_steps),
        )

    print(f"run={run_dir} episodes={len(jobs)} model={model_spec.model_id}")
    summary = Evaluator(
        runtime_factory=runtime_factory,
        store=store,
        max_workers=spec.concurrency,
        progress=_progress,
    ).run(jobs, resume=spec.resume)
    return 1 if summary["counts"]["failed"] else 0


def command_summarize(run_dir: str, requested: int | None) -> int:
    store = RunStore(Path(run_dir))
    if requested is None and store.manifest_path.exists():
        manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        jobs = manifest.get("plan", {}).get("jobs")
        if isinstance(jobs, list):
            requested = len(jobs)
    summary = summarize_traces(store.iter_traces(), requested=requested)
    store.write_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def command_trace2skill(action: str, config_path: str) -> int:
    if action in {"gate-a-plan", "gate-a"}:
        gate_a_spec = load_trace2skill_evaluation_config(config_path)
        if action == "gate-a-plan":
            payload = gate_a_plan(gate_a_spec)
        else:
            payload = run_gate_a(gate_a_spec, progress=_progress)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return int(payload.get("status") == "incomplete")

    spec = load_trace2skill_config(config_path)
    if action == "plan":
        payload = trace2skill_plan(spec)
    elif any(
        "REPLACE_WITH" in model
        for model in (spec.analyst_model.model, spec.compiler_model.model)
    ):
        raise ValueError(
            "replace the teacher-model placeholder before running Trace2Skill"
        )
    elif action == "analyze":
        payload = analyze_trajectories(spec, progress=print)
    elif action == "compile":
        payload = compile_initial_skill(spec, progress=print)
    elif action == "cold-start":
        payload = run_cold_start(spec, progress=print)
    else:  # pragma: no cover - argparse owns this boundary
        raise AssertionError(action)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shopsimrl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="validate config without API calls")
    plan.add_argument("config")
    run = subparsers.add_parser("run", help="sample/evaluate the configured model")
    run.add_argument("config")
    summarize = subparsers.add_parser("summarize", help="rebuild a run summary")
    summarize.add_argument("run_dir")
    summarize.add_argument("--requested", type=int)
    trace2skill = subparsers.add_parser(
        "trace2skill", help="build a train-only initial skill draft from traces"
    )
    trace2skill.add_argument(
        "action",
        choices=(
            "plan",
            "analyze",
            "compile",
            "cold-start",
            "gate-a-plan",
            "gate-a",
        ),
    )
    trace2skill.add_argument("config")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        return command_plan(args.config)
    if args.command == "run":
        return command_run(args.config)
    if args.command == "summarize":
        return command_summarize(args.run_dir, args.requested)
    if args.command == "trace2skill":
        return command_trace2skill(args.action, args.config)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
