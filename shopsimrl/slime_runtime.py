"""slime custom generation adapter for ShopSimulator agentic GRPO.

This module is imported inside slime rollout workers.  Slime imports stay lazy
so the rest of ShopSimRL (including CPU-only analysis and tests) does not gain a
runtime dependency on Ray, Torch, SGLang, or Megatron.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, replace
import json
import logging
import math
import os
from pathlib import Path
import traceback
from typing import Any, Sequence

from .curriculum import SkillAssignment, SkillCurriculum
from .environment import ShopSimulatorConfig, ShopSimulatorHTTPEnvironment
from .prompts import DEFAULT_SYSTEM_PROMPT, ShoppingPromptBuilder
from .runtime import (
    ACTION_PROTOCOL_VERSION,
    _action_tools,
    _environment_action,
    _protocol_feedback,
    _public_reset,
    _without_lease,
)
from .schemas import (
    TRACE_SCHEMA_VERSION,
    EpisodeJob,
    Skill,
    ToolCall,
    utc_now,
)
from .store import atomic_write_json, safe_name


logger = logging.getLogger(__name__)
SLIME_RUNTIME_VERSION = "shopsimrl-slime-agent-runtime-v2"
TRIAGE_SCHEMA_VERSION = "shopsimrl-training-failure-triage-v1"


@dataclass
class EpisodeRollout:
    trace: dict[str, Any]
    samples: list[Any]
    reward: dict[str, float]
    scored: bool

    @property
    def success(self) -> bool:
        return self.scored and self.reward.get("r_success") == 1.0


@dataclass
class _GroupOutcome:
    sample_index: int
    rollout: EpisodeRollout
    assignment: SkillAssignment


@dataclass
class _GroupEntry:
    expected: int
    curriculum: SkillCurriculum
    outcomes: dict[int, _GroupOutcome]
    running_retry: bool = False


_GROUPS: dict[tuple[str, int], _GroupEntry] = {}
_GROUP_LOCK: asyncio.Lock | None = None
_GROUP_LOCK_LOOP: asyncio.AbstractEventLoop | None = None
_EPISODE_SEMAPHORE: asyncio.Semaphore | None = None
_EPISODE_SEMAPHORE_LOOP: asyncio.AbstractEventLoop | None = None
_CURRICULUM_CACHE: dict[tuple[str, int], SkillCurriculum] = {}
_PROMPT_CACHE: dict[str, ShoppingPromptBuilder] = {}
_FAILED_GROUP_STREAK = 0


def _loop_lock() -> asyncio.Lock:
    global _GROUP_LOCK, _GROUP_LOCK_LOOP
    loop = asyncio.get_running_loop()
    if _GROUP_LOCK is None or _GROUP_LOCK_LOOP is not loop:
        _GROUP_LOCK = asyncio.Lock()
        _GROUP_LOCK_LOOP = loop
    return _GROUP_LOCK


def _episode_semaphore(args: Any) -> asyncio.Semaphore:
    global _EPISODE_SEMAPHORE, _EPISODE_SEMAPHORE_LOOP
    limit = getattr(args, "shopsim_episode_concurrency", 16)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("shopsim_episode_concurrency must be a positive integer")
    loop = asyncio.get_running_loop()
    if _EPISODE_SEMAPHORE is None or _EPISODE_SEMAPHORE_LOOP is not loop:
        _EPISODE_SEMAPHORE = asyncio.Semaphore(limit)
        _EPISODE_SEMAPHORE_LOOP = loop
    return _EPISODE_SEMAPHORE


def _required_arg(args: Any, name: str) -> Any:
    value = getattr(args, name, None)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"slime custom config must set {name}")
    return value


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    project_root = Path(os.environ.get("SHOPSIMRL_PROJECT_ROOT", Path.cwd()))
    return (project_root / path).resolve()


def _task_identity(sample: Any) -> tuple[int, str]:
    metadata = getattr(sample, "metadata", None) or {}
    task_id = metadata.get("task_id")
    split = metadata.get("split")
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        try:
            task_id = int(sample.prompt)
        except (TypeError, ValueError) as exc:
            raise ValueError("slime sample has no valid ShopSimulator task_id") from exc
    if not isinstance(split, str) or not split:
        split = "train"
    return task_id, split


def _curriculum(args: Any, group_index: int) -> SkillCurriculum:
    path = str(_project_path(_required_arg(args, "shopsim_curriculum_path")))
    key = (path, int(group_index))
    state = _CURRICULUM_CACHE.get(key)
    if state is None:
        state = SkillCurriculum.load(path)
        _CURRICULUM_CACHE[key] = state
    return state


def _prompt_builder(args: Any) -> ShoppingPromptBuilder:
    prompt_file = getattr(args, "shopsim_system_prompt_file", None)
    resolved_prompt = _project_path(prompt_file) if prompt_file else None
    cache_key = str(resolved_prompt) if resolved_prompt else "<default>"
    builder = _PROMPT_CACHE.get(cache_key)
    if builder is None:
        system_prompt = (
            resolved_prompt.read_text(encoding="utf-8")
            if resolved_prompt
            else DEFAULT_SYSTEM_PROMPT
        )
        builder = ShoppingPromptBuilder(system_prompt=system_prompt)
        _PROMPT_CACHE[cache_key] = builder
    return builder


def _environment(args: Any) -> ShopSimulatorHTTPEnvironment:
    return ShopSimulatorHTTPEnvironment(
        ShopSimulatorConfig(
            base_url=str(
                getattr(args, "shopsim_environment_base_url", "http://127.0.0.1:5700")
            ),
            persona=bool(getattr(args, "shopsim_environment_persona", True)),
            timeout=float(getattr(args, "shopsim_environment_timeout", 30.0)),
            trust_env=bool(getattr(args, "shopsim_environment_trust_env", False)),
        )
    )


def _max_context_tokens(args: Any) -> int:
    explicit = getattr(args, "rollout_max_context_len", None)
    if explicit is not None:
        return int(explicit)
    return int(args.context_parallel_size) * int(args.max_tokens_per_gpu)


def _reward(final: dict[str, Any] | None) -> tuple[dict[str, float], bool]:
    if not isinstance(final, dict) or final.get("done") is not True:
        return {"reward": 0.0, "r_strict": 0.0, "r_success": 0.0}, False
    reward = final.get("reward")
    detail = final.get("reward_detail")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not isinstance(detail, dict)
    ):
        return {"reward": 0.0, "r_strict": 0.0, "r_success": 0.0}, False
    payload = {"reward": float(reward)}
    for key, value in detail.items():
        if (
            key.startswith("r_")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            payload[key] = float(value)
    if "r_strict" not in payload or "r_success" not in payload:
        return {"reward": 0.0, "r_strict": 0.0, "r_success": 0.0}, False
    if any(not math.isfinite(value) for value in payload.values()):
        return {"reward": 0.0, "r_strict": 0.0, "r_success": 0.0}, False
    if payload["r_success"] not in {0.0, 1.0}:
        return {"reward": 0.0, "r_strict": 0.0, "r_success": 0.0}, False
    if abs(payload["reward"] - payload["r_strict"]) > 1e-12:
        raise RuntimeError("ShopSimulator reward does not equal r_strict")
    return payload, True


def _abort_sample(sample: Any, reason: str, reward: dict[str, float]) -> list[Any]:
    """Return slime's structurally valid, zero-loss placeholder shape."""

    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = reward
    sample.remove_sample = True
    sample.status = sample.Status.ABORTED
    sample.rollout_id = sample.index
    sample.metadata = {
        **dict(getattr(sample, "metadata", None) or {}),
        "shopsim_abort_reason": reason,
        "shopsim_scored": False,
    }
    return [sample]


def _tool_call_from_wire(
    parsed: Any, wire_message: dict[str, Any]
) -> ToolCall | None:
    if len(parsed.tool_uses) != 1:
        return None
    wire_calls = wire_message.get("tool_calls") or []
    if len(wire_calls) != 1:
        return None
    use = parsed.tool_uses[0]
    arguments = use.get("input")
    if not isinstance(arguments, dict):
        arguments = {"_raw_arguments": str(arguments)}
    wire = wire_calls[0]
    return ToolCall(
        call_id=str(wire.get("id", "call")),
        name=str(use.get("name", "")),
        arguments=arguments,
        raw_arguments=json.dumps(arguments, ensure_ascii=False, sort_keys=True),
    )


def _trace_path(args: Any, curriculum: SkillCurriculum, group: int, name: str) -> Path:
    root = _project_path(
        str(getattr(args, "shopsim_training_output_dir", "runs/slime-training"))
    )
    round_name = safe_name(curriculum.round_id)
    return root / round_name / f"group-{group:09d}" / f"{safe_name(name)}.json"


async def _run_episode(
    args: Any,
    base_sample: Any,
    sampling_params: dict[str, Any],
    *,
    skills: Sequence[Skill],
    assignment: SkillAssignment,
    diagnostic: bool,
) -> EpisodeRollout:
    # Queue before any blocking reset, and retain admission through close().
    # Group coordination is outside this scope; diagnostic retries re-enter it.
    async with _episode_semaphore(args):
        return await _run_episode_impl(
            args,
            base_sample,
            sampling_params,
            skills=skills,
            assignment=assignment,
            diagnostic=diagnostic,
        )


async def _run_episode_impl(
    args: Any,
    base_sample: Any,
    sampling_params: dict[str, Any],
    *,
    skills: Sequence[Skill],
    assignment: SkillAssignment,
    diagnostic: bool,
) -> EpisodeRollout:
    """Run one exact-token SGLang trajectory against a fresh environment."""

    # Lazy imports keep shopsimrl importable outside a slime environment.
    from slime.agent.adapters import OpenAIAdapter
    from slime.agent.adapters.common import call_sglang_generate
    from slime.agent.parsing import parse_model_output
    from slime.rollout.sglang_rollout import GenerateState

    state = GenerateState(args)
    tokenizer = state.tokenizer
    router_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    adapter = OpenAIAdapter(
        tokenizer=tokenizer,
        sglang_url=router_url,
        tool_parser=str(
            getattr(args, "shopsim_tool_parser", None)
            or getattr(args, "sglang_tool_call_parser", None)
            or "qwen3_coder"
        ),
        reasoning_parser=(
            getattr(args, "shopsim_reasoning_parser", None)
            or getattr(args, "sglang_reasoning_parser", None)
            or "qwen3"
        ),
        # The default manager demotes short rewritten assistant messages and
        # realigns short token drift, dropping previously sampled loss tokens.
        # Tool schemas and reasoning echoes change on each shopping turn: fork
        # on drift so every sampled response retains its exact training signal.
        fork_threshold_tokens=0,
    )
    task_id, split = _task_identity(base_sample)
    sample_index = int(base_sample.index)
    group_index = int(base_sample.group_index)
    session_id = (
        f"shopsim-{assignment.state_id[:12]}-{group_index}-{sample_index}"
        + ("-retry" if diagnostic else "")
    )
    adapter.open_session(
        session_id,
        sampling_defaults=dict(sampling_params),
        max_context_tokens=_max_context_tokens(args),
    )
    environment = _environment(args)
    prompt_builder = _prompt_builder(args)
    messages: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    reset: dict[str, Any] | None = None
    final: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    status = "failed"
    stage = "reset"
    started_at = utc_now()
    try:
        if state.aborted:
            raise RuntimeError("slime rollout was aborted before environment reset")
        reset = await asyncio.to_thread(environment.reset, task_id)
        messages = [
            {
                "role": "system",
                "content": prompt_builder.system_message(reset, skills),
            },
            {
                "role": "user",
                "content": prompt_builder.initial_observation(reset),
            },
        ]
        observation = prompt_builder.initial_observation(reset)
        observation_state = reset["observation_state"]
        max_steps = int(getattr(args, "shopsim_max_steps", 30))
        for step_index in range(1, max_steps + 1):
            if state.aborted:
                raise RuntimeError("slime rollout was aborted between agent turns")
            tools = _action_tools(observation_state)
            body = {
                "messages": messages,
                "tools": list(tools),
                "max_completion_tokens": int(
                    sampling_params.get(
                        "max_new_tokens",
                        getattr(args, "rollout_max_response_len", 2048),
                    )
                ),
            }
            for key in ("temperature", "top_p", "top_k"):
                if key in sampling_params:
                    body[key] = sampling_params[key]
            translated, tool_schema = adapter._translate(body)
            encoded = tokenizer.apply_chat_template(
                translated,
                tools=tool_schema,
                tokenize=True,
                add_generation_prompt=True,
            )
            prompt_ids = (
                list(encoded["input_ids"])
                if hasattr(encoded, "__getitem__") and "input_ids" in encoded
                else list(encoded)
            )
            stage = f"model_step_{step_index}"
            turn = await call_sglang_generate(
                prompt_ids,
                adapter.store[session_id],
                body,
                adapter=adapter,
                session_id=session_id,
            )
            if state.aborted or turn.finish_reason == "abort":
                raise RuntimeError("SGLang generation was aborted")
            if len(turn.output_ids) != len(turn.output_log_probs) or any(
                not math.isfinite(value) for value in turn.output_log_probs
            ):
                raise RuntimeError("SGLang response has invalid token log probabilities")
            raw_output = tokenizer.decode(turn.output_ids, skip_special_tokens=False)
            parsed = parse_model_output(
                raw_output,
                tools_schema=tool_schema,
                tool_parser_name=adapter.tool_parser,
                reasoning_parser_name=adapter.reasoning_parser,
            )
            reply = adapter._build_reply(parsed, turn.finish_reason, translated, tool_schema)
            turn = replace(turn, ill_formed=parsed.ill_formed)
            adapter.manager.record_turn(
                session_id,
                turn=turn,
                prompt_messages=translated,
                response_message=reply.manager_message,
                metadata={"sid": session_id, "step_index": step_index},
            )
            wire_message = reply.wire[0]
            tool_call = _tool_call_from_wire(parsed, wire_message)
            protocol_error = None
            if turn.finish_reason == "length" and tool_call is None:
                steps.append(
                    {
                        "step_index": step_index,
                        "observation": observation,
                        "tools": list(tools),
                        "model": {
                            "content": parsed.text or None,
                            "reasoning": parsed.reasoning or None,
                            "finish_reason": "length",
                            "raw_output": raw_output,
                            "policy_failure": {"code": "generation_length"},
                        },
                        "policy_failure": {"code": "generation_length"},
                        "action": None,
                        "environment": None,
                    }
                )
                final = _without_lease(
                    await asyncio.to_thread(environment.terminate, "generation_length")
                )
                break
            if len(parsed.tool_uses) != 1:
                protocol_error = {
                    "code": "tool_call_count",
                    "message": f"model must return one tool call, got {len(parsed.tool_uses)}",
                }
            elif parsed.text:
                protocol_error = {
                    "code": "mixed_content",
                    "message": "model mixed visible text with a tool call",
                }
            elif tool_call is None:
                protocol_error = {
                    "code": "invalid_tool_call",
                    "message": "model tool call could not be decoded",
                }
            elif tool_call.name not in {tool["function"]["name"] for tool in tools}:
                protocol_error = {
                    "code": "unavailable_tool",
                    "message": f"model called unavailable tool: {tool_call.name}",
                }
            else:
                try:
                    action = _environment_action(tool_call)
                except RuntimeError as exc:
                    protocol_error = {
                        "code": "invalid_tool_arguments",
                        "message": str(exc),
                    }

            model_record = {
                "content": parsed.text or None,
                "reasoning": parsed.reasoning or None,
                "tool_calls": [tool_call.to_dict()] if tool_call else [],
                "finish_reason": turn.finish_reason,
                "raw_output": raw_output,
                "protocol_error": protocol_error,
            }
            if protocol_error is not None:
                feedback = _protocol_feedback(protocol_error, tools)
                if wire_message.get("tool_calls"):
                    messages.append(wire_message)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": wire_message["tool_calls"][0]["id"],
                            "content": json.dumps(feedback, ensure_ascii=False),
                        }
                    )
                else:
                    if wire_message.get("content"):
                        messages.append(wire_message)
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(feedback, ensure_ascii=False),
                        }
                    )
                steps.append(
                    {
                        "step_index": step_index,
                        "observation": observation,
                        "tools": list(tools),
                        "model": model_record,
                        "protocol_error": protocol_error,
                        "protocol_feedback": feedback,
                        "action": None,
                        "environment": None,
                    }
                )
                continue

            messages.append(wire_message)
            stage = f"environment_step_{step_index}"
            environment_output = await asyncio.to_thread(environment.step, action)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.call_id,
                    "content": environment_output["observation"],
                }
            )
            steps.append(
                {
                    "step_index": step_index,
                    "observation": observation,
                    "tools": list(tools),
                    "model": model_record,
                    "action": action,
                    "environment": _without_lease(environment_output),
                }
            )
            if environment_output.get("done") is True:
                final = _without_lease(environment_output)
                break
            observation = environment_output["observation"]
            observation_state = environment_output["observation_state"]
        if final is None:
            final = _without_lease(
                await asyncio.to_thread(environment.terminate, "action_limit")
            )
        status = "completed"
    except Exception as exc:
        error = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        try:
            await asyncio.to_thread(environment.close)
        except Exception as exc:
            if error is None:
                error = {
                    "stage": "cleanup",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }

    reward, scored = _reward(final)
    trace = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "episode_id": (
            f"train-group-{group_index:09d}-sample-{sample_index:09d}"
            + ("-full-skill-retry" if diagnostic else "")
        ),
        "status": status,
        "job": EpisodeJob(
            task_id=task_id,
            sample_id=sample_index,
            seed=int(getattr(args, "rollout_seed", 0)),
            split=split,
        ).to_dict(),
        "provenance": {
            "environment": environment.identity(),
            "prompt": prompt_builder.identity(),
            "skills": {
                "curriculum_state_id": assignment.state_id,
                "assignment": assignment.to_dict(),
                "diagnostic_full_skill": diagnostic,
            },
            "runtime": {
                "adapter": SLIME_RUNTIME_VERSION,
                "action_protocol": ACTION_PROTOCOL_VERSION,
                "max_steps": int(getattr(args, "shopsim_max_steps", 30)),
            },
        },
        "started_at": started_at,
        "ended_at": utc_now(),
        "reset": _public_reset(reset) if reset is not None else None,
        "selected_skills": [skill.to_dict() for skill in skills],
        "conversation": messages,
        "steps": steps,
        "final": final,
        "error": error,
    }
    if error is not None and status != "completed":
        await adapter.drop_session(session_id)
        samples = _abort_sample(base_sample, error["type"], reward)
    else:
        samples = await adapter.finish_session(
            session_id,
            base_sample=base_sample,
            reward=reward,
            extra_metadata={
                "shopsim_task_id": task_id,
                "shopsim_group_index": group_index,
                "shopsim_assignment": assignment.to_dict(),
                "shopsim_diagnostic": diagnostic,
            },
        )
        if not samples:
            # A prompt that already exhausts the context window can yield no
            # trainable model token.  Preserve group shape, but exclude the
            # execution from both GRPO and failure-frontier classification.
            error = {
                "stage": "trajectory_finalize",
                "type": "EmptyTrajectory",
                "message": "slime adapter produced no trainable response tokens",
            }
            trace["status"] = "failed"
            trace["error"] = error
            reward = {"reward": 0.0, "r_strict": 0.0, "r_success": 0.0}
            scored = False
            samples = _abort_sample(base_sample, error["type"], reward)
        else:
            for result in samples:
                result.rollout_id = base_sample.index
                result.group_index = base_sample.group_index
                result.reward = reward
                result.status = (
                    result.Status.TRUNCATED
                    if final and final.get("termination_reason") == "generation_length"
                    else result.Status.COMPLETED
                )
    return EpisodeRollout(trace=trace, samples=samples, reward=reward, scored=scored)


async def _finish_group(
    args: Any,
    base_sample: Any,
    sampling_params: dict[str, Any],
    entry: _GroupEntry,
    group_index: int,
) -> dict[str, Any]:
    outcomes = [entry.outcomes[key] for key in sorted(entry.outcomes)]
    all_scored = all(outcome.rollout.scored for outcome in outcomes)
    all_wrong = all_scored and all(not outcome.rollout.success for outcome in outcomes)
    triage: dict[str, Any] = {
        "schema_version": TRIAGE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "group_index": group_index,
        "curriculum": entry.curriculum.identity(),
        "assignment": outcomes[0].assignment.to_dict(),
        "training_rollouts": [
            {
                "sample_index": outcome.sample_index,
                "episode_id": outcome.rollout.trace["episode_id"],
                "reward": outcome.rollout.reward,
                "scored": outcome.rollout.scored,
            }
            for outcome in outcomes
        ],
        "all_wrong": all_wrong,
        "classification": "not_frontier",
        "full_skill_retry": None,
    }
    if not all_scored:
        triage["classification"] = "environment_or_runtime_error"
    elif all_wrong and bool(getattr(args, "shopsim_full_skill_retry", True)):
        retry_assignment = SkillAssignment(
            state_id=entry.curriculum.state_id,
            group_key=str(group_index),
            mode="diagnostic_full_skill",
            skills=entry.curriculum.full_skills(),
        )
        retry_base = copy.deepcopy(base_sample)
        retry_base.index = max(outcome.sample_index for outcome in outcomes) + 1
        retry = await _run_episode(
            args,
            retry_base,
            sampling_params,
            skills=retry_assignment.skills,
            assignment=retry_assignment,
            diagnostic=True,
        )
        retry_path = _trace_path(
            args, entry.curriculum, group_index, "full-skill-retry"
        )
        await asyncio.to_thread(atomic_write_json, retry_path, retry.trace)
        triage["full_skill_retry"] = {
            "trace_path": str(retry_path.resolve()),
            "episode_id": retry.trace["episode_id"],
            "reward": retry.reward,
            "scored": retry.scored,
        }
        if not retry.scored:
            triage["classification"] = "environment_or_runtime_error"
        elif retry.success:
            triage["classification"] = "model_internalization_deficit"
        else:
            # This is an analysis queue state, not an automatic skill-deficit
            # verdict.  A gold-aware Failure Analyst may still return NO_PROPOSAL.
            triage["classification"] = "full_skill_failure_pending_analysis"
    elif all_wrong:
        triage["classification"] = "all_wrong_retry_disabled"

    triage_path = _trace_path(args, entry.curriculum, group_index, "triage")
    await asyncio.to_thread(atomic_write_json, triage_path, triage)
    return triage


async def _coordinate_group(
    args: Any,
    base_sample: Any,
    sampling_params: dict[str, Any],
    curriculum: SkillCurriculum,
    outcome: _GroupOutcome,
) -> dict[str, Any]:
    group_index = int(base_sample.group_index)
    key = (curriculum.state_id, group_index)
    expected = int(args.n_samples_per_prompt)
    leader = False
    async with _loop_lock():
        entry = _GROUPS.get(key)
        if entry is None:
            entry = _GroupEntry(
                expected=expected,
                curriculum=curriculum,
                outcomes={},
            )
            _GROUPS[key] = entry
        if entry.expected != expected:
            raise RuntimeError("inconsistent n_samples_per_prompt within one group")
        if outcome.assignment.state_id != entry.curriculum.state_id:
            raise RuntimeError("curriculum changed inside one GRPO group")
        previous_assignments = {
            prior.assignment.skill_ids for prior in entry.outcomes.values()
        }
        if previous_assignments and outcome.assignment.skill_ids not in previous_assignments:
            raise RuntimeError("skill mask changed inside one GRPO group")
        if outcome.sample_index in entry.outcomes:
            raise RuntimeError("duplicate sample completion in one GRPO group")
        entry.outcomes[outcome.sample_index] = outcome
        if len(entry.outcomes) == entry.expected and not entry.running_retry:
            entry.running_retry = True
            leader = True
    if not leader:
        # custom generate runs while holding slime's generation semaphore.  A
        # barrier here could deadlock when semaphore capacity < group size.
        # The default group gather still waits for the last (leader) task.
        return {"status": "pending_group_completion"}
    try:
        return await _finish_group(
            args, base_sample, sampling_params, entry, group_index
        )
    finally:
        async with _loop_lock():
            _GROUPS.pop(key, None)
            curriculum_path = str(
                _project_path(_required_arg(args, "shopsim_curriculum_path"))
            )
            _CURRICULUM_CACHE.pop((curriculum_path, group_index), None)


async def generate(
    args: Any, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Any]:
    """slime ``--custom-generate-function-path`` entry point."""

    if getattr(args, "partial_rollout", False):
        raise AssertionError("ShopSimulator agent trajectories do not support partial rollout")
    if evaluation or _task_identity(sample)[1] != "train":
        raise ValueError("use the standalone evaluator for val/test; this adapter is train-only")
    if getattr(args, "group_rm", False):
        raise ValueError("ShopSimulator supplies terminal rewards; group_rm must be disabled")
    if sampling_params.get("top_p", 1.0) != 1.0 or sampling_params.get("top_k", -1) != -1:
        raise ValueError("ShopSimulator exact-token training requires top_p=1.0 and top_k=-1")
    group_index = int(sample.group_index)
    curriculum = _curriculum(args, group_index)
    assignment = curriculum.assign(group_index)
    rollout = await _run_episode(
        args,
        sample,
        sampling_params,
        skills=assignment.skills,
        assignment=assignment,
        diagnostic=False,
    )
    trace_path = _trace_path(args, curriculum, group_index, f"rollout-{sample.index}")
    await asyncio.to_thread(atomic_write_json, trace_path, rollout.trace)
    for result in rollout.samples:
        result.metadata = {
            **dict(getattr(result, "metadata", None) or {}),
            "shopsim_trace_path": str(trace_path.resolve()),
            "shopsim_reward": rollout.reward,
            "shopsim_scored": rollout.scored,
        }
    triage = await _coordinate_group(
        args,
        sample,
        sampling_params,
        curriculum,
        _GroupOutcome(
            sample_index=int(sample.index),
            rollout=rollout,
            assignment=assignment,
        ),
    )
    if triage.get("status") != "pending_group_completion":
        retry = triage.get("full_skill_retry")
        triage_metrics = {
            "all_wrong": bool(triage.get("all_wrong")),
            "classification": str(triage.get("classification", "unknown")),
            "full_skill_retry": retry is not None,
            "full_skill_retry_scored": (
                bool(retry.get("scored")) if isinstance(retry, dict) else None
            ),
            "full_skill_retry_success": (
                bool((retry.get("reward") or {}).get("r_success") == 1.0)
                if isinstance(retry, dict)
                else None
            ),
        }
        # Only the last trajectory receives this group-level record.  It may
        # contain several token-contiguous Samples; the rollout logger
        # deduplicates them by group before counting the record.
        for result in rollout.samples:
            result.metadata["shopsim_group_triage"] = triage_metrics
    return rollout.samples


def _sample_leaves(node: Any):
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from _sample_leaves(child)
    else:
        yield node


def keep_fully_scored_group(args: Any, group: Sequence[Any]) -> bool:
    """Dynamic-sampling filter: technical failures invalidate the whole group."""

    global _FAILED_GROUP_STREAK

    samples = list(_sample_leaves(group))
    keep = bool(samples) and all(
        (getattr(sample, "metadata", None) or {}).get("shopsim_scored") is True
        for sample in samples
    )
    if keep:
        _FAILED_GROUP_STREAK = 0
        return True
    _FAILED_GROUP_STREAK += 1
    failure_limit = int(
        getattr(args, "shopsim_max_consecutive_failed_groups", 8) if args else 8
    )
    if failure_limit < 1:
        raise ValueError("shopsim_max_consecutive_failed_groups must be positive")
    if _FAILED_GROUP_STREAK >= failure_limit:
        raise RuntimeError(
            "ShopSimulator produced too many consecutive unscored groups; "
            "inspect environment/runtime traces before resuming"
        )
    return False


def group_has_reward_spread(args: Any, group: Sequence[Any]) -> bool:
    """Whether the group can produce a non-zero GRPO advantage.

    Fan-out siblings of one execution repeat that execution's reward, so the
    spread is measured over unique ``rollout_id`` values, matching how
    ``normalize_grpo_by_prompt_and_rollout`` centers advantages.
    """

    rewards: dict[int, float] = {}
    for sample in _sample_leaves(group):
        rollout_id = int(
            sample.rollout_id if sample.rollout_id is not None else sample.index
        )
        rewards.setdefault(rollout_id, float(sample.get_reward_value(args)))
    if len(rewards) < 2:
        return False
    return max(rewards.values()) - min(rewards.values()) > 1e-6


def fully_scored_group_filter(args: Any, group: Sequence[Any]) -> Any:
    """slime dynamic-filter entry point with an observable drop reason."""

    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    if not keep_fully_scored_group(args, group):
        # Unscored groups have no usable reward, so they are never salvageable.
        return DynamicFilterOutput(
            keep=False,
            reason="shopsim_unscored_technical_group",
        )
    if group_has_reward_spread(args, group):
        return DynamicFilterOutput(keep=True)
    # All siblings tied: the advantage is zero, so the group trains on nothing.
    # Resample instead, unless that would cost another rollout round.
    return DynamicFilterOutput(
        keep=False,
        reason="shopsim_zero_std_group",
        keep_when_insufficient=True,
    )


def normalize_grpo_by_prompt_and_rollout(
    args: Any, samples: Sequence[Any]
) -> tuple[list[float], list[float]]:
    """Normalize unique rollout rewards within each prompt group.

    Dynamic tool schemas can make one multi-turn execution fan out into several
    token-contiguous slime Samples.  All siblings share ``rollout_id``.  GRPO
    centering must count that execution once, then broadcast its advantage to
    the siblings, rather than weighting long trajectories more heavily.
    """

    import torch

    raw = [float(sample.get_reward_value(args)) for sample in samples]
    if not getattr(args, "rewards_normalization", True):
        return raw, raw
    grouped: dict[int, dict[int, list[int]]] = {}
    for position, sample in enumerate(samples):
        group = int(sample.group_index)
        rollout_id = int(
            sample.rollout_id if sample.rollout_id is not None else sample.index
        )
        grouped.setdefault(group, {}).setdefault(rollout_id, []).append(position)
    normalized = [0.0] * len(samples)
    for rollouts in grouped.values():
        rollout_ids = sorted(rollouts)
        rewards = []
        for rollout_id in rollout_ids:
            values = {raw[position] for position in rollouts[rollout_id]}
            if len(values) != 1:
                raise ValueError("fan-out siblings carry different trajectory rewards")
            rewards.append(values.pop())
        tensor = torch.tensor(rewards, dtype=torch.float)
        tensor = tensor - tensor.mean()
        if getattr(args, "grpo_std_normalization", True) and len(rewards) > 1:
            tensor = tensor / (tensor.std() + 1e-6)
        for rollout_id, advantage in zip(rollout_ids, tensor.tolist(), strict=True):
            for position in rollouts[rollout_id]:
                normalized[position] = advantage
    return raw, normalized
