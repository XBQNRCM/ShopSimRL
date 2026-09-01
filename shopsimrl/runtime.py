"""Single-episode agent runtime with complete, analysis-ready traces."""

from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from .environment import ShopEnvironment
from .model import ChatModel
from .prompts import ShoppingPromptBuilder
from .schemas import (
    TRACE_SCHEMA_VERSION,
    EpisodeJob,
    ModelOutput,
    Skill,
    ToolCall,
    utc_now,
)
from .skills import NoSkills, SkillProvider


ACTION_PROTOCOL_VERSION = "openai-function-tools-repair-v3"


@dataclass(frozen=True)
class RuntimeConfig:
    max_steps: int = 30

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")


def _without_lease(payload: dict[str, Any]) -> dict[str, Any]:
    hidden = {"env_idx", "lease_id"}
    return {key: value for key, value in payload.items() if key not in hidden}


def _public_reset(payload: dict[str, Any]) -> dict[str, Any]:
    hidden = {
        "env_idx",
        "lease_id",
        "private_task_instruction",
        "goal_options",
        "reason_key",
    }
    return {key: value for key, value in payload.items() if key not in hidden}


def _action_tools(state: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    tools = []
    if state.get("search_available") is True:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "使用关键词搜索商品。仅在当前页面提供此工具时调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "minLength": 1,
                                "description": "用于检索商品的简洁关键词。",
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    actions = state.get("actions")
    if not isinstance(actions, list) or any(
        not isinstance(value, str) or not value for value in actions
    ):
        raise RuntimeError("observation_state.actions must be a list of strings")
    if actions:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "点击当前页面的一个可用按钮、商品或规格选项。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "enum": actions,
                                "description": "必须逐字选择当前可点击值之一。",
                            }
                        },
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    if not tools:
        raise RuntimeError("current observation exposes no callable action tools")
    return tuple(tools)


def _environment_action(call: ToolCall) -> str:
    if call.name == "search":
        if set(call.arguments) != {"query"}:
            raise RuntimeError("search tool requires exactly one query argument")
        query = call.arguments["query"]
        if not isinstance(query, str) or not query.strip():
            raise RuntimeError("search query must be a non-empty string")
        return f"search[{query.strip()}]"
    if call.name == "click":
        if set(call.arguments) != {"value"}:
            raise RuntimeError("click tool requires exactly one value argument")
        value = call.arguments["value"]
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError("click value must be a non-empty string")
        return f"click[{value.strip()}]"
    raise RuntimeError(f"unsupported action tool: {call.name}")


def _protocol_feedback(
    error: dict[str, Any], tools: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    return {
        "type": "tool_protocol_error",
        "code": error["code"],
        "message": error["message"],
        "instruction": (
            "上一轮未执行任何环境动作。请基于同一页面重新作答，只调用一个"
            "当前提供的函数工具，不要输出可见文本。"
        ),
        "available_tools": [tool["function"]["name"] for tool in tools],
    }


def _append_protocol_feedback(
    messages: list[dict[str, Any]],
    model_output: ModelOutput,
    feedback: dict[str, Any],
) -> None:
    content = json.dumps(feedback, ensure_ascii=False, separators=(",", ":"))
    if model_output.tool_calls:
        messages.append(model_output.assistant_message())
        for call in model_output.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": content,
                }
            )
        return
    if model_output.content and model_output.content.strip():
        messages.append(
            {"role": "assistant", "content": model_output.content}
        )
    messages.append({"role": "user", "content": content})


class AgentRuntime:
    """Owns exactly one model and one environment for exactly one episode."""

    def __init__(
        self,
        *,
        model: ChatModel,
        environment: ShopEnvironment,
        prompt_builder: ShoppingPromptBuilder | None = None,
        skill_provider: SkillProvider | None = None,
        config: RuntimeConfig | None = None,
    ):
        self.model = model
        self.environment = environment
        self.prompt_builder = prompt_builder or ShoppingPromptBuilder()
        self.skill_provider = skill_provider or NoSkills()
        self.config = config or RuntimeConfig()

    def run(self, job: EpisodeJob) -> dict[str, Any]:
        started_at = utc_now()
        started = perf_counter()
        stage = "reset"
        reset: dict[str, Any] | None = None
        messages: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        selected_skills: tuple[Skill, ...] = ()
        final: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        status = "failed"

        try:
            reset = self.environment.reset(job.task_id)
            stage = "skill_selection"
            skill_context = {
                "task_id": job.task_id,
                "sample_id": job.sample_id,
                "split": job.split,
                "task_instruction": reset["task_instruction"],
                "user_persona": reset.get("user_persona"),
                "observation_state": reset.get("observation_state"),
                "task_mode": reset.get("task_mode"),
            }
            selected_skills = tuple(self.skill_provider.select(skill_context))
            system = self.prompt_builder.system_message(reset, selected_skills)
            messages.append({"role": "system", "content": system})
            observation = self.prompt_builder.initial_observation(reset)
            observation_state = reset["observation_state"]
            messages.append({"role": "user", "content": observation})

            for step_index in range(1, self.config.max_steps + 1):
                tools = _action_tools(observation_state)
                stage = f"model_step_{step_index}"
                model_output = self.model.generate(
                    messages,
                    seed=job.seed + step_index - 1,
                    tools=tools,
                )
                policy_failure = model_output.policy_failure
                if policy_failure is not None:
                    if policy_failure.get("code") != "generation_length":
                        raise RuntimeError(
                            f"unsupported model policy failure: {policy_failure!r}"
                        )
                    steps.append(
                        {
                            "step_index": step_index,
                            "observation": observation,
                            "tools": list(tools),
                            "model": model_output.to_dict(),
                            "policy_failure": policy_failure,
                            "action": None,
                            "environment": None,
                        }
                    )
                    stage = f"policy_termination_step_{step_index}"
                    final = _without_lease(
                        self.environment.terminate("generation_length")
                    )
                    break
                protocol_error = model_output.protocol_error
                if protocol_error is None:
                    if len(model_output.tool_calls) != 1:
                        protocol_error = {
                            "code": "tool_call_count",
                            "message": (
                                "model must return exactly one tool call, got "
                                f"{len(model_output.tool_calls)}"
                            ),
                        }
                    elif model_output.content and model_output.content.strip():
                        protocol_error = {
                            "code": "mixed_content",
                            "message": (
                                "model must not mix natural-language content with an "
                                "action tool call"
                            ),
                        }
                    else:
                        tool_call = model_output.tool_calls[0]
                        try:
                            action = _environment_action(tool_call)
                        except RuntimeError as exc:
                            protocol_error = {
                                "code": "invalid_tool_arguments",
                                "message": str(exc),
                            }
                if protocol_error is not None:
                    feedback = _protocol_feedback(protocol_error, tools)
                    _append_protocol_feedback(messages, model_output, feedback)
                    model_record = model_output.to_dict()
                    model_record["protocol_error"] = protocol_error
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
                messages.append(model_output.assistant_message())
                stage = f"environment_step_{step_index}"
                environment_output = self.environment.step(action)
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
                        "model": model_output.to_dict(),
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
                stage = "action_limit_termination"
                final = _without_lease(
                    self.environment.terminate("action_limit")
                )
            if final.get("done") is not True:
                raise RuntimeError("episode did not reach a terminal environment state")
            status = "completed"
        except Exception as exc:
            error = {
                "stage": stage,
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        finally:
            release_errors: list[str] = []
            try:
                self.environment.close()
            except Exception as exc:
                release_errors.append(f"environment.close: {type(exc).__name__}: {exc}")
            try:
                self.model.close()
            except Exception as exc:
                release_errors.append(f"model.close: {type(exc).__name__}: {exc}")
            if release_errors:
                if error is None:
                    error = {
                        "stage": "cleanup",
                        "type": "CleanupError",
                        "message": "; ".join(release_errors),
                    }
                else:
                    error["cleanup_errors"] = release_errors

        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "episode_id": job.episode_id,
            "status": status,
            "job": job.to_dict(),
            "provenance": {
                "model": self.model.identity(),
                "environment": self.environment.identity(),
                "prompt": self.prompt_builder.identity(),
                "skills": self.skill_provider.identity(),
                "runtime": {
                    "max_steps": self.config.max_steps,
                    "action_protocol": ACTION_PROTOCOL_VERSION,
                },
            },
            "started_at": started_at,
            "ended_at": utc_now(),
            "duration_ms": (perf_counter() - started) * 1000,
            "reset": _public_reset(reset) if reset is not None else None,
            "selected_skills": [skill.to_dict() for skill in selected_skills],
            "conversation": messages,
            "steps": steps,
            "final": final,
            "error": error,
        }
