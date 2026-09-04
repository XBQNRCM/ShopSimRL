"""Train-only Trace2Skill cold start with auditable, resumable artifacts."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import threading
from typing import Any, Callable, Iterable, Sequence

from .environment import ShopSimulatorConfig, ShopSimulatorHTTPEnvironment
from .model import ChatModel, OpenAICompatibleChatModel
from .runtime import _action_tools, _environment_action
from .schemas import TRACE_SCHEMA_VERSION, fingerprint, utc_now
from .store import atomic_write_json, safe_name
from .trace2skill_config import Trace2SkillSpec
from .trace2skill_prompts import (
    CLUSTER_TOOL,
    COMPILER_SYSTEM_PROMPT,
    CONSOLIDATION_SYSTEM_PROMPT,
    FAILURE_SYSTEM_PROMPT,
    FAILURE_TOOL,
    INITIAL_SKILL_TOOL,
    SUCCESS_SYSTEM_PROMPT,
    SUCCESS_TOOL,
)


TRACE2SKILL_SCHEMA_VERSION = "shopsimrl-trace2skill-cold-start-v3"
CARD_SCHEMA_VERSION = "shopsimrl-evidence-card-v3"
CLUSTER_SCHEMA_VERSION = "shopsimrl-evidence-cluster-v2"
INITIAL_SKILL_SCHEMA_VERSION = "shopsimrl-initial-skill-draft-v2"


class Trace2SkillError(RuntimeError):
    pass


class StructuredOutputError(Trace2SkillError):
    pass


Progress = Callable[[str], None]
ModelFactory = Callable[[Any], ChatModel]
EnvironmentFactory = Callable[[], Any]
InvalidOutputCallback = Callable[[dict[str, Any]], None]


def load_latest_traces(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load the last valid JSON object for every episode_id in a JSONL file."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"trajectory file not found: {source}")
    latest: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Trace2SkillError(
                    f"invalid JSON at {source}:{line_number}: {exc}"
                ) from exc
            episode_id = payload.get("episode_id") if isinstance(payload, dict) else None
            if not isinstance(episode_id, str) or not episode_id:
                raise Trace2SkillError(
                    f"trace at {source}:{line_number} has no episode_id"
                )
            latest[episode_id] = payload
    return tuple(latest[key] for key in sorted(latest))


def validate_cold_start_traces(
    traces: Sequence[dict[str, Any]], *, expected: int
) -> dict[str, int]:
    if len(traces) != expected:
        raise Trace2SkillError(
            f"expected {expected} unique trajectories, found {len(traces)}"
        )
    counts: Counter[str] = Counter()
    for trace in traces:
        episode_id = trace.get("episode_id", "<unknown>")
        if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise Trace2SkillError(f"{episode_id}: unsupported trace schema")
        job = trace.get("job") or {}
        if job.get("split") != "train":
            raise Trace2SkillError(f"{episode_id}: cold start only accepts train traces")
        final = trace.get("final")
        if (
            trace.get("status") != "completed"
            or not isinstance(final, dict)
            or final.get("done") is not True
        ):
            raise Trace2SkillError(f"{episode_id}: trajectory is not completed")
        success = (final.get("reward_detail") or {}).get("r_success")
        if success not in {0, 1, 0.0, 1.0}:
            raise Trace2SkillError(f"{episode_id}: missing binary r_success")
        counts["success" if float(success) == 1.0 else "failure"] += 1
    counts["total"] = len(traces)
    return dict(counts)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _redact_ids(text: str) -> str:
    text = re.sub(r"(?i)\bASIN\s*[:=]?\s*[A-Z0-9-]+", "product id: <redacted>", text)
    text = re.sub(r"\b\d{10,}\b", "<product_id>", text)
    text = re.sub(r"(?i)\btask[-_ ]?\d+\b", "<task_id>", text)
    return text


def _compact_persona(persona: Any) -> Any:
    if not isinstance(persona, dict):
        return persona
    hidden = {"用户ID", "注册时间", "最后更新时间"}
    return {key: value for key, value in persona.items() if key not in hidden}


def compact_trace(trace: dict[str, Any], *, include_gold: bool) -> dict[str, Any]:
    """Keep decision-relevant trace data while separating privileged gold data."""
    reset = trace.get("reset") or {}
    steps = []
    for raw_step in trace.get("steps") or []:
        model = raw_step.get("model") or {}
        environment = raw_step.get("environment") or {}
        state = environment.get("observation_state") or {}
        product = state.get("product") or {}
        step = {
            "step_index": raw_step.get("step_index"),
            "observation": _redact_ids(str(raw_step.get("observation", ""))),
            "reasoning": _redact_ids(str(model.get("reasoning") or "")),
            "action": _redact_ids(str(raw_step.get("action") or "")),
            "protocol_error": raw_step.get("protocol_error"),
            "action_feedback": environment.get("action_feedback"),
            "resulting_page_type": state.get("page_type"),
            "resulting_product": {
                key: _redact_ids(str(product[key]))
                for key in ("title", "brand", "category", "key_attributes", "price")
                if key in product
            },
            "selected_options": state.get("selected_options"),
            "missing_option_axes": state.get("missing_option_axes"),
            "selected_price": state.get("selected_price"),
        }
        steps.append(step)
    final = trace.get("final") or {}
    purchase = final.get("purchase") or {}
    public_purchase = (
        {
            key: purchase.get(key)
            for key in ("attributes", "category", "options", "price", "product_category")
            if key in purchase
        }
        if isinstance(purchase, dict)
        else {}
    )
    payload = {
        "source_trajectory_id": trace["episode_id"],
        "task_instruction": reset.get("task_instruction"),
        "user_persona": _compact_persona(reset.get("user_persona")),
        "trajectory": steps,
        "outcome": {
            "reward": final.get("reward"),
            "reward_detail": final.get("reward_detail"),
            "termination_reason": final.get("termination_reason"),
            "purchase": public_purchase,
        },
    }
    if include_gold:
        payload["privileged_gold_audit"] = {
            "goal": final.get("goal"),
            "chosen_purchase": final.get("purchase"),
        }
    return payload


def _tool_arguments(output: Any, expected_name: str) -> dict[str, Any]:
    if output.protocol_error is not None or output.policy_failure is not None:
        raise StructuredOutputError(
            f"model rejected structured output: "
            f"{output.protocol_error or output.policy_failure}"
        )
    if len(output.tool_calls) != 1 or output.tool_calls[0].name != expected_name:
        names = [call.name for call in output.tool_calls]
        raise StructuredOutputError(
            f"expected one {expected_name} call, received {names}"
        )
    return dict(output.tool_calls[0].arguments)


def _require_text_fields(
    payload: dict[str, Any], fields: Sequence[str], *, context: str
) -> None:
    invalid = [
        field
        for field in fields
        if not isinstance(payload.get(field), str) or not payload[field].strip()
    ]
    if invalid:
        raise StructuredOutputError(
            f"{context} has missing/empty text fields: {', '.join(invalid)}"
        )


def _validated_id_list(
    payload: dict[str, Any],
    field: str,
    *,
    allowed: set[str],
    context: str,
    require_nonempty: bool = False,
) -> list[str]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise StructuredOutputError(f"{context} {field} must be a list")
    if require_nonempty and not value:
        raise StructuredOutputError(f"{context} {field} must not be empty")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise StructuredOutputError(f"{context} {field} must contain text IDs")
    if len(value) != len(set(value)):
        raise StructuredOutputError(f"{context} {field} contains duplicate IDs")
    unknown = set(value) - allowed
    if unknown:
        raise StructuredOutputError(
            f"{context} {field} cited unknown IDs: {sorted(unknown)}"
        )
    return value


def _structured_call(
    model: ChatModel,
    messages: Sequence[dict[str, Any]],
    *,
    tool: dict[str, Any],
    expected_name: str,
    seed: int,
    attempts: int = 3,
    validate: Callable[[dict[str, Any]], None] | None = None,
    on_invalid: InvalidOutputCallback | None = None,
) -> dict[str, Any]:
    base_messages = list(messages)
    working_messages = base_messages
    last_error: StructuredOutputError | None = None
    for attempt in range(attempts):
        output = model.generate(
            working_messages,
            seed=seed + attempt,
            tools=(tool,),
        )
        arguments: dict[str, Any] | None = None
        try:
            arguments = _tool_arguments(output, expected_name)
            if validate is not None:
                validate(arguments)
            return arguments
        except StructuredOutputError as exc:
            last_error = exc
            invalid_tool_calls = [
                {"name": call.name, "arguments": call.arguments}
                for call in output.tool_calls
            ]
            event = {
                "attempt_number": attempt + 1,
                "max_attempts": attempts,
                "expected_tool": expected_name,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "arguments": arguments,
                "tool_calls": invalid_tool_calls,
                "protocol_error": output.protocol_error,
                "policy_failure": output.policy_failure,
            }
            if on_invalid is not None:
                on_invalid(event)
            if attempt + 1 < attempts:
                repair_payload = {
                    "structured_output_repair": {
                        "contract_error": str(exc),
                        "previous_invalid_arguments": arguments,
                        "previous_tool_calls": invalid_tool_calls,
                        "instruction": (
                            f"Correct the previous output, then call {expected_name} "
                            "exactly once with schema-valid arguments and no prose. "
                            "Preserve valid content, but resolve the stated contract "
                            "violation exactly."
                        ),
                    }
                }
                # Retain only the latest invalid draft. Accumulating every failed
                # draft wastes context and makes targeted correction ambiguous.
                working_messages = base_messages + [
                    {
                        "role": "user",
                        "content": json.dumps(
                            repair_payload,
                            ensure_ascii=False,
                            allow_nan=False,
                            separators=(",", ":"),
                        ),
                    }
                ]
    raise last_error or StructuredOutputError("structured output failed")


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _strings(child)


def _known_titles(trace: dict[str, Any]) -> set[str]:
    titles: set[str] = set()
    final = trace.get("final") or {}
    for record in (final.get("goal"), final.get("purchase")):
        if isinstance(record, dict):
            title = record.get("name") or record.get("title")
            if isinstance(title, str) and len(title.strip()) >= 6:
                titles.add(title.strip())
    for step in trace.get("steps") or []:
        environment = step.get("environment") or {}
        product = (environment.get("observation_state") or {}).get("product") or {}
        title = product.get("title")
        if isinstance(title, str) and len(title.strip()) >= 6:
            titles.add(title.strip())
    return titles


def _known_brands(trace: dict[str, Any]) -> set[str]:
    brands: set[str] = set()
    for step in trace.get("steps") or []:
        environment = step.get("environment") or {}
        product = (environment.get("observation_state") or {}).get("product") or {}
        brand = product.get("brand")
        if isinstance(brand, str) and len(brand.strip()) >= 3:
            brands.add(brand.strip())
    return brands


def _known_gold_options(trace: dict[str, Any]) -> set[str]:
    goal = (trace.get("final") or {}).get("goal") or {}
    options = goal.get("goal_options") if isinstance(goal, dict) else None
    if not isinstance(options, list):
        return set()
    return {
        option.strip()
        for option in options
        if isinstance(option, str) and len(option.strip()) >= 6
    }


def firewall_findings(value: Any, *, trace: dict[str, Any] | None = None) -> list[str]:
    findings: set[str] = set()
    combined = "\n".join(_strings(value))
    if re.search(r"(?i)\bASIN\b", combined):
        findings.add("mentions_asin")
    if re.search(r"\b\d{10,}\b", combined):
        findings.add("contains_long_identifier")
    if re.search(r"(?i)\btask[-_ ]?\d+\b", combined):
        findings.add("contains_task_id")
    if trace is not None:
        for title in _known_titles(trace):
            if title and title in combined:
                findings.add("copies_instance_product_title")
                break
        for brand in _known_brands(trace):
            if brand and brand in combined:
                findings.add("copies_instance_brand")
                break
        for option in _known_gold_options(trace):
            if option and option in combined:
                findings.add("copies_gold_option")
                break
    return sorted(findings)


class JsonlIndex:
    """Thread-safe append-only JSONL with latest-record resume semantics."""

    def __init__(self, path: Path, key: str):
        self.path = path
        self.key = key
        self.lock = threading.Lock()
        self.latest = self._load()

    def _load(self) -> dict[str, dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return records
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = payload.get(self.key) if isinstance(payload, dict) else None
                if isinstance(value, str):
                    records[value] = payload
        return records

    def append(self, payload: dict[str, Any]) -> None:
        key_value = payload.get(self.key)
        if not isinstance(key_value, str):
            raise ValueError(f"record has no string {self.key}")
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line)
                file.flush()
                os.fsync(file.fileno())
            self.latest[key_value] = payload


def _atomic_write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _validate_success_arguments(
    arguments: dict[str, Any], *, max_cards: int
) -> None:
    raw_cards = arguments.get("cards")
    if not isinstance(raw_cards, list):
        raise StructuredOutputError("success output cards must be a list")
    if len(raw_cards) > max_cards:
        raise StructuredOutputError("success analyst exceeded per-trajectory card cap")
    for card in raw_cards:
        if not isinstance(card, dict):
            raise StructuredOutputError("success card must be an object")
        _require_text_fields(
            card,
            (
                "mechanism", "applicable_when", "observed_evidence",
                "decisive_behavior", "generalizable_lesson",
            ),
            context="success card",
        )
        if not isinstance(card.get("related_chunk_ids"), list):
            raise StructuredOutputError("success related_chunk_ids must be a list")
        confidence = card.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            raise StructuredOutputError("success confidence must be in [0, 1]")


def _success_cards(
    trace: dict[str, Any],
    *,
    model_factory: ModelFactory,
    spec: Trace2SkillSpec,
) -> list[dict[str, Any]]:
    model = model_factory(spec.analyst_model)
    try:
        prompt = {
            "max_cards": spec.max_cards_per_trajectory,
            "trace": compact_trace(trace, include_gold=False),
            "current_skill": None,
        }
        arguments = _structured_call(
            model,
            [
                {"role": "system", "content": SUCCESS_SYSTEM_PROMPT},
                {"role": "user", "content": _json(prompt)},
            ],
            seed=spec.seed + int((trace.get("job") or {}).get("task_id", 0)),
            tool=SUCCESS_TOOL,
            expected_name="submit_success_cards",
            validate=lambda payload: _validate_success_arguments(
                payload, max_cards=spec.max_cards_per_trajectory
            ),
        )
    finally:
        model.close()
    raw_cards = arguments["cards"]
    records = []
    for index, card in enumerate(raw_cards, 1):
        findings = firewall_findings(card, trace=trace)
        records.append(
            {
                "schema_version": CARD_SCHEMA_VERSION,
                "card_id": f"success-{trace['episode_id']}-{index:02d}",
                "source_trajectory_id": trace["episode_id"],
                "channel": "success",
                **card,
                "firewall_findings": findings,
                "eligible_for_consolidation": not findings,
                "created_at": utc_now(),
            }
        )
    return records


def _analysis_fingerprint(trace: dict[str, Any], spec: Trace2SkillSpec) -> str:
    return fingerprint(
        {
            "pipeline": TRACE2SKILL_SCHEMA_VERSION,
            "trace": trace,
            "analyst_model": spec.analyst_model.identity(),
            "prompts": {
                "success": fingerprint(SUCCESS_SYSTEM_PROMPT),
                "failure": fingerprint(FAILURE_SYSTEM_PROMPT),
            },
            "tools": {
                "success": fingerprint(SUCCESS_TOOL),
                "failure": fingerprint(FAILURE_TOOL),
            },
            "max_cards": spec.max_cards_per_trajectory,
            "max_failure_steps": spec.max_failure_analysis_steps,
            "environment": {
                "base_url": spec.environment_base_url,
                "persona": spec.environment_persona,
            },
        }
    )


def _reported_model_identity(model_config: Any) -> dict[str, Any]:
    """Add transport-only settings to reports without changing artifact hashes."""
    identity = model_config.identity()
    identity["transport"] = {
        **identity.get("transport", {}),
        "stream": model_config.stream,
        "trust_env": model_config.trust_env,
    }
    return identity


def _available_action_tools(state: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    actions = state.get("actions")
    if state.get("search_available") is True or (
        isinstance(actions, list) and bool(actions)
    ):
        return _action_tools(state)
    return ()


def _failure_card(
    trace: dict[str, Any],
    *,
    model_factory: ModelFactory,
    environment_factory: EnvironmentFactory,
    spec: Trace2SkillSpec,
    current_skill: Sequence[dict[str, Any]] | None = None,
    stage: str = "cold_start",
    allowed_rewrite_targets: set[str] | None = None,
) -> dict[str, Any]:
    if stage not in {"cold_start", "online"}:
        raise ValueError(f"unsupported failure-analysis stage: {stage!r}")
    model = model_factory(spec.analyst_model)
    environment = environment_factory()
    grounded_actions = 0
    audit_actions: list[dict[str, Any]] = []
    audit_transcript: list[dict[str, Any]] = []
    terminal_trials: list[dict[str, Any]] = []
    trial_index = 1
    try:
        task_id = int((trace.get("job") or {})["task_id"])
        reset = environment.reset(task_id)
        state = reset["observation_state"]
        context = {
            "analysis_contract": {
                "current_skill": list(current_skill or ()),
                "stage": stage,
                "max_live_actions": spec.max_failure_analysis_steps - 1,
            },
            "failed_trace_and_privileged_gold": compact_trace(
                trace, include_gold=True
            ),
            "fresh_live_session": {
                "task_instruction": reset.get("task_instruction"),
                "user_persona": _compact_persona(reset.get("user_persona")),
                "observation": reset.get("observation"),
                "observation_state": state,
            },
        }
        messages = [
            {"role": "system", "content": FAILURE_SYSTEM_PROMPT},
            {"role": "user", "content": _json(context)},
        ]
        for step_index in range(1, spec.max_failure_analysis_steps + 1):
            # Reserve the final decision round for a diagnosis submission.
            tools = (
                (FAILURE_TOOL,)
                if step_index == spec.max_failure_analysis_steps
                else (*_available_action_tools(state), FAILURE_TOOL)
            )
            output = model.generate(
                messages,
                seed=spec.seed + task_id + step_index - 1,
                tools=tools,
            )
            if output.protocol_error is not None or output.policy_failure is not None:
                audit_transcript.append(
                    {
                        "trial_index": trial_index,
                        "step_index": step_index,
                        "type": "model_protocol_error",
                        "error": output.protocol_error or output.policy_failure,
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response violated the tool contract: "
                            f"{output.protocol_error or output.policy_failure}. "
                            "Use exactly one currently available tool and emit no prose."
                        ),
                    }
                )
                continue
            if len(output.tool_calls) != 1:
                raise StructuredOutputError("failure analyst must call one tool")
            call = output.tool_calls[0]
            if call.name == "submit_failure_card":
                arguments = dict(call.arguments)
                break
            if call.name not in {"search", "click"}:
                raise StructuredOutputError(f"unsupported failure audit tool: {call.name}")
            action = _environment_action(call)
            environment_output = environment.step(action)
            feedback = environment_output.get("action_feedback") or {}
            if feedback.get("valid") is True:
                grounded_actions += 1
            audit_actions.append(
                {
                    "trial_index": trial_index,
                    "step_index": step_index,
                    "action": action,
                    "valid": feedback.get("valid"),
                    "page_type": (environment_output.get("observation_state") or {}).get(
                        "page_type"
                    ),
                }
            )
            public_environment_result = {
                "trial_index": trial_index,
                "action": action,
                "action_feedback": feedback,
                "done": environment_output.get("done") is True,
                "observation": environment_output.get("observation"),
                "observation_state": environment_output.get("observation_state"),
                "reward": environment_output.get("reward"),
                "reward_detail": environment_output.get("reward_detail"),
                "purchase": environment_output.get("purchase"),
                "termination_reason": environment_output.get("termination_reason"),
            }
            audit_transcript.append(
                {
                    "trial_index": trial_index,
                    "step_index": step_index,
                    "type": "environment_action",
                    "model_reasoning": output.reasoning,
                    "tool_call": call.to_dict(),
                    "environment_result": public_environment_result,
                }
            )
            messages.append(output.assistant_message())
            if environment_output.get("done") is True:
                reward_detail = environment_output.get("reward_detail") or {}
                trial_result = {
                    "trial_index": trial_index,
                    "terminal_action": action,
                    "reward": environment_output.get("reward"),
                    "reward_detail": reward_detail,
                    "purchase": environment_output.get("purchase"),
                    "termination_reason": environment_output.get(
                        "termination_reason"
                    ),
                    "success": reward_detail.get("r_success") in {1, 1.0},
                }
                terminal_trials.append(trial_result)

                # A ShopSimulator purchase terminates one episode. The audit
                # runner opens a fresh episode for the same task so the analyst
                # can inspect another candidate or counterfactual branch.
                environment.close()
                next_environment = environment_factory()
                try:
                    next_reset = next_environment.reset(task_id)
                except Exception:
                    next_environment.close()
                    raise
                environment = next_environment
                trial_index += 1
                state = next_reset["observation_state"]
                public_environment_result["next_trial"] = {
                    "trial_index": trial_index,
                    "task_instruction": next_reset.get("task_instruction"),
                    "observation": next_reset.get("observation"),
                    "observation_state": state,
                }
            else:
                state = environment_output["observation_state"]
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": _json(public_environment_result),
                }
            )
        else:
            raise StructuredOutputError("failure analyst exhausted its step budget")
    finally:
        try:
            environment.close()
        finally:
            model.close()

    status = arguments.get("status")
    privileged = arguments.get("privileged_audit")
    deployable = arguments.get("deployable_abstraction")
    if status not in {"PROPOSE_ADD", "PROPOSE_REWRITE", "NO_PROPOSAL"}:
        raise StructuredOutputError(f"invalid failure card status: {status!r}")
    if not isinstance(privileged, dict) or not isinstance(deployable, dict):
        raise StructuredOutputError("failure card sections must be objects")
    _require_text_fields(
        privileged,
        (
            "failure_surface", "earliest_divergence", "oracle_comparison",
            "replay_evidence", "minimal_repair", "repair_validation_result",
        ),
        context="privileged failure audit",
    )
    if status != "NO_PROPOSAL":
        _require_text_fields(
            deployable,
            (
                "failure_mechanism", "applicable_when", "observable_trigger",
                "corrective_procedure", "verification_step", "proposed_content",
            ),
            context="deployable failure abstraction",
        )
    forced_reason = None
    findings = firewall_findings(deployable, trace=trace)
    successful_trials = [trial for trial in terminal_trials if trial["success"]]
    if status != "NO_PROPOSAL" and grounded_actions < 1:
        forced_reason = "proposal_without_grounded_live_action"
    if status != "NO_PROPOSAL" and not successful_trials:
        forced_reason = "proposal_without_successful_counterfactual"
    if status == "PROPOSE_REWRITE" and stage == "cold_start":
        forced_reason = "cold_start_has_no_rewrite_target"
    if status == "PROPOSE_REWRITE" and stage == "online":
        target = deployable.get("target_chunk_id")
        if (
            not isinstance(target, str)
            or not target
            or allowed_rewrite_targets is None
            or target not in allowed_rewrite_targets
        ):
            forced_reason = "invalid_rewrite_target"
    if findings:
        forced_reason = "gold_firewall_violation"
    if forced_reason is not None:
        status = "NO_PROPOSAL"
    if status == "NO_PROPOSAL":
        deployable = {
            "failure_mechanism": deployable.get("failure_mechanism", ""),
            "applicable_when": "",
            "observable_trigger": "",
            "corrective_procedure": "",
            "verification_step": "",
            "target_chunk_id": "",
            "proposed_content": "",
        }
    return {
        "schema_version": CARD_SCHEMA_VERSION,
        "card_id": f"failure-{trace['episode_id']}-01",
        "source_trajectory_id": trace["episode_id"],
        "channel": "failure",
        "stage": stage,
        "status": status,
        "privileged_audit": privileged,
        "deployable_abstraction": deployable,
        "audit_action_summary": audit_actions,
        "audit_transcript": audit_transcript,
        "terminal_trials": terminal_trials,
        "successful_counterfactual_trials": len(successful_trials),
        "grounded_live_actions": grounded_actions,
        "firewall_findings": findings,
        "forced_no_proposal_reason": forced_reason,
        "eligible_for_consolidation": status != "NO_PROPOSAL" and not findings,
        "created_at": utc_now(),
    }


def analyze_failure_trace(
    trace: dict[str, Any],
    *,
    model_factory: ModelFactory,
    environment_factory: EnvironmentFactory,
    spec: Trace2SkillSpec,
    current_skill: Sequence[dict[str, Any]],
    allowed_rewrite_targets: set[str],
) -> dict[str, Any]:
    """Analyze an online full-skill retry failure.

    This public boundary preserves the cold-start implementation's live replay,
    successful-counterfactual requirement, and gold firewall while enabling
    REWRITE only against a currently active logical chunk.
    """

    return _failure_card(
        trace,
        model_factory=model_factory,
        environment_factory=environment_factory,
        spec=spec,
        current_skill=current_skill,
        stage="online",
        allowed_rewrite_targets=set(allowed_rewrite_targets),
    )


def analyze_trajectories(
    spec: Trace2SkillSpec,
    *,
    model_factory: ModelFactory = OpenAICompatibleChatModel,
    environment_factory: EnvironmentFactory | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    traces = load_latest_traces(spec.traces_path)
    input_counts = validate_cold_start_traces(
        traces, expected=spec.expected_trajectories
    )
    output_dir = spec.output_dir
    analyses = JsonlIndex(
        output_dir / "trajectory_analyses.jsonl", "source_trajectory_id"
    )
    errors = JsonlIndex(output_dir / "analysis_errors.jsonl", "source_trajectory_id")
    if environment_factory is None:
        env_config = ShopSimulatorConfig(
            base_url=spec.environment_base_url,
            persona=spec.environment_persona,
            timeout=spec.environment_timeout,
        )
        environment_factory = lambda: ShopSimulatorHTTPEnvironment(env_config)

    pending = [
        trace
        for trace in traces
        if not (
            spec.resume
            and trace["episode_id"] in analyses.latest
            and analyses.latest[trace["episode_id"]].get("input_fingerprint")
            == _analysis_fingerprint(trace, spec)
        )
    ]

    def analyze_one(trace: dict[str, Any]) -> list[dict[str, Any]]:
        success = float(trace["final"]["reward_detail"]["r_success"]) == 1.0
        if success:
            records = _success_cards(trace, model_factory=model_factory, spec=spec)
            if not records:
                records = [
                    {
                        "schema_version": CARD_SCHEMA_VERSION,
                        "card_id": f"success-{trace['episode_id']}-empty",
                        "source_trajectory_id": trace["episode_id"],
                        "channel": "success",
                        "empty": True,
                        "firewall_findings": [],
                        "eligible_for_consolidation": False,
                        "created_at": utc_now(),
                    }
                ]
            return records
        return [
            _failure_card(
                trace,
                model_factory=model_factory,
                environment_factory=environment_factory,
                spec=spec,
            )
        ]

    completed = 0
    with ThreadPoolExecutor(max_workers=spec.concurrency) as executor:
        futures = {executor.submit(analyze_one, trace): trace for trace in pending}
        for future in as_completed(futures):
            trace = futures[future]
            episode_id = trace["episode_id"]
            try:
                records = future.result()
            except Exception as exc:
                errors.append(
                    {
                        "source_trajectory_id": episode_id,
                        "created_at": utc_now(),
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                if progress:
                    progress(f"analysis error {episode_id}: {type(exc).__name__}: {exc}")
                continue
            analyses.append(
                {
                    "source_trajectory_id": episode_id,
                    "input_fingerprint": _analysis_fingerprint(trace, spec),
                    "created_at": utc_now(),
                    "cards": records,
                }
            )
            completed += 1
            if progress:
                progress(
                    f"analyzed {completed}/{len(pending)} {episode_id} "
                    f"cards={len(records)}"
                )

    missing = [
        trace["episode_id"]
        for trace in traces
        if trace["episode_id"] not in analyses.latest
    ]
    flattened_cards = [
        card
        for episode_id in sorted(analyses.latest)
        for card in analyses.latest[episode_id].get("cards", [])
    ]
    _atomic_write_jsonl(output_dir / "evidence_cards.jsonl", flattened_cards)
    summary = {
        "schema_version": TRACE2SKILL_SCHEMA_VERSION,
        "created_at": utc_now(),
        "input": input_counts,
        "unique_analyzed": len(analyses.latest),
        "evidence_cards": len(flattened_cards),
        "missing": missing,
        "error_count": len(missing),
        "cards_path": str((output_dir / "evidence_cards.jsonl").resolve()),
    }
    atomic_write_json(output_dir / "analysis_summary.json", summary)
    if missing:
        raise Trace2SkillError(
            f"trajectory analysis incomplete: {len(missing)} missing; "
            f"see {output_dir / 'analysis_errors.jsonl'}"
        )
    return summary


def _all_card_records(path: Path) -> list[dict[str, Any]]:
    latest_by_card: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        raise FileNotFoundError(f"evidence cards not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            card_id = record.get("card_id") if isinstance(record, dict) else None
            if isinstance(card_id, str):
                latest_by_card[card_id] = record
    return [latest_by_card[key] for key in sorted(latest_by_card)]


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _deployable_card(card: dict[str, Any]) -> dict[str, Any]:
    common = {
        "evidence_card_id": card["card_id"],
        "channel": card["channel"],
    }
    if card["channel"] == "failure":
        return {**common, **dict(card.get("deployable_abstraction") or {})}
    return {
        **common,
        "mechanism": card.get("mechanism"),
        "applicable_when": card.get("applicable_when"),
        "observed_evidence": card.get("observed_evidence"),
        "decisive_behavior": card.get("decisive_behavior"),
        "generalizable_lesson": card.get("generalizable_lesson"),
        "confidence": card.get("confidence"),
    }


def _consolidation_fingerprint(
    cards: Sequence[dict[str, Any]], spec: Trace2SkillSpec
) -> str:
    return fingerprint(
        {
            "pipeline": TRACE2SKILL_SCHEMA_VERSION,
            "model": spec.compiler_model.identity(),
            "prompt": fingerprint(CONSOLIDATION_SYSTEM_PROMPT),
            "tool": fingerprint(CLUSTER_TOOL),
            "cards": [_deployable_card(card) for card in cards],
        }
    )


def _validate_consolidation_arguments(
    arguments: dict[str, Any],
    *,
    cards: Sequence[dict[str, Any]],
    max_clusters: int,
) -> None:
    raw_clusters = arguments.get("clusters")
    if not isinstance(raw_clusters, list):
        raise StructuredOutputError("consolidator clusters must be a list")
    if len(raw_clusters) > max_clusters:
        raise StructuredOutputError("consolidator exceeded its cluster budget")

    allowed_ids = {str(card["card_id"]) for card in cards}
    referenced: set[str] = set()
    for index, cluster in enumerate(raw_clusters, 1):
        context = f"evidence cluster {index}"
        if not isinstance(cluster, dict):
            raise StructuredOutputError(f"{context} must be an object")
        _require_text_fields(
            cluster,
            (
                "workflow_stage", "mechanism", "applicable_when", "procedure",
                "verification", "common_failure",
            ),
            context=context,
        )
        evidence_ids = _validated_id_list(
            cluster,
            "evidence_card_ids",
            allowed=allowed_ids,
            context=context,
            require_nonempty=True,
        )
        duplicated = referenced.intersection(evidence_ids)
        if duplicated:
            raise StructuredOutputError(
                f"evidence cards assigned to multiple clusters: {sorted(duplicated)}"
            )
        referenced.update(evidence_ids)
        findings = firewall_findings(cluster)
        if findings:
            raise StructuredOutputError(f"cluster failed gold firewall: {findings}")

    evidence_only = set(
        _validated_id_list(
            arguments,
            "evidence_only_card_ids",
            allowed=allowed_ids,
            context="consolidator output",
        )
    )
    overlap = referenced.intersection(evidence_only)
    if overlap:
        raise StructuredOutputError(
            f"cards cannot be both clustered and evidence-only: {sorted(overlap)}"
        )
    unaccounted = allowed_ids - referenced - evidence_only
    if unaccounted:
        raise StructuredOutputError(
            f"consolidator left cards unaccounted: {sorted(unaccounted)}"
        )


def _consolidate_batch(
    *,
    channel: str,
    batch_id: str,
    cards: Sequence[dict[str, Any]],
    spec: Trace2SkillSpec,
    model_factory: ModelFactory,
) -> dict[str, Any]:
    max_clusters = min(12, len(cards))
    model = model_factory(spec.compiler_model)
    try:
        prompt = {
            "channel": channel,
            "batch_id": batch_id,
            "max_clusters": max_clusters,
            "cards": [_deployable_card(card) for card in cards],
        }
        arguments = _structured_call(
            model,
            [
                {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": _json(prompt)},
            ],
            seed=spec.seed + int(fingerprint(batch_id)[:8], 16),
            tool=CLUSTER_TOOL,
            expected_name="submit_clusters",
            validate=lambda value: _validate_consolidation_arguments(
                value, cards=cards, max_clusters=max_clusters
            ),
        )
    finally:
        model.close()
    raw_clusters = arguments.get("clusters")
    evidence_only = arguments.get("evidence_only_card_ids")
    if not isinstance(raw_clusters, list) or not isinstance(evidence_only, list):
        raise StructuredOutputError("cluster output fields must be lists")
    if len(raw_clusters) > max_clusters:
        raise StructuredOutputError("consolidator exceeded its cluster budget")
    allowed_ids = {card["card_id"] for card in cards}
    referenced: set[str] = set()
    clusters = []
    for index, cluster in enumerate(raw_clusters, 1):
        if not isinstance(cluster, dict):
            raise StructuredOutputError("cluster must be an object")
        _require_text_fields(
            cluster,
            (
                "workflow_stage", "mechanism", "applicable_when", "procedure",
                "verification", "common_failure",
            ),
            context="evidence cluster",
        )
        evidence_ids = cluster.get("evidence_card_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise StructuredOutputError("cluster must cite evidence_card_ids")
        if not set(evidence_ids) <= allowed_ids:
            raise StructuredOutputError("cluster cited an unknown evidence card")
        findings = firewall_findings(cluster)
        if findings:
            raise StructuredOutputError(f"cluster failed gold firewall: {findings}")
        referenced.update(evidence_ids)
        clusters.append(
            {
                "schema_version": CLUSTER_SCHEMA_VERSION,
                "cluster_id": f"{batch_id}-cluster-{index:02d}",
                "channel": channel,
                "batch_id": batch_id,
                **cluster,
            }
        )
    if not set(evidence_only) <= allowed_ids:
        raise StructuredOutputError("evidence-only list cited an unknown card")
    return {
        "batch_id": batch_id,
        "channel": channel,
        "input_fingerprint": _consolidation_fingerprint(cards, spec),
        "clusters": clusters,
        "evidence_only_card_ids": sorted(set(evidence_only)),
        "created_at": utc_now(),
    }


def _validate_compiler_arguments(
    arguments: dict[str, Any],
    *,
    clusters: Sequence[dict[str, Any]],
    validation_dimension_cap: int,
) -> None:
    _require_text_fields(arguments, ("skill_title",), context="compiler output")
    raw_chunks = arguments.get("chunks")
    if not isinstance(raw_chunks, list):
        raise StructuredOutputError("compiler chunks must be a list")
    if not raw_chunks:
        raise StructuredOutputError("compiler produced an empty initial draft")
    if len(raw_chunks) > validation_dimension_cap or len(raw_chunks) > 16:
        raise StructuredOutputError("compiler exceeded validation dimension cap")

    allowed_clusters = {str(cluster["cluster_id"]) for cluster in clusters}
    assigned_chunk: dict[str, int] = {}
    duplicate_assignments: list[str] = []
    for index, chunk in enumerate(raw_chunks, 1):
        context = f"draft chunk {index}"
        if not isinstance(chunk, dict):
            raise StructuredOutputError(f"{context} must be an object")
        _require_text_fields(
            chunk,
            (
                "title", "workflow_stage", "applicable_when", "verification",
                "common_failure",
            ),
            context=context,
        )
        procedures = chunk.get("procedure")
        if (
            not isinstance(procedures, list)
            or not procedures
            or any(not isinstance(step, str) or not step.strip() for step in procedures)
        ):
            raise StructuredOutputError(
                f"{context} procedure must contain text steps"
            )
        evidence_ids = _validated_id_list(
            chunk,
            "evidence_cluster_ids",
            allowed=allowed_clusters,
            context=context,
            require_nonempty=True,
        )
        for evidence_id in evidence_ids:
            previous = assigned_chunk.get(evidence_id)
            if previous is None:
                assigned_chunk[evidence_id] = index
            else:
                duplicate_assignments.append(
                    f"{evidence_id} (draft chunks {previous} and {index})"
                )
        findings = firewall_findings(chunk)
        if findings:
            raise StructuredOutputError(
                f"draft chunk failed gold firewall: {findings}"
            )

    if duplicate_assignments:
        raise StructuredOutputError(
            "evidence clusters assigned to multiple chunks: "
            + ", ".join(duplicate_assignments)
        )
    referenced = set(assigned_chunk)

    evidence_only = set(
        _validated_id_list(
            arguments,
            "evidence_only_cluster_ids",
            allowed=allowed_clusters,
            context="compiler output",
        )
    )
    overlap = referenced.intersection(evidence_only)
    if overlap:
        raise StructuredOutputError(
            f"clusters cannot support chunks and be evidence-only: {sorted(overlap)}"
        )
    unaccounted = allowed_clusters - referenced - evidence_only
    if unaccounted:
        raise StructuredOutputError(
            f"compiler left clusters unaccounted: {sorted(unaccounted)}"
        )

    conflicts = arguments.get("conflicts_resolved")
    if not isinstance(conflicts, list) or any(
        not isinstance(item, str) or not item.strip() for item in conflicts
    ):
        raise StructuredOutputError(
            "compiler conflicts_resolved must be a list of non-empty strings"
        )


def _compile_skill(
    clusters: Sequence[dict[str, Any]],
    *,
    spec: Trace2SkillSpec,
    model_factory: ModelFactory,
) -> dict[str, Any]:
    sanitized = [
        {
            key: value
            for key, value in cluster.items()
            if key
            in {
                "cluster_id", "channel", "workflow_stage", "mechanism",
                "applicable_when", "procedure", "verification", "common_failure",
                "evidence_card_ids",
            }
        }
        for cluster in clusters
    ]
    invalid_store = JsonlIndex(
        spec.output_dir / "compiler_invalid_attempts.jsonl", "attempt_id"
    )
    compile_run_started_at = utc_now()
    compile_run_id = fingerprint(
        {
            "started_at": compile_run_started_at,
            "cluster_ids": [cluster["cluster_id"] for cluster in clusters],
        }
    )[:16]

    def record_invalid(event: dict[str, Any]) -> None:
        invalid_store.append(
            {
                "attempt_id": (
                    f"compiler-{compile_run_id}-{event['attempt_number']:02d}"
                ),
                "compile_run_id": compile_run_id,
                "compile_run_started_at": compile_run_started_at,
                "created_at": utc_now(),
                **event,
            }
        )

    model = model_factory(spec.compiler_model)
    try:
        prompt = {
            "validation_dimension_cap": spec.validation_dimension_cap,
            "cluster_summaries": sanitized,
        }
        arguments = _structured_call(
            model,
            [
                {"role": "system", "content": COMPILER_SYSTEM_PROMPT},
                {"role": "user", "content": _json(prompt)},
            ],
            seed=spec.seed,
            tool=INITIAL_SKILL_TOOL,
            expected_name="submit_initial_skill",
            attempts=5,
            validate=lambda value: _validate_compiler_arguments(
                value,
                clusters=clusters,
                validation_dimension_cap=spec.validation_dimension_cap,
            ),
            on_invalid=record_invalid,
        )
    finally:
        model.close()
    raw_chunks = arguments.get("chunks")
    if not isinstance(raw_chunks, list):
        raise StructuredOutputError("compiler chunks must be a list")
    if not raw_chunks:
        raise StructuredOutputError("compiler produced an empty initial draft")
    if len(raw_chunks) > spec.validation_dimension_cap or len(raw_chunks) > 16:
        raise StructuredOutputError("compiler exceeded validation dimension cap")
    allowed_clusters = {cluster["cluster_id"] for cluster in clusters}
    chunks = []
    used_ids: set[str] = set()
    for index, chunk in enumerate(raw_chunks, 1):
        if not isinstance(chunk, dict):
            raise StructuredOutputError("draft chunk must be an object")
        _require_text_fields(
            chunk,
            (
                "title", "workflow_stage", "applicable_when", "verification",
                "common_failure",
            ),
            context="draft chunk",
        )
        procedures = chunk.get("procedure")
        if (
            not isinstance(procedures, list)
            or not procedures
            or any(not isinstance(step, str) or not step.strip() for step in procedures)
        ):
            raise StructuredOutputError("draft chunk procedure must contain text steps")
        evidence_ids = chunk.get("evidence_cluster_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise StructuredOutputError("draft chunk must cite evidence clusters")
        if not set(evidence_ids) <= allowed_clusters:
            raise StructuredOutputError("draft chunk cited an unknown cluster")
        findings = firewall_findings(chunk)
        if findings:
            raise StructuredOutputError(f"draft chunk failed gold firewall: {findings}")
        title = str(chunk.get("title", f"chunk-{index}"))
        try:
            base_id = safe_name(title).lower()
        except ValueError:
            # Deployable fields are intentionally Simplified Chinese, while
            # artifact IDs remain portable ASCII. Use a semantic stable hash
            # when a title has no ASCII slug rather than altering the title.
            base_id = "chunk-" + fingerprint(
                {"title": title, "evidence_cluster_ids": sorted(evidence_ids)}
            )[:12]
        chunk_id = base_id
        suffix = 2
        while chunk_id in used_ids:
            chunk_id = f"{base_id}-{suffix}"
            suffix += 1
        used_ids.add(chunk_id)
        chunks.append({"chunk_id": chunk_id, "order": index, **chunk})
    evidence_only = arguments.get("evidence_only_cluster_ids")
    if not isinstance(evidence_only, list) or not set(evidence_only) <= allowed_clusters:
        raise StructuredOutputError("compiler returned invalid evidence-only clusters")
    return {
        "schema_version": INITIAL_SKILL_SCHEMA_VERSION,
        "created_at": utc_now(),
        "stage": "cold_start_draft_pre_validation",
        "gate_status": {"gate_a": "not_run", "gate_b": "not_run"},
        "validation_dimension_cap": spec.validation_dimension_cap,
        "active_skill_budget": spec.active_skill_budget,
        "skill_title": arguments.get("skill_title", "Shopping Skill"),
        "chunks": chunks,
        "evidence_only_cluster_ids": evidence_only,
        "conflicts_resolved": arguments.get("conflicts_resolved") or [],
    }


def _render_chunk(chunk: dict[str, Any]) -> str:
    procedures = "\n".join(f"  {index}. {step}" for index, step in enumerate(chunk["procedure"], 1))
    return (
        f"### {chunk['title']}\n\n"
        f"- Applies when: {chunk['applicable_when']}\n"
        f"- Procedure:\n{procedures}\n"
        f"- Verify: {chunk['verification']}\n"
        f"- Avoid: {chunk['common_failure']}"
    )


def render_initial_skill_markdown(draft: dict[str, Any]) -> str:
    body = "\n\n".join(_render_chunk(chunk) for chunk in draft["chunks"])
    return (
        f"# {draft['skill_title']}\n\n"
        "> Cold-start draft; not active until Gate A and Gate B pass.\n\n"
        f"{body}\n"
    )


def _skillbank(draft: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "shopsimrl-skillbank-v1",
        "bank_version": "cold-start-draft-pre-validation",
        "metadata": {
            "draft_schema_version": draft["schema_version"],
            "gate_status": draft["gate_status"],
            "active_skill_budget": draft["active_skill_budget"],
        },
        "skills": [
            {
                "skill_id": chunk["chunk_id"],
                "version": "draft-1",
                "enabled": True,
                "content": _render_chunk(chunk),
                "metadata": {
                    "workflow_stage": chunk["workflow_stage"],
                    "draft_only": True,
                    "evidence_cluster_ids": chunk["evidence_cluster_ids"],
                    "validation_result": None,
                },
            }
            for chunk in draft["chunks"]
        ],
    }


def compile_initial_skill(
    spec: Trace2SkillSpec,
    *,
    model_factory: ModelFactory = OpenAICompatibleChatModel,
    progress: Progress | None = None,
) -> dict[str, Any]:
    output_dir = spec.output_dir
    cards = _all_card_records(output_dir / "evidence_cards.jsonl")
    source_ids = {card.get("source_trajectory_id") for card in cards}
    if len(source_ids) != spec.expected_trajectories:
        raise Trace2SkillError(
            f"expected cards for {spec.expected_trajectories} trajectories, "
            f"found {len(source_ids)}"
        )
    eligible = [card for card in cards if card.get("eligible_for_consolidation") is True]
    if not eligible:
        raise Trace2SkillError("no evidence cards are eligible for consolidation")

    cluster_store = JsonlIndex(output_dir / "cluster_batches.jsonl", "batch_id")
    batch_records = []
    for channel in ("success", "failure"):
        channel_cards = [card for card in eligible if card.get("channel") == channel]
        for batch_number, batch in enumerate(
            _chunks(channel_cards, spec.consolidation_batch_size), 1
        ):
            batch_id = f"{channel}-batch-{batch_number:03d}"
            batch_fingerprint = _consolidation_fingerprint(batch, spec)
            if (
                spec.resume
                and batch_id in cluster_store.latest
                and cluster_store.latest[batch_id].get("input_fingerprint")
                == batch_fingerprint
            ):
                record = cluster_store.latest[batch_id]
            else:
                record = _consolidate_batch(
                    channel=channel,
                    batch_id=batch_id,
                    cards=batch,
                    spec=spec,
                    model_factory=model_factory,
                )
                cluster_store.append(record)
            batch_records.append(record)
            if progress:
                progress(
                    f"consolidated {batch_id}: {len(record['clusters'])} clusters"
                )
    clusters = [
        cluster for record in batch_records for cluster in record.get("clusters", [])
    ]
    if not clusters:
        raise Trace2SkillError("consolidation produced no cluster summaries")
    draft = _compile_skill(clusters, spec=spec, model_factory=model_factory)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "initial_skill_draft.json", draft)
    atomic_write_json(output_dir / "initial_skillbank.json", _skillbank(draft))
    (output_dir / "initial_skill_draft.md").write_text(
        render_initial_skill_markdown(draft), encoding="utf-8"
    )
    ledger_path = output_dir / "proposal_ledger.jsonl"
    ledger_lines = []
    for chunk in draft["chunks"]:
        ledger_lines.append(
            json.dumps(
                {
                    "candidate_id": chunk["chunk_id"],
                    "operation": "ADD",
                    "target_chunk_id": None,
                    "source_card_ids": sorted(
                        {
                            card_id
                            for cluster in clusters
                            if cluster["cluster_id"] in chunk["evidence_cluster_ids"]
                            for card_id in cluster["evidence_card_ids"]
                        }
                    ),
                    "failure_mechanism": chunk["common_failure"],
                    "proposal_checkpoint": "qwen35-4b-initial",
                    "stage": "cold_start",
                    "consolidation_cluster_id": chunk["evidence_cluster_ids"],
                    "replacement_family_id": None,
                    "validation_result": None,
                    "estimated_effect": None,
                    "contribution_rank": None,
                    "notes": "Awaiting initialization Gate A then Gate B",
                },
                ensure_ascii=False,
            )
        )
    ledger_path.write_text("\n".join(ledger_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": TRACE2SKILL_SCHEMA_VERSION,
        "created_at": utc_now(),
        "config": asdict(spec),
        "input": {
            "traces_sha256": fingerprint(
                list(load_latest_traces(spec.traces_path))
            ),
            "cards": len(cards),
            "eligible_cards": len(eligible),
            "clusters": len(clusters),
        },
        "output": {
            "draft_chunks": len(draft["chunks"]),
            "gate_a": "not_run",
            "gate_b": "not_run",
        },
        "models": {
            "analyst": _reported_model_identity(spec.analyst_model),
            "compiler": _reported_model_identity(spec.compiler_model),
        },
    }
    # Paths and dataclasses contain Path objects; keep the manifest portable JSON.
    manifest["config"] = json.loads(json.dumps(manifest["config"], default=str))
    atomic_write_json(output_dir / "trace2skill_manifest.json", manifest)
    if progress:
        progress(f"compiled initial draft with {len(draft['chunks'])} chunks")
    return draft


def run_cold_start(
    spec: Trace2SkillSpec,
    *,
    model_factory: ModelFactory = OpenAICompatibleChatModel,
    environment_factory: EnvironmentFactory | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    analyze_trajectories(
        spec,
        model_factory=model_factory,
        environment_factory=environment_factory,
        progress=progress,
    )
    return compile_initial_skill(spec, model_factory=model_factory, progress=progress)


def trace2skill_plan(spec: Trace2SkillSpec) -> dict[str, Any]:
    traces = load_latest_traces(spec.traces_path)
    counts = validate_cold_start_traces(traces, expected=spec.expected_trajectories)
    return {
        "schema_version": TRACE2SKILL_SCHEMA_VERSION,
        "traces": str(spec.traces_path),
        "output_dir": str(spec.output_dir),
        "counts": counts,
        "limits": {
            "max_cards_per_trajectory": spec.max_cards_per_trajectory,
            "failure_analysis_steps": spec.max_failure_analysis_steps,
            "consolidation_batch_size": spec.consolidation_batch_size,
            "validation_dimension_cap": spec.validation_dimension_cap,
            "active_skill_budget": spec.active_skill_budget,
        },
        "models": {
            "analyst": _reported_model_identity(spec.analyst_model),
            "compiler": _reported_model_identity(spec.compiler_model),
        },
    }
