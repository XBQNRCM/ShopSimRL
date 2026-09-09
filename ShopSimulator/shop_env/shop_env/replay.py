"""Read-only indexing for local ShopSimRL run artifacts."""

from __future__ import annotations

import json
import re
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from typing import Any


class ReplayError(ValueError):
    pass


@dataclass(frozen=True)
class TraceIndex:
    signature: tuple[int, int]
    offsets: dict[str, int]
    episodes: tuple[dict[str, Any], ...]
    total_records: int
    invalid_lines: int


_CACHE: dict[Path, TraceIndex] = {}
_CACHE_LOCK = threading.Lock()


def _utc_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReplayError(f"{path.name} must contain a JSON object")
    return payload


def resolve_run_path(value: str, *, project_root: Path, runs_root: Path) -> Path:
    raw_value = value.strip()
    if not raw_value:
        raise ReplayError("run path cannot be empty")
    raw = Path(raw_value)
    if raw.is_absolute():
        resolved = raw.resolve()
    elif raw.parts and raw.parts[0].lower() == "runs":
        resolved = (project_root / raw).resolve()
    else:
        resolved = (runs_root / raw).resolve()
    root = runs_root.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReplayError(f"run path must stay inside {root}") from exc
    if not resolved.is_dir():
        raise ReplayError(f"run directory does not exist: {resolved}")
    return resolved


def display_run_path(path: Path, *, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def list_runs(*, project_root: Path, runs_root: Path) -> list[dict[str, Any]]:
    if not runs_root.is_dir():
        return []
    records = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            continue
        artifact_paths = {
            name: path / name
            for name in ("manifest.json", "summary.json", "traces.jsonl")
        }
        artifacts = {name: artifact.is_file() for name, artifact in artifact_paths.items()}
        if not any(artifacts.values()):
            continue
        modified_at = max(
            artifact.stat().st_mtime
            for artifact in artifact_paths.values()
            if artifact.is_file()
        )
        records.append(
            {
                "name": path.name,
                "path": display_run_path(path, project_root=project_root),
                "modified_at": _utc_timestamp(modified_at),
                "artifacts": artifacts,
            }
        )
    return sorted(records, key=lambda record: record["modified_at"], reverse=True)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _episode_summary(trace: dict[str, Any]) -> dict[str, Any]:
    job = trace.get("job") if isinstance(trace.get("job"), dict) else {}
    final = trace.get("final") if isinstance(trace.get("final"), dict) else {}
    detail = (
        final.get("reward_detail")
        if isinstance(final.get("reward_detail"), dict)
        else {}
    )
    error = trace.get("error") if isinstance(trace.get("error"), dict) else {}
    reset = trace.get("reset") if isinstance(trace.get("reset"), dict) else {}
    steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    protocol_errors = 0
    policy_failures = 0
    invalid_actions = 0
    environment_actions = 0
    tokens: Counter[str] = Counter()
    for step in steps:
        if not isinstance(step, dict):
            continue
        model = step.get("model") if isinstance(step.get("model"), dict) else {}
        protocol_error = step.get("protocol_error") or model.get("protocol_error")
        policy_failure = step.get("policy_failure") or model.get("policy_failure")
        if protocol_error:
            protocol_errors += 1
        if policy_failure:
            policy_failures += 1
        environment = step.get("environment")
        if isinstance(environment, dict):
            environment_actions += 1
            feedback = environment.get("action_feedback")
            if isinstance(feedback, dict) and feedback.get("valid") is False:
                invalid_actions += 1
        usage = model.get("usage") if isinstance(model, dict) else None
        if not isinstance(usage, dict):
            continue
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                tokens[key] += value
        completion = usage.get("completion_tokens_details")
        if isinstance(completion, dict):
            value = completion.get("reasoning_tokens")
            if isinstance(value, int) and not isinstance(value, bool):
                tokens["reasoning_tokens"] += value
    reward = _number(final.get("reward"))
    success = _number(detail.get("r_success"))
    skills = trace.get("selected_skills")
    skill_ids = []
    if isinstance(skills, list):
        skill_ids = [
            str(skill["skill_id"])
            for skill in skills
            if isinstance(skill, dict) and skill.get("skill_id") is not None
        ]
    return {
        "episode_id": trace.get("episode_id"),
        "schema_version": trace.get("schema_version"),
        "task_id": job.get("task_id"),
        "sample_id": job.get("sample_id"),
        "split": job.get("split"),
        "seed": job.get("seed"),
        "status": trace.get("status", "unknown"),
        "reward": reward,
        "success": success,
        "termination_reason": final.get("termination_reason"),
        "steps": len(steps),
        "environment_actions": environment_actions,
        "protocol_errors": protocol_errors,
        "policy_failures": policy_failures,
        "invalid_actions": invalid_actions,
        "tokens": dict(tokens),
        "duration_ms": _number(trace.get("duration_ms")),
        "started_at": trace.get("started_at"),
        "ended_at": trace.get("ended_at"),
        "instruction": reset.get("task_instruction"),
        "task_mode": reset.get("task_mode"),
        "page_type": (
            reset.get("observation_state", {}).get("page_type")
            if isinstance(reset.get("observation_state"), dict)
            else None
        ),
        "skill_ids": skill_ids,
        "error": (
            {
                "type": error.get("type"),
                "stage": error.get("stage"),
                "message": error.get("message"),
            }
            if error
            else None
        ),
    }


def _trace_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _build_trace_index(path: Path) -> TraceIndex:
    signature = _trace_signature(path)
    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    total_records = 0
    invalid_lines = 0
    with path.open("rb") as file:
        while True:
            offset = file.tell()
            line = file.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                invalid_lines += 1
                continue
            if not isinstance(payload, dict):
                invalid_lines += 1
                continue
            episode_id = payload.get("episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                invalid_lines += 1
                continue
            total_records += 1
            latest[episode_id] = (offset, _episode_summary(payload))
    ordered = sorted(
        latest.items(),
        key=lambda item: (
            item[1][1].get("task_id") is None,
            item[1][1].get("task_id") or 0,
            item[1][1].get("sample_id") or 0,
            item[0],
        ),
    )
    return TraceIndex(
        signature=signature,
        offsets={episode_id: value[0] for episode_id, value in ordered},
        episodes=tuple(value[1] for _, value in ordered),
        total_records=total_records,
        invalid_lines=invalid_lines,
    )


def get_trace_index(path: Path) -> TraceIndex:
    signature = _trace_signature(path)
    with _CACHE_LOCK:
        cached = _CACHE.get(path)
        if cached is not None and cached.signature == signature:
            return cached
    built = _build_trace_index(path)
    with _CACHE_LOCK:
        _CACHE[path] = built
    return built


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _computed_summary(
    episodes: tuple[dict[str, Any], ...], *, requested: int
) -> dict[str, Any]:
    statuses: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    schemas: Counter[str] = Counter()
    rewards: list[float] = []
    successes: list[float] = []
    durations: list[float] = []
    steps: list[float] = []
    tokens: Counter[str] = Counter()
    protocol_errors = 0
    policy_failures = 0
    invalid_actions = 0
    environment_actions = 0
    for episode in episodes:
        statuses[str(episode.get("status", "unknown"))] += 1
        schemas[str(episode.get("schema_version", "unknown"))] += 1
        termination = episode.get("termination_reason")
        if termination:
            terminations[str(termination)] += 1
        error = episode.get("error")
        if isinstance(error, dict):
            label = f"{error.get('type', 'unknown')} @ {error.get('stage', 'unknown')}"
            errors[label] += 1
        reward = _number(episode.get("reward"))
        if reward is not None:
            rewards.append(reward)
        success = _number(episode.get("success"))
        if success is not None:
            successes.append(success)
        duration = _number(episode.get("duration_ms"))
        if duration is not None:
            durations.append(duration)
        steps.append(float(episode.get("steps") or 0))
        protocol_errors += int(episode.get("protocol_errors") or 0)
        policy_failures += int(episode.get("policy_failures") or 0)
        invalid_actions += int(episode.get("invalid_actions") or 0)
        environment_actions += int(episode.get("environment_actions") or 0)
        for key, value in (episode.get("tokens") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                tokens[key] += value
    recorded = len(episodes)
    return {
        "counts": {
            "requested": requested,
            "recorded": recorded,
            "pending": max(0, requested - recorded),
            "completed": statuses.get("completed", 0),
            "failed": statuses.get("failed", 0),
            "scored": len(rewards),
        },
        "primary": {
            "reward_mean": _mean(rewards),
            "success_rate": _mean(successes),
        },
        "behavior": {
            "steps_mean": _mean(steps),
            "protocol_errors": protocol_errors,
            "policy_failures": policy_failures,
            "invalid_actions": invalid_actions,
            "environment_actions": environment_actions,
            "invalid_action_rate": (
                invalid_actions / environment_actions if environment_actions else None
            ),
        },
        "latency": {
            "duration_ms_mean": _mean(durations),
            "duration_ms_p95": _percentile(durations, 0.95),
        },
        "tokens": dict(tokens),
        "statuses": dict(sorted(statuses.items())),
        "termination_reasons": dict(sorted(terminations.items())),
        "errors": dict(sorted(errors.items())),
        "schema_versions": dict(sorted(schemas.items())),
    }


def _artifact_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"present": False}
    stat = path.stat()
    return {
        "present": True,
        "bytes": stat.st_size,
        "modified_at": _utc_timestamp(stat.st_mtime),
    }


def load_run(path: Path, *, project_root: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    summary_path = path / "summary.json"
    traces_path = path / "traces.jsonl"
    manifest = _read_json(manifest_path)
    artifact_summary = _read_json(summary_path)
    if not traces_path.is_file():
        raise ReplayError(f"run has no traces.jsonl: {path}")
    index = get_trace_index(traces_path)
    jobs = (manifest or {}).get("plan", {}).get("jobs")
    requested = len(jobs) if isinstance(jobs, list) else len(index.episodes)
    return {
        "run": {
            "name": path.name,
            "path": display_run_path(path, project_root=project_root),
        },
        "artifacts": {
            "manifest": _artifact_info(manifest_path),
            "summary": _artifact_info(summary_path),
            "traces": _artifact_info(traces_path),
        },
        "manifest": manifest,
        "artifact_summary": artifact_summary,
        "computed_summary": _computed_summary(index.episodes, requested=requested),
        "trace_index": {
            "records": index.total_records,
            "unique_episodes": len(index.episodes),
            "duplicate_records": index.total_records - len(index.episodes),
            "invalid_lines": index.invalid_lines,
        },
        "episodes": list(index.episodes),
    }


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def _normalized(value: Any) -> str:
    return str(value).strip().casefold()


def _fuzzy_ratio(left: Any, right: Any) -> int:
    """Mirror the paper reward's dependency-free token-set ratio."""

    def tokens(value: Any) -> set[str]:
        normalized = re.sub(r"\W+", " ", _normalized(value), flags=re.UNICODE)
        return set(normalized.split())

    left_tokens, right_tokens = tokens(left), tokens(right)
    intersection = left_tokens & right_tokens
    left_only = left_tokens - intersection
    right_only = right_tokens - intersection
    common = " ".join(sorted(intersection)).strip()
    combined_left = " ".join(sorted(intersection | left_only)).strip()
    combined_right = " ".join(sorted(intersection | right_only)).strip()

    def ratio(first: str, second: str) -> int:
        return round(100 * SequenceMatcher(None, first, second).ratio())

    if not common:
        return ratio(combined_left, combined_right)
    return max(
        ratio(common, combined_left),
        ratio(common, combined_right),
        ratio(combined_left, combined_right),
    )


def _fuzzy_matches(left: Any, right: Any) -> bool:
    return _fuzzy_ratio(left, right) > 85


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _matching_mapping_value(mapping: dict[str, Any], key: str) -> Any:
    normalized_key = _normalized(key)
    for candidate, value in mapping.items():
        if _normalized(candidate) == normalized_key:
            return value
    return None


def _catalog_option_axes(product: dict[str, Any]) -> list[dict[str, Any]]:
    options = _mapping(product.get("options"))
    prices = _mapping(product.get("option_to_price"))
    images = _mapping(product.get("option_to_image"))
    axes = []
    for axis, raw_values in options.items():
        values = []
        for value in _string_list(raw_values):
            values.append(
                {
                    "value": value,
                    "price": _matching_mapping_value(prices, value),
                    "image": _matching_mapping_value(images, value),
                    "selected": False,
                }
            )
        axes.append({"name": str(axis), "values": values})
    return axes


def _selection_records(
    raw_selection: Any, option_axes: list[dict[str, Any]], *, fallback_axis: str
) -> list[dict[str, Any]]:
    if isinstance(raw_selection, dict):
        requested = [(str(axis), str(value)) for axis, value in raw_selection.items()]
    else:
        requested = [(None, value) for value in _string_list(raw_selection)]

    records = []
    for requested_axis, requested_value in requested:
        matches = []
        for axis in option_axes:
            if requested_axis is not None and _normalized(axis["name"]) != _normalized(
                requested_axis
            ):
                continue
            for option in axis["values"]:
                if _normalized(option["value"]) == _normalized(requested_value):
                    matches.append((axis, option))
        if not matches and requested_axis is None:
            for axis in option_axes:
                for option in axis["values"]:
                    if _normalized(option["value"]) == _normalized(requested_value):
                        matches.append((axis, option))
        if matches:
            axis, option = matches[0]
            option["selected"] = True
            records.append(
                {
                    "axis": axis["name"],
                    "value": option["value"],
                    "price": option.get("price"),
                    "catalog_match": True,
                }
            )
        else:
            axis_name = requested_axis or fallback_axis
            records.append(
                {
                    "axis": axis_name,
                    "value": requested_value,
                    "price": None,
                    "catalog_match": False,
                }
            )
            axis = next(
                (
                    item
                    for item in option_axes
                    if _normalized(item["name"]) == _normalized(axis_name)
                ),
                None,
            )
            if axis is None:
                axis = {"name": axis_name, "values": []}
                option_axes.append(axis)
            axis["values"].append(
                {
                    "value": requested_value,
                    "price": None,
                    "image": None,
                    "selected": True,
                    "catalog_match": False,
                }
            )
    return records


def _product_page(
    summary: dict[str, Any],
    *,
    raw_selection: Any,
    fallback_axis: str,
    product_catalog: dict[str, Any] | None,
) -> dict[str, Any] | None:
    asin = summary.get("asin")
    if asin is None:
        return None
    asin = str(asin)
    catalog_product = (
        product_catalog.get(asin)
        if isinstance(product_catalog, dict)
        and isinstance(product_catalog.get(asin), dict)
        else None
    )
    product = catalog_product or {}
    option_axes = _catalog_option_axes(product)
    selections = _selection_records(
        raw_selection, option_axes, fallback_axis=fallback_axis
    )
    selected_prices = [
        selection["price"]
        for selection in selections
        if _number(selection.get("price")) is not None
    ]
    images = _string_list(product.get("images"))
    main_image = product.get("MainImage")
    if isinstance(main_image, str) and main_image:
        images = [main_image, *[image for image in images if image != main_image]]
    return {
        "asin": asin,
        "catalog_available": catalog_product is not None,
        "title": product.get("title") or summary.get("name"),
        "brand": product.get("shop_name") or summary.get("brand"),
        "category": product.get("category") or summary.get("category"),
        "description": product.get("full_description"),
        "attributes": _string_list(
            product.get("attribute") or summary.get("attributes")
        ),
        "price": product.get("pricing"),
        "selected_price": summary.get("price") or (
            selected_prices[0] if len(selected_prices) == 1 else None
        ),
        "price_upper": summary.get("price_upper"),
        "images": images[:5],
        "option_axes": option_axes,
        "selected_options": selections,
    }


def _attribute_match_sets(
    required: list[str],
    actual: list[str],
    product: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    searchable_text = " ".join(
        [
            str(product.get("Title") or product.get("title") or ""),
            " ".join(map(str, product.get("BulletPoints") or [])),
            str(
                product.get("Description")
                or product.get("full_description")
                or ""
            ),
        ]
    ).casefold()
    matched = [
        value
        for value in required
        if any(_fuzzy_matches(candidate, value) for candidate in actual)
        or _normalized(value) in searchable_text
    ]
    missing = [value for value in required if value not in matched]
    matched_actual = [
        value
        for value in actual
        if any(_fuzzy_matches(value, candidate) for candidate in required)
    ]
    extra = [value for value in actual if value not in matched_actual]
    return matched, missing, matched_actual, extra


def _option_match_sets(
    required: list[str], actual: list[str]
) -> tuple[list[str], list[str], list[str], list[str]]:
    matched = [
        value
        for value in required
        if any(_fuzzy_matches(candidate, value) for candidate in actual)
    ]
    missing = [value for value in required if value not in matched]
    matched_actual = [
        value
        for value in actual
        if any(_fuzzy_matches(value, candidate) for candidate in required)
    ]
    extra = [value for value in actual if value not in matched_actual]
    return matched, missing, matched_actual, extra


def build_outcome_comparison(
    trace: dict[str, Any], *, product_catalog: dict[str, Any] | None = None
) -> dict[str, Any]:
    final = _mapping(trace.get("final"))
    goal = _mapping(final.get("goal"))
    purchase = _mapping(final.get("purchase"))
    target_options = _string_list(goal.get("goal_options") or goal.get("options"))
    purchased_option_mapping = _mapping(purchase.get("options"))
    purchased_options = [
        str(value) for value in purchased_option_mapping.values() if value is not None
    ]
    target_attributes = _string_list(goal.get("attributes"))
    purchased_attributes = _string_list(purchase.get("attributes"))
    target_asin = str(goal["asin"]) if goal.get("asin") is not None else None
    purchased_asin = (
        str(purchase["asin"]) if purchase.get("asin") is not None else None
    )
    purchased_product = (
        product_catalog.get(purchased_asin)
        if isinstance(product_catalog, dict)
        and purchased_asin is not None
        and isinstance(product_catalog.get(purchased_asin), dict)
        else purchase
    )
    (
        matched_attributes,
        missing_attributes,
        matched_purchased_attributes,
        extra_attributes,
    ) = _attribute_match_sets(
        target_attributes, purchased_attributes, purchased_product
    )
    (
        matched_options,
        missing_options,
        matched_purchased_options,
        extra_options,
    ) = _option_match_sets(target_options, purchased_options)
    return {
        "asin_match": (
            target_asin == purchased_asin
            if target_asin is not None and purchased_asin is not None
            else None
        ),
        "target": _product_page(
            goal,
            raw_selection=target_options,
            fallback_axis="Gold truth option",
            product_catalog=product_catalog,
        ),
        "purchased": _product_page(
            purchase,
            raw_selection=purchased_option_mapping,
            fallback_axis="Purchased option",
            product_catalog=product_catalog,
        ),
        "attributes": {
            "matched": matched_attributes,
            "missing": missing_attributes,
            "matched_purchased": matched_purchased_attributes,
            "extra": extra_attributes,
        },
        "options": {
            "gold": target_options,
            "purchased": purchased_options,
            "matched": matched_options,
            "missing": missing_options,
            "matched_purchased": matched_purchased_options,
            "extra": extra_options,
        },
    }


def load_episode(
    path: Path,
    episode_id: str,
    *,
    product_catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    traces_path = path / "traces.jsonl"
    if not traces_path.is_file():
        raise ReplayError(f"run has no traces.jsonl: {path}")
    index = get_trace_index(traces_path)
    offset = index.offsets.get(episode_id)
    if offset is None:
        raise ReplayError(f"episode does not exist in this run: {episode_id}")
    with traces_path.open("rb") as file:
        file.seek(offset)
        line = file.readline()
    try:
        payload = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot decode episode {episode_id}: {exc}") from exc
    return {
        "summary": _episode_summary(payload),
        "episode": payload,
        "outcome_comparison": build_outcome_comparison(
            payload, product_catalog=product_catalog
        ),
    }
