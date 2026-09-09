#!/usr/bin/env python3
"""Annotate persona budgets with Qwen and build cleaned ShopSimulator data."""

from __future__ import annotations

import argparse
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
import json
import os
from pathlib import Path
import sys
import threading
import time

import requests


SHOP_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOP_ENV))

from data_pipeline.dataset_builder import (  # noqa: E402
    DatasetBuildError,
    append_annotation_cache,
    full_instruction_task,
    load_annotation_cache,
    persona_annotation_tasks,
    read_products,
    resolve_persona_annotations,
    sha256_file,
    source_tasks,
    write_clean_artifacts,
)
from data_pipeline.price_annotation import (  # noqa: E402
    ANNOTATION_VERSION,
    AnnotationError,
    KINDS_WITH_UPPER_BOUND,
    convert_valid_v1_annotation,
    prompt_sha256,
    request_payload,
    validate_response,
)


DEFAULT_MODEL = "Qwen/Qwen3.5-27B"
DEFAULT_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_REQUESTS_PER_MINUTE = 32.0
DEFAULT_RATE_LIMIT_COOLDOWN = 65.0
DEFAULT_RATE_LIMIT_RETRIES = 20


class NonRetryableAPIError(DatasetBuildError):
    """An API error that another identical request cannot resolve."""


class RateLimitExhaustedError(DatasetBuildError):
    """Minute-scale rate limiting persisted beyond the configured retry budget."""


class AdaptiveRateLimiter:
    """Coordinate request starts and 429 cooldowns across all worker threads."""

    def __init__(
        self,
        *,
        requests_per_minute: float,
        cooldown_seconds: float,
        clock=None,
        sleeper=None,
    ) -> None:
        self._requests_per_minute = float(requests_per_minute)
        self._cooldown_seconds = float(cooldown_seconds)
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or time.sleep
        self._lock = threading.Lock()
        self._blocked_until = 0.0
        self._next_request_at = 0.0
        self._rate_limit_waves = 0

    def wait_for_slot(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                ready_at = max(self._blocked_until, self._next_request_at)
                delay = ready_at - now
                if delay <= 0:
                    interval = 60.0 / self._requests_per_minute
                    self._next_request_at = now + interval
                    return
            # Recheck once per second so a new shared cooldown is observed.
            self._sleeper(min(delay, 1.0))

    def on_rate_limit(self, response) -> dict:
        retry_after = _retry_after_seconds(response)
        with self._lock:
            now = self._clock()
            new_wave = now >= self._blocked_until
            if new_wave:
                self._rate_limit_waves += 1
                if self._rate_limit_waves == 1:
                    self._requests_per_minute = min(
                        self._requests_per_minute,
                        DEFAULT_REQUESTS_PER_MINUTE,
                    )
                else:
                    self._requests_per_minute = max(
                        1.0, self._requests_per_minute / 2.0
                    )
            cooldown = max(self._cooldown_seconds, retry_after)
            self._blocked_until = max(self._blocked_until, now + cooldown)
            self._next_request_at = max(
                self._next_request_at, self._blocked_until
            )
            return {
                "new_wave": new_wave,
                "cooldown_seconds": cooldown,
                "requests_per_minute": self._requests_per_minute,
                "rate_limit_waves": self._rate_limit_waves,
            }


def _retry_after_seconds(response) -> float:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return 0.0
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 0.0


def _rate_limit_kind(response) -> str | None:
    message = response.text.upper()
    for kind in ("TPM", "RPM", "RPH", "RPD", "TPD", "IPM", "IPD"):
        if kind in message:
            return kind
    return None


def _raise_for_non_retryable_status(response, api_key: str) -> None:
    status = response.status_code
    trace_id = response.headers.get("x-siliconcloud-trace-id") or "unavailable"
    if status == 401:
        raise NonRetryableAPIError(
            "SiliconFlow authentication failed (HTTP 401). Re-enter or rotate "
            "SILICONFLOW_API_KEY and verify the selected endpoint. "
            f"key_length={len(api_key)}, starts_with_sk={api_key.startswith('sk-')}, "
            f"trace_id={trace_id}"
        )
    if status == 403:
        raise NonRetryableAPIError(
            "SiliconFlow rejected this account or model (HTTP 403). "
            f"trace_id={trace_id}"
        )
    if status == 404:
        raise NonRetryableAPIError(
            "SiliconFlow endpoint or model was not found (HTTP 404). "
            f"trace_id={trace_id}"
        )
    if 400 <= status < 500 and status != 429:
        raise NonRetryableAPIError(
            f"SiliconFlow rejected the request (HTTP {status}). "
            f"response={response.text[:500]!r}, trace_id={trace_id}"
        )


def call_task(
    task: dict,
    *,
    api_key: str,
    api_url: str,
    model: str,
    timeout: float,
    retries: int,
    rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
    rate_limiter: AdaptiveRateLimiter | None = None,
) -> tuple[dict, dict]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error = None
    transient_failures = 0
    rate_limit_failures = 0
    rate_limit_exhausted = False
    http_attempts = 0
    while True:
        if rate_limiter is not None:
            rate_limiter.wait_for_slot()
        try:
            http_attempts += 1
            response = requests.post(
                api_url,
                headers=headers,
                json=request_payload(task, model),
                timeout=timeout,
            )
            _raise_for_non_retryable_status(response, api_key)
            if response.status_code == 429:
                kind = _rate_limit_kind(response)
                trace_id = (
                    response.headers.get("x-siliconcloud-trace-id")
                    or "unavailable"
                )
                if kind in {"RPH", "RPD", "TPD", "IPD"}:
                    raise NonRetryableAPIError(
                        f"SiliconFlow {kind} quota was reached (HTTP 429); "
                        "this is not a minute-scale limit, so stop and wait for "
                        f"the quota window to reset. trace_id={trace_id}"
                    )
                last_error = requests.HTTPError(
                    f"retryable HTTP 429 ({kind or 'unknown limit'}): "
                    f"{response.text[:500]}",
                    response=response,
                )
                rate_limit_failures += 1
                if rate_limit_failures > rate_limit_retries:
                    rate_limit_exhausted = True
                    break
                if rate_limiter is None:
                    time.sleep(
                        max(
                            DEFAULT_RATE_LIMIT_COOLDOWN,
                            _retry_after_seconds(response),
                        )
                    )
                else:
                    decision = rate_limiter.on_rate_limit(response)
                    if decision["new_wave"]:
                        print(
                            "rate limit reached "
                            f"({kind or 'unknown'}); pausing all workers for "
                            f"{decision['cooldown_seconds']:.0f}s, then pacing at "
                            f"{decision['requests_per_minute']:.2f} requests/min",
                            flush=True,
                        )
                continue
            if response.status_code in {500, 502, 503, 504}:
                raise requests.HTTPError(
                    f"retryable HTTP {response.status_code}: {response.text[:500]}",
                    response=response,
                )
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            annotation = validate_response(json.loads(content), task)
            return annotation, {
                "trace_id": response.headers.get("x-siliconcloud-trace-id"),
                "request_id": body.get("id"),
                "system_fingerprint": body.get("system_fingerprint"),
                "usage": body.get("usage", {}),
            }
        except NonRetryableAPIError:
            raise
        except (
            AnnotationError,
            KeyError,
            TypeError,
            ValueError,
            requests.RequestException,
        ) as exc:
            last_error = exc
            if transient_failures >= retries:
                break
            time.sleep(min(30.0, 2.0 ** transient_failures))
            transient_failures += 1
    message = (
        f"price annotation failed for task {task['task_id']} "
        f"({task['source_field']}) after {http_attempts} HTTP attempts: {last_error}"
    )
    if rate_limit_exhausted:
        raise RateLimitExhaustedError(message) from last_error
    raise DatasetBuildError(message) from last_error


def _cache_record(
    task: dict,
    annotation: dict,
    *,
    model: str,
    api_url: str,
    source_sha256: str,
    metadata: dict,
) -> dict:
    return {
        "annotation_version": ANNOTATION_VERSION,
        "model": model,
        "api_url": api_url,
        "prompt_sha256": prompt_sha256(),
        "source_archive_sha256": source_sha256,
        "task_id": task["task_id"],
        "task_uid": task["task_uid"],
        "source_field": task["source_field"],
        "annotation": annotation,
        **metadata,
    }


def import_legacy_cache(
    legacy_path: Path,
    *,
    cache_path: Path,
    cache: dict[tuple[int, str], dict],
    persona_tasks: list[dict],
    model: str,
    api_url: str,
    source_sha256: str,
) -> int:
    """Import only v1 records that remain valid under the current protocol."""
    if not legacy_path.is_file():
        return 0
    by_id = {task["task_id"]: task for task in persona_tasks}
    records = []
    with legacy_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            old = json.loads(line)
            if (
                old.get("annotation_version") != "qwen-price-upper-v1"
                or old.get("model") != model
                or old.get("source_archive_sha256") != source_sha256
            ):
                continue
            task = by_id.get(int(old["task_id"]))
            if task is None:
                continue
            source_field = old.get("annotation", {}).get("source_field")
            if source_field == "instruction":
                phase_task = full_instruction_task(task)
            elif source_field == "instruction_simple":
                phase_task = task
            else:
                continue
            key = (task["task_id"], source_field)
            if key in cache:
                continue
            converted = convert_valid_v1_annotation(old["annotation"], phase_task)
            if converted is None:
                continue
            cache[key] = converted
            records.append(
                _cache_record(
                    phase_task,
                    converted,
                    model=model,
                    api_url=api_url,
                    source_sha256=source_sha256,
                    metadata={
                        "imported_from": "qwen-price-upper-v1",
                        "request_id": old.get("request_id"),
                        "trace_id": old.get("trace_id"),
                        "system_fingerprint": old.get("system_fingerprint"),
                        "usage": old.get("usage", {}),
                    },
                )
            )
    append_annotation_cache(cache_path, records)
    return len(records)


def annotate_phase(
    tasks: list[dict],
    *,
    cache_path: Path,
    cache: dict[tuple[int, str], dict],
    api_key: str,
    api_url: str,
    model: str,
    workers: int,
    timeout: float,
    retries: int,
    rate_limit_retries: int,
    rate_limiter: AdaptiveRateLimiter,
    max_requests: int | None,
    source_sha256: str,
) -> int:
    missing = [
        task
        for task in tasks
        if (task["task_id"], task["source_field"]) not in cache
    ]
    if max_requests is not None:
        missing = missing[:max_requests]
    if not missing:
        return 0

    completed = 0

    def annotate_and_append(task: dict) -> None:
        nonlocal completed
        annotation, metadata = call_task(
            task,
            api_key=api_key,
            api_url=api_url,
            model=model,
            timeout=timeout,
            retries=retries,
            rate_limit_retries=rate_limit_retries,
            rate_limiter=rate_limiter,
        )
        record = _cache_record(
            task,
            annotation,
            model=model,
            api_url=api_url,
            source_sha256=source_sha256,
            metadata=metadata,
        )
        # The coordinator is the only writer. Each completed request becomes one
        # durable JSONL line before the in-memory cache is updated.
        append_annotation_cache(cache_path, [record])
        cache[(task["task_id"], task["source_field"])] = annotation
        completed += 1
        print(
            f"annotated {task['source_field']} task {task['task_id']} "
            f"({completed}/{len(missing)})",
            flush=True,
        )

    # Validate credentials, endpoint, model access, and response schema with one
    # paid call before allowing a systemic error to fan out to all workers.
    annotate_and_append(missing[0])
    remaining = iter(missing[1:])
    failures: list[tuple[dict, Exception]] = []
    fatal_error: Exception | None = None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending: dict[Future, dict] = {}

        def submit_next() -> bool:
            try:
                task = next(remaining)
            except StopIteration:
                return False
            future = executor.submit(
                call_task,
                task,
                api_key=api_key,
                api_url=api_url,
                model=model,
                timeout=timeout,
                retries=retries,
                rate_limit_retries=rate_limit_retries,
                rate_limiter=rate_limiter,
            )
            pending[future] = task
            return True

        for _ in range(workers):
            if not submit_next():
                break

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending.pop(future)
                try:
                    annotation, metadata = future.result()
                    record = _cache_record(
                        task,
                        annotation,
                        model=model,
                        api_url=api_url,
                        source_sha256=source_sha256,
                        metadata=metadata,
                    )
                    append_annotation_cache(cache_path, [record])
                    cache[(task["task_id"], task["source_field"])] = annotation
                    completed += 1
                    print(
                        f"annotated {task['source_field']} task {task['task_id']} "
                        f"({completed}/{len(missing)})",
                        flush=True,
                    )
                except CancelledError:
                    continue
                except Exception as exc:
                    failures.append((task, exc))
                    if isinstance(
                        exc, (NonRetryableAPIError, RateLimitExhaustedError)
                    ):
                        fatal_error = exc

            if fatal_error is not None:
                for future in pending:
                    future.cancel()
            else:
                for _ in range(len(done)):
                    if not submit_next():
                        break

    if fatal_error is not None:
        raise fatal_error
    if failures:
        examples = [
            f"{task['task_id']}:{task['source_field']} ({error})"
            for task, error in failures[:5]
        ]
        raise DatasetBuildError(
            f"{len(failures)} individual price annotation request(s) failed; "
            f"successful requests were appended to {cache_path}; examples={examples}"
        ) from failures[0][1]
    return len(missing)


def annotate_persona_tasks(
    persona_tasks: list[dict],
    *,
    cache_path: Path,
    legacy_cache_path: Path,
    api_key: str,
    api_url: str,
    model: str,
    workers: int,
    timeout: float,
    retries: int,
    rate_limit_retries: int,
    requests_per_minute: float,
    rate_limit_cooldown: float,
    max_requests: int | None,
    runtime_fallback_task_ids: set[int],
    source_sha256: str,
) -> dict[tuple[int, str], dict]:
    cache = load_annotation_cache(
        cache_path,
        model=model,
        api_url=api_url,
        source_sha256=source_sha256,
    )
    imported = import_legacy_cache(
        legacy_cache_path,
        cache_path=cache_path,
        cache=cache,
        persona_tasks=persona_tasks,
        model=model,
        api_url=api_url,
        source_sha256=source_sha256,
    )
    if imported:
        print(f"imported {imported} compatible records from the v1 cache")

    rate_limiter = AdaptiveRateLimiter(
        requests_per_minute=requests_per_minute,
        cooldown_seconds=rate_limit_cooldown,
    )

    llm_tasks = [
        task
        for task in persona_tasks
        if task["task_id"] not in runtime_fallback_task_ids
    ]
    used = annotate_phase(
        llm_tasks,
        cache_path=cache_path,
        cache=cache,
        api_key=api_key,
        api_url=api_url,
        model=model,
        workers=workers,
        timeout=timeout,
        retries=retries,
        rate_limit_retries=rate_limit_retries,
        rate_limiter=rate_limiter,
        max_requests=max_requests,
        source_sha256=source_sha256,
    )
    if any(
        (task["task_id"], "instruction_simple") not in cache
        for task in llm_tasks
    ):
        return cache

    fallback_tasks = []
    for task in llm_tasks:
        simple = cache[(task["task_id"], "instruction_simple")]
        if (
            simple["kind"] not in KINDS_WITH_UPPER_BOUND
            and task["instruction"] != task["instruction_simple"]
        ):
            fallback_tasks.append(full_instruction_task(task))
    remaining = None if max_requests is None else max(0, max_requests - used)
    if remaining != 0:
        annotate_phase(
            fallback_tasks,
            cache_path=cache_path,
            cache=cache,
            api_key=api_key,
            api_url=api_url,
            model=model,
            workers=workers,
            timeout=timeout,
            retries=retries,
            rate_limit_retries=rate_limit_retries,
            rate_limiter=rate_limiter,
            max_requests=remaining,
            source_sha256=source_sha256,
        )
    return cache


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a versioned corpus without modifying the raw archive. "
            "Only eligible persona tasks are sent to the paid LLM API."
        )
    )
    parser.add_argument(
        "--stage", choices=("plan", "annotate", "build", "all"), default="plan"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SHOP_ENV / "data" / "fine_items_eval_train_all.raw.json.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SHOP_ENV / "data" / "fine_items_eval_train_all.cleaned.v2.json.gz",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=SHOP_ENV / "data" / "price_annotations.persona.qwen3_5_27b.v3.jsonl",
    )
    parser.add_argument(
        "--legacy-cache",
        type=Path,
        default=SHOP_ENV / "data" / "price_annotations.qwen3_5_27b.v1.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SHOP_ENV / "configs" / "clean_dataset_manifest.v2.json",
    )
    parser.add_argument(
        "--exclusions",
        type=Path,
        default=SHOP_ENV / "configs" / "catalog_exclusions.v2.json",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=SHOP_ENV / "configs" / "task_splits.cleaned.v2.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=SHOP_ENV / "configs" / "persona_price_annotations.v2.json",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--api-key-env", default="SILICONFLOW_API_KEY")
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="maximum concurrent one-instruction API requests (default: 32)",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=DEFAULT_RATE_LIMIT_RETRIES,
        help=(
            "maximum minute-scale 429 retries per instruction "
            f"(default: {DEFAULT_RATE_LIMIT_RETRIES})"
        ),
    )
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help=(
            "globally pace request starts across all workers "
            f"(default: {DEFAULT_REQUESTS_PER_MINUTE:g})"
        ),
    )
    parser.add_argument(
        "--rate-limit-cooldown",
        type=float,
        default=DEFAULT_RATE_LIMIT_COOLDOWN,
        help=(
            "shared cooldown after a minute-scale HTTP 429 "
            f"(default: {DEFAULT_RATE_LIMIT_COOLDOWN:g}s)"
        ),
    )
    parser.add_argument(
        "--max-requests",
        "--max-batches",
        dest="max_requests",
        type=int,
        help=(
            "limit instruction records scheduled for a smoke test; retries may "
            "make more HTTP attempts; "
            "--max-batches is retained as a deprecated alias"
        ),
    )
    parser.add_argument(
        "--runtime-regex-task-id",
        action="append",
        type=int,
        default=[],
        help=(
            "explicit persona task ID whose missing LLM annotation may be stored "
            "as price_upper=null and handled by the runtime regex; repeatable"
        ),
    )
    parser.add_argument("--confirm-paid-api", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    if args.rate_limit_retries < 0:
        raise ValueError("--rate-limit-retries cannot be negative")
    if args.requests_per_minute <= 0:
        raise ValueError("--requests-per-minute must be positive")
    if args.rate_limit_cooldown <= 0:
        raise ValueError("--rate-limit-cooldown must be positive")
    if args.max_requests is not None and args.max_requests < 1:
        raise ValueError("--max-requests must be positive")
    if not args.source.is_file():
        raise FileNotFoundError(f"raw source archive not found: {args.source}")

    products = read_products(args.source)
    all_tasks = source_tasks(products)
    persona_tasks, exclusions = persona_annotation_tasks(products)
    persona_task_ids = {task["task_id"] for task in persona_tasks}
    runtime_fallback_task_ids = set(args.runtime_regex_task_id)
    invalid_fallback_ids = runtime_fallback_task_ids - persona_task_ids
    if invalid_fallback_ids:
        raise ValueError(
            "--runtime-regex-task-id must identify an eligible persona task; "
            f"invalid={sorted(invalid_fallback_ids)}"
        )
    source_digest = sha256_file(args.source)
    cache = load_annotation_cache(
        args.cache,
        model=args.model,
        api_url=args.api_url,
        source_sha256=source_digest,
    )
    excluded_tasks = sum(len(row["task_ids"]) for row in exclusions)
    active_tasks = len(all_tasks) - excluded_tasks
    cached_simple = sum(
        (task["task_id"], "instruction_simple") in cache
        for task in persona_tasks
        if task["task_id"] not in runtime_fallback_task_ids
    )
    llm_persona_count = len(persona_tasks) - len(runtime_fallback_task_ids)
    pending_full = sum(
        (task["task_id"], "instruction") not in cache
        and (task["task_id"], "instruction_simple") in cache
        and cache[(task["task_id"], "instruction_simple")]["kind"]
        not in KINDS_WITH_UPPER_BOUND
        and task["instruction"] != task["instruction_simple"]
        for task in persona_tasks
        if task["task_id"] not in runtime_fallback_task_ids
    )
    summary = {
        "stage": args.stage,
        "source_products": len(products),
        "source_tasks": len(all_tasks),
        "active_tasks_after_price_axis_filter": active_tasks,
        "excluded_multi_price_axis_products": len(exclusions),
        "excluded_multi_price_axis_tasks": excluded_tasks,
        "persona_tasks": len(persona_tasks),
        "persona_llm_tasks": llm_persona_count,
        "persona_runtime_regex_tasks": len(runtime_fallback_task_ids),
        "standard_regex_tasks": active_tasks - len(persona_tasks),
        "cached_annotation_phases": len(cache),
        "model": args.model,
        "request_granularity": "one_instruction",
        "workers": args.workers,
        "requests_per_minute": args.requests_per_minute,
        "rate_limit_cooldown_seconds": args.rate_limit_cooldown,
        "rate_limit_retries": args.rate_limit_retries,
        "runtime_regex_fallback_task_ids": sorted(runtime_fallback_task_ids),
        "initial_persona_requests": llm_persona_count - cached_simple,
        "pending_full_instruction_requests": pending_full,
        "prompt_sha256": prompt_sha256(),
        "source_will_be_modified": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.stage == "plan":
        return

    if args.stage in {"annotate", "all"}:
        if not args.confirm_paid_api:
            raise DatasetBuildError(
                "refusing paid API calls without --confirm-paid-api"
            )
        api_key = (os.environ.get(args.api_key_env) or "").strip()
        if not api_key:
            raise DatasetBuildError(
                f"set API key in environment variable {args.api_key_env!r}"
            )
        if not api_key.startswith("sk-") or len(api_key) < 20:
            raise DatasetBuildError(
                f"{args.api_key_env!r} does not look like a SiliconFlow API key; "
                f"length={len(api_key)}, starts_with_sk={api_key.startswith('sk-')}"
            )
        cache = annotate_persona_tasks(
            persona_tasks,
            cache_path=args.cache,
            legacy_cache_path=args.legacy_cache,
            api_key=api_key,
            api_url=args.api_url,
            model=args.model,
            workers=args.workers,
            timeout=args.timeout,
            retries=args.retries,
            rate_limit_retries=args.rate_limit_retries,
            requests_per_minute=args.requests_per_minute,
            rate_limit_cooldown=args.rate_limit_cooldown,
            max_requests=args.max_requests,
            runtime_fallback_task_ids=runtime_fallback_task_ids,
            source_sha256=source_digest,
        )

    if args.stage in {"build", "all"}:
        resolved = resolve_persona_annotations(
            persona_tasks,
            cache,
            runtime_fallback_task_ids=runtime_fallback_task_ids,
        )
        manifest = write_clean_artifacts(
            source=args.source,
            output=args.output,
            manifest_path=args.manifest,
            exclusions_path=args.exclusions,
            splits_path=args.splits,
            audit_path=args.audit,
            products=products,
            persona_annotations=resolved,
            model=args.model,
            api_url=args.api_url,
            force=args.force,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
