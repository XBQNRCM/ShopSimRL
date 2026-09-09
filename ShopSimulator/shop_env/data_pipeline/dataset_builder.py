"""Build a stable, filtered ShopSimulator corpus from immutable source data."""

from __future__ import annotations

from collections import Counter
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Mapping

from data_pipeline.price_annotation import ANNOTATION_VERSION, prompt_sha256
from web_agent_site.engine.variant_price import (
    VARIANT_PRICE_VERSION,
    price_affecting_axes,
)


DATASET_SCHEMA_VERSION = "shopsimulator-cleaned-v2"

# Product boundaries in the immutable 23,421-row source corpus.
SOURCE_SPLITS = {
    "persona_eval": (0, 1343, True),
    "standard_eval": (0, 1459, False),
    "persona_train": (1459, 4782, True),
    "standard_train": (4782, 23421, False),
    "all_train": (1459, 23421, None),
}


class DatasetBuildError(RuntimeError):
    """Raised when source data cannot produce a safe derived corpus."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_gzip_content(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_products(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise DatasetBuildError("source corpus must be a JSON array of objects")
    return payload


def source_tasks(products: Iterable[dict]) -> list[dict]:
    """Assign permanent IDs before any filtering and verify persona metadata."""
    tasks = []
    task_id = 0
    for product_index, product in enumerate(products):
        asin = str(product.get("asin", ""))
        instructions = product.get("instructions") or []
        if not isinstance(instructions, list):
            raise DatasetBuildError(
                f"product {product_index} has a non-list instructions field"
            )
        for instruction_index, instruction in enumerate(instructions):
            if not isinstance(instruction, dict):
                raise DatasetBuildError(
                    f"instruction {product_index}:{instruction_index} is not an object"
                )
            complete = str(instruction.get("instruction") or "")
            simple = str(instruction.get("instruction_simple") or "")
            has_persona = bool(product.get("user_persona"))
            if bool(simple) != has_persona:
                raise DatasetBuildError(
                    "instruction_simple and user_persona disagree for source product "
                    f"{product_index}"
                )
            uid_material = json.dumps(
                [asin, instruction_index, complete],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            tasks.append(
                {
                    "task_id": task_id,
                    "task_uid": "shopsim-" + hashlib.sha256(uid_material).hexdigest()[:24],
                    "source_product_index": product_index,
                    "instruction_index": instruction_index,
                    "asin": asin,
                    "is_persona": has_persona,
                    "instruction_simple": simple,
                    "instruction": complete,
                }
            )
            task_id += 1
    return tasks


def catalog_partition(products: list[dict]) -> tuple[set[int], list[dict]]:
    eligible = set()
    exclusions = []
    tasks_by_product = {}
    for task in source_tasks(products):
        tasks_by_product.setdefault(task["source_product_index"], []).append(task)
    for source_index, product in enumerate(products):
        axes = price_affecting_axes(product)
        if len(axes) >= 2:
            exclusions.append(
                {
                    "source_product_index": source_index,
                    "asin": str(product.get("asin", "")),
                    "task_ids": [
                        task["task_id"]
                        for task in tasks_by_product.get(source_index, [])
                    ],
                    "reason": "multiple_price_affecting_axes",
                    "price_affecting_axes": axes,
                }
            )
        else:
            eligible.add(source_index)
    return eligible, exclusions


def persona_annotation_tasks(products: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return only eligible persona tasks; standard tasks never reach the LLM."""
    eligible_products, exclusions = catalog_partition(products)
    persona_tasks = []
    for task in source_tasks(products):
        if (
            task["source_product_index"] in eligible_products
            and task["is_persona"]
        ):
            persona_tasks.append(
                {
                    **task,
                    "source_field": "instruction_simple",
                    "annotation_text": task["instruction_simple"],
                }
            )
    return persona_tasks, exclusions


def full_instruction_task(task: Mapping[str, object]) -> dict:
    return {
        **task,
        "source_field": "instruction",
        "annotation_text": str(task.get("instruction") or ""),
    }


def load_annotation_cache(
    path: Path,
    *,
    model: str,
    api_url: str,
    source_sha256: str,
) -> dict[tuple[int, str], dict]:
    if not path.is_file():
        return {}
    expected_prompt = prompt_sha256()
    annotations = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetBuildError(
                    f"invalid cache JSON at {path}:{line_number}"
                ) from exc
            if (
                record.get("annotation_version") != ANNOTATION_VERSION
                or record.get("model") != model
                or record.get("api_url") != api_url
                or record.get("prompt_sha256") != expected_prompt
                or record.get("source_archive_sha256") != source_sha256
            ):
                continue
            source_field = str(record.get("source_field"))
            if source_field not in {"instruction_simple", "instruction"}:
                continue
            annotations[(int(record["task_id"]), source_field)] = record["annotation"]
    return annotations


def append_annotation_cache(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def resolve_persona_annotations(
    persona_tasks: Iterable[dict],
    cache: Mapping[tuple[int, str], dict],
    *,
    runtime_fallback_task_ids: Iterable[int] = (),
) -> dict[int, dict]:
    """Apply simple-first/full-fallback deterministically outside the model."""
    runtime_fallback_ids = {int(task_id) for task_id in runtime_fallback_task_ids}
    resolved = {}
    missing = []
    for task in persona_tasks:
        task_id = task["task_id"]
        simple = cache.get((task_id, "instruction_simple"))
        if simple is None:
            if task_id in runtime_fallback_ids:
                resolved[task_id] = _runtime_regex_fallback_annotation(
                    "instruction_simple"
                )
                continue
            missing.append(f"{task_id}:instruction_simple")
            continue
        if simple.get("price_upper") is not None:
            resolved[task_id] = simple
            continue
        if task["instruction"] == task["instruction_simple"]:
            resolved[task_id] = simple
            continue
        complete = cache.get((task_id, "instruction"))
        if complete is None:
            if task_id in runtime_fallback_ids:
                resolved[task_id] = _runtime_regex_fallback_annotation(
                    "instruction"
                )
                continue
            missing.append(f"{task_id}:instruction")
            continue
        resolved[task_id] = complete
    if missing:
        raise DatasetBuildError(
            f"persona annotation cache is incomplete ({len(missing)} phases missing); "
            f"examples={missing[:10]}"
        )
    return resolved


def _runtime_regex_fallback_annotation(missing_source_field: str) -> dict:
    return {
        "kind": "runtime_regex_fallback",
        "amount": None,
        "evidence": "",
        "source_field": "runtime_regex",
        "price_upper": None,
        "annotation_version": "runtime-regex-fallback-v1",
        "approximate_upper_multiplier": None,
        "fallback_reason": "explicitly_ignored_llm_failure",
        "missing_source_field": missing_source_field,
    }


def _write_json(path: Path, payload: object, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_deterministic_gzip(path: Path, payload: object, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".building")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                json.dump(
                    payload,
                    text,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                text.write("\n")
    temporary.replace(path)


def build_clean_corpus(
    products: list[dict],
    persona_annotations: Mapping[int, dict],
) -> tuple[list[dict], list[dict], dict, list[dict]]:
    tasks = source_tasks(products)
    eligible_products, exclusions = catalog_partition(products)
    expected_persona_ids = {
        task["task_id"]
        for task in tasks
        if task["is_persona"] and task["source_product_index"] in eligible_products
    }
    if set(persona_annotations) != expected_persona_ids:
        raise DatasetBuildError(
            "persona annotation coverage mismatch; "
            f"missing={len(expected_persona_ids - set(persona_annotations))}, "
            f"extra={len(set(persona_annotations) - expected_persona_ids)}"
        )

    tasks_by_product = {}
    for task in tasks:
        tasks_by_product.setdefault(task["source_product_index"], []).append(task)

    cleaned = []
    active_task_ids = []
    annotation_counts = Counter()
    audit = []
    standard_regex_tasks = 0
    for source_index, source_product in enumerate(products):
        if source_index not in eligible_products:
            continue
        product = copy.deepcopy(source_product)
        product["source_product_index"] = source_index
        for task in tasks_by_product.get(source_index, []):
            instruction = product["instructions"][task["instruction_index"]]
            instruction["task_id"] = task["task_id"]
            instruction["task_uid"] = task["task_uid"]
            instruction["instruction_index"] = task["instruction_index"]
            if task["is_persona"]:
                annotation = copy.deepcopy(persona_annotations[task["task_id"]])
                instruction["price_upper"] = annotation["price_upper"]
                annotation_counts[annotation["kind"]] += 1
                audit.append(
                    {
                        "task_id": task["task_id"],
                        "task_uid": task["task_uid"],
                        "source_product_index": source_index,
                        "asin": task["asin"],
                        **annotation,
                    }
                )
            else:
                # Standard tasks intentionally use the runtime regex fallback.
                instruction["price_upper"] = None
                standard_regex_tasks += 1
            instruction.pop("price_constraint", None)
            active_task_ids.append(task["task_id"])
        cleaned.append(product)

    persona_runtime_regex_tasks = annotation_counts.get(
        "runtime_regex_fallback", 0
    )
    stats = {
        "source_product_count": len(products),
        "source_task_count": len(tasks),
        "cleaned_product_count": len(cleaned),
        "active_task_count": len(active_task_ids),
        "excluded_product_count": len(exclusions),
        "excluded_task_count": len(tasks) - len(active_task_ids),
        "persona_task_count": len(persona_annotations),
        "persona_llm_task_count": (
            len(persona_annotations) - persona_runtime_regex_tasks
        ),
        "persona_runtime_regex_task_count": persona_runtime_regex_tasks,
        "persona_runtime_regex_task_ids": sorted(
            task_id
            for task_id, annotation in persona_annotations.items()
            if annotation["kind"] == "runtime_regex_fallback"
        ),
        "standard_regex_task_count": standard_regex_tasks,
        "persona_annotation_kind_counts": dict(sorted(annotation_counts.items())),
        "active_task_ids": active_task_ids,
    }
    return cleaned, exclusions, stats, audit


def explicit_task_splits(cleaned_products: Iterable[dict], source_task_count: int) -> dict:
    active_by_product = {}
    for product in cleaned_products:
        source_index = int(product["source_product_index"])
        active_by_product[source_index] = [
            int(instruction["task_id"])
            for instruction in product.get("instructions") or []
        ]
    split_payload = {
        "version": "shopsimulator-stable-task-splits-v2",
        "id_semantics": "IDs assigned before filtering; gaps are intentional",
        "source_task_count": source_task_count,
        "active_task_count": sum(len(ids) for ids in active_by_product.values()),
        "splits": {},
    }
    for name, (start, end, persona) in SOURCE_SPLITS.items():
        task_ids = []
        for source_index in range(start, end):
            task_ids.extend(active_by_product.get(source_index, []))
        split_payload["splits"][name] = {
            "task_ids": task_ids,
            "persona": persona,
        }
    return split_payload


def write_clean_artifacts(
    *,
    source: Path,
    output: Path,
    manifest_path: Path,
    exclusions_path: Path,
    splits_path: Path,
    audit_path: Path,
    products: list[dict],
    persona_annotations: Mapping[int, dict],
    model: str,
    api_url: str,
    force: bool = False,
) -> dict:
    source_resolved = source.resolve()
    destinations = [output, manifest_path, exclusions_path, splits_path, audit_path]
    if any(path.resolve() == source_resolved for path in destinations):
        raise DatasetBuildError("a derived artifact must not overwrite the source archive")

    cleaned, exclusions, stats, audit = build_clean_corpus(
        products, persona_annotations
    )
    splits = explicit_task_splits(cleaned, stats["source_task_count"])
    _write_deterministic_gzip(output, cleaned, force=force)
    _write_json(exclusions_path, exclusions, force=force)
    _write_json(splits_path, splits, force=force)
    _write_json(audit_path, audit, force=force)

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "annotation_version": ANNOTATION_VERSION,
        "variant_price_version": VARIANT_PRICE_VERSION,
        "model": model,
        "api_url": api_url,
        "prompt_sha256": prompt_sha256(),
        "source_archive": source.name,
        "source_archive_sha256": sha256_file(source),
        "output_archive": output.name,
        "output_archive_sha256": sha256_file(output),
        "output_data_sha256": sha256_gzip_content(output),
        "exclusions_sha256": sha256_file(exclusions_path),
        "task_splits_sha256": sha256_file(splits_path),
        "persona_price_audit_sha256": sha256_file(audit_path),
        **{key: value for key, value in stats.items() if key != "active_task_ids"},
    }
    _write_json(manifest_path, manifest, force=force)
    return manifest
