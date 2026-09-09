#!/usr/bin/env python3
"""Build the frozen ShopSimRL persona train/val/test manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SHOP_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOP_ENV))

# Frozen defaults. Every value can also be overridden on the command line.
CLEANED_ARCHIVE = SHOP_ENV / "data/fine_items_eval_train_all.cleaned.v2.json.gz"
CLEAN_DATASET_MANIFEST = SHOP_ENV / "configs/clean_dataset_manifest.v2.json"
SOURCE_SPLIT_MANIFEST = SHOP_ENV / "configs/task_splits.cleaned.v2.json"
OUTPUT_MANIFEST = SHOP_ENV / "configs/persona_splits.v1.json"
SEED = 20260829
VALIDATION_SIZE = 400
TEST_SIZE = 400
HELD_OUT_MAX_GROUP_SIZE = 20

from data_pipeline.dataset_builder import (  # noqa: E402
    read_products,
    sha256_file,
    sha256_gzip_content,
)
from data_pipeline.project_splits import build_project_split_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-archive", type=Path, default=CLEANED_ARCHIVE)
    parser.add_argument(
        "--clean-dataset-manifest", type=Path, default=CLEAN_DATASET_MANIFEST
    )
    parser.add_argument(
        "--source-split-manifest", type=Path, default=SOURCE_SPLIT_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_MANIFEST)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--validation-size", type=int, default=VALIDATION_SIZE)
    parser.add_argument("--test-size", type=int, default=TEST_SIZE)
    parser.add_argument(
        "--held-out-max-group-size", type=int, default=HELD_OUT_MAX_GROUP_SIZE
    )
    parser.add_argument("--test-exclude-task-id", action="append", type=int, default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_manifest = json.loads(
        args.clean_dataset_manifest.read_text(encoding="utf-8")
    )
    source_splits = json.loads(
        args.source_split_manifest.read_text(encoding="utf-8")
    )
    archive_sha = sha256_file(args.cleaned_archive)
    data_sha = sha256_gzip_content(args.cleaned_archive)
    source_splits_sha = sha256_file(args.source_split_manifest)
    if archive_sha != clean_manifest["output_archive_sha256"]:
        raise RuntimeError("cleaned archive hash does not match clean dataset manifest")
    if data_sha != clean_manifest["output_data_sha256"]:
        raise RuntimeError("cleaned data hash does not match clean dataset manifest")
    if source_splits_sha != clean_manifest["task_splits_sha256"]:
        raise RuntimeError("source split hash does not match clean dataset manifest")

    source = {
        "cleaned_archive": args.cleaned_archive.name,
        "cleaned_archive_sha256": archive_sha,
        "cleaned_data_sha256": data_sha,
        "clean_dataset_manifest": args.clean_dataset_manifest.name,
        "clean_dataset_manifest_sha256": sha256_file(args.clean_dataset_manifest),
        "source_split_manifest": args.source_split_manifest.name,
        "source_split_manifest_sha256": source_splits_sha,
        "persona_task_count": int(clean_manifest["persona_task_count"]),
    }
    manifest = build_project_split_manifest(
        read_products(args.cleaned_archive),
        source_splits,
        seed=args.seed,
        validation_size=args.validation_size,
        test_size=args.test_size,
        held_out_max_group_size=args.held_out_max_group_size,
        source=source,
        test_excluded_task_ids=args.test_exclude_task_id,
    )
    if manifest["checks"]["persona_task_count"] != source["persona_task_count"]:
        raise RuntimeError("project split count does not match clean dataset manifest")

    serialized = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") == serialized:
            print(f"Project split manifest is ready: {args.output}")
            return
        if not args.force:
            raise FileExistsError(
                f"refusing to replace {args.output}; pass --force after reviewing changes"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".building")
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    temporary.replace(args.output)
    counts = {
        name: split["task_count"] for name, split in manifest["splits"].items()
    }
    print(f"Wrote project split manifest: {args.output}")
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
