#!/usr/bin/env python3
"""Materialize the cleaned corpus and its deterministic search index."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


SHOP_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOP_ENV))

from scripts.build_index import iter_json_array  # noqa: E402
from web_agent_site.engine.search import (  # noqa: E402
    MultiFieldBM25Searcher,
    SearchIndexError,
    build_index,
    sha256_file,
)


CLEANED_ARCHIVE = SHOP_ENV / "data" / "fine_items_eval_train_all.cleaned.v2.json.gz"
CLEANED_MANIFEST = SHOP_ENV / "configs" / "clean_dataset_manifest.v2.json"


def decompress_verified(
    source: Path,
    target: Path,
    *,
    expected_sha256: str,
    force: bool = False,
) -> str:
    if target.is_file() and not force:
        digest = sha256_file(target)
        if digest == expected_sha256:
            print(f"Product corpus is ready: {target}")
            return digest
        raise RuntimeError(
            f"existing product corpus has unexpected SHA-256: {digest}; "
            "use --force only if you intend to replace it"
        )
    if not source.is_file():
        raise FileNotFoundError(f"bundled product archive not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".preparing")
    digest = hashlib.sha256()
    try:
        with gzip.open(source, "rb") as compressed, temporary.open("wb") as output:
            while chunk := compressed.read(1024 * 1024):
                digest.update(chunk)
                output.write(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise RuntimeError(
                "decompressed product corpus SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual}"
            )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"Prepared product corpus: {target}")
    return actual


def ensure_index(products: Path, index: Path, product_sha: str, *, force: bool = False):
    if index.is_file() and not force:
        try:
            searcher = MultiFieldBM25Searcher(
                index, expected_product_sha256=product_sha
            )
            manifest = dict(searcher.manifest)
            searcher.close()
            print(f"Search index is ready: {index}")
            return manifest
        except SearchIndexError as exc:
            raise RuntimeError(
                f"existing search index is incompatible: {exc}; "
                "run again with --force-index"
            ) from exc

    manifest = build_index(
        iter_json_array(products),
        index,
        product_data_sha256=product_sha,
    )
    manifest_path = index.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Built search index with {manifest['product_count']} products: {index}")
    return manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=CLEANED_ARCHIVE,
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=CLEANED_MANIFEST,
        help="clean-dataset manifest containing archive and decompressed SHA-256",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=SHOP_ENV / "data" / "items_eval_train.json",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=SHOP_ENV / "search_engine" / "products.sqlite3",
    )
    parser.add_argument("--force-products", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.dataset_manifest.is_file():
        raise FileNotFoundError(
            f"clean dataset manifest not found: {args.dataset_manifest}"
        )
    manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "shopsimulator-cleaned-v2":
        raise RuntimeError("unsupported clean dataset manifest")
    expected_product_sha = str(manifest["output_data_sha256"])
    expected_archive_sha = str(manifest["output_archive_sha256"])
    actual_archive_sha = sha256_file(args.archive)
    if actual_archive_sha != expected_archive_sha:
        raise RuntimeError(
            "cleaned archive SHA-256 mismatch: "
            f"expected {expected_archive_sha}, got {actual_archive_sha}"
        )
    product_sha = decompress_verified(
        args.archive,
        args.products,
        expected_sha256=expected_product_sha,
        force=args.force_products,
    )
    if not args.skip_index:
        ensure_index(
            args.products,
            args.index,
            product_sha,
            force=args.force_index,
        )


if __name__ == "__main__":
    main()
