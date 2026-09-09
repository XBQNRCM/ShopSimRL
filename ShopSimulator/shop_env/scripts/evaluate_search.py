#!/usr/bin/env python3
"""Evaluate Gold-ASIN retrieval on a stable ShopSimulator task split."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading


SHOP_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOP_ENV))

from scripts.build_index import iter_json_array  # noqa: E402
from shop_env.task_splits import task_ids  # noqa: E402
from web_agent_site.engine.search import (  # noqa: E402
    MultiFieldBM25Searcher,
    sha256_file,
)


_THREAD_LOCAL = threading.local()


def iter_retrieval_tasks(products_path: Path):
    """Yield tasks in the same product/instruction order as goal construction."""
    seen_asins = set()
    for product in iter_json_array(products_path):
        asin = str(product.get("asin", "")).strip()
        if not asin or asin == "nan" or len(asin) > 20 or asin in seen_asins:
            continue
        seen_asins.add(asin)
        for instruction in product.get("instructions") or []:
            if not (instruction.get("attributes") or []):
                continue
            complete = str(instruction.get("instruction") or "")
            yield {
                "asin": asin,
                "simple": str(instruction.get("instruction_simple") or complete),
                "full": complete,
            }


def load_split_tasks(products_path: Path, split_name: str) -> list[dict]:
    selected_ids = task_ids(split_name)
    start, stop = selected_ids.start, selected_ids.stop
    selected = []
    for task_id, task in enumerate(iter_retrieval_tasks(products_path)):
        if task_id >= stop:
            break
        if task_id >= start:
            selected.append(task)
    if len(selected) != len(selected_ids):
        raise RuntimeError(
            f"split {split_name!r} expected {len(selected_ids)} tasks, "
            f"loaded {len(selected)}"
        )
    return selected


def _thread_searcher(index_path: Path) -> MultiFieldBM25Searcher:
    searcher = getattr(_THREAD_LOCAL, "searcher", None)
    if searcher is None:
        searcher = MultiFieldBM25Searcher(index_path)
        _THREAD_LOCAL.searcher = searcher
    return searcher


def gold_rank(task: dict, query_mode: str, index_path: Path, top_k: int):
    hits = _thread_searcher(index_path).search(task[query_mode], k=top_k)
    return next(
        (hit.rank for hit in hits if hit.asin == task["asin"]),
        None,
    )


def summarize(ranks: list[int | None], top_k: int) -> dict:
    count = len(ranks)
    return {
        "queries": count,
        "recall_at_1": sum(rank == 1 for rank in ranks) / count,
        "recall_at_20": sum(
            rank is not None and rank <= 20 for rank in ranks
        )
        / count,
        f"recall_at_{top_k}": sum(rank is not None for rank in ranks) / count,
        f"mrr_at_{top_k}": sum(1.0 / rank if rank else 0.0 for rank in ranks)
        / count,
    }


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--split", default="persona_eval")
    parser.add_argument(
        "--query-mode",
        action="append",
        choices=("simple", "full"),
        help="May be supplied more than once; defaults to both modes.",
    )
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k < 20 or args.workers <= 0:
        raise ValueError("top-k must be at least 20 and workers must be positive")
    modes = list(dict.fromkeys(args.query_mode or ("simple", "full")))
    tasks = load_split_tasks(args.products, args.split)

    product_sha = sha256_file(args.products)
    probe = MultiFieldBM25Searcher(
        args.index,
        expected_product_sha256=product_sha,
    )
    manifest = dict(probe.manifest)
    probe.close()
    result = {
        "search_version": manifest["search_version"],
        "product_data_sha256": manifest["product_data_sha256"],
        "split": args.split,
        "top_k": args.top_k,
        "metrics": {},
        "metric_scope": "Gold-ASIN retrieval proxy; alternatives may also be valid",
    }
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for mode in modes:
            ranks = list(
                executor.map(
                    lambda task: gold_rank(task, mode, args.index, args.top_k),
                    tasks,
                )
            )
            result["metrics"][mode] = summarize(ranks, args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
