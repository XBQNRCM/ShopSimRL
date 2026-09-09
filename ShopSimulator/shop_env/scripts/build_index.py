#!/usr/bin/env python3
"""Build the reproducible Environment v3 multi-field BM25 index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SHOP_ENV = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SHOP_ENV))

from web_agent_site.engine.search import build_index, sha256_file  # noqa: E402


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024):
    """Stream a top-level JSON array without loading the product corpus at once."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open(encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            eof = not chunk
            buffer += chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("product file must contain a top-level JSON array")
                    started = True
                    position += 1
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) and buffer[position] == "]":
                    return
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    break
                if not isinstance(value, dict):
                    raise ValueError("every product entry must be a JSON object")
                yield value
                position = end
            buffer = buffer[position:]
            if eof:
                if buffer.strip():
                    raise ValueError("incomplete product JSON array")
                raise ValueError("product JSON array has no closing bracket")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--products",
        type=Path,
        default=SHOP_ENV / "data" / "items_eval_train.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SHOP_ENV / "search_engine" / "products.sqlite3",
    )
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    product_sha = sha256_file(args.products)
    manifest = build_index(
        iter_json_array(args.products),
        args.output,
        product_data_sha256=product_sha,
    )
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
