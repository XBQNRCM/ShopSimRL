import json
from pathlib import Path
import tempfile
import unittest

from web_agent_site.engine.search import (
    DEFAULT_FIELD_WEIGHTS,
    FTS_COLUMNS,
    MultiFieldBM25Searcher,
    SearchIndexError,
    build_index,
    normalize_query,
    product_fields,
    search_tokens,
)


PRODUCTS = [
    {
        "asin": "000000000003",
        "title": "迷你单门冰箱",
        "shop_name": "小熊",
        "category": "家电›冰箱",
        "attribute": ["宿舍", "静音"],
        "customization_options": {"容量": [{"value": "45升"}]},
    },
    {
        "asin": "000000000001",
        "title": "宿舍小冰箱",
        "shop_name": "海尔",
        "category": "家电›冰箱",
        "attribute": ["节能"],
        "customization_options": {"颜色": [{"value": "白色"}]},
    },
    {
        "asin": "000000000002",
        "title": "开放式耳机",
        "category": "数码›耳机",
        "attribute": ["低夹耳压力"],
    },
]


class SearchV3Test(unittest.TestCase):
    def test_normalizer_and_tokens_are_replayable(self):
        self.assertEqual(normalize_query("  迷你，冰箱 100 CNY "), "迷你 冰箱 100元")
        self.assertEqual(search_tokens("小冰箱"), ("小冰", "冰箱"))

    def test_product_fields_do_not_include_goal_or_reward(self):
        fields = product_fields(
            {
                **PRODUCTS[0],
                "instructions": [{"instruction": "hidden"}],
                "reward_detail": {"answer": True},
            }
        )
        rendered = json.dumps(fields, ensure_ascii=False)
        self.assertNotIn("hidden", rendered)
        self.assertNotIn("answer", rendered)

    def test_option_document_contains_only_public_axes_and_values(self):
        fields = product_fields(
            {
                "title": "台灯",
                "customization_options": {
                    "颜色": [
                        {
                            "value": "暖白",
                            "price": 199,
                            "price_string": "199元",
                            "image": "https://metadata.invalid/secret.jpg",
                            "asin": "variant-secret-asin",
                            "is_available": True,
                        }
                    ]
                },
            }
        )
        self.assertEqual(fields["options"], "颜色 暖白")
        self.assertNotIn("metadata", fields["options"])
        self.assertNotIn("199", fields["options"])
        self.assertNotIn("variant-secret-asin", fields["options"])

    def test_multifield_search_and_asin_tie_break_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            manifest = build_index(PRODUCTS, path, product_data_sha256="abc")
            self.assertEqual(manifest["product_count"], 3)
            self.assertRegex(manifest["python_version"], r"^\d+\.\d+\.\d+$")
            self.assertRegex(manifest["sqlite_version"], r"^\d+\.\d+\.\d+$")
            searcher = MultiFieldBM25Searcher(path, expected_product_sha256="abc")
            hits = searcher.search("宿舍 冰箱", k=3)
            self.assertEqual(hits[0].asin, "000000000001")
            self.assertEqual([hit.rank for hit in hits], list(range(1, len(hits) + 1)))
            self.assertEqual(
                [hit.asin for hit in hits],
                [hit.asin for hit in searcher.search("宿舍 冰箱", k=3)],
            )
            searcher.close()

    def test_bm25_weights_align_with_the_declared_fts_columns(self):
        products = [
            {
                "asin": "000000000001",
                "title": "plain",
                "brand": "needle",
            },
            {
                "asin": "000000000002",
                "title": "needle",
                "brand": "plain",
            },
        ]
        weights = dict(DEFAULT_FIELD_WEIGHTS)
        weights.update(title=10.0, brand=1.0, category=1.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            manifest = build_index(
                products,
                path,
                product_data_sha256="abc",
                field_weights=weights,
            )
            self.assertEqual(manifest["fts_columns"], list(FTS_COLUMNS))
            self.assertEqual(manifest["bm25_column_weights"]["asin"], 0.0)
            searcher = MultiFieldBM25Searcher(path)
            hits = searcher.search("needle", k=2)
            self.assertEqual(hits[0].asin, "000000000002")
            self.assertGreater(hits[0].score, hits[1].score)
            searcher.close()

    def test_product_sha_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            build_index(PRODUCTS, path, product_data_sha256="abc")
            with self.assertRaisesRegex(SearchIndexError, "SHA-256"):
                MultiFieldBM25Searcher(path, expected_product_sha256="wrong")


if __name__ == "__main__":
    unittest.main()
