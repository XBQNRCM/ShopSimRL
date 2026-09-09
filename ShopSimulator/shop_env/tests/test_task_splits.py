import json
from pathlib import Path
import unittest
import tempfile

from shop_env.task_splits import load_task_splits, task_ids


class TaskSplitTest(unittest.TestCase):
    def test_frozen_ranges_match_bundled_corpus_layout(self):
        payload = load_task_splits()
        self.assertEqual(
            payload.get("active_task_count", payload.get("corpus_size")),
            22726,
        )
        self.assertEqual(
            payload.get("source_task_count", payload.get("source_corpus_size")),
            23421,
        )
        self.assertEqual(len(task_ids("persona_eval")), 1274)
        self.assertEqual(len(task_ids("standard_eval")), 1381)
        self.assertEqual(len(task_ids("persona_train")), 3252)
        self.assertEqual(len(task_ids("standard_train")), 18093)
        self.assertEqual(len(task_ids("all_train")), 21345)

    def test_clean_manifest_and_exclusions_match_frozen_corpus(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs"
        manifest_path = config_dir / "clean_dataset_manifest.v2.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_product_count"], 23421)
        self.assertEqual(manifest["cleaned_product_count"], 22726)
        self.assertEqual(manifest["excluded_product_count"], 695)
        exclusions = json.loads(
            (config_dir / "catalog_exclusions.v2.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(exclusions), 695)
        self.assertTrue(
            all(
                len(product["price_affecting_axes"]) >= 2
                for product in exclusions
            )
        )

    def test_explicit_stable_ids_may_contain_gaps(self):
        payload = {
            "version": "test",
            "splits": {
                "train": {"task_ids": [0, 2, 5], "persona": False},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(task_ids("train", path), (0, 2, 5))


if __name__ == "__main__":
    unittest.main()
