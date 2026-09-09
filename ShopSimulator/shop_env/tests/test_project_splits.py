import hashlib
import json
from pathlib import Path
import unittest

from data_pipeline.project_splits import build_project_split_manifest


PROJECT_SPLIT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "configs" / "persona_splits.v1.json"
)


def task_ids_sha256(task_ids):
    payload = json.dumps(task_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def persona_product(task_id, *, user_id=None):
    return {
        "asin": f"asin-{task_id}",
        "user_persona": {
            "用户ID": user_id or f"user-{task_id}",
            "preference": f"preference-{task_id}",
        },
        "instructions": [
            {
                "task_id": task_id,
                "task_uid": f"task-{task_id}",
                "instruction": f"完整需求 {task_id}",
                "instruction_simple": f"简化需求 {task_id}",
            }
        ],
    }


class ProjectSplitTest(unittest.TestCase):
    def test_repository_manifest_is_frozen_and_self_consistent(self):
        manifest = json.loads(PROJECT_SPLIT_MANIFEST.read_text(encoding="utf-8"))
        splits = manifest["splits"]
        expected = {
            "train": (
                3726,
                "6dd57881e170a459770c1977cdf06dd184d4dacab8eb1101ebd52c695a85cac9",
            ),
            "val": (
                400,
                "103c72fd5da76870d2406420a8d7eab0c616780cb6fb8ac27875b5424dae872e",
            ),
            "test": (
                400,
                "4137a041471b29310c7e349732c55f04e8ea1c9a517cd60475d048ca1466c257",
            ),
        }

        self.assertEqual(manifest["schema_version"], "shopsimrl-persona-splits-v1")
        self.assertEqual(manifest["policy"]["version"], "exact-linked-components-v1")
        self.assertEqual(manifest["checks"]["persona_task_count"], 4526)
        all_ids = set()
        for name, (expected_count, expected_hash) in expected.items():
            split = splits[name]
            task_ids = split["task_ids"]
            self.assertEqual(split["role"], name)
            self.assertEqual(split["task_count"], expected_count)
            self.assertEqual(len(task_ids), expected_count)
            self.assertEqual(task_ids_sha256(task_ids), expected_hash)
            self.assertEqual(split["task_ids_sha256"], expected_hash)
            self.assertFalse(all_ids.intersection(task_ids))
            all_ids.update(task_ids)
        self.assertEqual(len(all_ids), 4526)
        self.assertLessEqual(splits["val"]["max_group_size"], 20)
        self.assertLessEqual(splits["test"]["max_group_size"], 20)

    def test_split_is_exact_deterministic_and_group_disjoint(self):
        products = [persona_product(task_id) for task_id in range(14)]
        products[1]["user_persona"] = products[0]["user_persona"].copy()
        products[1]["instructions"][0]["instruction"] = "另一个完整需求"
        products[1]["instructions"][0]["instruction_simple"] = "另一个简化需求"
        source_splits = {
            "splits": {
                "persona_eval": {
                    "persona": True,
                    "task_ids": list(range(7)),
                },
                "persona_train": {
                    "persona": True,
                    "task_ids": list(range(7, 14)),
                },
                "standard_train": {"persona": False, "task_ids": []},
            }
        }
        kwargs = {
            "seed": 42,
            "validation_size": 3,
            "test_size": 3,
            "held_out_max_group_size": 2,
            "source": {"test": "source"},
        }

        first = build_project_split_manifest(products, source_splits, **kwargs)
        second = build_project_split_manifest(products, source_splits, **kwargs)

        self.assertEqual(first, second)
        self.assertEqual(first["splits"]["train"]["task_count"], 8)
        self.assertEqual(first["splits"]["val"]["task_count"], 3)
        self.assertEqual(first["splits"]["test"]["task_count"], 3)
        owners = {
            task_id: name
            for name, split in first["splits"].items()
            for task_id in split["task_ids"]
        }
        self.assertEqual(set(owners), set(range(14)))
        self.assertEqual(owners[0], owners[1])

    def test_large_linked_group_stays_out_of_held_out_splits(self):
        products = [persona_product(task_id) for task_id in range(12)]
        for task_id in (0, 1, 2):
            products[task_id]["user_persona"]["用户ID"] = "shared-user"
        source_splits = {
            "splits": {
                "persona_eval": {
                    "persona": True,
                    "task_ids": list(range(6)),
                },
                "persona_train": {
                    "persona": True,
                    "task_ids": list(range(6, 12)),
                },
            }
        }

        manifest = build_project_split_manifest(
            products,
            source_splits,
            seed=7,
            validation_size=2,
            test_size=2,
            held_out_max_group_size=2,
            source={},
        )

        self.assertTrue({0, 1, 2} <= set(manifest["splits"]["train"]["task_ids"]))
        self.assertLessEqual(manifest["splits"]["val"]["max_group_size"], 2)
        self.assertLessEqual(manifest["splits"]["test"]["max_group_size"], 2)


if __name__ == "__main__":
    unittest.main()
