import json
import tempfile
import unittest
from pathlib import Path

from shop_env.replay import (
    ReplayError,
    list_runs,
    load_episode,
    load_run,
    resolve_run_path,
)


def _trace(*, episode_id, task_id, reward, status="completed", protocol=False):
    return {
        "schema_version": "shopsimrl-episode-v4",
        "episode_id": episode_id,
        "status": status,
        "job": {"task_id": task_id, "sample_id": 0, "split": "test", "seed": 7},
        "duration_ms": 2500,
        "reset": {"task_instruction": f"task {task_id}", "task_mode": "persona"},
        "steps": [
            {
                "step_index": 1,
                "model": {
                    "protocol_error": {"type": "missing_tool_call"} if protocol else None,
                    "policy_failure": None,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
                },
                "environment": {
                    "action_feedback": {"valid": not protocol},
                    "observation": "next",
                },
            }
        ],
        "conversation": [],
        "final": {
            "reward": reward,
            "termination_reason": "purchase",
            "reward_detail": {"r_success": int(reward > 0)},
        },
        "error": None,
    }


class ReplayArtifactsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp.name)
        self.runs_root = self.project_root / "runs"
        self.run_path = self.runs_root / "example"
        self.run_path.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def _write_run(self):
        manifest = {
            "plan": {
                "jobs": [
                    {"task_id": 1, "sample_id": 0},
                    {"task_id": 2, "sample_id": 0},
                    {"task_id": 3, "sample_id": 0},
                ]
            }
        }
        (self.run_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        records = [
            _trace(episode_id="task-1", task_id=1, reward=0.0),
            _trace(episode_id="task-2", task_id=2, reward=1.0, protocol=True),
            _trace(episode_id="task-1", task_id=1, reward=0.5),
        ]
        with (self.run_path / "traces.jsonl").open("w", encoding="utf-8") as file:
            for record in records:
                file.write(json.dumps(record) + "\n")
            file.write('{"incomplete":')

    def test_resolve_and_list_stay_inside_runs_root(self):
        self._write_run()
        resolved = resolve_run_path(
            "runs/example", project_root=self.project_root, runs_root=self.runs_root
        )
        self.assertEqual(resolved, self.run_path.resolve())
        self.assertEqual(list_runs(project_root=self.project_root, runs_root=self.runs_root)[0]["name"], "example")
        with self.assertRaises(ReplayError):
            resolve_run_path(
                str(self.project_root), project_root=self.project_root, runs_root=self.runs_root
            )

    def test_latest_duplicate_wins_and_summary_is_computed(self):
        self._write_run()
        payload = load_run(self.run_path, project_root=self.project_root)
        self.assertEqual(payload["trace_index"]["records"], 3)
        self.assertEqual(payload["trace_index"]["unique_episodes"], 2)
        self.assertEqual(payload["trace_index"]["duplicate_records"], 1)
        self.assertEqual(payload["trace_index"]["invalid_lines"], 1)
        self.assertEqual(payload["computed_summary"]["counts"]["pending"], 1)
        self.assertEqual(payload["computed_summary"]["primary"]["reward_mean"], 0.75)
        self.assertEqual(payload["computed_summary"]["behavior"]["protocol_errors"], 1)
        self.assertEqual(payload["computed_summary"]["behavior"]["invalid_actions"], 1)

        episode = load_episode(self.run_path, "task-1")
        self.assertEqual(episode["episode"]["final"]["reward"], 0.5)
        self.assertIsNone(episode["outcome_comparison"]["target"])

    def test_episode_builds_catalog_backed_target_purchase_comparison(self):
        trace = _trace(episode_id="task-5", task_id=5, reward=0.0)
        trace["final"].update(
            {
                "goal": {
                    "asin": "target-asin",
                    "name": "目标商品",
                    "attributes": ["防水", "卷翘"],
                    "goal_options": ["棕色"],
                    "price_upper": 60.0,
                },
                "purchase": {
                    "asin": "purchased-asin",
                    "name": "实际商品",
                    "attributes": ["卷翘", "纤长"],
                    "options": {"颜色分类": "黑色"},
                    "price": 29.0,
                },
            }
        )
        with (self.run_path / "traces.jsonl").open("w", encoding="utf-8") as file:
            file.write(json.dumps(trace, ensure_ascii=False) + "\n")
        catalog = {
            "target-asin": {
                "asin": "target-asin",
                "title": "目录中的目标商品",
                "shop_name": "目标店铺",
                "attribute": ["防水", "卷翘"],
                "options": {"颜色分类": ["黑色", "棕色"]},
                "option_to_price": {"黑色": 50.0, "棕色": 55.0},
                "option_to_image": {},
                "images": ["https://example.test/target.jpg"],
            },
            "purchased-asin": {
                "asin": "purchased-asin",
                "title": "目录中的实际商品",
                "attribute": ["卷翘", "纤长"],
                "options": {"颜色分类": ["黑色", "棕色"]},
                "option_to_price": {"黑色": 29.0, "棕色": 32.0},
                "option_to_image": {},
            },
        }

        payload = load_episode(
            self.run_path, "task-5", product_catalog=catalog
        )
        comparison = payload["outcome_comparison"]
        self.assertFalse(comparison["asin_match"])
        self.assertEqual(comparison["attributes"]["matched"], ["卷翘"])
        self.assertEqual(comparison["attributes"]["missing"], ["防水"])
        self.assertEqual(comparison["options"]["missing"], ["棕色"])
        self.assertTrue(comparison["target"]["catalog_available"])
        self.assertEqual(comparison["target"]["selected_price"], 55.0)
        self.assertTrue(
            comparison["target"]["option_axes"][0]["values"][1]["selected"]
        )
        self.assertTrue(
            comparison["purchased"]["option_axes"][0]["values"][0]["selected"]
        )

    def test_missing_trace_is_rejected(self):
        with self.assertRaises(ReplayError):
            load_run(self.run_path, project_root=self.project_root)


if __name__ == "__main__":
    unittest.main()
