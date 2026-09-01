from __future__ import annotations

import json
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from shopsimrl.cli import command_run
from shopsimrl.schemas import ModelOutput, ToolCall


class FakeModel:
    def __init__(self, config):
        self.config = config

    def identity(self):
        return self.config.identity()

    def generate(self, messages, *, seed=None, tools=None):
        return ModelOutput(
            reasoning="buy the selected product",
            tool_calls=(
                ToolCall(
                    call_id="call-1",
                    name="click",
                    arguments={"value": "buy now"},
                    raw_arguments='{"value":"buy now"}',
                ),
            ),
        )

    def close(self):
        pass


class FakeEnvironment:
    def __init__(self, config):
        self.config = config

    def identity(self):
        return self.config.identity()

    def reset(self, task_id):
        return {
            "task_instruction": f"task {task_id}",
            "observation": "page",
            "observation_state": {
                "observation_version": "test-v1",
                "search_available": False,
                "actions": ["buy now"],
            },
            "user_persona": {"style": "simple"},
            "task_mode": "persona",
        }

    def step(self, action):
        return {
            "done": True,
            "observation": "done",
            "observation_state": {
                "observation_version": "test-v1",
                "search_available": False,
                "actions": [],
            },
            "reward": 1.0,
            "reward_detail": {"r_success": 1, "r_strict": 1.0},
            "termination_reason": "purchase",
        }

    def terminate(self, reason):
        raise AssertionError("episode should finish before the action limit")

    def close(self):
        pass


class CliTest(unittest.TestCase):
    def test_single_model_run_writes_one_flat_artifact_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "splits.json").write_text(
                json.dumps(
                    {
                        "version": "test-v1",
                        "splits": {
                            "val": {"task_ids": [7], "persona": True}
                        },
                    }
                ),
                encoding="utf-8",
            )
            config = root / "experiment.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    experiment:
                      name: single-run
                      output_dir: runs
                    tasks:
                      split_file: splits.json
                      split: val
                    model:
                      id: checkpoint-1
                      name: fake
                      base_url: http://model/v1
                      api_key_env: null
                    """
                ),
                encoding="utf-8",
            )
            with patch("shopsimrl.cli.OpenAICompatibleChatModel", FakeModel), patch(
                "shopsimrl.cli.ShopSimulatorHTTPEnvironment", FakeEnvironment
            ):
                self.assertEqual(command_run(str(config)), 0)

            run = root / "runs" / "single-run"
            self.assertTrue((run / "manifest.json").is_file())
            self.assertTrue((run / "traces.jsonl").is_file())
            self.assertTrue((run / "summary.json").is_file())
            self.assertFalse((run / "comparison.json").exists())
            self.assertFalse((run / "checkpoint-1").exists())


if __name__ == "__main__":
    unittest.main()
