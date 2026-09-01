from __future__ import annotations

import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from shopsimrl.config import load_experiment_config
from shopsimrl.prompts import DEFAULT_SYSTEM_PROMPT
from shopsimrl.tasks import load_task_split


ROOT = Path(__file__).resolve().parents[1]


class ConfigTest(unittest.TestCase):
    def test_qwen_configs_are_single_model_runs_with_aligned_prompt(self):
        expected = {
            "train": (3726, 0.6, "auto"),
            "val": (400, 0.6, "auto"),
            "test": (400, 0.6, "auto"),
        }
        prompt_path = ROOT / "configs" / "prompts" / "persona_single_turn.txt"
        self.assertEqual(
            prompt_path.read_text(encoding="utf-8").strip(),
            DEFAULT_SYSTEM_PROMPT.strip(),
        )

        for split, (task_count, temperature, tool_choice) in expected.items():
            spec = load_experiment_config(
                ROOT / "configs" / f"qwen35_4b_{split}.yaml"
            )
            self.assertEqual(spec.split, split)
            self.assertEqual(spec.model.model_id, "qwen35-4b")
            self.assertEqual(spec.model.config.model, "Qwen/Qwen3.5-4B")
            self.assertEqual(spec.model.config.api_key_env, "SILICONFLOW_API_KEY")
            self.assertEqual(spec.model.config.temperature, temperature)
            self.assertEqual(spec.model.config.tool_choice, tool_choice)
            self.assertEqual(spec.model.config.max_tokens, 2048)
            self.assertTrue(spec.model.config.extra_body["enable_thinking"])
            self.assertNotIn("thinking_budget", spec.model.config.extra_body)
            self.assertEqual(spec.system_prompt.strip(), DEFAULT_SYSTEM_PROMPT.strip())
            self.assertEqual(
                len(load_task_split(spec.split_file, split).task_ids), task_count
            )

    def test_config_loads_dotenv_without_overwriting_process_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "SHOPSIMRL_TEST_KEY=from-file\n", encoding="utf-8"
            )
            config = root / "experiment.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    experiment:
                      dotenv: .env
                    tasks:
                      split_file: splits.json
                      split: val
                    model:
                      name: test-model
                      base_url: http://model/v1
                      api_key_env: SHOPSIMRL_TEST_KEY
                    """
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"SHOPSIMRL_TEST_KEY": "from-process"}):
                spec = load_experiment_config(config)
                self.assertEqual(os.environ["SHOPSIMRL_TEST_KEY"], "from-process")
        self.assertEqual(spec.model.config.api_key_env, "SHOPSIMRL_TEST_KEY")

    def test_equipped_test_config_uses_full_selected_skill_and_bare_sampling(self):
        spec = load_experiment_config(
            ROOT / "configs" / "qwen35_4b_test_trace2skill_equipped.yaml"
        )
        self.assertEqual(spec.split, "test")
        self.assertEqual(spec.repeats, 1)
        self.assertIsNone(spec.sample_size)
        self.assertEqual(spec.model.config.tool_choice, "auto")
        self.assertIsNotNone(spec.skillbank_path)
        self.assertEqual(spec.skillbank_path.name, "selected_skillbank.json")
        self.assertIsNone(spec.max_skills)

    def test_val_config_does_not_cap_gate_a_draft_chunks(self):
        spec = load_experiment_config(ROOT / "configs" / "qwen35_4b_val.yaml")
        self.assertIsNone(spec.max_skills)

    def test_multi_model_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "experiment.yaml"
            config.write_text(
                textwrap.dedent(
                    """
                    tasks:
                      split_file: splits.json
                      split: val
                    models:
                      - name: first
                        base_url: http://model/v1
                    """
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "one model mapping"):
                load_experiment_config(config)


if __name__ == "__main__":
    unittest.main()
