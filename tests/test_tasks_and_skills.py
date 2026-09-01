from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shopsimrl.skills import JsonSkillBank
from shopsimrl.tasks import load_task_split


class TasksAndSkillsTest(unittest.TestCase):
    def test_explicit_split_preserves_stable_ids_with_gaps(self):
        payload = {
            "version": "test-v1",
            "splits": {"train": {"task_ids": [1, 4, 8], "persona": True}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            split = load_task_split(path, "train")
        self.assertEqual(split.task_ids, (1, 4, 8))
        self.assertTrue(split.persona)

    def test_skillbank_selects_global_and_task_scoped_records(self):
        payload = {
            "schema_version": "shopsimrl-skillbank-v1",
            "bank_version": "draft-3",
            "skills": [
                {"skill_id": "global", "content": "always"},
                {"skill_id": "task-4", "content": "only four", "task_ids": [4]},
                {"skill_id": "task-9", "content": "only nine", "task_ids": [9]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skills.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            bank = JsonSkillBank(path)
            selected = bank.select({"task_id": 4})
        self.assertEqual([skill.skill_id for skill in selected], ["global", "task-4"])
        self.assertEqual(bank.identity()["bank_version"], "draft-3")


if __name__ == "__main__":
    unittest.main()
