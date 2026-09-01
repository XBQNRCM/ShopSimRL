from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from shopsimrl.config import ExperimentSpec, ModelSpec
from shopsimrl.model import OpenAICompatibleConfig
from shopsimrl.schemas import TRACE_SCHEMA_VERSION, EpisodeJob
from shopsimrl.trace2skill_evaluation import (
    build_mask_assignments,
    fit_main_effect_ols,
    run_gate_a,
)
from shopsimrl.trace2skill_evaluation_config import GateASpec


def _experiment(root: Path, task_ids: list[int]) -> ExperimentSpec:
    split_path = root / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "version": "test-v1",
                "splits": {"val": {"task_ids": task_ids, "persona": True}},
            }
        ),
        encoding="utf-8",
    )
    return ExperimentSpec(
        name="bare-val",
        output_dir=root / "unused",
        dotenv_path=None,
        seed=123,
        concurrency=4,
        resume=True,
        split_file=split_path,
        split="val",
        sample_size=None,
        repeats=1,
        environment_base_url="http://environment",
        environment_timeout=10.0,
        environment_persona=None,
        max_steps=5,
        system_prompt="test prompt",
        skillbank_path=None,
        max_skills=None,
        model=ModelSpec(
            model_id="model",
            config=OpenAICompatibleConfig(
                model="fake",
                base_url="http://model/v1",
                api_key_env=None,
            ),
        ),
    )


def _write_draft(root: Path) -> tuple[Path, Path]:
    chunks = [
        {
            "chunk_id": "chunk-positive",
            "order": 1,
            "title": "positive",
        },
        {
            "chunk_id": "chunk-negative",
            "order": 2,
            "title": "negative",
        },
    ]
    draft = {
        "schema_version": "shopsimrl-initial-skill-draft-v2",
        "stage": "cold_start_draft_pre_validation",
        "active_skill_budget": 1,
        "skill_title": "test",
        "chunks": chunks,
    }
    draft_path = root / "initial_skill_draft.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")
    bank = {
        "schema_version": "shopsimrl-skillbank-v1",
        "bank_version": "draft",
        "skills": [
            {
                "skill_id": chunk["chunk_id"],
                "content": f"content {chunk['chunk_id']}",
                "metadata": {"draft_only": True},
            }
            for chunk in chunks
        ],
    }
    bank_path = root / "initial_skillbank.json"
    bank_path.write_text(json.dumps(bank), encoding="utf-8")
    return draft_path, bank_path


class _EffectRuntime:
    def __init__(self, provider):
        self.provider = provider

    def run(self, job: EpisodeJob):
        skills = tuple(
            self.provider.select(
                {
                    "split": job.split,
                    "task_id": job.task_id,
                    "sample_id": job.sample_id,
                }
            )
        )
        ids = {skill.skill_id for skill in skills}
        reward = 0.4
        reward += 0.3 if "chunk-positive" in ids else 0.0
        reward -= 0.2 if "chunk-negative" in ids else 0.0
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "episode_id": job.episode_id,
            "status": "completed",
            "job": job.to_dict(),
            "provenance": {},
            "duration_ms": 1.0,
            "selected_skills": [skill.to_dict() for skill in skills],
            "steps": [],
            "final": {
                "done": True,
                "reward": reward,
                "reward_detail": {
                    "r_strict": reward,
                    "r_success": int(reward >= 0.5),
                },
                "termination_reason": "purchase",
            },
            "error": None,
        }


class Trace2SkillEvaluationTest(unittest.TestCase):
    def test_hash_mask_assignments_are_stable_and_task_specific(self):
        jobs = [
            EpisodeJob(task_id, 0, task_id, "val") for task_id in range(20)
        ]
        first = build_mask_assignments(
            jobs, ("a", "b", "c"), mask_seed=7, draft_sha256="draft"
        )
        second = build_mask_assignments(
            jobs, ("a", "b", "c"), mask_seed=7, draft_sha256="draft"
        )
        self.assertEqual(first["assignment_sha256"], second["assignment_sha256"])
        self.assertEqual(
            [record["mask"] for record in first["assignments"]],
            [record["mask"] for record in second["assignments"]],
        )
        self.assertGreater(
            len({tuple(record["mask"]) for record in first["assignments"]}), 1
        )

    def test_main_effect_ols_recovers_intercept_and_coefficients(self):
        masks = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, 1],
        ]
        values = [0.4 + 0.3 * row[0] - 0.2 * row[1] for row in masks]
        intercept, coefficients = fit_main_effect_ols(masks, values)
        self.assertAlmostEqual(intercept, 0.4)
        self.assertAlmostEqual(coefficients[0], 0.3)
        self.assertAlmostEqual(coefficients[1], -0.2)

    def test_gate_a_runs_masked_eval_and_freezes_only_positive_top_k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            draft_path, bank_path = _write_draft(root)
            spec = GateASpec(
                name="gate-a",
                output_dir=root / "gate-a",
                experiment=_experiment(root, list(range(40))),
                draft_path=draft_path,
                draft_skillbank_path=bank_path,
                mask_seed=19,
                mask_probability=0.5,
                active_skill_budget=1,
            )
            manifest = run_gate_a(
                spec,
                runtime_factory=lambda provider: _EffectRuntime(provider),
            )
            self.assertEqual(manifest["status"], "complete")
            selected = json.loads(
                (spec.output_dir / "selected_skillbank.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                [record["skill_id"] for record in selected["skills"]],
                ["chunk-positive"],
            )
            contributions = json.loads(
                (spec.output_dir / "contributions.json").read_text(
                    encoding="utf-8"
                )
            )
            by_id = {
                row["chunk_id"]: row for row in contributions["contributions"]
            }
            self.assertAlmostEqual(by_id["chunk-positive"]["coefficient"], 0.3)
            self.assertAlmostEqual(by_id["chunk-negative"]["coefficient"], -0.2)
            self.assertEqual(by_id["chunk-negative"]["status"], "non_positive")
            self.assertTrue((spec.output_dir / "mask_assignments.json").is_file())
            self.assertTrue((spec.output_dir / "traces.jsonl").is_file())



if __name__ == "__main__":
    unittest.main()
