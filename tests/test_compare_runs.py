from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.compare_shopsimrl_runs import (
    ComparisonError,
    compare_runs,
    validate_plan_alignment,
)
from shopsimrl.schemas import TRACE_SCHEMA_VERSION, EpisodeJob
from shopsimrl.store import RunStore


def _plan(*, skills: dict) -> dict:
    jobs = [EpisodeJob(task_id, 0, task_id, "test") for task_id in range(10)]
    return {
        "pipeline_version": "test-v1",
        "experiment": "test",
        "model_id": "model",
        "model": {"sampling": {"temperature": 0.6}},
        "task_split": {"name": "test", "task_count": len(jobs)},
        "jobs": [job.to_dict() for job in jobs],
        "environment": {"persona": True},
        "runtime": {
            "max_steps": 30,
            "concurrency": 1,
            "action_protocol": "tools",
        },
        "prompt": {"version": "prompt"},
        "skills": skills,
    }


def _trace(job: EpisodeJob, reward: float, *, equipped: bool) -> dict:
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "episode_id": job.episode_id,
        "status": "completed",
        "job": job.to_dict(),
        "provenance": {
            "model": {"model": "fake"},
            "environment": {"version": "fake"},
            "prompt": {"version": "fake"},
            "runtime": {"version": "fake"},
            "skills": {
                "provider": "json_skillbank" if equipped else "none"
            },
        },
        "duration_ms": 1.0,
        "selected_skills": (
            [{"skill_id": "selected", "content": "strategy"}]
            if equipped
            else []
        ),
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


class CompareRunsTest(unittest.TestCase):
    def test_report_compares_existing_runs_with_paired_bootstrap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bare = RunStore(root / "bare")
            equipped = RunStore(root / "equipped")
            bare.initialize(_plan(skills={"provider": "none"}))
            equipped.initialize(
                _plan(
                    skills={
                        "provider": "json_skillbank",
                        "bank_sha256": "selected",
                    }
                )
            )
            for task_id in range(10):
                job = EpisodeJob(task_id, 0, task_id, "test")
                bare.save_trace(_trace(job, 0.25, equipped=False))
                equipped.save_trace(_trace(job, 0.75, equipped=True))

            report = compare_runs(
                bare.root,
                equipped.root,
                bootstrap_samples=100,
                bootstrap_seed=5,
            )

            reward = report["paired"]["metrics"]["reward"]
            self.assertEqual(reward["mean_difference"], 0.5)
            self.assertEqual(reward["confidence_interval"], [0.5, 0.5])
            self.assertEqual(
                report["alignment"]["traces"]["equipped_chunk_ids"],
                ["selected"],
            )

    def test_plan_alignment_rejects_non_skill_model_difference(self):
        bare = {"plan_fingerprint": "bare", "plan": _plan(skills={"provider": "none"})}
        equipped = {
            "plan_fingerprint": "equipped",
            "plan": _plan(skills={"provider": "json_skillbank"}),
        }
        validate_plan_alignment(bare, equipped)
        equipped["plan"] = json.loads(json.dumps(equipped["plan"]))
        equipped["plan"]["model"]["sampling"]["temperature"] = 0.0
        with self.assertRaises(ComparisonError):
            validate_plan_alignment(bare, equipped)


if __name__ == "__main__":
    unittest.main()
