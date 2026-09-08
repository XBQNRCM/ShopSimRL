from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shopsimrl.evaluation import EvaluationPlan, Evaluator, build_jobs, summarize_traces
from shopsimrl.schemas import TRACE_SCHEMA_VERSION, EpisodeJob
from shopsimrl.store import RunStore


def completed_trace(job: EpisodeJob, reward=1.0, valid=True):
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "episode_id": job.episode_id,
        "status": "completed",
        "job": job.to_dict(),
        "provenance": {},
        "duration_ms": 20.0,
        "steps": [
            {
                "model": {
                    "usage": {
                        "total_tokens": 7,
                        "completion_tokens_details": {"reasoning_tokens": 3},
                    }
                },
                "environment": {"action_feedback": {"valid": valid}},
            }
        ],
        "final": {
            "done": True,
            "reward": reward,
            "reward_detail": {
                "r_success": int(reward == 1.0),
                "r_strict": reward,
                "r_loose": reward,
            },
            "termination_reason": "purchase",
        },
        "error": None,
    }


class FakeRuntime:
    def __init__(self, calls):
        self.calls = calls

    def run(self, job):
        self.calls.append(job.episode_id)
        return completed_trace(job)


class EvaluationTest(unittest.TestCase):
    def test_sampling_is_deterministic_and_supports_repeats(self):
        plan = EvaluationPlan("train", (2, 5, 9, 11), seed=42, sample_size=2, repeats=3)
        first = build_jobs(plan)
        second = build_jobs(plan)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual({job.sample_id for job in first}, {0, 1, 2})
        self.assertEqual(len({job.seed for job in first}), 6)

    def test_summary_reports_coverage_behavior_and_reward(self):
        jobs = [EpisodeJob(1, 0, 1, "eval"), EpisodeJob(2, 0, 2, "eval")]
        traces = [completed_trace(jobs[0], 1.0), completed_trace(jobs[1], 0.0, False)]
        summary = summarize_traces(traces, requested=2)
        self.assertEqual(summary["counts"]["coverage"], 1.0)
        self.assertEqual(summary["primary"]["reward_mean"], 0.5)
        self.assertEqual(summary["primary"]["success_rate"], 0.5)
        self.assertEqual(summary["behavior"]["model_steps"], 2)
        self.assertEqual(summary["behavior"]["protocol_errors"], 0)
        self.assertEqual(summary["behavior"]["policy_failures"], 0)
        self.assertEqual(summary["behavior"]["invalid_action_rate"], 0.5)
        self.assertEqual(summary["cost"]["tokens"]["total_tokens"], 14)
        self.assertEqual(summary["cost"]["tokens"]["reasoning_tokens"], 6)

    def test_evaluator_resumes_only_complete_traces(self):
        job = EpisodeJob(7, 0, 9, "eval")
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "run")
            store.initialize({"jobs": [job.to_dict()]})
            evaluator = Evaluator(
                runtime_factory=lambda: FakeRuntime(calls),
                store=store,
                max_workers=2,
            )
            evaluator.run([job], resume=True)
            evaluator.run([job], resume=True)
            self.assertEqual(calls, [job.episode_id])
            self.assertTrue(store.is_complete(job))
            self.assertEqual(len(store.traces_path.read_text(encoding="utf-8").splitlines()), 1)

    def test_evaluator_retries_failed_episodes_then_keeps_success(self):
        job = EpisodeJob(7, 0, 9, "eval")
        calls = []

        class Flaky:
            def __init__(self):
                self.seen = 0

            def run(self, current):
                calls.append(current.episode_id)
                self.seen += 1
                if self.seen < 3:
                    return {
                        "schema_version": TRACE_SCHEMA_VERSION,
                        "episode_id": current.episode_id,
                        "status": "failed",
                        "job": current.to_dict(),
                        "provenance": {},
                        "final": None,
                        "error": {"type": "timeout"},
                    }
                return completed_trace(current)

        worker = Flaky()
        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "run")
            store.initialize({"jobs": [job.to_dict()]})
            summary = Evaluator(
                runtime_factory=lambda: worker,
                store=store,
                max_workers=1,
            ).run([job], resume=True)
            self.assertEqual(calls, [job.episode_id] * 3)
            self.assertTrue(store.is_complete(job))
            self.assertEqual(summary["counts"]["failed"], 0)
            self.assertEqual(summary["counts"]["completed"], 1)
            self.assertEqual(
                len(store.traces_path.read_text(encoding="utf-8").splitlines()), 3
            )

    def test_evaluator_stops_after_two_retries(self):
        job = EpisodeJob(8, 0, 10, "eval")
        calls = []

        class AlwaysFail:
            def run(self, current):
                calls.append(current.episode_id)
                return {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "episode_id": current.episode_id,
                    "status": "failed",
                    "job": current.to_dict(),
                    "provenance": {},
                    "final": None,
                    "error": {"type": "timeout"},
                }

        with tempfile.TemporaryDirectory() as directory:
            store = RunStore(Path(directory) / "run")
            store.initialize({"jobs": [job.to_dict()]})
            summary = Evaluator(
                runtime_factory=AlwaysFail,
                store=store,
                max_workers=1,
            ).run([job], resume=True)
            self.assertEqual(calls, [job.episode_id] * 3)
            self.assertFalse(store.is_complete(job))
            self.assertEqual(summary["counts"]["failed"], 1)
            self.assertEqual(summary["counts"]["completed"], 0)
            self.assertEqual(summary["primary"]["reward_mean"], None)

    def test_summary_separates_protocol_steps_from_environment_actions(self):
        job = EpisodeJob(3, 0, 3, "eval")
        trace = completed_trace(job)
        trace["steps"].insert(
            0,
            {
                "model": {"usage": {"total_tokens": 5}},
                "protocol_error": {"code": "tool_call_count"},
                "action": None,
                "environment": None,
            },
        )

        summary = summarize_traces([trace], requested=1)

        self.assertEqual(summary["behavior"]["model_steps"], 2)
        self.assertEqual(summary["behavior"]["protocol_errors"], 1)
        self.assertEqual(summary["behavior"]["protocol_error_rate"], 0.5)
        self.assertEqual(summary["behavior"]["total_actions"], 1)

    def test_summary_counts_generation_length_as_scored_policy_failure(self):
        job = EpisodeJob(4, 0, 4, "train")
        trace = completed_trace(job, reward=0.0)
        trace["steps"] = [
            {
                "model": {"usage": {"completion_tokens": 2048}},
                "policy_failure": {"code": "generation_length"},
                "action": None,
                "environment": None,
            }
        ]
        trace["final"]["termination_reason"] = "generation_length"

        summary = summarize_traces([trace], requested=1)

        self.assertEqual(summary["counts"]["completed"], 1)
        self.assertEqual(summary["counts"]["scored"], 1)
        self.assertEqual(summary["primary"]["reward_mean"], 0.0)
        self.assertEqual(summary["behavior"]["policy_failures"], 1)
        self.assertEqual(summary["behavior"]["policy_failure_rate"], 1.0)
        self.assertEqual(summary["behavior"]["total_actions"], 0)
        self.assertEqual(
            summary["behavior"]["termination_reasons"],
            {"generation_length": 1},
        )


if __name__ == "__main__":
    unittest.main()
