from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from shopsimrl.model import OpenAICompatibleConfig
from shopsimrl.schemas import ModelOutput, ToolCall
from shopsimrl.trace2skill import (
    StructuredOutputError,
    _compile_skill,
    _consolidate_batch,
    _success_cards,
    _validate_compiler_arguments,
    firewall_findings,
    load_latest_traces,
    run_cold_start,
    validate_cold_start_traces,
)
from shopsimrl.trace2skill_config import Trace2SkillSpec


def _call(name: str, arguments: dict) -> ModelOutput:
    raw = json.dumps(arguments, ensure_ascii=False)
    return ModelOutput(
        tool_calls=(
            ToolCall(
                call_id=f"call-{name}",
                name=name,
                arguments=arguments,
                raw_arguments=raw,
            ),
        ),
        finish_reason="tool_calls",
    )


def _trace(episode_id: str, task_id: int, success: int) -> dict:
    goal = {
        "asin": f"{task_id:012d}",
        "name": f"Private target title {task_id}",
        "goal_options": ["target option"],
    }
    return {
        "schema_version": "shopsimrl-episode-v4",
        "episode_id": episode_id,
        "status": "completed",
        "job": {
            "task_id": task_id,
            "sample_id": 0,
            "seed": task_id,
            "split": "train",
        },
        "reset": {
            "task_instruction": "Buy an item under budget",
            "user_persona": {"preference": "simple"},
        },
        "steps": [],
        "final": {
            "done": True,
            "reward": float(success),
            "reward_detail": {"r_success": success, "r_strict": float(success)},
            "termination_reason": "purchase",
            "purchase": {"asin": "999999999999", "name": "Chosen item"},
            "goal": goal,
        },
    }


class FakeTrace2SkillModel:
    def __init__(self, config):
        self.config = config
        self.failure_turn = 0

    def identity(self):
        return self.config.identity()

    def generate(self, messages, *, seed=None, tools=None):
        names = {tool["function"]["name"] for tool in tools or ()}
        if "submit_success_cards" in names:
            return _call(
                "submit_success_cards",
                {
                    "cards": [
                        {
                            "mechanism": "Constraint-first query",
                            "applicable_when": "The request has hard constraints",
                            "observed_evidence": "The result set retained feasible items",
                            "decisive_behavior": "Searched by type and a hard constraint",
                            "generalizable_lesson": "Start with high-recall hard constraints",
                            "related_chunk_ids": [],
                            "confidence": 0.8,
                        }
                    ]
                },
            )
        if "submit_failure_card" in names:
            if self.failure_turn == 0 and "search" in names:
                self.failure_turn += 1
                return _call("search", {"query": "item budget"})
            if self.failure_turn == 1 and "click" in names:
                self.failure_turn += 1
                return _call("click", {"value": "buy now"})
            return _call(
                "submit_failure_card",
                {
                    "status": "PROPOSE_ADD",
                    "privileged_audit": {
                        "gold_asin": "000000000002",
                        "chosen_asin": "999999999999",
                        "failure_surface": "candidate verification",
                        "earliest_divergence": "The first candidate was accepted too early",
                        "oracle_comparison": "A visible hard constraint differed",
                        "replay_evidence": "A broader search exposed alternatives",
                        "minimal_repair": "Compare the unresolved hard constraint",
                        "repair_validation_result": "A feasible candidate remained",
                    },
                    "deployable_abstraction": {
                        "failure_mechanism": "Premature candidate commitment",
                        "applicable_when": "More than one hard constraint remains",
                        "observable_trigger": "The detail page leaves a constraint unchecked",
                        "corrective_procedure": "Return and compare another candidate",
                        "verification_step": "Check every hard constraint before purchase",
                        "target_chunk_id": "",
                        "proposed_content": "Verify all hard constraints before purchase",
                    },
                },
            )
        if "submit_clusters" in names:
            payload = json.loads(messages[-1]["content"])
            ids = [card["evidence_card_id"] for card in payload["cards"]]
            return _call(
                "submit_clusters",
                {
                    "clusters": [
                        {
                            "workflow_stage": "candidate inspection",
                            "mechanism": f"{payload['channel']} constraint handling",
                            "applicable_when": "Hard constraints are present",
                            "procedure": "Inspect constraints before committing",
                            "verification": "All hard constraints are checked",
                            "common_failure": "Premature commitment",
                            "evidence_card_ids": ids,
                        }
                    ],
                    "evidence_only_card_ids": [],
                },
            )
        if "submit_initial_skill" in names:
            payload = json.loads(messages[-1]["content"])
            if "active_skill_budget_after_validation" in payload:
                raise AssertionError("Compiler must not receive the Gate B budget")
            cluster_ids = [
                cluster["cluster_id"] for cluster in payload["cluster_summaries"]
            ]
            return _call(
                "submit_initial_skill",
                {
                    "skill_title": "Shopping Skill",
                    "chunks": [
                        {
                            "title": "Verify hard constraints",
                            "workflow_stage": "candidate inspection",
                            "applicable_when": "The request specifies hard constraints",
                            "procedure": [
                                "List unresolved hard constraints",
                                "Check them on the candidate page",
                            ],
                            "verification": "No hard constraint remains unresolved",
                            "common_failure": "Buying the first plausible result",
                            "evidence_cluster_ids": cluster_ids,
                        }
                    ],
                    "evidence_only_cluster_ids": [],
                    "conflicts_resolved": [],
                },
            )
        raise AssertionError(f"unexpected tools: {names}")

    def close(self):
        pass


class RepairingConsolidatorModel:
    calls = 0

    def __init__(self, config):
        self.config = config

    def generate(self, messages, *, seed=None, tools=None):
        type(self).calls += 1
        payload = json.loads(messages[1]["content"])
        evidence_id = payload["cards"][0]["evidence_card_id"]
        if type(self).calls == 1:
            return _call(
                "submit_clusters",
                {"clusters": [], "evidence_only_card_ids": []},
            )
        return _call(
            "submit_clusters",
            {
                "clusters": [
                    {
                        "workflow_stage": "candidate inspection",
                        "mechanism": "约束核验",
                        "applicable_when": "任务包含硬约束",
                        "procedure": "购买前逐项核验",
                        "verification": "所有硬约束均有页面证据",
                        "common_failure": "过早购买首个候选",
                        "evidence_card_ids": [evidence_id],
                    }
                ],
                "evidence_only_card_ids": [],
            },
        )

    def close(self):
        pass


class RepairingSuccessModel:
    calls = 0
    repair_prompts: list[str] = []

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _valid_card() -> dict:
        return {
            "mechanism": "约束优先搜索",
            "applicable_when": "任务包含明确硬约束",
            "observed_evidence": "结果保留了满足约束的候选",
            "decisive_behavior": "先按品类与硬约束搜索",
            "generalizable_lesson": "先处理可检索的硬约束",
            "related_chunk_ids": [],
            "confidence": 0.8,
        }

    def generate(self, messages, *, seed=None, tools=None):
        type(self).calls += 1
        if type(self).calls > 1:
            type(self).repair_prompts.append(messages[-1]["content"])
        if type(self).calls == 1:
            return _call("submit_success_cards", {"cards": {}})
        if type(self).calls == 2:
            card = self._valid_card()
            return _call("submit_success_cards", {"cards": [card, card, card]})
        return _call("submit_success_cards", {"cards": [self._valid_card()]})

    def close(self):
        pass


class RepairingCompilerModel:
    calls = 0
    repair_prompt: dict | None = None

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _chunk(cluster_ids: list[str], title: str) -> dict:
        return {
            "title": title,
            "workflow_stage": "候选核验",
            "applicable_when": "存在尚未核验的硬约束",
            "procedure": ["逐项核验硬约束"],
            "verification": "所有硬约束都有页面证据",
            "common_failure": "过早购买首个候选",
            "evidence_cluster_ids": cluster_ids,
        }

    def generate(self, messages, *, seed=None, tools=None):
        type(self).calls += 1
        if type(self).calls == 1:
            return _call(
                "submit_initial_skill",
                {
                    "skill_title": "购物技能",
                    "chunks": [
                        self._chunk(["cluster-1"], "核验约束一"),
                        self._chunk(["cluster-1"], "核验约束二"),
                    ],
                    "evidence_only_cluster_ids": [],
                    "conflicts_resolved": [],
                },
            )
        type(self).repair_prompt = json.loads(messages[-1]["content"])
        return _call(
            "submit_initial_skill",
            {
                "skill_title": "购物技能",
                "chunks": [self._chunk(["cluster-1"], "核验约束")],
                "evidence_only_cluster_ids": [],
                "conflicts_resolved": [],
            },
        )

    def close(self):
        pass


class FakeAuditEnvironment:
    def __init__(self):
        self.page = "search"

    def reset(self, task_id):
        self.page = "search"
        return {
            "task_instruction": "Buy an item under budget",
            "user_persona": {"preference": "simple"},
            "observation": "search page",
            "observation_state": {
                "search_available": True,
                "actions": [],
                "observation_version": "test-v1",
            },
        }

    def step(self, action):
        self.action = action
        if action == "search[item budget]":
            self.page = "product"
            return {
                "done": False,
                "reward": 0.0,
                "reward_detail": {},
                "purchase": {},
                "observation": "matching product with buy now",
                "observation_state": {
                    "search_available": False,
                    "actions": ["buy now"],
                    "observation_version": "test-v1",
                    "page_type": "product_detail",
                },
                "action_feedback": {"valid": True},
            }
        self.page = "terminal"
        return {
            "done": True,
            "reward": 1.0,
            "reward_detail": {"r_success": 1, "r_strict": 1.0},
            "purchase": {"asin": "000000000002", "price": 10.0},
            "termination_reason": "purchase",
            "observation": "purchase completed",
            "observation_state": {
                "search_available": False,
                "actions": [],
                "observation_version": "test-v1",
                "page_type": "terminal",
            },
            "action_feedback": {"valid": True},
        }

    def close(self):
        pass


class Trace2SkillTest(unittest.TestCase):
    def test_latest_trace_wins_and_train_contract_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            old = _trace("episode-1", 1, 0)
            latest = _trace("episode-1", 1, 1)
            path.write_text(
                json.dumps(old) + "\n" + json.dumps(latest) + "\n",
                encoding="utf-8",
            )
            traces = load_latest_traces(path)
        self.assertEqual(len(traces), 1)
        self.assertEqual(validate_cold_start_traces(traces, expected=1)["success"], 1)

    def test_firewall_rejects_identifiers_and_instance_titles(self):
        trace = _trace("episode-1", 1, 0)
        findings = firewall_findings(
            {"rule": "Choose ASIN 000000000001 Private target title 1"},
            trace=trace,
        )
        self.assertIn("mentions_asin", findings)
        self.assertIn("contains_long_identifier", findings)
        self.assertIn("copies_instance_product_title", findings)

    def test_cold_start_emits_pre_validation_chunk_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            traces_path = root / "traces.jsonl"
            traces_path.write_text(
                "\n".join(
                    json.dumps(trace)
                    for trace in (
                        _trace("episode-success", 1, 1),
                        _trace("episode-failure", 2, 0),
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            model_config = OpenAICompatibleConfig(
                model="fake", base_url="http://fake/v1", api_key_env=None
            )
            spec = Trace2SkillSpec(
                source_path=root / "config.yaml",
                traces_path=traces_path,
                output_dir=root / "trace2skill",
                expected_trajectories=2,
                seed=7,
                concurrency=2,
                resume=True,
                environment_base_url="http://fake",
                environment_timeout=1,
                environment_persona=True,
                max_failure_analysis_steps=3,
                max_cards_per_trajectory=2,
                consolidation_batch_size=10,
                validation_dimension_cap=16,
                active_skill_budget=10,
                analyst_model=model_config,
                compiler_model=model_config,
            )
            draft = run_cold_start(
                spec,
                model_factory=FakeTrace2SkillModel,
                environment_factory=FakeAuditEnvironment,
            )
            output = spec.output_dir
            self.assertEqual(len(draft["chunks"]), 1)
            self.assertEqual(draft["gate_status"]["gate_a"], "not_run")
            self.assertTrue((output / "evidence_cards.jsonl").is_file())
            self.assertTrue((output / "cluster_batches.jsonl").is_file())
            self.assertTrue((output / "initial_skill_draft.md").is_file())
            self.assertTrue((output / "initial_skillbank.json").is_file())
            skillbank = json.loads(
                (output / "initial_skillbank.json").read_text(encoding="utf-8")
            )
            self.assertTrue(skillbank["skills"][0]["metadata"]["draft_only"])
            analyses = [
                json.loads(line)
                for line in (output / "trajectory_analyses.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            failure = next(
                record for record in analyses
                if record["source_trajectory_id"] == "episode-failure"
            )["cards"][0]
            self.assertEqual(failure["successful_counterfactual_trials"], 1)
            self.assertEqual(failure["terminal_trials"][0]["reward"], 1.0)
            self.assertEqual(
                failure["audit_transcript"][-1]["environment_result"]
                ["next_trial"]["trial_index"],
                2,
            )

    def test_consolidation_repairs_semantically_unaccounted_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_config = OpenAICompatibleConfig(
                model="fake", base_url="http://fake/v1", api_key_env=None
            )
            spec = Trace2SkillSpec(
                source_path=root / "config.yaml",
                traces_path=root / "traces.jsonl",
                output_dir=root / "trace2skill",
                expected_trajectories=1,
                seed=7,
                concurrency=1,
                resume=True,
                environment_base_url="http://fake",
                environment_timeout=1,
                environment_persona=True,
                max_failure_analysis_steps=3,
                max_cards_per_trajectory=2,
                consolidation_batch_size=10,
                validation_dimension_cap=16,
                active_skill_budget=10,
                analyst_model=model_config,
                compiler_model=model_config,
            )
            card = {
                "card_id": "success-1-card-01",
                "channel": "success",
                "mechanism": "Constraint verification",
                "applicable_when": "Hard constraints are present",
                "observed_evidence": "The selected item met every constraint",
                "decisive_behavior": "Checked the detail page",
                "generalizable_lesson": "Verify before purchase",
                "confidence": 0.8,
            }
            RepairingConsolidatorModel.calls = 0
            record = _consolidate_batch(
                channel="success",
                batch_id="success-batch-001",
                cards=[card],
                spec=spec,
                model_factory=RepairingConsolidatorModel,
            )
        self.assertEqual(RepairingConsolidatorModel.calls, 2)
        self.assertEqual(len(record["clusters"]), 1)
        self.assertEqual(
            record["clusters"][0]["evidence_card_ids"], [card["card_id"]]
        )

    def test_success_analysis_repairs_semantically_invalid_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_config = OpenAICompatibleConfig(
                model="fake", base_url="http://fake/v1", api_key_env=None
            )
            spec = Trace2SkillSpec(
                source_path=root / "config.yaml",
                traces_path=root / "traces.jsonl",
                output_dir=root / "trace2skill",
                expected_trajectories=1,
                seed=7,
                concurrency=1,
                resume=True,
                environment_base_url="http://fake",
                environment_timeout=1,
                environment_persona=True,
                max_failure_analysis_steps=3,
                max_cards_per_trajectory=2,
                consolidation_batch_size=10,
                validation_dimension_cap=16,
                active_skill_budget=10,
                analyst_model=model_config,
                compiler_model=model_config,
            )
            RepairingSuccessModel.calls = 0
            RepairingSuccessModel.repair_prompts = []
            cards = _success_cards(
                _trace("episode-success", 1, 1),
                model_factory=RepairingSuccessModel,
                spec=spec,
            )

        self.assertEqual(RepairingSuccessModel.calls, 3)
        self.assertEqual(len(cards), 1)
        self.assertIn(
            "success output cards must be a list",
            RepairingSuccessModel.repair_prompts[0],
        )
        self.assertIn(
            "success analyst exceeded per-trajectory card cap",
            RepairingSuccessModel.repair_prompts[1],
        )
        first_repair = json.loads(RepairingSuccessModel.repair_prompts[0])
        self.assertEqual(
            first_repair["structured_output_repair"]
            ["previous_invalid_arguments"],
            {"cards": {}},
        )

    def test_compiler_repair_sees_invalid_draft_and_persists_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_config = OpenAICompatibleConfig(
                model="fake", base_url="http://fake/v1", api_key_env=None
            )
            spec = Trace2SkillSpec(
                source_path=root / "config.yaml",
                traces_path=root / "traces.jsonl",
                output_dir=root / "trace2skill",
                expected_trajectories=1,
                seed=7,
                concurrency=1,
                resume=True,
                environment_base_url="http://fake",
                environment_timeout=1,
                environment_persona=True,
                max_failure_analysis_steps=3,
                max_cards_per_trajectory=2,
                consolidation_batch_size=10,
                validation_dimension_cap=16,
                active_skill_budget=10,
                analyst_model=model_config,
                compiler_model=model_config,
            )
            cluster = {
                "cluster_id": "cluster-1",
                "channel": "success",
                "workflow_stage": "候选核验",
                "mechanism": "硬约束核验",
                "applicable_when": "任务含硬约束",
                "procedure": "逐项核验",
                "verification": "约束均有证据",
                "common_failure": "过早购买",
                "evidence_card_ids": ["card-1"],
            }
            RepairingCompilerModel.calls = 0
            RepairingCompilerModel.repair_prompt = None
            draft = _compile_skill(
                [cluster], spec=spec, model_factory=RepairingCompilerModel
            )
            invalid_records = [
                json.loads(line)
                for line in (spec.output_dir / "compiler_invalid_attempts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(RepairingCompilerModel.calls, 2)
        self.assertEqual(len(draft["chunks"]), 1)
        repair = RepairingCompilerModel.repair_prompt["structured_output_repair"]
        self.assertIn("draft chunks 1 and 2", repair["contract_error"])
        self.assertEqual(
            repair["previous_invalid_arguments"]["skill_title"], "购物技能"
        )
        self.assertEqual(len(invalid_records), 1)
        self.assertEqual(invalid_records[0]["attempt_number"], 1)
        self.assertEqual(
            invalid_records[0]["arguments"]["chunks"][1]
            ["evidence_cluster_ids"],
            ["cluster-1"],
        )

    def test_compiler_duplicate_error_reports_both_chunk_positions(self):
        cluster = {"cluster_id": "cluster-1"}
        chunk = RepairingCompilerModel._chunk(["cluster-1"], "核验约束")
        with self.assertRaisesRegex(
            StructuredOutputError,
            r"cluster-1 \(draft chunks 1 and 2\)",
        ):
            _validate_compiler_arguments(
                {
                    "skill_title": "购物技能",
                    "chunks": [chunk, dict(chunk)],
                    "evidence_only_cluster_ids": [],
                    "conflicts_resolved": [],
                },
                clusters=[cluster],
                validation_dimension_cap=16,
            )


if __name__ == "__main__":
    unittest.main()
