import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from shopsimrl.curriculum import SkillCurriculum, build_curriculum_state
from shopsimrl.slime_metrics import compute_shopsim_rollout_metrics, enrich_rollout_metrics
from shopsimrl.wandb_reporting import (
    analysis_metrics,
    gate_metrics,
    report_round_to_wandb,
)


def _curriculum(tmp_path) -> SkillCurriculum:
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "skills": [
                    {
                        "skill_id": "a",
                        "content": "rule a",
                        "metadata": {"estimated_effect": 0.2},
                    },
                    {
                        "skill_id": "b",
                        "content": "rule b",
                        "metadata": {"estimated_effect": 0.1},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return SkillCurriculum.from_dict(
        build_curriculum_state(
            bank,
            round_id="round-000",
            seed=3,
            skill_free_probability=0.2,
            rho_min=0.2,
            rho_max=0.8,
        )
    )


def _sample(group, rollout, reward, assignment, **metadata):
    return SimpleNamespace(
        group_index=group,
        rollout_id=rollout,
        index=rollout,
        reward=reward,
        metadata={
            "shopsim_reward": reward,
            "shopsim_scored": True,
            "shopsim_assignment": assignment,
            **metadata,
        },
    )


def test_rollout_metrics_deduplicate_fanout_and_report_triage(tmp_path):
    curriculum = _curriculum(tmp_path)
    assisted = {
        "state_id": curriculum.state_id,
        "group_key": "1",
        "mode": "assisted",
        "skill_ids": ["a"],
    }
    free = {
        "state_id": curriculum.state_id,
        "group_key": "2",
        "mode": "skill_free",
        "skill_ids": [],
    }
    first = _sample(1, 10, {"reward": 1.0, "r_strict": 0.5, "r_success": 1.0}, assisted)
    # A second token-contiguous Sample from the same trajectory must not change
    # means or counts.
    first_fanout = _sample(1, 10, {"reward": 1.0, "r_strict": 0.5, "r_success": 1.0}, assisted)
    second = _sample(
        2,
        20,
        {"reward": 0.0, "r_strict": 0.0, "r_success": 0.0},
        free,
        shopsim_group_triage={
            "all_wrong": True,
            "classification": "full_skill_failure_pending_analysis",
            "full_skill_retry": True,
            "full_skill_retry_scored": True,
            "full_skill_retry_success": False,
        },
    )
    metrics = compute_shopsim_rollout_metrics(
        [first, first_fanout, second], curriculum
    )
    assert metrics["rollout/shopsim/trajectory_count"] == 2
    assert metrics["rollout/shopsim/reward_mean"] == 0.5
    assert metrics["rollout/shopsim/success_rate"] == 0.5
    assert metrics["rollout/shopsim/skill_free_group_rate"] == 0.5
    assert metrics["rollout/shopsim/skills_per_assisted_group_mean"] == 1.0
    assert metrics["rollout/shopsim/chunk/a/inclusion_rate"] == 1.0
    assert metrics["rollout/shopsim/chunk/b/inclusion_rate"] == 0.0
    assert metrics["rollout/shopsim/q"] == 0.2
    assert metrics["rollout/shopsim/rho_max"] == 0.8
    assert metrics["rollout/shopsim/pending_analysis_rate"] == 1.0
    assert metrics["rollout/shopsim/full_skill_retry_success_rate"] == 0.0


def test_rollout_hook_adds_dynamic_filter_drop_rate(tmp_path):
    curriculum = _curriculum(tmp_path)
    curriculum_path = tmp_path / "curriculum.json"
    curriculum_path.write_text(
        json.dumps(
            build_curriculum_state(
                tmp_path / "bank.json",
                round_id="round-000",
                seed=3,
                skill_free_probability=0.2,
                rho_min=0.2,
                rho_max=0.8,
            )
        ),
        encoding="utf-8",
    )
    assignment = {
        "state_id": curriculum.state_id,
        "group_key": "1",
        "mode": "assisted",
        "skill_ids": ["a"],
    }
    samples = [
        _sample(1, 10, {"reward": 1.0, "r_success": 1.0}, assignment),
        _sample(1, 11, {"reward": 0.0, "r_success": 0.0}, assignment),
    ]
    extra = {
        "rollout/dynamic_filter/drop_shopsim_unscored_technical_group": 1
    }
    assert (
        enrich_rollout_metrics(
            0,
            SimpleNamespace(shopsim_curriculum_path=str(curriculum_path)),
            samples,
            extra,
            1.0,
        )
        is False
    )
    assert extra["rollout/shopsim/technical_group_drop_count"] == 1.0
    assert extra["rollout/shopsim/technical_group_drop_rate"] == 0.5


def test_round_metric_extractors():
    assert analysis_metrics(
        {
            "triage_groups": 4,
            "analyzed_cards": 3,
            "eligible_cards": 2,
            "candidates": 1,
            "proposal_history_records": 7,
            "evidence_only_card_ids": ["x", "y"],
        }
    ) == {
        "analysis/triage_groups": 4.0,
        "analysis/analyzed_cards": 3.0,
        "analysis/eligible_cards": 2.0,
        "analysis/candidates": 1.0,
        "analysis/proposal_history_records": 7.0,
        "analysis/evidence_only_cards": 2.0,
    }
    metrics = gate_metrics(
        {
            "status": "complete",
            "summary": {
                "counts": {"coverage": 1.0, "failed": 0},
                "primary": {"reward_mean": 0.6},
                "reward_components": {"r_success": 0.7},
            },
            "input": {"bare_baseline": {"mean_outcomes": {"reward": 0.3}}},
        },
        {
            "selected_intervention_ids": ["a"],
            "contributions": [
                {"coefficient": 0.4},
                {"coefficient": -0.1},
            ],
            "estimates": {
                "reward": {
                    "bare_mean": 0.3,
                    "masked_mean": 0.6,
                    "delta_mean": 0.3,
                }
            },
        },
    )
    assert metrics["gate/complete"] == 1.0
    assert metrics["gate/selected_chunk_count"] == 1.0
    assert metrics["gate/positive_chunk_count"] == 1.0
    assert metrics["gate/reward/delta_mean"] == 0.3


def test_round_report_uses_one_run_with_table_and_artifact(tmp_path, monkeypatch):
    output = tmp_path / "gate"
    output.mkdir()
    manifest = {
        "status": "complete",
        "summary": {"counts": {"coverage": 1.0}, "primary": {}},
        "input": {"bare_baseline": {"mean_outcomes": {"reward": 0.2}}},
    }
    (output / "online_gate_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    contributions = {
        "selected_intervention_ids": ["a"],
        "estimates": {"reward": {"delta_mean": 0.1}},
        "contributions": [
            {
                "intervention_id": "a",
                "logical_chunk_id": "a",
                "operation": "KEEP",
                "coefficient": 0.1,
                "rank": 1,
                "status": "selected",
            }
        ],
    }
    (output / "contributions.json").write_text(
        json.dumps(contributions), encoding="utf-8"
    )

    captured = SimpleNamespace(init=None, logs=[], artifacts=[])

    class Run:
        def log(self, payload):
            captured.logs.append(payload)

        def log_artifact(self, artifact):
            captured.artifacts.append(artifact)

        def finish(self):
            captured.finished = True

    class Artifact:
        def __init__(self, name, type):
            self.name, self.type, self.files = name, type, []

        def add_file(self, path, name):
            self.files.append((path, name))

    module = ModuleType("wandb")
    module.init = lambda **kwargs: (setattr(captured, "init", kwargs) or Run())
    module.Table = lambda **kwargs: kwargs
    module.Artifact = Artifact
    monkeypatch.setitem(sys.modules, "wandb", module)

    result = report_round_to_wandb(
        stage="gate",
        name="round-000",
        output_dir=output,
        payload=manifest,
        project="shopsimrl",
        group="experiment-a",
        mode="offline",
    )
    assert captured.init["project"] == "shopsimrl"
    assert captured.init["group"] == "experiment-a"
    assert captured.init["mode"] == "offline"
    assert "gate/chunk_contributions" in captured.logs[0]
    assert {name for _, name in captured.artifacts[0].files} == {
        "online_gate_manifest.json",
        "contributions.json",
    }
    assert result["metrics"]["gate/selected_chunk_count"] == 1.0
    assert captured.finished
