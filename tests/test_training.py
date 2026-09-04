import json
import math
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from shopsimrl.curriculum import (
    SkillCurriculum,
    build_curriculum_state,
    write_slime_task_data,
)
from shopsimrl.online_validation import (
    _proposal_ledger_rows,
    build_candidate_pool,
    build_online_assignments,
    estimate_online_contributions,
    selected_skillbank,
)
from shopsimrl.schemas import TRACE_SCHEMA_VERSION, EpisodeJob
from shopsimrl import slime_runtime
from shopsimrl.slime_runtime import (
    keep_fully_scored_group,
    normalize_grpo_by_prompt_and_rollout,
)


def _write_bank(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "shopsimrl-skillbank-v1",
                "bank_version": "test",
                "skills": [
                    {
                        "skill_id": "a",
                        "content": "old a",
                        "metadata": {"estimated_effect": 0.2},
                    },
                    {
                        "skill_id": "b",
                        "content": "rule b",
                        "metadata": {"estimated_effect": 0.1},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_curriculum_is_group_consistent_and_hierarchical(tmp_path):
    bank_path = tmp_path / "bank.json"
    _write_bank(bank_path)
    payload = build_curriculum_state(
        bank_path,
        round_id="round-1",
        seed=7,
        skill_free_probability=0.0,
        rho_min=0.2,
        rho_max=0.8,
    )
    curriculum = SkillCurriculum.from_dict(payload)
    assert curriculum.assign(11) == curriculum.assign(11)
    assert curriculum.assign(11).mode in {"assisted", "assisted_empty"}
    probabilities = {
        chunk.skill.skill_id: chunk.inclusion_probability
        for chunk in curriculum.chunks
    }
    assert probabilities == {"a": 0.8, "b": pytest.approx(0.5)}

    forced_free = dict(payload)
    core = {
        key: forced_free[key]
        for key in (
            "round_id",
            "seed",
            "skill_free_probability",
            "chunks",
            "source",
        )
    }
    core["skill_free_probability"] = 1.0
    from shopsimrl.schemas import fingerprint

    forced_free.update(core)
    forced_free["state_id"] = fingerprint(core)
    free_curriculum = SkillCurriculum.from_dict(forced_free)
    assert free_curriculum.assign(11).mode == "skill_free"
    assert free_curriculum.assign(11).skills == ()


def test_prepare_slime_data_uses_frozen_task_ids(tmp_path):
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "splits": {"train": {"task_ids": [7, 2], "persona": True}},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "train.jsonl"
    summary = write_slime_task_data(split_path, "train", output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert summary["records"] == 2
    assert [row["metadata"]["task_id"] for row in rows] == [7, 2]
    assert [row["prompt"] for row in rows] == ["7", "2"]


def _completed_trace(assignment, reward, *, bare=False):
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "episode_id": assignment["episode_id"],
        "status": "completed",
        "job": {key: assignment[key] for key in ("split", "task_id", "sample_id", "seed")},
        "provenance": {
            "model": {"model": "fake", "checkpoint_id": "checkpoint-1"},
            "environment": {"version": "test"},
            "prompt": {"version": "test"},
            "runtime": {"max_steps": 5},
            "skills": {"provider": "none" if bare else "assigned"},
        },
        "selected_skills": [
            {"skill_id": skill_id}
            for skill_id in ([] if bare else assignment["included_chunk_ids"])
        ],
        "final": {
            "done": True,
            "reward": reward,
            "reward_detail": {
                "r_strict": reward,
                "r_success": float(reward > 0.5),
            },
        },
    }


def test_online_gate_keeps_rewrite_versions_mutually_exclusive(tmp_path):
    bank_path = tmp_path / "bank.json"
    _write_bank(bank_path)
    pool = build_candidate_pool(
        bank_path,
        [
            {
                "candidate_id": "a-new",
                "operation": "REWRITE",
                "target_chunk_id": "a",
                "content": "new a",
            },
            {
                "candidate_id": "c",
                "operation": "ADD",
                "target_chunk_id": None,
                "content": "rule c",
            },
        ],
        round_id="round-2",
        proposal_checkpoint="checkpoint-1",
    )
    jobs = tuple(
        EpisodeJob(task_id=index, sample_id=0, seed=index, split="val")
        for index in range(96)
    )
    assignments = build_online_assignments(jobs, pool, mask_seed=13)
    for assignment in assignments["assignments"]:
        included = set(assignment["included_chunk_ids"])
        assert not {"a", "a-new"} <= included

    traces = []
    for assignment in assignments["assignments"]:
        included = set(assignment["included_chunk_ids"])
        reward = (
            (0.1 + (assignment["task_id"] % 5) * 0.01)
            + 0.2 * ("a" in included)
            + 0.6 * ("a-new" in included)
            + 0.3 * ("b" in included)
            - 0.2 * ("c" in included)
        )
        traces.append(_completed_trace(assignment, reward))
    contributions = estimate_online_contributions(
        pool=pool,
        assignments=assignments,
        traces=traces,
        bare_traces=[
            _completed_trace(row, 0.1 + (row["task_id"] % 5) * 0.01, bare=True)
            for row in assignments["assignments"]
        ],
        active_skill_budget=2,
    )
    results = {
        row["intervention_id"]: row for row in contributions["contributions"]
    }
    assert results["a"]["status"] == "rewrite_loser"
    assert results["a-new"]["status"] == "selected"
    assert results["b"]["status"] == "selected"
    assert results["c"]["status"] == "non_positive"
    assert contributions["estimates"]["reward"]["intercept"] == 0.0
    for name, expected in {"a": 0.2, "a-new": 0.6, "b": 0.3, "c": -0.2}.items():
        assert results[name]["coefficient"] == pytest.approx(expected)
    bank = selected_skillbank(pool, contributions)
    assert [record["skill_id"] for record in bank["skills"]] == ["a", "b"]
    assert bank["skills"][0]["content"] == "new a"
    ledger = {
        row["candidate_id"]: row
        for row in _proposal_ledger_rows(pool, contributions)
    }
    assert ledger["a-new"]["validation_result"] == "selected"
    assert ledger["c"]["validation_result"] == "non_positive"
    assert "a" not in ledger
    cumulative = _proposal_ledger_rows(
        pool,
        contributions,
        [
            {
                "candidate_id": "past",
                "operation": "ADD",
                "content": "past rule",
                "validation_result": "non_positive",
            }
        ],
    )
    assert [row["candidate_id"] for row in cumulative] == [
        "past",
        "a-new",
        "c",
    ]
    assert cumulative[0]["validation_result"] == "non_positive"
    lifecycle = _proposal_ledger_rows(
        pool,
        contributions,
        [
            {
                "candidate_id": "a",
                "operation": "ADD",
                "validation_result": "selected",
            }
        ],
    )
    assert lifecycle[0]["candidate_id"] == "a"
    assert lifecycle[0]["validation_result"] == "retired"
    assert lifecycle[0]["validation_history"][-1]["intervention_id"] == "a"


class _Sample:
    def __init__(self, group, rollout, reward):
        self.group_index = group
        self.rollout_id = rollout
        self.index = rollout
        self.reward = {"reward": reward}

    def get_reward_value(self, args):
        return self.reward[args.reward_key]


def test_grpo_normalization_counts_fanout_rollout_once():
    samples = [
        _Sample(0, 10, 0.0),
        _Sample(0, 10, 0.0),
        _Sample(0, 11, 1.0),
        _Sample(1, 20, 0.25),
        _Sample(1, 21, 0.25),
    ]
    args = SimpleNamespace(
        reward_key="reward",
        rewards_normalization=True,
        grpo_std_normalization=True,
    )
    raw, normalized = normalize_grpo_by_prompt_and_rollout(args, samples)
    assert raw == [0.0, 0.0, 1.0, 0.25, 0.25]
    assert normalized[0] == normalized[1]
    assert normalized[0] == pytest.approx(-1 / math.sqrt(2), abs=1e-5)
    assert normalized[2] == pytest.approx(1 / math.sqrt(2), abs=1e-5)
    assert normalized[3:] == [0.0, 0.0]


def test_dynamic_filter_rejects_whole_group_on_unscored_segment():
    scored = SimpleNamespace(metadata={"shopsim_scored": True})
    failed = SimpleNamespace(metadata={"shopsim_scored": False})
    slime_runtime._FAILED_GROUP_STREAK = 0
    assert keep_fully_scored_group(None, [[scored], [scored]]) is True
    args = SimpleNamespace(shopsim_max_consecutive_failed_groups=2)
    assert keep_fully_scored_group(args, [[scored], [failed]]) is False
    with pytest.raises(RuntimeError, match="too many consecutive"):
        keep_fully_scored_group(args, [[scored], [failed]])
    slime_runtime._FAILED_GROUP_STREAK = 0


def test_group_coordinator_does_not_barrier_early_rollouts(tmp_path, monkeypatch):
    bank_path = tmp_path / "bank.json"
    _write_bank(bank_path)
    curriculum = SkillCurriculum.from_dict(
        build_curriculum_state(
            bank_path,
            round_id="round-coordinator",
            seed=1,
            skill_free_probability=0.0,
            rho_min=1.0,
            rho_max=1.0,
        )
    )
    curriculum_path = tmp_path / "curriculum.json"
    curriculum_path.write_text("{}", encoding="utf-8")
    calls = []

    async def fake_finish(args, sample, sampling, entry, group_index):
        calls.append((group_index, sorted(entry.outcomes)))
        return {"status": "complete"}

    monkeypatch.setattr(slime_runtime, "_finish_group", fake_finish)
    args = SimpleNamespace(
        n_samples_per_prompt=2,
        shopsim_curriculum_path=str(curriculum_path),
    )
    assignment = curriculum.assign(5)

    def outcome(index):
        return slime_runtime._GroupOutcome(
            sample_index=index,
            assignment=assignment,
            rollout=slime_runtime.EpisodeRollout(
                trace={"episode_id": str(index)},
                samples=[],
                reward={"reward": 0.0, "r_strict": 0.0, "r_success": 0.0},
                scored=True,
            ),
        )

    async def run():
        first = SimpleNamespace(group_index=5)
        second = SimpleNamespace(group_index=5)
        early = await slime_runtime._coordinate_group(
            args, first, {}, curriculum, outcome(10)
        )
        assert early == {"status": "pending_group_completion"}
        final = await slime_runtime._coordinate_group(
            args, second, {}, curriculum, outcome(11)
        )
        assert final == {"status": "complete"}

    asyncio.run(run())
    assert calls == [(5, [10, 11])]
