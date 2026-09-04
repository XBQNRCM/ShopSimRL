from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from shopsimrl.model import OpenAICompatibleConfig
from shopsimrl.online_validation import (
    OnlineGateSpec, build_candidate_pool, load_online_gate_config,
    online_gate_plan, run_online_gate,
)
from shopsimrl.paired_validation import (
    GateEvaluationError, load_bare_validation, paired_observations,
    read_validation_traces, unique_traces,
)
from shopsimrl.schemas import EpisodeJob
from shopsimrl.trace2skill_evaluation import build_mask_assignments, fit_paired_delta_ols, gate_a_plan, run_gate_a
from shopsimrl.trace2skill_evaluation_config import GateASpec, load_trace2skill_evaluation_config
from test_trace2skill_evaluation import _EffectRuntime, _experiment, _write_bare, _write_draft


@pytest.fixture
def pair(tmp_path):
    draft, bank = _write_draft(tmp_path)
    spec = GateASpec(
        "gate", tmp_path / "gate", _experiment(tmp_path, list(range(40))),
        tmp_path / "bare", draft, bank, 19, 0.5, 1,
    )
    plan = _write_bare(spec)
    bare = read_validation_traces(spec.bare_run_dir / "traces.jsonl")
    jobs = [EpisodeJob(**row) for row in plan["jobs"]]
    assignments = build_mask_assignments(jobs, ["a", "b"], mask_seed=13)
    masked = deepcopy(bare)
    by_id = {row["episode_id"]: row for row in masked}
    for row in assignments["assignments"]:
        trace = by_id[row["episode_id"]]
        delta = 0.3 * row["mask"][0] - 0.2 * row["mask"][1]
        trace["selected_skills"] = [{"skill_id": key} for key in row["included_chunk_ids"]]
        trace["provenance"]["skills"] = {"provider": "assigned"}
        trace["final"]["reward"] += delta
        trace["final"]["reward_detail"]["r_strict"] += delta
        trace["final"]["reward_detail"]["r_success"] += delta
    return spec, plan, assignments, masked, bare


def test_task_delta_alignment_and_auxiliary_outcomes(pair):
    _, _, assignments, masked, bare = pair
    masks, outcomes, rows = paired_observations(assignments, reversed(masked), reversed(bare))
    assert len(rows) == 40
    for metric in ("reward", "r_strict", "r_success"):
        intercept, coefficients = fit_paired_delta_ols(masks, outcomes[metric])
        assert intercept == 0
        assert coefficients == pytest.approx((0.3, -0.2))
    assert len({row["bare_reward"] for row in rows}) > 1
    assert rows[0]["delta_reward"] == pytest.approx(rows[0]["reward"] - rows[0]["bare_reward"])


@pytest.mark.parametrize("case", ["missing", "extra", "failed", "duplicate", "task", "split", "seed", "mask", "bare_skill", "missing_provenance", "checkpoint", "sampling", "prompt", "environment", "runtime", "nan", "component", "strict"])
def test_reject_unsafe_pairs(pair, case):
    _, _, assignments, masked, bare = pair
    trace = masked[0]
    if case == "missing":
        bare.pop()
    elif case == "extra":
        extra = deepcopy(bare[0])
        extra["episode_id"] = "extra"
        bare.append(extra)
    elif case == "failed":
        bare[0]["status"] = "failed"
    elif case == "duplicate":
        bare.append(deepcopy(bare[0]))
    elif case in {"task", "split", "seed"}:
        key = {"task": "task_id", "split": "split", "seed": "seed"}[case]
        bare[0]["job"][key] = "test" if case == "split" else -100
    elif case == "mask":
        trace["selected_skills"] = [{"skill_id": "unexpected"}]
    elif case == "bare_skill":
        bare[0]["provenance"]["skills"] = {"provider": "assigned"}
    elif case == "missing_provenance":
        trace["provenance"] = {}
    elif case in {"checkpoint", "sampling"}:
        key = "checkpoint_id" if case == "checkpoint" else "sampling"
        trace["provenance"]["model"][key] = "different"
    elif case in {"prompt", "environment", "runtime"}:
        trace["provenance"][case]["different"] = True
    elif case == "nan":
        bare[0]["final"]["reward"] = float("nan")
    elif case == "component":
        bare[0]["final"]["reward_detail"]["r_extra"] = 0.2
    elif case == "strict":
        bare[0]["final"]["reward"] += 0.1
    with pytest.raises(GateEvaluationError):
        paired_observations(assignments, masked, bare)


def test_failed_attempt_can_resume_but_completed_replicates_are_not_latest_selected(pair):
    _, _, _, _, bare = pair
    failed = deepcopy(bare[0])
    failed["status"] = "failed"
    assert unique_traces([failed, bare[0]])[bare[0]["episode_id"]] == bare[0]
    with pytest.raises(GateEvaluationError, match="duplicate completed"):
        unique_traces([bare[0], bare[0]])
    with pytest.raises(GateEvaluationError, match="follows a completed"):
        unique_traces([bare[0], failed])


def test_unsent_seeds_need_not_match_but_sample_ids_must(pair):
    _, _, assignments, masked, bare = pair
    for trace in masked + bare:
        trace["provenance"]["model"]["sampling"]["send_seed"] = False
    bare[0]["job"]["seed"] += 1
    paired_observations(assignments, masked, bare)
    bare[0]["job"]["sample_id"] += 1
    with pytest.raises(GateEvaluationError, match="job mismatch"):
        paired_observations(assignments, masked, bare)


def test_gate_a_failed_attempt_is_resumed_and_plan_fingerprint_agrees(pair):
    spec, plan, _, _, _ = pair
    expected = gate_a_plan(spec)["semantic_plan_fingerprint"]
    failed_task = plan["jobs"][0]["task_id"]
    calls = []

    class Runtime(_EffectRuntime):
        def run(self, job):
            calls.append(job.task_id)
            trace = super().run(job)
            if job.task_id == failed_task:
                trace["status"] = "failed"
                trace["final"]["done"] = False
            return trace

    manifest = run_gate_a(spec, runtime_factory=lambda provider: Runtime(provider, plan))
    assert manifest["status"] == "incomplete"
    assert len(calls) == 40
    assert not (spec.output_dir / "selected_skillbank.json").exists()
    assert json.loads((spec.output_dir / "manifest.json").read_text())["plan_fingerprint"] == expected
    calls.clear()

    class Retry(_EffectRuntime):
        def run(self, job):
            calls.append(job.task_id)
            return super().run(job)

    result = run_gate_a(spec, runtime_factory=lambda provider: Retry(provider, plan))
    assert result["status"] == "complete"
    assert calls == [failed_task]
    # Raw failed + completed attempts are retained, but only one completed pair contributes.
    assert len((spec.output_dir / "traces.jsonl").read_text().splitlines()) == 41
    contributions = json.loads((spec.output_dir / "contributions.json").read_text())
    assert contributions["observations"] == 40


def test_baseline_preflight_matches_manifest_and_traces(pair):
    spec, plan, _, _, _ = pair
    records, info = load_bare_validation(spec.bare_run_dir, plan, require_checkpoint=True)
    assert len(records) == 40
    assert info["checkpoint_id"] == "checkpoint-0"
    for field in ("model", "runtime", "prompt", "environment", "task_split"):
        wrong = deepcopy(plan)
        wrong[field]["changed"] = True
        with pytest.raises(GateEvaluationError, match="mismatch"):
            load_bare_validation(spec.bare_run_dir, wrong)
    changed_jobs = deepcopy(plan)
    changed_jobs["jobs"][0]["split"] = "test"
    with pytest.raises(GateEvaluationError, match="job coverage"):
        load_bare_validation(spec.bare_run_dir, changed_jobs)
    # Transport settings, concurrency and the local path of the split do not define treatment.
    equivalent = deepcopy(plan)
    equivalent["runtime"]["concurrency"] = 16
    equivalent["model"]["transport"]["timeout_seconds"] = 999
    equivalent["task_split"]["source_path"] = "/relocated/splits.json"
    load_bare_validation(spec.bare_run_dir, equivalent)
    missing_checkpoint = deepcopy(plan)
    missing_checkpoint["model"].pop("checkpoint_id")
    with pytest.raises(GateEvaluationError, match="checkpoint_id"):
        load_bare_validation(spec.bare_run_dir, missing_checkpoint, require_checkpoint=True)
    with (spec.bare_run_dir / "traces.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("not json\n")
    with pytest.raises(GateEvaluationError, match="invalid validation JSON"):
        load_bare_validation(spec.bare_run_dir, plan)


def test_no_constant_column_or_zero_mask_special_case():
    # This matrix has full rank without an intercept but not with one.
    intercept, coefficients = fit_paired_delta_ols([[1, 0], [0, 1]] * 3, [0.3, -0.2] * 3)
    assert intercept == 0
    assert coefficients == pytest.approx((0.3, -0.2))
    _, coefficients = fit_paired_delta_ols([[0], [1], [1]], [0.8, 0.2, 0.4])
    assert coefficients == pytest.approx((0.3,))
    with pytest.raises(GateEvaluationError, match="rank deficient"):
        fit_paired_delta_ols([[0, 0], [1, 1], [1, 1]], [0, 1, 1])


def test_online_orchestration_reuses_bare_and_resumes(pair):
    spec, plan, _, _, _ = pair
    # Online ordinary current slots use the same shared paired estimator.
    bank = spec.draft_skillbank_path
    pool = build_candidate_pool(bank, [], round_id="round-1", proposal_checkpoint="checkpoint-0")
    pool_path = spec.output_dir.parent / "candidate_pool.json"
    pool_path.write_text(json.dumps(pool), encoding="utf-8")
    online = OnlineGateSpec("online", spec.output_dir.parent / "online", spec.experiment,
                            spec.bare_run_dir, pool_path, 19, 1)
    assert online_gate_plan(online)["bare_baseline"]["observations"] == 40
    manifest = run_online_gate(online, runtime_factory=lambda provider: _EffectRuntime(provider, plan))
    assert manifest["status"] == "complete"
    result = json.loads((online.output_dir / "contributions.json").read_text())
    assert result["selected_intervention_ids"] == ["chunk-positive"]
    assert result["estimates"]["reward"]["coefficients"]["chunk-positive"] == pytest.approx(0.3)
    assert result["estimates"]["reward"]["coefficients"]["chunk-negative"] == pytest.approx(-0.2)
    assert result["bare_baseline"]["checkpoint_id"] == "checkpoint-0"
    from shopsimrl.curriculum import SkillCurriculum, build_curriculum_state

    next_state = build_curriculum_state(
        online.output_dir / "selected_skillbank.json", round_id="round-2", seed=1,
        skill_free_probability=0.3, rho_min=0.2, rho_max=0.9,
    )
    assert [chunk.skill.skill_id for chunk in SkillCurriculum.from_dict(next_state).chunks] == ["chunk-positive"]
    never_called = Mock()
    run_online_gate(online, runtime_factory=never_called)
    never_called.assert_not_called()
    with pytest.raises(GateEvaluationError, match="bare validation run missing"):
        run_online_gate(replace(online, bare_run_dir=spec.bare_run_dir / "missing"), runtime_factory=never_called)
    never_called.assert_not_called()


@pytest.mark.parametrize("online", [False, True])
def test_gate_config_requires_bare_run_dir(tmp_path, online):
    root = Path(__file__).resolve().parents[1]
    name = "trace2skill_online_gate.example.yaml" if online else "trace2skill_gate_a.yaml"
    loader = load_online_gate_config if online else load_trace2skill_evaluation_config
    spec = loader(root / "configs" / name)
    assert spec.bare_run_dir.is_absolute()
    payload = yaml.safe_load((root / "configs" / name).read_text())
    record = payload["online_gate" if online else "gate_a"]
    record["experiment_config"] = str(root / "configs" / record["experiment_config"])
    record.pop("bare_run_dir")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="bare_run_dir.*required"):
        loader(path)


def test_checkpoint_identity_is_metadata_only_and_legacy_identity_unchanged():
    config = OpenAICompatibleConfig("fake", "http://model/v1")
    assert "checkpoint_id" not in config.identity()
    assert replace(config, checkpoint_id="weights-1").identity()["checkpoint_id"] == "weights-1"
    with pytest.raises(ValueError, match="checkpoint_id"):
        replace(config, checkpoint_id=" ")
