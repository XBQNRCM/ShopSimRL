"""Exercise the real slime adapter/trajectory/Sample with CPU model/environment doubles."""

import asyncio
from collections import Counter
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from shopsimrl import slime_runtime
from shopsimrl.curriculum import SkillAssignment


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "slime"))
    from slime.agent.adapters import common
    from slime.agent import parsing
    from slime.agent.trajectory import TurnRecord
    from slime.utils.types import Sample

    calls, generated, environments = [], [], []
    outputs = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            # Dynamic tool schemas change the prefix, just as in the real prompt.
            return [1, len(calls) + 2, 3]

        def decode(self, ids, **kwargs):
            return "sampled output"

    state = SimpleNamespace(tokenizer=Tokenizer(), aborted=False, purchase_reward=1.0)
    module = ModuleType("slime.rollout.sglang_rollout")
    module.GenerateState = lambda args: state
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def parse(*args, **kwargs):
        name, arguments, finish = outputs[len(calls) - 1]
        return parsing.ParsedModelOutput(
            reasoning="reasoning that the OpenAI adapter echoes into history",
            text="",
            tool_uses=[{"name": name, "input": arguments}] if name else [],
        )

    async def sample(prompt_ids, session, body, **kwargs):
        calls.append(body.copy())
        _, _, finish = outputs[len(calls) - 1]
        ids = [1000 + len(calls) * 10 + i for i in range(3)]
        generated.extend(ids)
        return TurnRecord(prompt_ids, ids, finish, [-0.1, -0.2, -0.3])

    class Environment:
        def __init__(self):
            self.actions, self.closed = [], False
            environments.append(self)

        def identity(self):
            return {"adapter": "cpu_test"}

        def reset(self, task_id):
            return {"task_instruction": "buy", "observation": "search page",
                    "observation_state": {"search_available": True, "actions": []}}

        def step(self, action):
            self.actions.append(action)
            if action == "click[buy]":
                return self.terminate("purchase")
            return {"observation": "product", "done": False,
                    "observation_state": {"search_available": False, "actions": ["buy"]}}

        def terminate(self, reason):
            success = float(reason == "purchase") * state.purchase_reward
            return {"done": True, "observation": "terminal", "reward": success,
                    "reward_detail": {"r_strict": success, "r_success": success},
                    "termination_reason": reason}

        def close(self):
            self.closed = True

    monkeypatch.setattr(common, "call_sglang_generate", sample)
    monkeypatch.setattr(parsing, "parse_model_output", parse)
    monkeypatch.setattr(slime_runtime, "_environment", lambda args: Environment())
    args = SimpleNamespace(
        sglang_router_ip="localhost", sglang_router_port=30000,
        rollout_max_context_len=32768, shopsim_max_steps=4,
    )

    def run():
        return asyncio.run(slime_runtime._run_episode_impl(
            args, Sample(index=3, group_index=1, prompt="7", metadata={"task_id": 7, "split": "train"}),
            {"temperature": 0.6, "top_p": 1.0, "top_k": -1, "max_new_tokens": 2048},
            skills=(), assignment=SkillAssignment("frozen", "1", "skill_free", ()), diagnostic=False,
        ))

    return SimpleNamespace(run=run, args=args, state=state, outputs=outputs, Sample=Sample,
                           calls=calls, generated=generated, environments=environments)


def test_real_adapter_keeps_every_sampled_token_and_logprob(harness):
    harness.outputs[:] = [("search", {"query": "shoes"}, "stop"), ("click", {"value": "buy"}, "stop")]
    result = harness.run()
    assert result.scored and result.success, result.trace["error"]
    trained = []
    for sample in result.samples:
        assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length
        tail = sample.tokens[-sample.response_length:]
        trained.extend(token for token, mask in zip(tail, sample.loss_mask) if mask)
        assert [lp for lp, mask in zip(sample.rollout_log_probs, sample.loss_mask) if mask] == [-0.1, -0.2, -0.3]
        assert sample.rollout_id == 3 and sample.group_index == 1
    assert Counter(trained) == Counter(harness.generated)
    assert harness.environments[0].closed


def test_unavailable_tool_is_repaired_without_executing_environment_action(harness):
    harness.outputs[:] = [("search", {"query": "shoes"}, "stop"),
                          ("search", {"query": "again"}, "stop"),
                          ("click", {"value": "buy"}, "stop")]
    result = harness.run()
    assert result.success, result.trace["error"]
    assert harness.environments[0].actions == ["search[shoes]", "click[buy]"]
    assert result.trace["steps"][1]["protocol_error"]["code"] == "unavailable_tool"


def test_sglang_abort_is_not_scored_as_a_policy_failure(harness):
    harness.outputs[:] = [(None, {}, "abort")] * 4
    result = harness.run()
    assert not result.scored
    assert result.trace["status"] == "failed"
    assert len(harness.calls) == 1
    assert harness.environments[0].closed


def test_stopped_rollout_does_not_start_more_model_requests(harness):
    harness.state.aborted = True
    result = harness.run()
    assert not result.scored
    assert not harness.calls


def test_length_without_action_remains_a_scored_policy_failure(harness):
    harness.outputs[:] = [(None, {}, "length")]
    result = harness.run()
    assert result.scored and not result.success
    assert result.trace["final"]["termination_reason"] == "generation_length"
    assert result.samples[0].status == result.samples[0].Status.TRUNCATED


def test_failed_group_retry_flows_into_online_analysis(harness, tmp_path, monkeypatch):
    from shopsimrl import training_analysis
    from shopsimrl.curriculum import build_curriculum_state, write_curriculum_state
    from shopsimrl.online_validation import load_candidate_pool

    bank_path = tmp_path / "bank.json"
    bank_path.write_text(json.dumps({"skills": [{
        "skill_id": "a", "version": "1", "content": "check constraints",
        "metadata": {"estimated_effect": 0.2},
    }]}), encoding="utf-8")
    curriculum_path = tmp_path / "curriculum.json"
    curriculum = build_curriculum_state(bank_path, round_id="round-000", seed=1,
                                        skill_free_probability=1.0, rho_min=0.2, rho_max=0.9)
    write_curriculum_state(curriculum_path, curriculum)
    harness.args.shopsim_curriculum_path = str(curriculum_path)
    harness.args.shopsim_training_output_dir = str(tmp_path / "training")
    harness.args.n_samples_per_prompt = 2
    harness.state.purchase_reward = 0.0
    harness.outputs[:] = [("search", {"query": "shoes"}, "stop"),
                          ("click", {"value": "buy"}, "stop")] * 3

    async def run_group():
        train_samples = []
        for index in range(2):
            base = harness.Sample(index=index, group_index=0, prompt="7", metadata={"task_id": 7, "split": "train"})
            train_samples.extend(await slime_runtime.generate(harness.args, base, {"top_p": 1.0, "top_k": -1}))
        return train_samples

    samples = asyncio.run(run_group())
    assert {sample.rollout_id for sample in samples} == {0, 1}
    group_triage = [
        (sample.rollout_id, sample.metadata.get("shopsim_group_triage"))
        for sample in samples
        if sample.metadata.get("shopsim_group_triage") is not None
    ]
    assert {rollout_id for rollout_id, _ in group_triage} == {1}
    assert {
        record["classification"] for _, record in group_triage
    } == {"full_skill_failure_pending_analysis"}
    round_dir = tmp_path / "training" / "round-000"
    triage = json.loads((round_dir / "group-000000000" / "triage.json").read_text(encoding="utf-8"))
    assert triage["classification"] == "full_skill_failure_pending_analysis"
    assert len(harness.environments) == 3
    seen = []

    def analyze(trace, **kwargs):
        seen.append(trace)
        assert kwargs["allowed_rewrite_targets"] == {"a"}
        assert [row["skill_id"] for row in trace["selected_skills"]] == ["a"]
        return {"source_trajectory_id": trace["episode_id"], "eligible_for_consolidation": False}

    monkeypatch.setattr(training_analysis, "analyze_failure_trace", analyze)
    model = SimpleNamespace(identity=lambda: {"model": "cpu-analyst"})
    spec = training_analysis.OnlineAnalysisSpec(
        name="round-000", training_round_dir=round_dir, current_skillbank_path=bank_path,
        output_dir=tmp_path / "analysis", proposal_ledger_history_path=None,
        proposal_checkpoint="checkpoint-0", max_candidates=6, resume=True,
        trace2skill=SimpleNamespace(environment_base_url="http://unused", environment_persona=True,
                                   environment_timeout=30, concurrency=1, analyst_model=model,
                                   compiler_model=model, max_failure_analysis_steps=20),
    )
    summary = training_analysis.analyze_training_failures(spec)
    assert len(seen) == 1
    assert summary["analyzed_cards"] == 1 and summary["candidates"] == 0
    pool = load_candidate_pool(Path(summary["candidate_pool_path"]))
    assert [row["skill_id"] for row in pool["current_skills"]] == ["a"]
    assert training_analysis.analyze_training_failures(spec) == summary
    assert len(seen) == 1
    drain = training_analysis.drain_training_failure_cards(spec)
    assert drain["newly_analyzed"] == 0
    assert training_analysis.compile_training_failure_cards(spec) == summary


@pytest.mark.parametrize("sampling,split,evaluation", [
    ({"top_p": 0.95}, "train", False), ({"top_k": 20}, "train", False),
    ({}, "test", False), ({}, "val", True),
])
def test_training_contract_rejects_incompatible_sampling_and_evaluation(harness, sampling, split, evaluation):
    sample = harness.Sample(index=0, group_index=0, prompt="7", metadata={"task_id": 7, "split": split})
    with pytest.raises(ValueError):
        asyncio.run(slime_runtime.generate(harness.args, sample, sampling, evaluation=evaluation))
    assert not harness.calls


def _filter_sample(rollout_id, reward, scored=True):
    return SimpleNamespace(
        index=rollout_id,
        rollout_id=rollout_id,
        metadata={"shopsim_scored": scored},
        get_reward_value=lambda args, reward=reward: reward,
    )


def test_dynamic_filter_reports_technical_drop_reason(harness):
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    slime_runtime._FAILED_GROUP_STREAK = 0
    kept = slime_runtime.fully_scored_group_filter(
        harness.args, [[_filter_sample(0, 1.0)], [_filter_sample(1, 0.0)]]
    )
    dropped = slime_runtime.fully_scored_group_filter(
        harness.args, [[_filter_sample(0, 1.0)], [_filter_sample(1, 0.0, scored=False)]]
    )
    assert isinstance(kept, DynamicFilterOutput) and kept.keep and kept.reason is None
    assert isinstance(dropped, DynamicFilterOutput) and not dropped.keep
    assert dropped.reason == "shopsim_unscored_technical_group"
    assert dropped.keep_when_insufficient is False
    slime_runtime._FAILED_GROUP_STREAK = 0


def test_dynamic_filter_drops_tied_group_but_yields_when_batch_runs_short(harness):
    from slime.rollout.filter_hub.base_types import DynamicFilterOutput

    slime_runtime._FAILED_GROUP_STREAK = 0
    tied = slime_runtime.fully_scored_group_filter(
        harness.args, [[_filter_sample(0, 1.0)], [_filter_sample(1, 1.0)]]
    )
    assert isinstance(tied, DynamicFilterOutput) and not tied.keep
    assert tied.reason == "shopsim_zero_std_group"
    assert tied.keep_when_insufficient is True
    slime_runtime._FAILED_GROUP_STREAK = 0


def test_reward_spread_counts_executions_not_fan_out_segments(harness):
    # Two segments of one execution repeat its reward and cannot disagree.
    single = [_filter_sample(0, 1.0), _filter_sample(0, 1.0)]
    assert slime_runtime.group_has_reward_spread(harness.args, single) is False
    two = single + [_filter_sample(1, 0.0), _filter_sample(1, 0.0)]
    assert slime_runtime.group_has_reward_spread(harness.args, two) is True


def test_slime_dataset_accepts_task_ids_when_checkpoint_has_processor(harness, tmp_path, monkeypatch):
    # Dataset does not use Ray when reading prompts; only its training getter does.
    monkeypatch.setitem(sys.modules, "ray", ModuleType("ray"))
    processing = ModuleType("slime.utils.processing_utils")
    processing.process_vision_info = lambda messages, processor: {"images": None, "videos": None}
    monkeypatch.setitem(sys.modules, processing.__name__, processing)
    from slime.utils.data import Dataset

    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"prompt": "7", "metadata": {"task_id": 7, "split": "train"}}) + "\n", encoding="utf-8")
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: "rendered prompt")
    kwargs = dict(tokenizer=tokenizer, processor=object(), max_length=None, prompt_key="prompt")
    with pytest.raises(AssertionError, match="prompt must be a list"):
        Dataset(str(path), apply_chat_template=False, **kwargs)
    data = Dataset(str(path), apply_chat_template=True, **kwargs)
    assert slime_runtime._task_identity(data.samples[0]) == (7, "train")
