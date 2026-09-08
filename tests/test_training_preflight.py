import json
from types import SimpleNamespace

import pytest
import yaml

from shopsimrl.curriculum import build_curriculum_state, write_curriculum_state, write_slime_task_data
from shopsimrl.training_analysis import (
    OnlineAnalysisSpec,
    _pending_triage_records,
    drain_training_failure_cards,
    watch_training_failure_cards,
    write_analysis_failure_fallback,
)
from shopsimrl.training_preflight import check_training_inputs


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPSIMRL_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "bank.json").write_text(json.dumps({"skills": [{
        "skill_id": "a", "content": "rule", "metadata": {"estimated_effect": 0.2},
    }]}), encoding="utf-8")
    state = build_curriculum_state(tmp_path / "bank.json", round_id="round-0", seed=1,
                                   skill_free_probability=0.2, rho_min=0.2, rho_max=0.9)
    write_curriculum_state(tmp_path / "curriculum.json", state)
    (tmp_path / "split.json").write_text(json.dumps({
        "splits": {"train": {"task_ids": [7, 8], "persona": True}},
    }), encoding="utf-8")
    write_slime_task_data(tmp_path / "split.json", "train", tmp_path / "data.jsonl")
    (tmp_path / "prompt.txt").write_text("prompt", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({
        "shopsim_curriculum_path": "curriculum.json", "shopsim_skillbank_path": "bank.json",
        "shopsim_system_prompt_file": "prompt.txt", "shopsim_environment_persona": True,
        "shopsim_episode_concurrency": 2, "shopsim_max_steps": 30,
        "shopsim_max_consecutive_failed_groups": 8,
    }), encoding="utf-8")
    return tmp_path


def test_offline_training_preflight(inputs):
    result = check_training_inputs("config.yaml", "data.jsonl", "split.json")
    assert result["status"] == "ready"
    assert result["training_tasks"] == 2 and result["active_chunks"] == 1


@pytest.mark.parametrize("change", ["task", "split", "duplicate", "bank", "curriculum"])
def test_preflight_rejects_changed_inputs(inputs, change):
    if change in {"task", "split", "duplicate"}:
        path = inputs / "data.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if change == "task":
            rows[0]["metadata"]["task_id"] = 999
            rows[0]["prompt"] = "999"
        elif change == "split":
            rows[0]["metadata"]["split"] = "test"
        else:
            rows.append(rows[0])
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    else:
        path = inputs / f"{change}.json"
        payload = json.loads(path.read_text())
        if change == "bank":
            payload["skills"][0]["content"] = "changed"
        else:
            payload["skill_free_probability"] = 0.9
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        check_training_inputs("config.yaml", "data.jsonl", "split.json")


def test_analysis_does_not_silently_accept_missing_round(tmp_path):
    with pytest.raises(FileNotFoundError):
        _pending_triage_records(tmp_path / "missing")
    with pytest.raises(ValueError, match="no completed group"):
        _pending_triage_records(tmp_path)


def test_analysis_failure_keeps_current_skills_without_new_candidates(tmp_path):
    bank = tmp_path / "bank.json"
    bank.write_text(
        json.dumps(
            {
                "skills": [
                    {
                        "skill_id": "a",
                        "content": "rule a",
                        "metadata": {"estimated_effect": 0.2},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    spec = OnlineAnalysisSpec(
        name="round-000",
        training_round_dir=tmp_path / "missing-round",
        current_skillbank_path=bank,
        output_dir=tmp_path / "analysis",
        proposal_ledger_history_path=None,
        proposal_checkpoint="ckpt",
        max_candidates=6,
        resume=True,
        trace2skill=SimpleNamespace(),
    )
    summary = write_analysis_failure_fallback(spec, RuntimeError("analyst down"))
    assert summary["status"] == "failed"
    assert summary["submitted_candidates"] is False
    assert summary["candidates"] == 0
    pool = json.loads((tmp_path / "analysis" / "candidate_pool.json").read_text())
    assert pool["candidates"] == []
    assert pool["current_skills"][0]["skill_id"] == "a"


def _analysis_spec(tmp_path, round_dir):
    bank = tmp_path / "bank.json"
    if not bank.is_file():
        bank.write_text(
            json.dumps(
                {
                    "skills": [
                        {
                            "skill_id": "a",
                            "content": "rule a",
                            "metadata": {"estimated_effect": 0.2},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    model = SimpleNamespace(identity=lambda: {"model": "cpu-analyst"})
    return OnlineAnalysisSpec(
        name="round-000",
        training_round_dir=round_dir,
        current_skillbank_path=bank,
        output_dir=tmp_path / "analysis",
        proposal_ledger_history_path=None,
        proposal_checkpoint="ckpt",
        max_candidates=6,
        resume=True,
        trace2skill=SimpleNamespace(
            environment_base_url="http://unused",
            environment_persona=True,
            environment_timeout=30,
            concurrency=1,
            analyst_model=model,
            compiler_model=model,
            max_failure_analysis_steps=20,
        ),
    )


def test_drain_allows_empty_live_round(tmp_path):
    spec = _analysis_spec(tmp_path, tmp_path / "missing-round")
    payload = drain_training_failure_cards(spec, allow_empty=True)
    assert payload["triage_groups"] == 0
    assert payload["newly_analyzed"] == 0
    assert payload["analyzed_cards"] == 0


def test_watch_analyze_stops_on_stop_file(tmp_path):
    spec = _analysis_spec(tmp_path, tmp_path / "missing-round")
    stop = tmp_path / "analyst.stop"
    stop.write_text("stop", encoding="utf-8")
    payload = watch_training_failure_cards(spec, poll_seconds=0.05, stop_path=stop)
    assert payload["status"] == "watched"
    assert payload["triage_groups"] == 0
    assert payload["newly_analyzed"] == 0
