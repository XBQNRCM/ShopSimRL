#!/usr/bin/env python3
"""Build publication data from frozen artifacts; --refresh-local imports raw runs.

No model calls, environment calls, or changes to historical experiment artifacts.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts/analysis"
ARMS = {
    "base_free": "qwen35-4b-test-base-free",
    "base_s0": "qwen35-4b-test-base-s0",
    "iter80_free": "qwen35-4b-test-iter80-free",
    "iter80_st": "qwen35-4b-test-iter80-st",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def refresh_local():
    """Extract small, allowlisted numerical records, never raw conversations."""
    from shopsimrl.evaluation import summarize_traces

    steps, sources = {}, {}
    pattern = re.compile(r"perf (\d+): (\{.*\})")
    for path in sorted((ROOT / "runs/wandb").rglob("output*.log")):
        for line in path.open(encoding="utf-8", errors="replace"):
            if "rollout/shopsim/reward_mean" not in line:
                continue
            match = pattern.search(line)
            if not match:
                raise ValueError(f"Unparsed metric record in {path.name}")
            step, raw = int(match[1]), ast.literal_eval(match[2])
            if not 1 <= step <= 80:
                continue
            row = {k: v for k, v in raw.items() if isinstance(v, (int, float)) and
                   (k.startswith("rollout/shopsim/") or k.startswith("rollout/dynamic_filter/"))}
            if step in steps and steps[step] != row:
                raise ValueError(f"Conflicting duplicate at step {step}")
            steps[step] = row
            sources.setdefault(step, set()).add(path.relative_to(ROOT).as_posix())
    if not set(range(1, 81)).issubset(steps):
        raise ValueError("Incomplete primary 1–80 training history")
    rows = [{"step": step, "round": (step - 1) // 20, "primary_run": int(step <= 80), **steps[step]}
            for step in sorted(steps)]
    csv_write(OUT / "training_steps.csv", rows)
    write(OUT / "training_sources.json", {str(k): sorted(v) for k, v in sources.items()})

    groups = []
    for path in sorted((ROOT / "runs/slime-training").glob("round-00[0-3]/group-*/triage.json")):
        d = read(path)
        rollouts = d["training_rollouts"]
        scored = [x["reward"] for x in rollouts if x["scored"]]
        retry = d.get("full_skill_retry")
        groups.append({"round": int(d["curriculum"]["round_id"].split("-")[-1]),
                       "group_index": d["group_index"], "mode": d["assignment"]["mode"],
                       "classification": d["classification"], "all_wrong": int(d["all_wrong"]),
                       "rollouts": len(rollouts), "scored": len(scored),
                       "reward_mean": float(np.mean([x["reward"] for x in scored])) if scored else "",
                       "success_mean": float(np.mean([x["r_success"] for x in scored])) if scored else "",
                       "retry_present": int(retry is not None)})
    if len(groups) != 2560:
        raise ValueError(f"Expected 2560 primary groups; found {len(groups)}")
    csv_write(OUT / "training_groups.csv", groups)

    outcomes, audit = {}, {}
    for arm, name in ARMS.items():
        path = ROOT / "runs" / name / "traces.jsonl"
        completed, attempts = {}, Counter()
        for line in path.open(encoding="utf-8"):
            trace = json.loads(line)
            job = trace["job"]
            key = (job["split"], job["task_id"], job["sample_id"])
            attempts[trace["status"]] += 1
            if trace["status"] == "completed":
                if key in completed:
                    raise ValueError(f"Duplicate completed task in {name}: {key}")
                completed[key] = trace
        if len(completed) != 400:
            raise ValueError(f"Incomplete {name}")
        summary = summarize_traces(completed.values(), requested=400)
        released = read(ROOT / "artifacts/eval" / name / "summary.json")
        for metric in ["reward_mean", "success_rate"]:
            if not np.isclose(summary["primary"][metric], released["primary"][metric], atol=1e-12):
                raise ValueError(f"Released summary mismatch: {name}/{metric}")
        audit[arm] = {"attempt_status_counts": dict(attempts), "completed_unique": 400,
                      "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "summary_recomputed": True}
        for key, trace in sorted(completed.items()):
            final = trace["final"]
            task = outcomes.setdefault(str(key[1]), {"task_id": key[1], "split": key[0], "sample_id": key[2]})
            task[arm] = {"reward": final["reward"], "success": final["reward_detail"]["r_success"]}
    if len(outcomes) != 400 or any(not all(a in row for a in ARMS) for row in outcomes.values()):
        raise ValueError("Test arms do not share the same 400 tasks")
    write(OUT / "test_task_outcomes.json", list(outcomes.values()))
    write(OUT / "local_extraction_audit.json", audit)


def paired(a, b, metric="reward"):
    from scripts.compare_shopsimrl_runs import _bootstrap_mean_ci
    delta = np.asarray(b) - np.asarray(a)
    source_metric = "r_success" if metric == "success" else metric
    seed = int.from_bytes(hashlib.sha256(f"20260901:{source_metric}".encode()).digest()[:8], "big")
    interval = _bootstrap_mean_ci(delta.tolist(), samples=10000, confidence_level=.95, seed=seed)
    return {"n": len(delta), "a": float(np.mean(a)), "b": float(np.mean(b)),
            "difference": float(np.mean(delta)), "ci95": interval}


def analyze():
    outcomes = read(OUT / "test_task_outcomes.json")
    with (OUT / "training_steps.csv").open(encoding="utf-8", newline="") as handle:
        steps = [{k: float(v) if v else None for k, v in row.items()} for row in csv.DictReader(handle)]
    with (OUT / "training_groups.csv").open(encoding="utf-8", newline="") as handle:
        groups = list(csv.DictReader(handle))
    primary = [row for row in steps if row["primary_run"] == 1]
    if [int(row["step"]) for row in primary] != list(range(1, 81)):
        raise ValueError("Primary metrics must be exactly steps 1 through 80")
    summaries = {a: read(ROOT / "artifacts/eval" / n / "summary.json") for a, n in ARMS.items()}
    comparisons = {}
    for name, a, b in [("training_free", "base_free", "iter80_free"),
                        ("initial_skill", "base_free", "base_s0"),
                        ("final_skill", "iter80_free", "iter80_st"),
                        ("final_pair_vs_base", "base_free", "iter80_st")]:
        comparisons[name] = {m: paired([t[a][m] for t in outcomes], [t[b][m] for t in outcomes], m)
                             for m in ["reward", "success"]}
        comparisons[name]["success_transitions"] = dict(Counter(
            f"{int(t[a]['success'])}->{int(t[b]['success'])}" for t in outcomes))
    rounds, coefficients, banks = [], [], []
    cold = read(ROOT / "artifacts/cold-start/selected_skillbank.json")
    banks.append({"label": "S0", "step": 0, "skills": cold["skills"]})
    for i in range(4):
        root = ROOT / f"artifacts/round-{i:03}"
        gate, analysis = read(root / "gate/contributions.json"), read(root / "analysis/analysis_summary.json")
        curriculum = read(ROOT / "artifacts/cold-start/curriculum.json" if i == 0 else root / "curriculum.json")
        selected = read(root / "gate/selected_skillbank.json")
        observations = gate["observations_table"]
        # Refit every published main effect from the paired observations.
        x = np.asarray([o["mask"] for o in observations], dtype=float)
        y = np.asarray([o["reward"] - o["bare_reward"] for o in observations])
        beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        expected = [gate["estimates"]["reward"]["coefficients"][c["intervention_id"]] for c in gate["contributions"]]
        if rank != x.shape[1] or not np.allclose(beta, expected, atol=1e-9, rtol=1e-8):
            raise ValueError(f"Gate refit mismatch in round {i}")
        round_groups = [g for g in groups if int(g["round"]) == i]
        metrics = [r for r in primary if int(r["round"]) == i]
        statuses = Counter(g["classification"] for g in round_groups)
        round_row = {"round": i, "step": 20 * (i + 1), "q": curriculum["skill_free_probability"],
                     "input_chunks": len(curriculum["chunks"]), "selected_chunks": len(selected["skills"]),
                     "bare": gate["bare_baseline"]["mean_outcomes"],
                     "masked": {k: v["masked_mean"] for k, v in gate["estimates"].items()},
                     "gate_delta": paired([o["bare_reward"] for o in observations], [o["reward"] for o in observations]),
                     "candidate_count": analysis["candidates"], "eligible_cards": analysis["eligible_cards"],
                     "analyzed_cards": analysis["analyzed_cards"], "group_count": len(round_groups),
                     "classifications": dict(statuses), "generated_episodes": sum(int(g["rollouts"]) for g in round_groups),
                     "all_wrong_groups": sum(int(g["all_wrong"]) for g in round_groups),
                     "trainer_episodes": int(sum(r["rollout/shopsim/train_trajectory_count"] for r in metrics)),
                     "rho": [{"id": c["skill_id"], "rho": c["inclusion_probability"]} for c in curriculum["chunks"]],
                     "gate_refit_max_error": float(np.max(np.abs(beta - expected)))}
        rounds.append(round_row)
        coefficients.extend({"round": i, **c} for c in gate["contributions"])
        banks.append({"label": f"S{i+1}", "step": (i+1)*20, "skills": selected["skills"]})
    result = {"schema_version": "shopsimrl-public-analysis-v1", "checkpoint": "qwen35-4b-iter80",
              "primary_steps": [1, 80],
              "bootstrap": {"method": "task-level paired percentile", "samples": 10000, "seed": 20260901,
                            "rng": "random.Random; SHA256-derived per-metric seed (same as compare_shopsimrl_runs.py)",
                            "confidence": .95, "unit": "task; conditional on this training run"},
              "test": summaries, "comparisons": comparisons, "rounds": rounds,
              "training_steps": primary, "coefficients": coefficients, "skill_banks": banks,
              "ranking": read(ROOT / "artifacts/eval/ranking/ranking_test_comparison.json")}
    write(OUT / "analysis.json", result)
    csv_write(OUT / "rounds.csv", [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rounds])
    csv_write(OUT / "chunk_contributions.csv", [{k: v for k, v in c.items() if k != "auxiliary_coefficients"} for c in coefficients])
    sources = [p for p in (ROOT / "artifacts").rglob("*.json") if OUT not in p.parents]
    sources += [OUT / p for p in ["training_steps.csv", "training_groups.csv", "test_task_outcomes.json"]]
    write(OUT / "provenance.json", {"script": "scripts/analyze_training.py", "numpy_version": np.__version__,
          "input_hash_normalization": "SHA-256 after CRLF-to-LF normalization of JSON/CSV text; historical Git checkout EOLs vary",
          "inputs": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(sources)}})
    print(f"Verified 80 steps, {len(outcomes)} paired test tasks and four gate refits. Wrote {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-local", action="store_true", help="Import local raw runs before rebuilding analysis")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.refresh_local:
        refresh_local()
    analyze()
