"""Recover the already-frozen positive SkillBank from its recorded rollout context.

This reads selection metadata only, never test rewards or success outcomes.
It does not refit coefficients or claim to reconstruct the missing validation run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shopsimrl.paired_validation import PAIRED_ESTIMATOR
from shopsimrl.schemas import fingerprint
from shopsimrl.store import atomic_write_json


def recover(comparison_path: Path, run_dir: Path, draft_path: Path, output: Path) -> dict:
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    plan = manifest["plan"]
    if fingerprint(plan) != manifest["plan_fingerprint"]:
        raise ValueError("source manifest fingerprint mismatch")
    identity = comparison["alignment"]["plan"]["positive_skills"]
    expected_ids = comparison["alignment"]["traces"]["positive_chunk_ids"]
    if plan["skills"] != identity or manifest["plan_fingerprint"] != comparison["alignment"]["plan"]["positive_plan_fingerprint"]:
        raise ValueError("positive run does not match the recorded comparison")
    if not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("invalid recorded selection")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    draft_by_id = {row["skill_id"]: row for row in draft["skills"]}
    selected = None
    count = 0
    with (run_dir / "traces.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            trace = json.loads(line)
            rows = trace.get("selected_skills")
            # An attempt that failed before skill selection contains no context.
            if not rows:
                continue
            if trace["provenance"]["skills"] != identity:
                raise ValueError("trace SkillBank identity differs from the positive run")
            if [row["skill_id"] for row in rows] != expected_ids:
                raise ValueError("recorded skill context changed within the positive run")
            if selected is None:
                selected = rows
            elif rows != selected:
                raise ValueError("recorded chunk text/version/metadata changed within the run")
            count += 1
    if selected is None:
        raise ValueError("no recorded positive skill context found")
    records = []
    for row in selected:
        metadata = dict(row["metadata"])
        effect = metadata.get("estimated_effect")
        if (metadata.get("estimator") != PAIRED_ESTIMATOR
                or metadata.get("validation_result") != "paired_delta_positive_bundle"
                or isinstance(effect, bool) or not isinstance(effect, (float, int))
                or not math.isfinite(effect) or effect <= 0):
            raise ValueError(f"chunk {row['skill_id']} has no positive paired-delta estimate")
        if row["content"] != draft_by_id[row["skill_id"]]["content"]:
            raise ValueError("recorded skill text differs from the cold-start draft")
        metadata.pop("selection_scope", None)
        records.append({**row, "enabled": True, "metadata": metadata})
    recovery = {
        "method": "restore_frozen_selection_from_recorded_context",
        "comparison_path": str(comparison_path.resolve()),
        "source_run": str(run_dir.resolve()),
        "source_skillbank": identity,
        "source_draft_sha256": fingerprint(draft),
        "matching_trace_contexts": count,
        "selected_chunk_ids": expected_ids,
        "uses_test_outcomes": False,
        "validation_refitted": False,
    }
    bank = {
        "schema_version": "shopsimrl-skillbank-v1",
        "bank_version": "cold-start-paired-delta-positive-recovered-v1",
        "metadata": {
            "selected_count": len(records), "selected_chunk_ids": expected_ids,
            "estimator": {"name": PAIRED_ESTIMATOR, "fit_intercept": False},
            "recovery": recovery,
        },
        "skills": records,
    }
    bank_path = output / "selected_skillbank.json"
    if bank_path.exists() and json.loads(bank_path.read_text(encoding="utf-8")) != bank:
        raise ValueError("output already contains a different SkillBank; choose a new directory")
    atomic_write_json(bank_path, bank)
    atomic_write_json(output / "recovery_manifest.json", {
        **recovery, "recovered_skillbank_sha256": fingerprint(bank),
    })
    (output / "selected_skill.md").write_text(
        "# Shopping Skill — paired-delta positive selection\n\n"
        "> Restored from the frozen skill context recorded in the positive run; "
        "no test outcomes were used for selection.\n\n"
        + "\n\n".join(row["content"] for row in records) + "\n", encoding="utf-8",
    )
    return {"selected_count": len(records), "matching_trace_contexts": count,
            "skillbank_path": str(bank_path.resolve()), "skillbank_sha256": fingerprint(bank)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--positive-run", type=Path, required=True)
    parser.add_argument("--draft-skillbank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(recover(args.comparison, args.positive_run, args.draft_skillbank, args.output_dir), indent=2))
