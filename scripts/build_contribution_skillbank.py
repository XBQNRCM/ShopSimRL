"""Build a control SkillBank from Gate A chunk contribution signs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _matches(coefficient: float, sign: str) -> bool:
    if sign == "negative":
        return coefficient < 0.0
    if sign == "positive":
        return coefficient > 0.0
    raise ValueError(f"unsupported contribution sign: {sign}")


def build_skillbank(
    *,
    draft_skillbank: dict[str, Any],
    contributions: dict[str, Any],
    sign: str,
) -> dict[str, Any]:
    rows = contributions.get("contributions")
    records = draft_skillbank.get("skills")
    if not isinstance(rows, list) or not isinstance(records, list):
        raise ValueError("invalid contribution report or draft SkillBank")

    contribution_by_id: dict[str, dict[str, Any]] = {}
    selected_ids: set[str] = set()
    for row in rows:
        skill_id = row.get("chunk_id")
        coefficient = row.get("coefficient")
        if not isinstance(skill_id, str) or not isinstance(
            coefficient, (int, float)
        ):
            raise ValueError("invalid chunk contribution row")
        contribution_by_id[skill_id] = row
        if _matches(float(coefficient), sign):
            selected_ids.add(skill_id)

    selected: list[dict[str, Any]] = []
    for record in records:
        skill_id = record.get("skill_id")
        if skill_id not in selected_ids:
            continue
        row = contribution_by_id[skill_id]
        metadata = dict(record.get("metadata") or {})
        metadata.pop("draft_only", None)
        metadata.update(
            {
                "validation_result": f"{sign}_contribution_control",
                "estimated_effect": row["coefficient"],
                "contribution_rank": row["rank"],
                "draft_order": row["draft_order"],
            }
        )
        selected.append(
            {
                "skill_id": skill_id,
                "version": f"gate-a-{sign}-control-1",
                "enabled": True,
                "content": record["content"],
                "metadata": metadata,
            }
        )

    if len(selected) != len(selected_ids):
        missing = sorted(selected_ids - {row["skill_id"] for row in selected})
        raise ValueError(f"contribution chunks missing from draft SkillBank: {missing}")
    if not selected:
        raise ValueError(f"no {sign} contribution chunks found")

    selected_chunk_ids = [record["skill_id"] for record in selected]
    return {
        "schema_version": "shopsimrl-skillbank-v1",
        "bank_version": f"cold-start-gate-a-{sign}-control-v1",
        "metadata": {
            "skill_title": f"多约束电商购物技能（Gate A {sign} contribution control）",
            "gate_status": {"gate_a": "control", "gate_b": "not_run"},
            "draft_sha256": contributions.get("draft_sha256"),
            "assignment_sha256": contributions.get("assignment_sha256"),
            "contribution_sign": sign,
            "selected_count": len(selected),
            "selected_chunk_ids": selected_chunk_ids,
            "injection_order": "canonical_draft_order",
            "estimator": contributions.get("estimator"),
        },
        "skills": selected,
    }


def _write_markdown(path: Path, skillbank: dict[str, Any]) -> None:
    title = skillbank["metadata"]["skill_title"]
    body = "\n\n".join(record["content"] for record in skillbank["skills"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {title}\n\n"
        "> Control skill assembled from Gate A chunks whose estimated reward "
        "contribution is strictly negative.\n\n"
        f"{body}\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-skillbank", type=Path, required=True)
    parser.add_argument("--contributions", type=Path, required=True)
    parser.add_argument("--sign", choices=("negative", "positive"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()

    skillbank = build_skillbank(
        draft_skillbank=_read_object(args.draft_skillbank),
        contributions=_read_object(args.contributions),
        sign=args.sign,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(skillbank, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output is not None:
        _write_markdown(args.markdown_output, skillbank)
    print(
        f"wrote {len(skillbank['skills'])} {args.sign} chunks to "
        f"{args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
