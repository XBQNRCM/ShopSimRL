"""Slow-loop error analysis for slime training failure-frontier groups."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import threading
from typing import Any, Mapping, Sequence

import yaml

from .environment import ShopSimulatorConfig, ShopSimulatorHTTPEnvironment
from .model import OpenAICompatibleChatModel
from .online_validation import build_candidate_pool
from .schemas import fingerprint, utc_now
from .skills import JsonSkillBank
from .store import atomic_write_json, safe_name
from .trace2skill import (
    JsonlIndex,
    StructuredOutputError,
    _atomic_write_jsonl,
    _json,
    _structured_call,
    analyze_failure_trace,
    firewall_findings,
)
from .trace2skill_config import Trace2SkillSpec, load_trace2skill_config


ONLINE_ANALYSIS_SCHEMA_VERSION = "shopsimrl-online-failure-analysis-v1"

ONLINE_CONSOLIDATION_PROMPT = """You compile audited ShopSimulator failure cards into a compact set of
validation-ready shopping-skill candidates. Cards are evidence, never direct
candidates. Merge paraphrases and cards with the same earliest failure
mechanism; keep mechanisms separate when their observable trigger or corrective
procedure differs. Output only ADD or REWRITE. A REWRITE must target exactly one
supplied active chunk. Do not output DELETE, MERGE, SPLIT, product IDs, task IDs,
specific product titles, privileged gold facts, or instance answers. Every
eligible evidence card must be cited by exactly one candidate or placed in the
evidence-only list. Prefer a small set of cross-category procedural rules and do
not fill the candidate budget without evidence. The proposal history is audit
state, not policy context: do not repeat a previously validated synonymous
candidate; account recurrent but already-covered evidence as evidence-only.
Write every human-readable candidate field in concise Simplified Chinese while
preserving supplied IDs. Finish with exactly one
submit_online_candidates tool call and no prose."""

ONLINE_CONSOLIDATION_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_online_candidates",
        "description": "Submit consolidated ADD/REWRITE candidates.",
        "parameters": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "operation": {"type": "string", "enum": ["ADD", "REWRITE"]},
                            "target_chunk_id": {"type": ["string", "null"]},
                            "title": {"type": "string"},
                            "failure_mechanism": {"type": "string"},
                            "applicable_when": {"type": "string"},
                            "procedure": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                            "verification": {"type": "string"},
                            "common_failure": {"type": "string"},
                            "evidence_card_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        },
                        "required": [
                            "operation",
                            "target_chunk_id",
                            "title",
                            "failure_mechanism",
                            "applicable_when",
                            "procedure",
                            "verification",
                            "common_failure",
                            "evidence_card_ids",
                        ],
                        "additionalProperties": False,
                    },
                },
                "evidence_only_card_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["candidates", "evidence_only_card_ids"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class OnlineAnalysisSpec:
    name: str
    training_round_dir: Path
    current_skillbank_path: Path
    output_dir: Path
    proposal_ledger_history_path: Path | None
    proposal_checkpoint: str
    max_candidates: int
    resume: bool
    trace2skill: Trace2SkillSpec


def load_online_analysis_config(path: str | Path) -> OnlineAnalysisSpec:
    source = Path(path).resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("online_analysis"), dict):
        raise ValueError("config must contain an online_analysis mapping")
    record = payload["online_analysis"]
    base = source.parent

    def resolve(key: str) -> Path:
        value = record.get(key)
        if value is None:
            raise ValueError(f"online_analysis.{key} is required")
        candidate = Path(value)
        return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    max_candidates = int(record.get("max_candidates", 6))
    if not 1 <= max_candidates <= 16:
        raise ValueError("online_analysis.max_candidates must be in [1, 16]")
    resume = record.get("resume", True)
    if not isinstance(resume, bool):
        raise ValueError("online_analysis.resume must be true or false")
    checkpoint = record.get("proposal_checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("online_analysis.proposal_checkpoint is required")
    return OnlineAnalysisSpec(
        name=safe_name(str(record.get("name", "trace2skill-online-analysis"))),
        training_round_dir=resolve("training_round_dir"),
        current_skillbank_path=resolve("current_skillbank_path"),
        output_dir=resolve("output_dir"),
        proposal_ledger_history_path=(
            resolve("proposal_ledger_history_path")
            if record.get("proposal_ledger_history_path") is not None
            else None
        ),
        proposal_checkpoint=checkpoint,
        max_candidates=max_candidates,
        resume=resume,
        trace2skill=load_trace2skill_config(resolve("trace2skill_config")),
    )


def _pending_triage_records(
    round_dir: Path, *, allow_empty: bool = False
) -> list[dict[str, Any]]:
    if not round_dir.is_dir():
        if allow_empty:
            return []
        raise FileNotFoundError(f"training round directory not found: {round_dir}")
    paths = sorted(round_dir.glob("group-*/triage.json"))
    if not paths:
        if allow_empty:
            return []
        raise ValueError(f"training round contains no completed group triage: {round_dir}")
    records = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("classification") != "full_skill_failure_pending_analysis":
            continue
        retry = payload.get("full_skill_retry")
        if not isinstance(retry, dict) or not isinstance(retry.get("trace_path"), str):
            raise ValueError(f"pending triage has no retry trace path: {path}")
        records.append({**payload, "triage_path": str(path.resolve())})
    return records


def _render_candidate_content(candidate: Mapping[str, Any]) -> str:
    procedure = "\n".join(
        f"  {index}. {step}"
        for index, step in enumerate(candidate["procedure"], 1)
    )
    return (
        f"### {candidate['title']}\n\n"
        f"- Applies when: {candidate['applicable_when']}\n"
        f"- Procedure:\n{procedure}\n"
        f"- Verify: {candidate['verification']}\n"
        f"- Avoid: {candidate['common_failure']}"
    )


def _load_proposal_history(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"proposal ledger history not found: {path}")
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid proposal ledger JSON at {path}:{line_number}") from exc
        candidate_id = record.get("candidate_id") if isinstance(record, dict) else None
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"proposal ledger row has no candidate_id at {path}:{line_number}")
        if candidate_id not in latest:
            order.append(candidate_id)
        latest[candidate_id] = record
    return [latest[candidate_id] for candidate_id in order]


def _validate_consolidation(
    payload: dict[str, Any],
    *,
    cards: Sequence[dict[str, Any]],
    active_ids: set[str],
    max_candidates: int,
    historical_signatures: set[str],
) -> None:
    candidates = payload.get("candidates")
    evidence_only = payload.get("evidence_only_card_ids")
    if not isinstance(candidates, list) or not isinstance(evidence_only, list):
        raise StructuredOutputError("candidate output fields must be lists")
    if len(candidates) > max_candidates:
        raise StructuredOutputError("online compiler exceeded candidate budget")
    allowed_cards = {card["card_id"] for card in cards}
    accounted: set[str] = set()
    signatures: set[str] = set()
    for index, candidate in enumerate(candidates, 1):
        if not isinstance(candidate, dict):
            raise StructuredOutputError(f"candidate {index} must be an object")
        operation = candidate.get("operation")
        target = candidate.get("target_chunk_id")
        if operation not in {"ADD", "REWRITE"}:
            raise StructuredOutputError(f"candidate {index} has invalid operation")
        if operation == "ADD" and target not in {None, ""}:
            raise StructuredOutputError("ADD candidate cannot have a target")
        if operation == "REWRITE" and target not in active_ids:
            raise StructuredOutputError("REWRITE candidate has an inactive target")
        for field in (
            "title",
            "failure_mechanism",
            "applicable_when",
            "verification",
            "common_failure",
        ):
            if not isinstance(candidate.get(field), str) or not candidate[field].strip():
                raise StructuredOutputError(f"candidate {index} has empty {field}")
        procedure = candidate.get("procedure")
        evidence = candidate.get("evidence_card_ids")
        if (
            not isinstance(procedure, list)
            or not procedure
            or any(not isinstance(step, str) or not step.strip() for step in procedure)
        ):
            raise StructuredOutputError(f"candidate {index} has invalid procedure")
        if (
            not isinstance(evidence, list)
            or not evidence
            or any(card_id not in allowed_cards for card_id in evidence)
        ):
            raise StructuredOutputError(f"candidate {index} has invalid evidence IDs")
        overlap = accounted.intersection(evidence)
        if overlap:
            raise StructuredOutputError(f"cards cited by multiple candidates: {sorted(overlap)}")
        accounted.update(evidence)
        findings = firewall_findings(candidate)
        if findings:
            raise StructuredOutputError(f"candidate {index} failed gold firewall: {findings}")
        signature = fingerprint(
            {
                "operation": operation,
                "target_chunk_id": target,
                "content": _render_candidate_content(candidate),
            }
        )
        if signature in signatures:
            raise StructuredOutputError("online compiler emitted duplicate candidates")
        if signature in historical_signatures:
            raise StructuredOutputError("online compiler repeated a historical candidate")
        signatures.add(signature)
    if any(card_id not in allowed_cards for card_id in evidence_only):
        raise StructuredOutputError("evidence-only list cites an unknown card")
    if len(evidence_only) != len(set(evidence_only)):
        raise StructuredOutputError("evidence-only list contains duplicates")
    overlap = accounted.intersection(evidence_only)
    if overlap:
        raise StructuredOutputError(f"cards both cited and evidence-only: {sorted(overlap)}")
    missing = allowed_cards - accounted - set(evidence_only)
    if missing:
        raise StructuredOutputError(f"compiler left cards unaccounted: {sorted(missing)}")


def _consolidate(
    spec: OnlineAnalysisSpec,
    cards: Sequence[dict[str, Any]],
    current_skills: Sequence[dict[str, Any]],
    proposal_history: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not cards:
        return [], []
    model = OpenAICompatibleChatModel(spec.trace2skill.compiler_model)
    try:
        prompt = {
            "max_candidates": spec.max_candidates,
            "active_chunks": [
                {
                    "skill_id": record["skill_id"],
                    "content": record["content"],
                }
                for record in current_skills
            ],
            "failure_cards": [
                {
                    "card_id": card["card_id"],
                    "status": card["status"],
                    "deployable_abstraction": card["deployable_abstraction"],
                }
                for card in cards
            ],
            "proposal_history": [
                {
                    key: record.get(key)
                    for key in (
                        "candidate_id",
                        "operation",
                        "target_chunk_id",
                        "failure_mechanism",
                        "content",
                        "proposal_checkpoint",
                        "validation_result",
                        "estimated_effect",
                    )
                }
                for record in proposal_history
            ],
        }
        historical_signatures = {
            fingerprint(
                {
                    "operation": record.get("operation"),
                    "target_chunk_id": record.get("target_chunk_id"),
                    "content": record["content"],
                }
            )
            for record in proposal_history
            if record.get("operation") in {"ADD", "REWRITE"}
            and isinstance(record.get("content"), str)
            and record["content"].strip()
        }
        arguments = _structured_call(
            model,
            [
                {"role": "system", "content": ONLINE_CONSOLIDATION_PROMPT},
                {"role": "user", "content": _json(prompt)},
            ],
            tool=ONLINE_CONSOLIDATION_TOOL,
            expected_name="submit_online_candidates",
            seed=spec.trace2skill.seed + int(fingerprint(prompt)[:8], 16),
            attempts=5,
            validate=lambda value: _validate_consolidation(
                value,
                cards=cards,
                active_ids={record["skill_id"] for record in current_skills},
                max_candidates=spec.max_candidates,
                historical_signatures=historical_signatures,
            ),
        )
    finally:
        model.close()
    normalized = []
    for candidate in arguments["candidates"]:
        content = _render_candidate_content(candidate)
        candidate_id = "online-" + fingerprint(
            {
                "operation": candidate["operation"],
                "target_chunk_id": candidate.get("target_chunk_id"),
                "content": content,
            }
        )[:16]
        normalized.append(
            {
                "candidate_id": candidate_id,
                "operation": candidate["operation"],
                "target_chunk_id": candidate.get("target_chunk_id") or None,
                "content": content,
                "metadata": {
                    "title": candidate["title"],
                    "failure_mechanism": candidate["failure_mechanism"],
                    "evidence_card_ids": list(candidate["evidence_card_ids"]),
                    "consolidation_cluster_id": "cluster-" + candidate_id.removeprefix("online-"),
                    "proposal_checkpoint": spec.proposal_checkpoint,
                },
            }
        )
    return normalized, list(arguments["evidence_only_card_ids"])


@dataclass
class _OnlineCardPipeline:
    spec: OnlineAnalysisSpec
    bank: JsonSkillBank
    current_skills: list[dict[str, Any]]
    active_ids: set[str]
    card_store: JsonlIndex
    environment_config: ShopSimulatorConfig

    def load_retry_trace(self, record: dict[str, Any]) -> dict[str, Any]:
        retry_record = record["full_skill_retry"]
        trace_path = Path(retry_record["trace_path"])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        episode_id = trace.get("episode_id")
        if episode_id != retry_record.get("episode_id"):
            raise ValueError(f"retry episode mismatch in {trace_path}")
        final = trace.get("final")
        r_success = ((final or {}).get("reward_detail") or {}).get("r_success")
        skills_provenance = (trace.get("provenance") or {}).get("skills") or {}
        selected_skill_state = [
            (row.get("skill_id"), row.get("version"), row.get("content"))
            for row in trace.get("selected_skills") or []
            if isinstance(row, dict)
        ]
        expected_skill_state = [
            (
                skill_record["skill_id"],
                str(skill_record.get("version", "1")),
                skill_record["content"],
            )
            for skill_record in self.current_skills
        ]
        if (
            trace.get("status") != "completed"
            or (trace.get("job") or {}).get("split") != "train"
            or not isinstance(final, dict)
            or final.get("done") is not True
            or r_success not in {0, 0.0}
            or skills_provenance.get("diagnostic_full_skill") is not True
            or selected_skill_state != expected_skill_state
        ):
            raise ValueError(
                f"trace is not a failed retry under the configured full skill: {trace_path}"
            )
        return trace

    def input_fingerprint(self, record: dict[str, Any]) -> str:
        return fingerprint(
            {
                "pipeline": ONLINE_ANALYSIS_SCHEMA_VERSION,
                "trace": self.load_retry_trace(record),
                "active_skillbank_sha256": self.bank.bank_sha256,
                "analyst_model": self.spec.trace2skill.analyst_model.identity(),
                "max_failure_steps": self.spec.trace2skill.max_failure_analysis_steps,
            }
        )

    def episode_id(self, record: dict[str, Any]) -> str:
        episode_id = record["full_skill_retry"]["episode_id"]
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("pending triage has no retry episode_id")
        return episode_id

    def pending_records(
        self,
        *,
        allow_empty: bool,
        skip_ids: set[str] | None = None,
        retry_errors: bool = False,
    ) -> list[dict[str, Any]]:
        skip = skip_ids or set()
        records = []
        for record in _pending_triage_records(
            self.spec.training_round_dir, allow_empty=allow_empty
        ):
            episode_id = self.episode_id(record)
            if episode_id in skip:
                continue
            stored = self.card_store.latest.get(episode_id)
            if not self.spec.resume or stored is None:
                records.append(record)
                continue
            if stored.get("status") == "ANALYST_ERROR" and not retry_errors:
                continue
            try:
                digest = self.input_fingerprint(record)
            except Exception:
                records.append(record)
                continue
            if stored.get("input_fingerprint") != digest:
                records.append(record)
        return records

    def analyze(self, record: dict[str, Any]) -> dict[str, Any]:
        trace = self.load_retry_trace(record)
        card = analyze_failure_trace(
            trace,
            model_factory=OpenAICompatibleChatModel,
            environment_factory=lambda: ShopSimulatorHTTPEnvironment(
                self.environment_config
            ),
            spec=self.spec.trace2skill,
            current_skill=self.current_skills,
            allowed_rewrite_targets=self.active_ids,
        )
        card["input_fingerprint"] = self.input_fingerprint(record)
        return card

    def analyze_safe(self, record: dict[str, Any]) -> dict[str, Any]:
        episode_id = self.episode_id(record)
        try:
            return self.analyze(record)
        except Exception as exc:
            digest = None
            try:
                digest = self.input_fingerprint(record)
            except Exception:
                pass
            return {
                "source_trajectory_id": episode_id,
                "eligible_for_consolidation": False,
                "status": "ANALYST_ERROR",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "input_fingerprint": digest,
            }


def _online_card_pipeline(spec: OnlineAnalysisSpec) -> _OnlineCardPipeline:
    bank = JsonSkillBank(spec.current_skillbank_path, max_skills=None)
    current_skills = [
        record for record in bank.records if record.get("enabled", True) is not False
    ]
    return _OnlineCardPipeline(
        spec=spec,
        bank=bank,
        current_skills=current_skills,
        active_ids={record["skill_id"] for record in current_skills},
        card_store=JsonlIndex(spec.output_dir / "failure_cards.jsonl", "source_trajectory_id"),
        environment_config=ShopSimulatorConfig(
            base_url=spec.trace2skill.environment_base_url,
            persona=spec.trace2skill.environment_persona,
            timeout=spec.trace2skill.environment_timeout,
        ),
    )


def drain_training_failure_cards(
    spec: OnlineAnalysisSpec, *, allow_empty: bool = False
) -> dict[str, Any]:
    """Analyze pending full-skill retry traces. Does not run the compiler."""
    pipeline = _online_card_pipeline(spec)
    pending = pipeline.pending_records(
        allow_empty=allow_empty, retry_errors=True
    )
    if pending:
        workers = max(1, int(spec.trace2skill.concurrency))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(pipeline.analyze_safe, record) for record in pending]
            for future in as_completed(futures):
                pipeline.card_store.append(future.result())
    triage_records = _pending_triage_records(
        spec.training_round_dir, allow_empty=allow_empty
    )
    source_ids = {pipeline.episode_id(record) for record in triage_records}
    return {
        "triage_groups": len(triage_records),
        "newly_analyzed": len(pending),
        "analyzed_cards": len(
            [source_id for source_id in source_ids if source_id in pipeline.card_store.latest]
        ),
    }


def watch_training_failure_cards(
    spec: OnlineAnalysisSpec,
    *,
    poll_seconds: float = 5.0,
    stop_path: str | Path | None = None,
) -> dict[str, Any]:
    """Poll a live training round for pending retries. Does not run the compiler."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    pipeline = _online_card_pipeline(spec)
    stop = threading.Event()
    marker = Path(stop_path).resolve() if stop_path is not None else None

    def request_stop(_signum: int, _frame: Any) -> None:
        stop.set()

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    previous_int = signal.signal(signal.SIGINT, request_stop)
    in_flight: dict[str, Any] = {}
    newly = 0
    workers = max(1, int(spec.trace2skill.concurrency))
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:

            def submit_new() -> None:
                for record in pipeline.pending_records(
                    allow_empty=True, skip_ids=set(in_flight)
                ):
                    episode_id = pipeline.episode_id(record)
                    in_flight[episode_id] = executor.submit(
                        pipeline.analyze_safe, record
                    )

            def harvest(futures: Sequence[Any]) -> None:
                nonlocal newly
                done_ids = [
                    episode_id
                    for episode_id, future in list(in_flight.items())
                    if future in futures
                ]
                for episode_id in done_ids:
                    future = in_flight.pop(episode_id)
                    pipeline.card_store.append(future.result())
                    newly += 1

            while not stop.is_set() and (marker is None or not marker.is_file()):
                submit_new()
                if not in_flight:
                    stop.wait(poll_seconds)
                    continue
                done, _ = wait(
                    list(in_flight.values()),
                    timeout=poll_seconds,
                    return_when=FIRST_COMPLETED,
                )
                harvest(list(done))
            submit_new()
            if in_flight:
                harvest(list(as_completed(list(in_flight.values()))))
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
    triage_records = _pending_triage_records(spec.training_round_dir, allow_empty=True)
    source_ids = {pipeline.episode_id(record) for record in triage_records}
    return {
        "status": "watched",
        "triage_groups": len(triage_records),
        "newly_analyzed": newly,
        "analyzed_cards": len(
            [source_id for source_id in source_ids if source_id in pipeline.card_store.latest]
        ),
    }


def compile_training_failure_cards(spec: OnlineAnalysisSpec) -> dict[str, Any]:
    """Consolidate already-written eligible cards. Does not call the analyst."""
    pipeline = _online_card_pipeline(spec)
    proposal_history = _load_proposal_history(spec.proposal_ledger_history_path)
    triage_records = _pending_triage_records(spec.training_round_dir)
    source_ids = {pipeline.episode_id(record) for record in triage_records}
    cards = [
        card
        for source_id, card in sorted(pipeline.card_store.latest.items())
        if source_id in source_ids and card.get("eligible_for_consolidation") is True
    ]
    compilation_input_sha256 = fingerprint(
        {
            "cards": cards,
            "active_skillbank_sha256": pipeline.bank.bank_sha256,
            "compiler_model": spec.trace2skill.compiler_model.identity(),
            "max_candidates": spec.max_candidates,
            "proposal_checkpoint": spec.proposal_checkpoint,
            "proposal_history": proposal_history,
        }
    )
    summary_path = spec.output_dir / "analysis_summary.json"
    pool_path = spec.output_dir / "candidate_pool.json"
    if spec.resume and summary_path.is_file() and pool_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("compilation_input_sha256") == compilation_input_sha256:
            return previous
    candidates, evidence_only = _consolidate(
        spec, cards, pipeline.current_skills, proposal_history
    )
    pool = build_candidate_pool(
        spec.current_skillbank_path,
        candidates,
        round_id=spec.name,
        proposal_checkpoint=spec.proposal_checkpoint,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(pool_path, pool)
    ledger_by_id = {
        record["candidate_id"]: dict(record) for record in proposal_history
    }
    ledger_order = [record["candidate_id"] for record in proposal_history]
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        if candidate_id not in ledger_by_id:
            ledger_order.append(candidate_id)
        metadata = candidate["metadata"]
        ledger_by_id[candidate_id] = {
            "candidate_id": candidate_id,
            "operation": candidate["operation"],
            "target_chunk_id": candidate["target_chunk_id"],
            "source_card_ids": metadata["evidence_card_ids"],
            "failure_mechanism": metadata["failure_mechanism"],
            "proposal_checkpoint": spec.proposal_checkpoint,
            "stage": "online",
            "consolidation_cluster_id": metadata["consolidation_cluster_id"],
            "replacement_family_id": (
                candidate["target_chunk_id"]
                if candidate["operation"] == "REWRITE"
                else None
            ),
            "content": candidate["content"],
            "validation_result": None,
            "estimated_effect": None,
            "contribution_rank": None,
            "notes": None,
        }
    ledger_records = [ledger_by_id[candidate_id] for candidate_id in ledger_order]
    _atomic_write_jsonl(spec.output_dir / "proposal_ledger.jsonl", ledger_records)
    summary = {
        "schema_version": ONLINE_ANALYSIS_SCHEMA_VERSION,
        "created_at": utc_now(),
        "triage_groups": len(triage_records),
        "analyzed_cards": len(
            [source for source in source_ids if source in pipeline.card_store.latest]
        ),
        "eligible_cards": len(cards),
        "candidates": len(candidates),
        "proposal_history_records": len(proposal_history),
        "evidence_only_card_ids": evidence_only,
        "candidate_pool_path": str((spec.output_dir / "candidate_pool.json").resolve()),
        "current_skillbank_sha256": pipeline.bank.bank_sha256,
        "compilation_input_sha256": compilation_input_sha256,
    }
    atomic_write_json(summary_path, summary)
    return summary


def analyze_training_failures(spec: OnlineAnalysisSpec) -> dict[str, Any]:
    """Catch up any remaining cards, then compile. Used after a training round."""
    drain_training_failure_cards(spec, allow_empty=False)
    return compile_training_failure_cards(spec)


def write_analysis_failure_fallback(
    spec: OnlineAnalysisSpec, error: BaseException
) -> dict[str, Any]:
    """Keep the current SkillBank only so a failed Analyst cannot block val/gate."""
    bank = JsonSkillBank(spec.current_skillbank_path, max_skills=None)
    pool = build_candidate_pool(
        spec.current_skillbank_path,
        [],
        round_id=spec.name,
        proposal_checkpoint=spec.proposal_checkpoint,
    )
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = spec.output_dir / "candidate_pool.json"
    atomic_write_json(pool_path, pool)
    summary = {
        "schema_version": ONLINE_ANALYSIS_SCHEMA_VERSION,
        "created_at": utc_now(),
        "status": "failed",
        "submitted_candidates": False,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "triage_groups": 0,
        "analyzed_cards": 0,
        "eligible_cards": 0,
        "candidates": 0,
        "proposal_history_records": 0,
        "evidence_only_card_ids": [],
        "candidate_pool_path": str(pool_path.resolve()),
        "current_skillbank_sha256": bank.bank_sha256,
        "compilation_input_sha256": None,
    }
    atomic_write_json(spec.output_dir / "analysis_summary.json", summary)
    return summary
