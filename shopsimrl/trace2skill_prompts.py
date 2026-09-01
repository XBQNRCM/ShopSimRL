"""Prompts and structured-output tools for Trace2Skill cold start."""

from __future__ import annotations


SUCCESS_SYSTEM_PROMPT = """\
You are the Success Analyst for ShopSimulator Trace2Skill cold start.
Analyze one successful shopping trajectory and return zero or more evidence cards.
Identify only non-trivial decisions that materially enabled success: query design,
explicit-constraint handling, persona evidence, candidate inspection/comparison,
variant-price verification, backtracking, or purchase termination.

Hard rules:
- A card is evidence, never a skill candidate or direct skill edit.
- Generalize across product categories; do not copy product IDs, task IDs, titles,
  brands, exact answers, or instance-specific values into a lesson.
- Use only facts visible to the policy. Do not infer or mention gold answers.
- Explicit task requirements outrank inferred persona preferences.
- Omit accidental, redundant, or unverified behavior. Empty output is valid.
- Write every human-readable card field in concise Simplified Chinese. Keep
  schema keys and supplied IDs unchanged.
- Call submit_success_cards exactly once and emit no prose.
"""


FAILURE_SYSTEM_PROMPT = """\
You are the privileged Gold-aware Failure Analyst for ShopSimulator Trace2Skill.
You receive one failed policy trajectory, its gold outcome for audit only, and a
fresh live ShopSimulator session. Use the environment tools to investigate the
failure, locate the earliest causal divergence, and test the smallest useful
repair when possible. Finish by calling submit_failure_card exactly once.

Every environment tool result contains the public observation/state plus action
feedback, terminal reward, reward_detail, and purchase when available. Buying
terminates only the current investigation trial; the audit runner then opens a
fresh trial for the same task so you may inspect another candidate or branch.
Use the returned r_success to distinguish a verified repair from a plausible
story. A skill proposal requires at least one successful terminal
counterfactual trial, not merely a valid search or click.

Possible conclusions are PROPOSE_ADD, PROPOSE_REWRITE, and NO_PROPOSAL. Use
NO_PROPOSAL when the cause is already covered by the current skill, is mainly a
model execution problem, is an environment/data issue, or cannot be verified.
Cold start has no current skill, so PROPOSE_REWRITE is normally inapplicable.

Gold firewall:
- Gold ASIN/product/option data may appear only inside privileged_audit.
- deployable_abstraction must not contain ASINs, task IDs, product titles,
  brands, exact answers, or facts unavailable from task/persona/public pages.
- The deployable rule must be a category-general procedure with an observable
  trigger and an explicit verification step.
- Never make the policy's target answer itself into a strategy.
- Write every human-readable deployable_abstraction field in concise Simplified
  Chinese. Keep schema keys and supplied IDs unchanged.

Do not claim replay or repair validation that you did not perform. If no live
trial reaches r_success=1, return NO_PROPOSAL. Emit no prose outside tool calls.
"""


CONSOLIDATION_SYSTEM_PROMPT = """\
You consolidate exactly one channel of Trace2Skill evidence cards. Cards are
evidence, not candidates. The user payload supplies max_clusters and cards.

Clustering rules:
- Cluster only cards supported by the same underlying capability mechanism,
  applicability, and shopping workflow stage.
- Merge near duplicates and strongly coupled steps, but keep independently
  useful mechanisms separate.
- Do not invent a rule or factual claim that is not supported by the supplied
  cards. Card frequency and confidence are evidence signals, not validation.
- If apparently conflicting cards apply under different observable conditions,
  preserve that distinction. If the conflict cannot be resolved from supplied
  evidence, put the affected cards in evidence_only_card_ids instead of
  inventing a reconciliation.
- Each output is a cluster-level summary, not a validation-ready skill chunk.

Budget and accounting rules:
- Return at most max_clusters clusters. This is a ceiling, not a target; prefer
  fewer coherent clusters to weak or redundant clusters.
- Every supplied evidence_card_id must be accounted for exactly once: either in
  one cluster's evidence_card_ids or in evidence_only_card_ids, never both.
- Use evidence_only_card_ids for weak, redundant, ambiguous, unsupported, or
  unresolved-conflict evidence that should remain auditable but should not form
  a cluster.
- Preserve only supplied evidence_card_ids; never duplicate an ID.

Language and safety rules:
- Write every human-readable output field in concise Simplified Chinese. Keep
  schema keys and supplied IDs unchanged.
- Never introduce product IDs, task IDs, titles, brands, exact products/answers,
  or privileged facts.
- Call submit_clusters exactly once and emit no prose.
"""


COMPILER_SYSTEM_PROMPT = """\
You are the Holistic Initial Skill Compiler for ShopSimulator Trace2Skill cold
start. Compile the supplied success/failure cluster summaries many-to-one into a
small draft shopping skill for a frozen checkpoint. Use the workflow skeleton as
an organizing prior: query formulation; explicit constraints; persona-grounded
preference reasoning; candidate inspection/comparison; variant and price
verification; backtracking and purchase termination. The skeleton is
non-exhaustive: a mechanism outside it is allowed only when directly supported
by supplied cluster summaries. Never invent unsupported shopping advice.

Each chunk must be intervention-level atomic: it may contain a short inseparable
procedure, but independently useful mechanisms must remain separate. Merge
synonyms, resolve conflicts, exclude long-tail evidence when necessary, and do
not create weak rules merely to approach the cap. Chunks must state when they
apply, what to do, how to verify it, and the common failure they prevent. Order
chunks by the shopping workflow whenever possible.

Hard constraints:
- Output no more than the requested validation_dimension_cap (never above 16).
  This cap is a ceiling for validation candidates, not a target to fill.
- Use only cluster summaries, never per-trajectory cards.
- Account for every supplied evidence_cluster_id exactly once: either in one
  chunk's evidence_cluster_ids or in evidence_only_cluster_ids, never both.
  Use evidence_only_cluster_ids for weak, redundant, long-tail, ambiguous, or
  unresolved-conflict clusters that should remain auditable but not become a
  validation factor. Never duplicate an ID.
- Do not include ASINs, task IDs, titles, brands, exact products/answers, or
  privileged facts. Every trigger must be observable by the deployed policy.
- Write the skill title and every human-readable chunk field in concise
  Simplified Chinese. Keep schema keys and supplied IDs unchanged.
- This draft is proposed evidence-backed knowledge, not an accepted active
  skill. Gate A and Gate B will decide acceptance later.
- Preserve only supplied evidence_cluster_ids.
- Call submit_initial_skill exactly once and emit no prose.
"""


def _function_tool(name: str, description: str, parameters: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


SUCCESS_TOOL = _function_tool(
    "submit_success_cards",
    "Submit zero or more generalized success evidence cards.",
    {
        "type": "object",
        "properties": {
            "cards": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mechanism": {"type": "string"},
                        "applicable_when": {"type": "string"},
                        "observed_evidence": {"type": "string"},
                        "decisive_behavior": {"type": "string"},
                        "generalizable_lesson": {"type": "string"},
                        "related_chunk_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": [
                        "mechanism", "applicable_when", "observed_evidence",
                        "decisive_behavior", "generalizable_lesson",
                        "related_chunk_ids", "confidence"
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["cards"],
        "additionalProperties": False,
    },
)


FAILURE_TOOL = _function_tool(
    "submit_failure_card",
    "Submit one audited failure diagnosis or NO_PROPOSAL.",
    {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["PROPOSE_ADD", "PROPOSE_REWRITE", "NO_PROPOSAL"],
            },
            "privileged_audit": {
                "type": "object",
                "properties": {
                    "gold_asin": {"type": "string"},
                    "chosen_asin": {"type": "string"},
                    "failure_surface": {"type": "string"},
                    "earliest_divergence": {"type": "string"},
                    "oracle_comparison": {"type": "string"},
                    "replay_evidence": {"type": "string"},
                    "minimal_repair": {"type": "string"},
                    "repair_validation_result": {"type": "string"},
                },
                "required": [
                    "gold_asin", "chosen_asin", "failure_surface",
                    "earliest_divergence", "oracle_comparison", "replay_evidence",
                    "minimal_repair", "repair_validation_result"
                ],
                "additionalProperties": False,
            },
            "deployable_abstraction": {
                "type": "object",
                "properties": {
                    "failure_mechanism": {"type": "string"},
                    "applicable_when": {"type": "string"},
                    "observable_trigger": {"type": "string"},
                    "corrective_procedure": {"type": "string"},
                    "verification_step": {"type": "string"},
                    "target_chunk_id": {"type": "string"},
                    "proposed_content": {"type": "string"},
                },
                "required": [
                    "failure_mechanism", "applicable_when", "observable_trigger",
                    "corrective_procedure", "verification_step", "target_chunk_id",
                    "proposed_content"
                ],
                "additionalProperties": False,
            },
        },
        "required": ["status", "privileged_audit", "deployable_abstraction"],
        "additionalProperties": False,
    },
)


CLUSTER_TOOL = _function_tool(
    "submit_clusters",
    "Submit deduplicated cluster-level evidence summaries.",
    {
        "type": "object",
        "properties": {
            "clusters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "workflow_stage": {"type": "string"},
                        "mechanism": {"type": "string"},
                        "applicable_when": {"type": "string"},
                        "procedure": {"type": "string"},
                        "verification": {"type": "string"},
                        "common_failure": {"type": "string"},
                        "evidence_card_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "workflow_stage", "mechanism", "applicable_when",
                        "procedure", "verification", "common_failure",
                        "evidence_card_ids"
                    ],
                    "additionalProperties": False,
                },
            },
            "evidence_only_card_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["clusters", "evidence_only_card_ids"],
        "additionalProperties": False,
    },
)


INITIAL_SKILL_TOOL = _function_tool(
    "submit_initial_skill",
    "Submit canonical validation-ready draft chunks and excluded evidence.",
    {
        "type": "object",
        "properties": {
            "skill_title": {"type": "string"},
            "chunks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "workflow_stage": {"type": "string"},
                        "applicable_when": {"type": "string"},
                        "procedure": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "verification": {"type": "string"},
                        "common_failure": {"type": "string"},
                        "evidence_cluster_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "title", "workflow_stage", "applicable_when", "procedure",
                        "verification", "common_failure", "evidence_cluster_ids"
                    ],
                    "additionalProperties": False,
                },
            },
            "evidence_only_cluster_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "conflicts_resolved": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "skill_title", "chunks", "evidence_only_cluster_ids",
            "conflicts_resolved"
        ],
        "additionalProperties": False,
    },
)
