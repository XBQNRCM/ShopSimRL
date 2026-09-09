# Method: what co-evolves, and why

IntraSkill investigates whether a shopping policy and its external procedural guidance can improve together. The policy learns through GRPO; a separate loop investigates failures, proposes reusable guidance, measures its usefulness at the current checkpoint, and revises the active skill. The released experiment tests this complete system in one training run. It does not isolate its advantage over ordinary RL.

## One task family, several procedural chunks

The task family is personalized shopping: interpret a request and persona, search the catalog, inspect products, choose variants, and purchase. Shoes and electronics are surface variations within this workflow, rather than separate skills by definition. A **chunk** is a unit that can be independently included or withheld: query formulation, evidence-based persona reasoning, candidate verification, option selection, or a final price check.

The active skill is a structured collection of these chunks. It should contain transferable procedures, with conditions for applying them and known failure modes. A product identifier or a private target answer is not a reusable procedure. This distinction matters because a successful diagnostic retry can reveal either a general strategy or a task-specific shortcut.

## The residual competence frontier

Model parameters encode learned behavior. An external chunk may still help where the current checkpoint has not reliably acquired a procedure. We call this checkpoint-relative usefulness the *residual competence frontier*. A chunk can become less useful as the policy improves, or become useful again under a different failure pattern. It is not assigned a permanent quality score.

This is a research interpretation, not a direct measurement of internal knowledge. A near-zero regression coefficient can also arise from noise, interactions, redundant guidance, or insufficient statistical power. A shrinking skill bank alone does not demonstrate internalization. In fact, this run's active sizes were 10 → 7 → 5 → 8 → 8.

## Fast loop: group-consistent assistance

Each GRPO group contains eight trajectories for one task. A group first draws a skill-free decision with probability **q = 0.20**. Otherwise, each active chunk is sampled using its inclusion probability rho. The complete mask is shared by all eight siblings. This keeps their comparison conditional on the same guidance; giving different siblings different chunks would mix treatment differences with policy variation in the group advantage.

The released run keeps q fixed. Within assisted groups, positive contributions are mapped linearly into rho in [0.20, 0.90], with the largest positive contribution reaching the upper bound. These values are frozen in a hashed curriculum between gates. An assisted draw may include no chunks; the runtime records this separately from a skill-free draw.

```text
task + frozen curriculum
  └─ one group mask
       ├─ trajectory 1 ─ reward 1
       ├─ ...
       └─ trajectory 8 ─ reward 8
             └─ group-relative advantage → policy update
```

The runtime preserves actual generated token IDs and rollout log probabilities for training. Prompt tokens, environment observations, and tool results have zero loss mask; generated tokens carry the loss. A changing tool schema can split an episode into multiple contiguous token segments. Reward normalization first deduplicates by original rollout identity so a longer, segmented episode does not count several times in its group's mean.

### Dynamic filtering and its denominator

The full-training launcher generates 32 groups per rollout step and admits 16 groups, or 128 episodes. A global batch of 64 yields two optimizer updates per rollout step. Across 80 steps this gives 20,480 generated episodes and 10,240 admitted episodes.

Groups with technical failures are excluded and replenished. Zero-variance groups are deprioritized, but a fallback can retain them to fill the batch. Consequently, the logged zero-variance **drop rate** is a filtering statistic; it is not the fraction of all generated groups with identical rewards. Behavior analysis covers generated, scored episodes and does not claim to reconstruct the exact admitted sample distribution.

## Slow loop: investigate failures before writing guidance

Only an all-wrong group whose eight trajectories are normally scored triggers an extra full-current-skill retry. That diagnostic episode is excluded from GRPO. A successful retry suggests an internalization deficit: existing guidance can solve a case the masked group missed. A failed retry is only a request for further analysis, not automatic proof that the skill lacks a procedure.

Trace2Skill analyzes failed diagnostic trajectories. An analyst can inspect privileged diagnostic information, but deployable guidance must be abstracted from public observations and reusable actions. Private target facts, identifiers, and gold answers stay out of the policy's skill text. An ADD proposal needs a real successful counterfactual trial; REWRITE must target a currently active chunk. The analyst may return NO_PROPOSAL.

At round end, a compiler merges eligible failure cards into at most six ADD/REWRITE candidates, deduplicating mechanisms. Many cards can support one candidate. The proposal ledger records their history but is not injected into the policy prompt. Online candidate counts were 6, 6, 6, and 4; eligible cards were 61, 22, 20, and 11.

## Paired randomized validation

Every 20 rollout steps, the checkpoint is frozen. The generic evaluator runs all 400 validation tasks without skills. A separate gate evaluates the same tasks at that checkpoint with randomized masks over current chunks and proposed candidates. Task, sample, model identity, decoding settings, prompts, environment, and action protocol must match the pairing contract.

For task i, Bᵢ is its bare reward, Rᵢ its masked reward, and Cᵢ its actual version-inclusion vector. The gate fits:

```text
delta_i = R_i − B_i
beta_hat = argmin_beta Σ_i (delta_i − C_iᵀ beta)²
```

There is no fitted intercept, no centered treatment matrix, and no subtraction of a global bare mean. An all-zero mask remains a valid observation: two stochastic bare-equivalent rollouts can still have different rewards. The primary outcome is strict reward; each complete reward component receives its own paired-delta fit.

For an ordinary chunk slot, absent/present are equiprobable. A rewrite family uses mutually exclusive absent/old/new states: two competing versions never coexist in the prompt. First, the highest-coefficient version wins within each family, with ties favoring the old version. Then global strictly-positive top-K selection, K ≤ 10, determines the active bank; the bank is not padded with nonpositive chunks. The selected bank determines the next curriculum.

### What the gate establishes

Recomputing the four saved observation tables reproduces their coefficients within 2 × 10⁻¹⁶. That confirms numerical consistency. It does not establish an interaction-free causal model, the correctness of the served checkpoint, or reliable ranking on unseen tasks.

The masked validation mean describes a randomized current-plus-candidate pool. It is **not** the reward of the final selected full skill. The fixed validation split is reused for selection across rounds, so it cannot serve as an untouched final test. Pairing reduces task-difficulty variation only to the extent that the noisy bare and masked outcomes covary; it does not guarantee lower variance.

## What this run answers, and what remains open

The trained bare policy substantially outperforms its initial checkpoint on the frozen test set. Its generation behavior also changes on fixed validation tasks. Those observations are compatible with learning useful procedures. The experiment lacks plain-GRPO, fixed-skill, no-mask, no-online-gate, alternative-q, and multiple-training-seed controls. They are needed to attribute gains specifically to co-evolution and to test the proposed internalization mechanism.

Continue with [experimental protocol](protocol.html), [behavior analysis](behavior.html), or the original [method proposal](../reference/docs/proposal.html), [training contract](../reference/docs/training.html), and [paired gate specification](../reference/docs/paired_validation.html).
