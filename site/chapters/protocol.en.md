# Experimental protocol and runtime

This report covers the frozen Qwen3.5-4B Iter80 experiment. It distinguishes the research proposal, the implemented contracts, and the settings actually used by this run. The local environment is included as an ordinary `ShopSimulator/` directory.

## Task, observation, and reward

The setting is ShopSimulator single-turn personalized shopping. A task supplies a purchase request and persona context. The agent sees public shopping observations and the tools available at its current page, then searches, opens products, selects options, and buys. The frozen persona split contains **3,726 train, 400 validation, and 400 test tasks**.

The run uses environment version `shopsimulator-paper-aligned-v8`, observation contract `shopping-observation-v7`, and action protocol `openai-function-tools-repair-v3`. Strict reward (`r_strict`, also exposed as `reward`) is the optimization target. Success means full credit, not merely reaching a terminal purchase page. Attribute, option, type, price, and personalization components help diagnose partial matches; their exact definitions remain in the [environment contract](../reference/ShopSimulator/docs/ENVIRONMENT.html).

A model turn can fail to produce a valid tool call. The runtime supplies protocol feedback and may ask for repair without executing an environment action. An executed but invalid environment action is a separate event. A purchase can terminate successfully at the protocol level while failing the task's reward criteria.

## Frozen settings

| Item | Released run |
|---|---|
| Policy | Qwen3.5-4B; final evaluation identity `qwen35-4b-iter80` |
| Schedule | Four rounds × 20 rollout steps |
| Siblings | Eight trajectories per group; shared mask |
| Generated / admitted | 32 / 16 groups per rollout step |
| Global batch | 64 episodes; two optimizer updates per rollout step |
| Learning rate | Constant 1 × 10⁻⁶ in both optimizer parameter groups, verified across all W&B updates |
| Guidance | q fixed at 0.20; contribution-mapped rho in [0.20, 0.90] |
| Skill budget | At most 10 active chunks; at most 6 online candidates per round |
| Gate | 400 bare + 400 randomly masked trajectories at each checkpoint |
| Sampling | Temperature 0.6, top_p 1.0, thinking enabled |
| Limits | 4,096 generated tokens per turn; 30 model steps per episode |
| Seeds | `send_seed=false`; matching task seed is not shared decoder randomness |

The launcher defaults to four GPUs, trainer tensor parallelism 2, and rollout tensor parallelism 1. These are execution settings rather than claims of hardware-independent cost. Training uses slime and SGLang extension points; the policy runtime and general evaluator share the shopping contract. Exact current commands and topology constraints are in the [training guide](../reference/docs/training.html).

## Three kinds of evaluation

**Training generation** follows the current policy, sampled training tasks, and changing assistance masks. It includes groups subsequently filtered out. It is useful for diagnosing training behavior but is not an independent held-out performance estimate.

**Validation** uses the fixed 400 tasks at the base, Iter20, Iter40, Iter60, and Iter80 checkpoints. Bare validation supports longitudinal behavior comparisons. Random-mask validation at the four trained checkpoints estimates chunk contributions and selects the next bank. These repeated selection uses matter when interpreting validation gains.

**Final test** compares Base free, Base + initial S₀, Iter80 free, and Iter80 + final Sₜ on the same 400 tasks, one completed rollout per task and condition. All four conditions have full coverage. Base free also contains 21 failed attempts that were subsequently retried; they are not extra test samples. Duplicate completed records are rejected, rather than silently taking the latest.

## Evidence boundaries

The public analysis covers exactly four completed rounds, rollout steps 1–80. Group summaries distinguish 20,480 generated episodes from 10,240 trainer-admitted episodes. Diagnostic full-skill retries are excluded from those training sample counts.

The initial S₀ was recovered from historical frozen context, with `uses_test_outcomes=false` and `validation_refitted=false`. It is not represented as a freshly rerun Gate A satisfying every current provenance check. Online gates retain their saved paired observation tables, selected banks, and curriculum states.

For behavioral tokens, training traces save output text but not the original sampled token IDs. We re-encode that text with the official Qwen tokenizer and record its SHA-256, while marking the match to the historical tokenizer hash as unverified. Evaluation traces instead contain API `completion_tokens`. These two sources are labeled separately; neither is treated as a substitute for the exact token stream used by the optimizer.

## Data and source traceability

Each analysis extract records input hashes. Public numeric episode and turn tables omit persona text, full conversations, and catalog bodies. Selected examples contain action categories and costs with an explicit selection rule. Rebuilding aggregate figures requires no GPU, model endpoint, W&B key, or raw trace directory. Refreshing the raw extraction requires the local experiment traces and tokenizer; refreshing W&B requires read access to those runs.

See [results](results.html), [behavior](behavior.html), [reproduction](reproduce.html), and the [original provenance notes](../reference/docs/provenance.html).
