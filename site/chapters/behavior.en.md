# How shopping behavior changes during training

Reward alone cannot distinguish shorter reasoning, better tool use, more persistent search, or a different mix of solved tasks. We extracted **20,480 training episodes / 110,997 model turns**, plus **3,600 completed evaluation episodes** from nine conditions. The analysis is retrospective and introduces no new model sampling.

## Measurement dictionary

| Metric | Definition |
|---|---|
| Model steps | Saved model turns, including unsuccessful protocol repairs |
| Actions / valid actions | Submitted actions / actions explicitly accepted by environment feedback |
| Tokens per turn | First divide each complete episode's generated tokens by model turns, then average episodes |
| Pooled tokens per turn | Total tokens divided by total turns; longer episodes have more weight |
| Training tokens | Re-encoded saved `raw_output`, including reasoning and serialized tool-call markup; not original sampled IDs |
| Evaluation tokens | Recorded API `completion_tokens`, including reasoning; no input tokens |
| Searches / unique queries | Valid executed search calls / whitespace-normalized, case-folded distinct queries |
| Product opens | Valid search-result → product-detail transitions |
| Unique products / revisits | Distinct opened product IDs / repeat opens of an already opened ID |
| Option selections / changes | Valid option clicks / switching an already selected axis to a different value |
| Backtracking / pagination | Return from detail or restart search / navigation between result pages |
| Protocol errors / invalid actions | Failed model-output parsing / explicit invalid environment feedback |

Selecting a color or size refreshes the same product page; it does not increment product opens. Products merely visible in search results are not counted as visited product pages. Tokens are available for every scored training episode and completed evaluation episode; technical episodes remain separately labeled. Failed-attempt request cost is not reconstructed here.

## Successful paths stay short; remaining failures have longer tails

{{behavior_rounds}}

Successful training episodes remain around 4.5–4.8 turns, with a median of four in every round. Failure means rise from 6.76 to 8.97 turns between R0 and R3, peaking at 10.28 in R2; their 90th percentiles are 13, 22, 22, and 18.2. The shrinking failure population is increasingly selective. This is descriptive evidence about residual failures, not proof that training makes a fixed failing task harder.

![Generation and interaction by outcome](../figures/behavior_outcomes.svg)

![Cumulative distribution of episode lengths](../figures/behavior_step_distribution.svg)

R0 includes 59 technically unscored episodes. They are excluded from success/failure means, not converted into reward-zero policy failures. The compared R0 and R3 generated-task sets have no overlapping task IDs, so a within-task training-round contrast cannot be estimated. We use fixed validation tasks for the longitudinal comparison below.

## Less generation is clearer than fewer interactions

On the same 400 bare-validation tasks, Base → Iter80 reduces average tokens per turn from **282.39 to 159.46 (−43.5%)** and tokens per episode from **1,704.93 to 1,021.50 (−40.1%)**. First-turn generation falls from 216.61 to 129.37 tokens, before the agent has executed its first search. This suggests a changed generation style as well as any difference in later navigation.

Average model steps fall from 6.01 to 5.56, but the paired interval spans zero. Search counts (1.298 → 1.258) and unique opened products (1.160 → 1.095) also lack clear nonzero paired differences. We therefore do not describe the overall change as conclusively “fewer searches” or “fewer steps.”

{{behavior_pairs}}

![Fixed validation across five checkpoints](../figures/behavior_fixed_validation.svg)

The 211 tasks successful at both Base and Iter80 provide a useful additional slice: generation falls by 92.32 tokens per turn, 95% interval [−102.08, −83.79], while step reduction remains uncertain. This slice is selected using both outcomes and is descriptive, not an unbiased causal subgroup. All intervals use 5,000 task-paired percentile bootstrap resamples and quantify task variation conditional on this run. The exploratory metric family has no multiple-comparison adjustment.

![Paired changes with uncertainty intervals](../figures/behavior_paired_effects.svg)

## Better environment validity, more parsing trouble

Invalid environment actions fall from 0.4025 to 0.0875 per fixed validation episode; protocol parsing errors rise from 0.1475 to 0.3475. These are distinct failure surfaces. The former asks whether a submitted action is valid in the environment; the latter asks whether a model response can be interpreted as the required tool call. A policy can improve one while worsening the other.

The first opened product's recorded result rank moves from 4.96 to 1.30 on the 399 tasks with an observed first open in both conditions. Pagination falls from 0.245 to 0.0175 per task. Together these show a stronger tendency toward early-ranked results. They do not by themselves establish better query semantics or prove that fewer comparisons are desirable; relevance and task outcomes must be considered jointly.

## Where generation changes

We classify each turn by the page state **before** its action: search home, search results, product detail, or protocol repair. This separates initial planning, choosing a result, checking variants, and repairing format. Turn-weighted means and 90th percentiles reveal long outputs that an episode mean can hide.

![Token lengths by page phase](../figures/behavior_phases.svg)

Training generation is not monotonic: episode-average tokens per turn are 277.30, 255.02, 190.09, and 221.26 across rounds. Skill content and sampled tasks change at round boundaries. These curves cannot isolate a parameter-only effect or directly share an absolute token scale with the API-based evaluation series.

## Trace examples with explicit selection rules

The examples below are selected after analysis to illustrate mechanisms. They are not a random sample or evidence of prevalence. We include both improvements and a regression. Each path uses action categories rather than full private task context; ties are resolved by lower task ID.

{{behavior_cases}}

## Download and extend

The [episode table](../data/behavior/train_episodes.csv.gz), [turn table](../data/behavior/train_turns.csv.gz), [evaluation episodes](../data/behavior/eval_episodes.csv.gz), and [evaluation turns](../data/behavior/eval_turns.csv.gz) support additional slicing. The [summary JSON](../data/behavior/behavior.json) includes all 80 steps, means/medians/p90, success/failure distributions, skill-free/assisted round summaries, phase statistics, and pairing metadata.

Assistance comparisons within training are observational: q, task assignment, selection, and current skill content affect them. They are not a randomized estimate of the effect of adding the final skill. The [test results](results.html) and [W&B diagnostics](optimization.html) complement these behavior measurements.
