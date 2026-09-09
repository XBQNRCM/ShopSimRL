# IntraSkill

**Skill–Model Co-evolution for Shopping Agents** · ShopSimRL

[English](README.md) · [简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Experiment report](docs/experiment_report.md)

IntraSkill studies how an agent's policy and its external procedural guidance can evolve together. Built on ShopSimulator and slime, this project trains Qwen3.5-4B to search, inspect product variants, satisfy persona-grounded preferences, and purchase through function tools.

The released run improves **skill-free test success from 51.75% to 84.25%**. With its final skill, the trained model reaches **86.25%**. These are results from one training run on the project's local protocol, not a comparison against the upstream benchmark leaderboard.

![Four-condition test results](artifacts/analysis/figures/test_results.png)

## How it works

1. **Initialize a shopping skill.** Trace2Skill distills training trajectories into independently maskable procedural chunks.
2. **Train with shared masks.** Each GRPO group contains eight trajectories under one shared context. A fixed probability `q = 0.20` removes all skills; assisted groups sample chunks using contribution-dependent inclusion probabilities.
3. **Investigate the failure frontier.** All-wrong groups receive a diagnostic full-skill retry. Remaining failures feed the analyst and compiler, which propose `ADD` or `REWRITE` candidates.
4. **Validate every 20 rollout steps.** The frozen checkpoint runs bare and randomized-mask validation on the same 400 tasks. The gate fits `R_i − B_i = C_iᵀβ + ε_i` without an intercept, resolves rewrite families, and retains positive-coefficient winners up to the chunk budget.

Model updates compare trajectories within a fixed context; skill updates compare interventions at a frozen checkpoint. See the [method proposal](docs/proposal.md) and [paired-validation contract](docs/paired_validation.md).

## Released experiment

| Model | Skill context | Strict reward | Success |
|---|---|---:|---:|
| Base | None | 0.5422 | 51.75% |
| Base | Initial S₀ | 0.5663 | 54.50% |
| Iter80 | None | 0.8501 | 84.25% |
| Iter80 | Final Sₜ | 0.8671 | 86.25% |

All four conditions complete the same **400 test tasks**, one rollout per task. The persona split is **3726 train / 400 validation / 400 test**. The release comprises **80 rollout steps in four rounds**, with **20,480 generated episodes** and **10,240 episodes admitted to the trainer**. With global batch size 64, each 128-episode rollout batch yields two optimizer updates; rollout steps and optimizer updates are different units.

![Training dynamics](artifacts/analysis/figures/training.png)

The active skill changes from **10 → 7 → 5 → 8 → 8 chunks**. Skill-free validation success increases from **74.75% at step 20 to 86.00% at step 80**. The contribution gate uses validation, while the four-condition table uses test.

**What this establishes:** the trained checkpoint performs substantially better without external skills. The paired 95% confidence interval for its success gain is **+27.50 to +37.50 percentage points**.

**What remains open:** the final skill's additional success gain is **+2.00 points**, with a paired 95% interval of **−0.50 to +4.75**. There is no plain-GRPO, fixed-skill, no-mask, or repeated-training-seed control in this release. Contribution ranking is also inconclusive: the base-model appendix has top4 / bottom4 / one random4 success of 51.75% / 55.00% / 54.25%. These results do not isolate the causal benefit of co-evolution over ordinary RL.

## Behavior and optimization

The analysis covers **110,997 saved training turns**, 20,480 generated episodes, and 3,600 completed evaluations. On the fixed 400-task bare validation set, Base → Iter80 reduces average generation from **282.39 to 159.46 tokens per turn (−43.5%)**. Model steps change from 6.01 to 5.56, but their paired interval spans zero. Successful training paths stay around 4.5–4.8 steps; remaining failures are longer and search more often. Invalid environment actions decrease while protocol parsing errors increase.

Four complete W&B histories add **160 optimizer updates**. Their 3,118 overlapping numerical values match local logs exactly. The report separates token entropy, PPO diagnostics, gradient norms, admitted Sample length, and throughput from episode-level behavior.

Read the [behavior analysis](docs/behavior_analysis.md) and [optimization analysis](docs/optimization_analysis.md). The bilingual project site contains six full research chapters, an interactive behavior explorer, all skill versions, a searchable source-document library, and downloadable episode/turn data.

## Reproduce the analysis — no GPU required

Python 3.10 or later:

```bash
python -m pip install -e ".[analysis,site]"
python scripts/analyze_training.py
python scripts/analyze_behavior.py
python scripts/analyze_wandb.py
python scripts/plot_training.py
python scripts/plot_behavior.py
python scripts/plot_wandb.py
python scripts/build_site.py
python scripts/check_publication.py
python -m http.server 8000 --bind 127.0.0.1 --directory _site
```

Open [the local project page](http://localhost:8000). It includes English/Chinese text, an interactive training chart, a skill-version explorer, and downloadable data. The static build is compatible with GitHub Pages; see [publishing instructions](docs/project_page.md).

The analysis rebuilds from the committed [numerical extracts](artifacts/analysis/README.md). It checks test pairing and refits all four published gates. To refresh these extracts from a complete local `runs/`, use `python scripts/analyze_training.py --refresh-local`. Full traces, model weights, W&B logs, and credentials are not required to view or rebuild the public analysis.

## Run the agent or train a model

1. Prepare and start the bundled environment using the [ShopSimulator setup guide](ShopSimulator/README.md). `ShopSimulator/` is a regular directory, not a submodule.
2. Serve the appropriate base or trained checkpoint on the endpoint configured in the evaluation YAML. The checkpoint ID records provenance; it does **not** load or verify the served weights.
3. Inspect a plan, then run an evaluation:

```bash
python scripts/run_shopsimrl.py plan configs/qwen35_4b_test.yaml
python scripts/run_shopsimrl.py run configs/qwen35_4b_test.yaml
```

| Condition | Configuration |
|---|---|
| Base, no skill | [`qwen35_4b_test.yaml`](configs/qwen35_4b_test.yaml) |
| Base + S₀ | [`qwen35_4b_test_trace2skill_equipped.yaml`](configs/qwen35_4b_test_trace2skill_equipped.yaml) |
| Iter80, no skill | [`qwen35_4b_test_final_free.yaml`](configs/qwen35_4b_test_final_free.yaml) |
| Iter80 + Sₜ | [`qwen35_4b_test_final_pair.yaml`](configs/qwen35_4b_test_final_pair.yaml) |

Use a fresh experiment name/output directory for a new run; do not overwrite frozen runs or reuse manifests after changing weights. GPU training additionally requires the slime/SGLang/Megatron stack, model weights, and an analyst API. Follow the [training guide](docs/training.md), not the lightweight analysis installation above.

## Repository map

| Path | Purpose |
|---|---|
| `shopsimrl/` | Agent runtime, evaluation, curriculum, training adapter, Trace2Skill |
| `ShopSimulator/` | Bundled local environment, frozen splits, corpus preparation, replay UI |
| `slime/` | Training framework checkout |
| `configs/`, `scripts/` | Experiment entry points and analysis/build commands |
| `artifacts/` | Frozen skill banks, evaluation excerpts, reproducible analysis and figures |
| `site/` | Bilingual static project-page source |
| `docs/`, `tests/` | Research/engineering documentation and regression tests |
| `runs/`, `data/` | Local full experiments and generated data; ignored by Git |

## Provenance and acknowledgements

This project builds on [ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator), [slime](https://github.com/THUDM/slime), and Qwen3.5, and uses a Trace2Skill-style analysis pipeline. The local environment differs from upstream; its reward and observation contract is documented in [ENVIRONMENT.md](ShopSimulator/docs/ENVIRONMENT.md).

The released S₀ was restored from frozen recorded skill contexts without using test outcomes for selection. Its [recovery manifest](artifacts/cold-start/recovery_manifest.json) is not a newly completed calibration run. Historical artifacts remain unchanged. Read [provenance and reuse notes](docs/provenance.md) before interpreting or redistributing the bundle.
