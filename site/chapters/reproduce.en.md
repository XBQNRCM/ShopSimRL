# Reproduce, inspect, and extend

The site is a static research report with downloadable numerical evidence. Its primary pages work without access to the source repository. Source documents are rendered in the [reference library](../library.html); original Chinese engineering documents are labeled as such rather than silently machine-translated.

## Rebuild the frozen analysis

From the repository root, with Python 3.10 or later:

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

Open `http://127.0.0.1:8000`. Default aggregation reads committed numerical extracts and frozen gate artifacts; it does not query models or W&B. Behavior bootstrap uses 5,000 task resamples, while primary test intervals use the existing comparison implementation with 10,000. Each records its random-number convention.

## Refresh source measurements

`analyze_training.py --refresh-local` reads local training logs, triage files, and evaluation traces. `analyze_behavior.py --refresh-local --tokenizer /path/to/tokenizer.json` additionally parses all saved turns and re-encodes training outputs. This requires the ignored `runs/` directory and the appropriate tokenizer; it is separate from ordinary reproduction.

To refresh the four authorized W&B histories, install the optional `wandb` extra and run `python scripts/fetch_wandb.py`. It reads `WANDB_API_KEY` from the environment or project `.env` only in memory. The script exports numeric history and minimal source metadata; use `scan_history()` to avoid the default sampling of `history()`.

## Run the project

The repository contains the shopping environment, policy runtime, evaluation CLI, training adapters, Trace2Skill pipeline, configs, and frozen artifacts. Data corpora, model checkpoints, large raw traces, local service credentials, and machine caches are not included in the site bundle.

Start with the [environment guide](../reference/ShopSimulator/README.html), then [runtime and evaluation](../reference/docs/runtime_eval.html). The [training guide](../reference/docs/training.html) covers data preparation, frozen curriculum checks, GPU topology, slime launching, failure analysis, bare validation, and online gates. Use a new output directory for new experiments so historical artifacts remain traceable.

## Check the analysis contracts

The extraction tests distinguish a product visit from an option refresh, count protocol repair as a model turn but not an action, reject partial token denominators, keep technical failures separate, and verify task-level aggregation before pairing. Publication checks validate bilingual pages, local links and anchors, source hashes, figure XML, four-round W&B coverage, and agreement of reported test totals.

## Data package

| Evidence | Download |
|---|---|
| Primary analysis and skill bank texts | [analysis.json](../data/analysis.json) |
| Generated-group summaries | [training_groups.csv](../data/training_groups.csv) |
| Primary rollout metrics | [training_steps.csv](../data/training_steps.csv) |
| Per-task four-condition outcomes | [test_task_outcomes.json](../data/test_task_outcomes.json) |
| Gate version coefficients | [chunk_contributions.csv](../data/chunk_contributions.csv) |
| Behavioral episode / turn extracts | [behavior report](behavior.html#download-and-extend) |
| Optimizer and systems history | [W&B diagnostics](optimization.html#reproducible-source-data) |
| Input hashes | [provenance.json](../data/provenance.json) |

## Publication and reuse

GitHub Pages builds `_site/` using an explicit asset allowlist; credentials, raw run directories, and corpus files are outside the package. The build contains a checksum manifest. Deployment is configured through the repository's Pages workflow and requires the repository/account to support the chosen visibility.

ShopSimulator retains upstream provenance and notices. The project does not invent a new license for upstream materials or imply that local experimental use grants unrestricted redistribution rights. See the [provenance record](../reference/docs/provenance.html) and existing repository notices.
