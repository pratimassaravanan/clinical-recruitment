# Theme #2 Completion Checklist

This file replaces the older aspirational roadmap-complete narrative. It is the deliberately conservative source of truth for what the current repo supports today and what outward-facing docs are safe to claim.

## Claim Rules

- Treat something as implemented only when it exists in the current code and is part of the benchmark path, the repo baselines, the tests, or the freshly generated reports.
- Treat auxiliary modules as experimental unless they are directly exercised by the current benchmark or backed by current artifacts.
- Prefer narrower wording when there is any doubt.

## Current Core Status

| Area | Status | Primary evidence |
|------|--------|------------------|
| `180`-step environment | present in benchmark path | `env.py`, `openenv.yaml` |
| `3` public tasks | present in benchmark path | `openenv.yaml`, `app.py` |
| Typed models | present in repo | `models.py` |
| `8`-action interface | present in benchmark path | `models.py`, `training/neural_policy.py` |
| Action-specific candidate pools | present in benchmark path | `models.py`, `env.py`, `train_agents.py`, `full_sweep.py` |
| `37`-dimensional feature vector | present in repo baseline stack | `training/neural_policy.py` |
| Four repo baselines | present in repo | `research/methods/*_agent.py` |
| Fresh sweep and stats tooling | present, with current generated outputs | `experiments/full_sweep.py`, `experiments/reproducibility.py`, `data/sweep_results/benchmark_report.json` |
| Live HF Space URL | present in submission path | `README.md`, `server/app.py` |
| Committed pilot LLM training artifact | present, but limited | `data/training_outputs/sft_grpo_results.json` |
| Offline progressive training artifacts | present for lightweight baselines | `data/training/training_history.csv`, `data/training/training_eval.csv` |
| Benchmark diagrams | generated artifacts present | `scripts/generate_docs_diagrams.py`, `docs/images/` |
| Local validation suites | test files present in repo | `test_env.py`, `test_agents.py`, `test_research_modules.py`, `test_local_serving.py` |
| Anonymous paper source and PDF | present in repo | `paper/main.tex`, `paper/main.pdf` |

## Safe Claims

### Environment and interface

It is safe to claim that the current benchmark path provides:

- `180`-step deterministic episodes
- `easy_bench`, `medium_bench`, and `hard_bench`
- Typed `Observation`, `Action`, `State`, and `StepResult` models
- `8` implemented action types
- Action-specific pools for screening, recontact, and allocation
- Milestones, constraints, delayed effects, and site-level metrics
- Explicit plan state and indexed-memory state
- Token budget, token usage, and token-efficiency signals
- Observation fields for counterfactual hints and simple rollout estimates

### Repo baselines

It is safe to claim that the repo includes four trainable baselines built on a shared pure-NumPy actor-critic stack:

- `HCAPO`
- `MiRA`
- `KLong`
- `MemexRL`

It is **not** safe to describe these as fully validated reproductions of external named methods.

### Reporting and verification

It is safe to claim that the repo currently provides:

- `experiments/train_agents.py` for single-run training
- `experiments/full_sweep.py` for multi-seed sweeps
- `experiments/reproducibility.py` for paired tests, Wilcoxon, effect sizes, and confidence intervals
- `scripts/generate_docs_diagrams.py` for the three main benchmark diagrams
- `data/sweep_results/benchmark_report.{md,json}` as the current benchmark summary
- Four main local test suites in the repo

### Training evidence

It is safe to claim that the repo currently provides:

- a live judge-facing environment URL at `https://pratimassaravanan-clinical-recruitment.hf.space`
- a committed pilot LLM training artifact in `data/training_outputs/sft_grpo_results.json`
- measurable pilot SFT improvement in that artifact: loss `0.858 -> 0.745` and action diversity `1 -> 5`
- explicit documentation that the pilot GRPO phase failed because the older reward path did not receive the required `environments` context
- separate progressive offline-policy training CSVs in `data/training/`

It is **not** safe to collapse those into one generic "training success" claim. The pilot artifact is early evidence, and the offline-policy CSVs are not LLM fine-tuning evidence.

## Fresh Generated Evidence

The current benchmark numbers come from `data/sweep_results/benchmark_report.json`.

The exact `5`-seed sweep values in the current report use seeds `1`, `7`, `21`, `42`, and `123`.

| Baseline | Mean | Std | 95% CI |
|----------|------|-----|--------|
| `HCAPO` | `0.2215` | `0.0127` | `[0.2100, 0.2303]` |
| `KLong` | `0.2152` | `0.0222` | `[0.1977, 0.2286]` |
| `MemexRL` | `0.2148` | `0.0270` | `[0.1943, 0.2352]` |
| `MiRA` | `0.2094` | `0.0095` | `[0.2023, 0.2165]` |

Interpretation that is safe to repeat:

- `HCAPO` is highest mean in the current sweep.
- No pairwise comparison reaches `p < 0.05`.
- The current baseline suite is clustered.
- Integration checks pass `30/30` across the three tasks.
- This is benchmark evidence for the current repo state, not evidence of clear baseline separation.

## Present in Repo, but Describe Carefully

The following files or areas exist, but should be described as auxiliary or experimental unless a specific claim is backed by tests or generated artifacts:

- `research/advanced_features.py`
- `training/async_rl.py`
- `training/progressive_rl.py`
- `experiments/ablate_features.py`
- `experiments/ablate_horizon.py`
- `experiments/appendix_report.py`
- `notebooks/training_grpo_openenv.ipynb`

These files may be useful, but they are not the core of the current benchmark claim set.

For the notebook specifically: describe it only as a hackathon bootstrap path. Do not present it as benchmark evidence, reproduced training evidence, or performance validation.

## Claims to Avoid

Do **not** claim any of the following in outward-facing docs:

- A `10`-action interface
- Complete roadmap coverage across the entire Theme #2 wishlist
- Externally validated reproductions or benchmark-leading status for named external methods
- A statistically significant `HCAPO` win from the current sweep
- Provider-grounded token accounting or external billing integration
- Notebook, TRL, or Colab workflows as validated evidence for the benchmark results
- The paper PDF as independent evidence beyond what the code and generated benchmark artifacts support
- The current post-fix `5k`-trace T4/Colab rerun as complete unless new outputs are checked in

## Practical File Map

- `models.py`: typed benchmark interface
- `env.py`: environment dynamics and reward path
- `training/neural_policy.py`: `ACTION_SPACE`, feature extractor, `STATE_DIM = 37`
- `research/methods/`: repo baselines and helper modules
- `experiments/full_sweep.py`: current multi-seed report generator
- `data/sweep_results/`: current report, raw outputs, and charts
- `data/training_outputs/sft_grpo_results.json`: committed pilot LLM training artifact
- `data/training/`: lightweight progressive offline-policy training outputs
- `docs/images/`: regenerated diagrams and refreshed sweep charts
- `paper/main.tex`: current anonymous paper source
- `notebooks/training_grpo_openenv.ipynb`: hackathon bootstrap notebook only, not benchmark evidence

## Remaining Work

- Stronger empirical separation will require larger training budgets, more seeds, or sharper ablations.
- Public docs should continue to follow the benchmark path and generated artifacts rather than older roadmap language.
- The GRPO starter notebook should be treated as a hackathon bootstrap path, not as benchmark evidence by itself.
