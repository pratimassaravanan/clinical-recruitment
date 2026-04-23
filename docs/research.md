# Research Stack

This directory captures offline benchmark experimentation that is intentionally kept out of the serving path.
It should be read as supporting research documentation for the repo's own baselines and analysis scaffolds, not as a claim of externally validated reproductions.

## Layout

- `research/policies.py`: local research baselines
- `research/runner.py`: deterministic offline experiment runner
- `research/replay.py`: frontier replay buffer scaffold
- `research/skills.py`: offline skill inference helpers
- `research/preferences.py`: offline preference ranking helpers
- `research/goal_discovery.py`: summary-to-goal discovery helpers
- `research/world_models/site_model.py`: lightweight site-value world model
- `research/privacy/simulator.py`: anonymization helper for offline analysis
- `research/methods/salt.py`: SALT-style step advantage scaffold
- `research/methods/oversight.py`: oversight and recovery summaries
- `experiments/run_research.py`: CLI entrypoint that writes CSV results into `data/`
- `experiments/run_progressive_training.py`: staged-horizon evaluation entrypoint
- `experiments/train_offline_policy.py`: trainable offline baseline entrypoint
- `experiments/ablate_horizon.py`: turn-restricted versus full-horizon ablations
- `experiments/ablate_features.py`: model/feature ablations
- `experiments/reproducibility.py`: multi-seed reproducibility sweep
- `experiments/pareto_report.py`: Pareto summary export
- `experiments/appendix_report.py`: appendix-style markdown report
- `training/progressive_rl.py`: progressive-horizon curriculum helpers
- `training/offline_policy.py`: lightweight feature-based policy model
- `training/train_offline_policy.py`: training loop and weight export helpers
- `training/trajectory_splitter.py`: KLong-style trajectory chunking helpers
- `training/curriculum.py`: confidence-aware and Thompson-sampling curriculum helpers
- `training/async_rl.py`: async-style training scaffold
- `scripts/generate_charts.py`: chart generation from experiment CSVs into `docs/images/`
- `scripts/plot_trajectories.py`: before/after trajectory comparison plot
- `scripts/plot_training_curves.py`: reward-curve dashboard plot
- `scripts/plot_pareto_frontier.py`: standalone Pareto frontier plot
- `test_research_modules.py`: focused verification for the offline research scaffolds

## Default study

The default offline experiment runner compares four local research policies across the benchmark's `3` public tasks:

- `greedy_screen`
- `conservative_retention`
- `site_negotiation`
- `rule_based_memory`

These are separate from the four named repo baselines used in the fresh sweep (`HCAPO`, `MiRA`, `KLong`, `MemexRL`).

Current benchmark facts that matter for interpreting any result:

- Episodes run for up to `180` steps.
- The benchmark exposes `8` implemented actions.
- Shared neural-policy features are `37`-dimensional.
- Theme #2 remains a best-fit framing for the long-horizon structure here, not a claim of complete coverage or solved planning ability.

Run locally:

```bash
python experiments/run_research.py --episodes 3
python experiments/pareto_report.py
python experiments/run_progressive_training.py
python experiments/train_offline_policy.py --epochs 6
python experiments/ablate_horizon.py
python experiments/ablate_features.py
python experiments/reproducibility.py
python experiments/appendix_report.py
python scripts/generate_charts.py
python scripts/plot_trajectories.py
python scripts/plot_training_curves.py
python test_research_modules.py
```

The aggregated CSVs now also track planning, indexed-memory, milestone-potential, trajectory-splitting, token-efficiency, Pareto frontier size, replay size, oversight ratio, SALT advantages, inferred skills, discovered goals, and site-value summaries so serving-side long-horizon scaffolds can be analyzed offline without polluting the runtime path.

Privacy anonymization, curriculum managers, and async-style training remain offline-only scaffolds rather than serving-path features.

For baseline interpretation, the current multi-seed sweep is the `5`-seed run with seeds `1 7 21 42 123`; its fresh report shows `HCAPO 0.2215 +/- 0.0127`, `KLong 0.2152 +/- 0.0222`, `MemexRL 0.2148 +/- 0.0270`, and `MiRA 0.2094 +/- 0.0095`, with no pairwise comparison reaching `p < 0.05`.
