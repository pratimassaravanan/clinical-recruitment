# Research Method Baselines

This directory contains the repo baselines and helper modules used by the long-horizon experiments.
They support a Theme #2 style long-horizon benchmark interpretation, but only as repo-local baselines over the current environment rather than externally validated reproductions.

## Baselines

| File | Baseline | Main mechanism |
|------|----------|----------------|
| `hcapo_agent.py` | `HCAPO` | Hierarchical subgoals and hindsight relabeling |
| `mira_agent.py` | `MiRA` | Milestone-aware potential shaping |
| `klong_agent.py` | `KLong` | Multi-scale temporal aggregation and TD(lambda)-style credit assignment |
| `memex_agent.py` | `MemexRL` | Episodic memory with learned write/read behavior |

These are described throughout the docs as repo baselines inspired by long-horizon RL ideas. They are not presented as externally validated reproductions of external named methods.

## Shared training stack

All four baselines depend on `training/neural_policy.py`, which currently provides:

- benchmark episodes that run for up to `180` steps across `3` public tasks
- `ACTION_SPACE` with the benchmark's `8` implemented actions
- `extract_state_features()` for the `37`-dimensional numeric state vector
- A shared pure-NumPy actor-critic backbone

## Fresh sweep snapshot

Current benchmark numbers come from `data/sweep_results/neurips_report.{md,json}` and should be read as the current repo sweep snapshot, not as a reproduction claim.

| Baseline | Mean | Std | 95% CI |
|----------|------|-----|--------|
| `HCAPO` | `0.2215` | `0.0127` | `[0.2100, 0.2303]` |
| `KLong` | `0.2152` | `0.0222` | `[0.1977, 0.2286]` |
| `MemexRL` | `0.2148` | `0.0270` | `[0.1943, 0.2352]` |
| `MiRA` | `0.2094` | `0.0095` | `[0.2023, 0.2165]` |

No pairwise comparison reaches `p < 0.05` in the current `5`-seed sweep.

## Reproduce training and reports

```bash
python experiments/train_agents.py --agent all --episodes 50
python experiments/full_sweep.py --seeds 1 7 21 42 123 --episodes 30 --eval-episodes 5
```

Single-run training artifacts are written under `data/trained_agents/` and `data/training_results/`.
The multi-seed benchmark report is written under `data/sweep_results/`.

## Other files in this directory

- `registry.py`: repo-local provenance labels for the baseline families
- `site_agents.py`: site negotiation helper module
- `salt.py`: auxiliary step-level advantage helper
- `oversight.py`: auxiliary oversight helper

These helpers exist in the repo, but the main benchmark claims are centered on the four baseline agents plus the generated sweep artifacts.
