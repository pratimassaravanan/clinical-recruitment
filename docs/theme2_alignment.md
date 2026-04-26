# Theme #2 Alignment

This note maps the current `clinical-recruitment-env` repository to Theme #2 style long-horizon planning and data-driven decision-making requirements. It is intentionally narrow: it only describes behavior that is visible in the current benchmark path, current repo baselines, or freshly generated artifacts, and it should be read as a best-fit Theme #2 mapping rather than a claim of complete requirement coverage.

## Verified benchmark surfaces

- `180`-step episodes in `env.py`
- `3` public tasks in `openenv.yaml` and `app.py`
- Typed `Observation`, `Action`, and `State` models in `models.py`
- `8` implemented action types in `models.py` and `training/neural_policy.py`
- `37`-dimensional numeric feature vector in `training/neural_policy.py`
- Fresh sweep outputs in `data/sweep_results/`
- Regenerated diagrams in `docs/images/`

## Theme #2 evidence map

| Theme #2 idea | Current repo evidence | Notes |
|---------------|-----------------------|-------|
| Long-horizon episodes | `env.py`, `openenv.yaml` | Episodes run for up to `180` steps with delayed effects and final graded scores |
| Delayed feedback | `env.py` delayed-effects queue, milestone updates, dropout handling | The benchmark includes explicit lagged consequences rather than only immediate rewards |
| Goal decomposition | `plan_next_phase`, `current_plan`, phase targeting in `env.py` | This is evidence of planner-facing scaffolding, not proof of strong hierarchical planning performance |
| Extended state tracking | `Observation` fields, `State`, patient memory summary, indexed memory summary | The environment carries structured context beyond the scalar reward, but not every field is shown to be equally decision-critical |
| Recovery from mistakes | `recontact`, `recovery` phase, constraint handling, plan refresh logic | Recovery-oriented actions are part of the interface; this does not imply broad robustness to all failure modes |
| Durable representations | Indexed-memory actions plus repo baselines such as `MemexRL` | The repo exposes memory-oriented state and action surfaces, not a validated durable-memory solution |
| Multi-scale temporal reasoning | `KLong` baseline and milestone/frontier features | Present in repo baselines and numeric features, not claimed as solved or statistically dominant |
| Lightweight world-state modeling | `world_type`, `hypothesis_accuracy`, `uncertainty_components`, `counterfactual_rollout` | The core path exposes structured world-state signals and what-if summaries, but not a full scientific workflow loop or general-purpose world model |
| Business workflow structure | Screening, conversion, allocation, retention, and recovery phases | The benchmark is workflow-shaped and operational, but still a bounded simulation with `3` tasks |
| Token-aware efficiency | `token_budget_remaining`, `token_usage_so_far`, `token_efficiency_score` | Internal accounting exists, but it is not provider-grounded external billing |

## Theme-relevant observation surfaces

The current `Observation` model exposes the following categories that are relevant to long-horizon planning analysis:

- Action-specific candidate pools: `available_patients`, `recontact_candidates`, `allocation_candidates`
- Site state: `site_performance`
- Long-horizon state: `milestones`, `active_constraints`, `delayed_effects_pending`
- Planning and memory: `current_plan`, `indexed_memory_summary`, `retrieved_memory_context`
- Difficulty and uncertainty: `difficulty`, `uncertainty_level`, `uncertainty_components`
- Token-aware signals: `token_budget_remaining`, `token_usage_so_far`, `token_efficiency_score`
- Counterfactual guidance: `counterfactual_hint`, `counterfactual_rollout`

## Repo baselines in scope

The current docs treat the following as repo baselines, not as externally validated reproductions or comprehensive ablations:

| Baseline | Main mechanism |
|----------|----------------|
| `HCAPO` | Hierarchical subgoals with hindsight relabeling |
| `MiRA` | Milestone-aware potential shaping |
| `KLong` | Multi-scale temporal aggregation |
| `MemexRL` | Episodic memory write/read behavior |

All four use the shared pure-NumPy actor-critic stack in `training/neural_policy.py`.

## Current empirical status

Fresh `5`-seed sweep summary from `data/sweep_results/benchmark_report.json`:

| Baseline | Mean | Std | 95% CI |
|----------|------|-----|--------|
| `HCAPO` | `0.2215` | `0.0127` | `[0.2100, 0.2303]` |
| `KLong` | `0.2152` | `0.0222` | `[0.1977, 0.2286]` |
| `MemexRL` | `0.2148` | `0.0270` | `[0.1943, 0.2352]` |
| `MiRA` | `0.2094` | `0.0095` | `[0.2023, 0.2165]` |

Important interpretation:

- `HCAPO` is the highest-mean baseline in the current run.
- No pairwise comparison reaches `p < 0.05`.
- The repo therefore supports only a conservative benchmark claim: the environment exposes long-horizon planning structure and remains nontrivial for the current baseline suite, but the current sweep does not establish a clear statistical winner.

## What this file does not claim

- It does not claim a `10`-action interface.
- It does not claim full `50/50` roadmap completion.
- It does not claim external-token-grounded Mercor accounting.
- It does not claim that every auxiliary research module is a validated benchmark contribution.
- It does not claim that the current state/action surfaces amount to complete Theme #2 coverage.
- It does not claim that planning, memory, or recovery are solved capabilities in this repo.
- It does not claim significant baseline separation from the current sweep.

## Bottom line

Theme #2 is the best fit for the current repository because the benchmark is long-horizon, typed, workflow-shaped, and centered on strategic resource allocation, with explicit planning, memory, delayed-effects, recovery-oriented surfaces, and lightweight world-state signals. The strongest repo-truth is about benchmark structure, a live evaluation surface, and reproducible reports, not complete Theme #2 coverage or leaderboard dominance.
