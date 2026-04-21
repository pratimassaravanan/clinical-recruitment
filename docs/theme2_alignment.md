# Theme #2 Alignment

This note maps the current `clinical-recruitment-env` repository to Theme #2 style long-horizon planning requirements. It is intentionally narrower than the older docs: it only describes behavior that is visible in the current benchmark path, current repo baselines, or freshly generated artifacts.

## Verified benchmark surfaces

- `180`-step episodes in `env.py`
- `3` public tasks in `openenv.yaml` and `app.py`
- Typed `Observation`, `Action`, and `State` models in `models.py`
- `8` implemented action types in `models.py` and `training/neural_policy.py`
- `37`-dimensional numeric feature vector in `training/neural_policy.py`
- Fresh sweep outputs in `data/sweep_results/`
- Regenerated diagrams in `docs/images/`

## Requirement mapping

| Theme #2 idea | Current repo evidence | Notes |
|---------------|-----------------------|-------|
| Long-horizon episodes | `env.py`, `openenv.yaml` | Episodes run for up to `180` steps with delayed effects and final graded scores |
| Delayed feedback | `env.py` delayed-effects queue, milestone updates, dropout handling | Actions can trigger downstream consequences several steps later |
| Goal decomposition | `plan_next_phase`, `current_plan`, phase targeting in `env.py` | The benchmark has explicit plan state and planner-followthrough shaping |
| Extended state tracking | `Observation` fields, `State`, patient memory summary, indexed memory summary | The environment carries structured context beyond the scalar reward |
| Recovery from mistakes | `recontact`, `recovery` phase, constraint handling, plan refresh logic | Recovery is part of the benchmark interface, not just a post-hoc analysis idea |
| Durable representations | Indexed-memory actions plus repo baselines such as `MemexRL` | The environment and baselines both expose memory-oriented behavior |
| Multi-scale temporal reasoning | `KLong` baseline and milestone/frontier features | Present in repo baselines and numeric features, not claimed as solved |
| Business workflow structure | Screening, conversion, allocation, retention, and recovery phases | The task is structured around an operational funnel instead of a short game loop |
| Token-aware efficiency | `token_budget_remaining`, `token_usage_so_far`, `token_efficiency_score` | Internal accounting exists, but it is not provider-grounded external billing |

## Theme-relevant observation surfaces

The current `Observation` model exposes the following categories that matter for long-horizon planning:

- Action-specific candidate pools: `available_patients`, `recontact_candidates`, `allocation_candidates`
- Site state: `site_performance`
- Long-horizon state: `milestones`, `active_constraints`, `delayed_effects_pending`
- Planning and memory: `current_plan`, `indexed_memory_summary`, `retrieved_memory_context`
- Difficulty and uncertainty: `difficulty`, `uncertainty_level`, `uncertainty_components`
- Token-aware signals: `token_budget_remaining`, `token_usage_so_far`, `token_efficiency_score`
- Counterfactual guidance: `counterfactual_hint`, `counterfactual_rollout`

## Repo baselines in scope

The current docs treat the following as repo baselines, not as externally validated reproductions:

| Baseline | Main mechanism |
|----------|----------------|
| `HCAPO` | Hierarchical subgoals with hindsight relabeling |
| `MiRA` | Milestone-aware potential shaping |
| `KLong` | Multi-scale temporal aggregation |
| `MemexRL` | Episodic memory write/read behavior |

All four use the shared pure-NumPy actor-critic stack in `training/neural_policy.py`.

## Current empirical status

Fresh `5`-seed sweep summary from `data/sweep_results/neurips_report.json`:

| Baseline | Mean | Std | 95% CI |
|----------|------|-----|--------|
| `HCAPO` | `0.2215` | `0.0127` | `[0.2100, 0.2303]` |
| `KLong` | `0.2152` | `0.0222` | `[0.1977, 0.2286]` |
| `MemexRL` | `0.2148` | `0.0270` | `[0.1943, 0.2352]` |
| `MiRA` | `0.2094` | `0.0095` | `[0.2023, 0.2165]` |

Important interpretation:

- `HCAPO` is the highest-mean baseline in the current run.
- No pairwise comparison reaches `p < 0.05`.
- The repo therefore supports a conservative benchmark claim: the environment is active and challenging, but the current baseline suite does not yet show a clear statistical winner.

## What this file does not claim

- It does not claim a `10`-action interface.
- It does not claim full `50/50` roadmap completion.
- It does not claim external-token-grounded Mercor accounting.
- It does not claim that every auxiliary research module is a validated benchmark contribution.
- It does not claim significant baseline separation from the current sweep.

## Bottom line

The repository aligns with Theme #2 because it provides a long-horizon, typed, workflow-shaped benchmark with explicit planning, memory, delayed effects, and structured recovery surfaces. The strongest current claim is benchmark design and reproducibility, not leaderboard dominance.
