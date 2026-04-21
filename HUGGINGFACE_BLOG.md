# Adaptive Clinical Recruitment: What the Repo Actually Implements Today

**Adaptive Clinical Recruitment** is a long-horizon benchmark for sequential trial-planning decisions. It models the patient funnel over `180` simulated days, exposes typed observations and actions, and lets agents balance screening, follow-up, site allocation, planning, memory use, and budget pressure.

This post is intentionally conservative. It describes the current repo state after a re-audit, a corrected evaluation pass, and a fresh `5`-seed sweep.

## TL;DR

- The benchmark has `3` public tasks and `8` implemented action types.
- The trainable baselines use a `37`-dimensional numeric feature vector.
- The repo includes four baseline agents: `HCAPO`, `MiRA`, `KLong`, and `MemexRL`.
- After rerunning the corrected sweep, `HCAPO` has the highest mean score at `0.2215`.
- No pairwise comparison reaches `p < 0.05`, so the current results do **not** support a strong winner narrative.

## What the benchmark exposes

At each step, the environment returns a typed `Observation` with:

- Funnel state and action-specific candidate pools
- Per-site performance metrics
- Milestones and delayed-effects state
- Constraint and uncertainty summaries
- Plan state and indexed-memory summaries
- Token accounting and token-efficiency signals
- Counterfactual hints and simple rollout estimates

The current action interface contains exactly these `8` actions:

1. `screen_patient`
2. `recontact`
3. `allocate_to_site`
4. `adjust_strategy`
5. `plan_next_phase`
6. `summarize_and_index`
7. `retrieve_relevant_history`
8. `stop_recruitment`

Site negotiation is represented through `adjust_strategy` values such as `negotiate_site_A`, not as a separate ninth or tenth action.

## What changed during the re-audit

Before regenerating results, we fixed several issues that made the older docs too optimistic.

1. The experiment path previously used `available_patients` for `recontact` and `allocate_to_site`, even though those actions should draw from their own candidate pools.
2. The sweep charts could fail on small seed counts because of malformed error bars.
3. The chart refresh path updated `data/sweep_results/` but could leave `docs/images/` stale.
4. Several public docs still described a `10`-action interface and repeated an outdated significance claim.

The current docs and charts now follow the corrected benchmark path.

## Fresh 5-seed sweep

The regenerated report lives in `data/sweep_results/neurips_report.{md,json}`.

| Baseline | Mean | Std | 95% CI |
|----------|------|-----|--------|
| `HCAPO` | `0.2215` | `0.0127` | `[0.2100, 0.2303]` |
| `KLong` | `0.2152` | `0.0222` | `[0.1977, 0.2286]` |
| `MemexRL` | `0.2148` | `0.0270` | `[0.1943, 0.2352]` |
| `MiRA` | `0.2094` | `0.0095` | `[0.2023, 0.2165]` |

Pairwise tests from the same report show:

- `HCAPO vs MiRA`: `p = 0.1823`
- `HCAPO vs KLong`: `p = 0.3849`
- `HCAPO vs MemexRL`: `p = 0.6370`
- no comparison reaches `p < 0.05`

That means the honest headline is not "hierarchical planning wins." The honest headline is:

> The benchmark is active, reproducible, and non-trivial, but the current baseline suite remains tightly clustered.

## Why that is still useful

This repo is strongest as a benchmark package, not as a settled leaderboard.

- The environment interface is typed and deterministic.
- The action construction path now matches the observation schema.
- The main diagrams and sweep charts are regenerated from code.
- The integration checks pass `30/30` across `easy_bench`, `medium_bench`, and `hard_bench`.

For a benchmark, that is still a useful result. It means future work can focus on stronger training budgets, ablations, and better long-horizon methods without inheriting stale claims from the docs.

## Trying the benchmark

### Local API

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Minimal Python loop

```python
from env import ClinicalRecruitmentEnv
from models import Action

env = ClinicalRecruitmentEnv()
result = env.reset(task="medium_bench")
obs = result.observation

action = Action(
    action_type="screen_patient",
    patient_id=obs.available_patients[0]["id"],
    hypothesis="noise_dominant",
    confidence=0.7,
)

result = env.step(action)
print(result.reward)
```

### Reproducing the sweep

```bash
python experiments/full_sweep.py --seeds 1 7 21 42 123 --episodes 30 --eval-episodes 5
```

## What this post does not claim

- It does not claim a `10`-action interface.
- It does not claim that all `50` roadmap features are implemented and validated.
- It does not claim externally validated reproductions or benchmark-leading status for external named methods.
- It does not claim a statistically significant `HCAPO` win.
- It does not use notebook or TRL claims as evidence for the benchmark results.

## Key files

- `README.md`: current repo overview
- `docs/theme2_alignment.md`: conservative Theme #2 mapping
- `docs/theme2_completion_checklist.md`: reality-based status file
- `data/sweep_results/neurips_report.md`: fresh benchmark summary
- `paper/main.pdf`: current anonymous NeurIPS E&D paper build

## Links

- GitHub repository: `https://github.com/pratimassaravanan/clinical-recruitment-env`
- Hugging Face Space: `https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment-env`
