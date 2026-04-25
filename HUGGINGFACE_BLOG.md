# Adaptive Clinical Recruitment: A Workflow-Shaped OpenEnv Benchmark With Honest Training Evidence

**Adaptive Clinical Recruitment** is a long-horizon OpenEnv benchmark for data-driven trial-planning decisions. It models the patient funnel over `180` simulated steps, exposes typed observations and actions, and lets agents balance screening, follow-up, site allocation, planning, memory use, and budget pressure from a live environment URL.

This writeup stays on Theme #2 only: what the repo currently supports as a benchmark package, what training evidence is committed today, and what is still missing from the strongest possible submission story.

This post is intentionally conservative. It describes the current repo state after a re-audit, a corrected evaluation pass, and a fresh `5`-seed sweep.

## TL;DR

- The benchmark has `3` public tasks and `8` implemented action types.
- The trainable baselines use a `37`-dimensional numeric feature vector.
- The repo includes four baseline agents: `HCAPO`, `MiRA`, `KLong`, and `MemexRL`.
- The live environment is already hosted at `https://pratimassaravanan-clinical-recruitment.hf.space`.
- The repo contains a committed pilot T4 training artifact showing measurable SFT improvement, but not end-to-end task success yet.
- After rerunning the corrected sweep, `HCAPO` has the highest mean score at `0.2215`.
- No pairwise comparison reaches `p < 0.05`, so the current results do **not** support a strong winner narrative.

## Why this is an interesting training target

Clinical recruitment is not a one-step prediction problem. It is a workflow-shaped resource-allocation loop:

- which patients to screen first
- when to spend effort on recontact
- which sites deserve scarce capacity
- when to change strategy under budget and dropout pressure

Those decisions play out over weeks or months of simulated time, with delayed effects, constraint pressure, and recovery actions. That makes the environment a better fit for Theme #2 than a short-horizon reward toy.

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

The regenerated report lives in `data/sweep_results/benchmark_report.{md,json}`.

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

> The repo exposes a real benchmark surface, but the current baseline suite remains too tightly clustered to support a winner claim.

## Current training evidence

The repository also contains committed training artifacts, but they need to be described carefully.

- `data/training_outputs/sft_grpo_results.json` captures an earlier small-scale Tesla T4 `SFT -> GRPO` pilot.
- Safe conclusions from that file:
  - SFT loss fell from `0.858` to `0.745` (`13.2%`)
  - the model expanded from `1` repeated action to `5` action types
  - the GRPO phase ran without usable reward because the old TRL/OpenEnv path did not pass the required `environments` context
  - enrollment after training remained `0`, so this is evidence of format learning and action diversification, not full task mastery
- `data/training/training_history.csv` and `data/training/training_eval.csv` provide separate progressive-horizon offline-policy training diagnostics.

So the current repo does satisfy the hackathon's "show the training" spirit in a limited way: there is real, committed training evidence on disk. But the most valuable missing artifact is still a post-fix `5k`-trace T4/Colab rerun aligned with the current `train.py` and Kaggle notebook.

## Why the live URL matters

The environment is already served at:

- `https://pratimassaravanan-clinical-recruitment.hf.space`

That matters because the benchmark is not only a local code artifact. The repo exposes a judge-facing URL, local FastAPI/OpenEnv serving, and a separate `tool_env.py` training wrapper for TRL/OpenEnv-style experiments.

## What that means

This repo currently reads best as a benchmark package with early training evidence, not as a settled leaderboard.

- The environment interface is typed and deterministic.
- The action construction path now matches the observation schema.
- The main diagrams and sweep charts are regenerated from code.
- The integration checks pass `30/30` across `easy_bench`, `medium_bench`, and `hard_bench`.
- A pilot LLM training artifact is committed, but the strongest corrected large-trace rerun is still pending.

For Theme #2, that is the relevant result. The environment is usable for evaluation work and early training experiments, but the present numbers do not justify a stronger performance claim than that.

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
- It does not claim that the current post-fix `5k`-trace T4 rerun has already been completed.

## Key files

- `README.md`: current repo overview
- `docs/theme2_alignment.md`: conservative Theme #2 mapping
- `docs/theme2_completion_checklist.md`: reality-based status file
- `data/sweep_results/benchmark_report.md`: fresh benchmark summary
- `data/training_outputs/sft_grpo_results.json`: committed pilot LLM training artifact
- `paper/main.pdf`: current anonymous paper build

## Links

- GitHub repository: `https://github.com/pratimassaravanan/clinical-recruitment`
- Hugging Face Space: `https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment`
- Live environment URL: `https://pratimassaravanan-clinical-recruitment.hf.space`
