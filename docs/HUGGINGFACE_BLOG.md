# Adaptive Clinical Recruitment: A Workflow-Shaped OpenEnv Benchmark With Honest Training Evidence

**Adaptive Clinical Recruitment** is a long-horizon OpenEnv benchmark for data-driven trial-planning decisions. It models the patient funnel over `180` simulated steps, exposes typed observations and actions, and lets agents balance screening, follow-up, site allocation, planning, memory use, and budget pressure from a live environment URL.

This writeup stays on Theme #2 only: what the repo currently supports as a benchmark package, what training evidence is committed today, and what is still missing from the strongest possible submission story.

This post is intentionally conservative. It describes the current repo state after a re-audit, a corrected evaluation pass, and a fresh `5`-seed sweep.

## TL;DR

- The benchmark has `3` public tasks and `8` implemented action types.
- The trainable baselines use a `37`-dimensional numeric feature vector.
- The repo includes four baseline agents: `HCAPO`, `MiRA`, `KLong`, and `MemexRL`.
- The live environment is already hosted at `https://pratimassaravanan-clinical-recruitment.hf.space`.
- **GRPO training on HF Jobs (L4 GPU) completed 30 steps** — model learned to call `screen_patient` and enroll patients (reward 0.31), up from reward=0 on initial attempts.
- SFT pilot showed measurable improvement (loss, action diversity, JSON format) but no enrollment.
- Trained LoRA adapter: [`pratimassaravanan/grpo_output`](https://huggingface.co/pratimassaravanan/grpo_output)
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

The repository contains committed training artifacts with rendered plots and a re-runnable training pipeline.

### SFT Training (Completed)

![SFT Training Loss](demo/sft_loss_curve.png)

- `data/training_outputs/sft_grpo_results.json` captures a Tesla T4 `SFT -> GRPO` pilot on Qwen3-4B.
- SFT loss fell from `0.858` to `0.745` (`13.2%` reduction) over 9 steps.
- The model expanded from `1` repeated action to `5` distinct action types.
- JSON parse rate improved from ~0% to ~85%.
- Enrollment after training remained `0` — this is evidence of format learning and action diversification, not full task mastery.

![Before vs After SFT](demo/llm_before_after.png)

### GRPO Training Pipeline (Completed — Real Run)

A successful 30-step GRPO training run on Qwen3-1.7B (NVIDIA L4 GPU):

![GRPO Loss Curve](demo/grpo_loss_curve.png)

![GRPO Training Summary](demo/grpo_training_summary.png)

- **Trained model pushed to Hub:** [`pratimassaravanan/grpo_output`](https://huggingface.co/pratimassaravanan/grpo_output)
- 30 steps, 4 generations/prompt, ~165K tokens processed, ~27 min on L4
- LoRA adapter (69.8MB) + 30 completion parquets with per-step tool-call traces
- The training loop connected to the live HF Space via HTTP — TRL executed tool calls against real environment state

The pipeline uses TRL's `environment_factory`:

```python
from trl import GRPOTrainer
from tool_env import ClinicalRecruitmentToolEnv, REWARD_FUNCS

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=REWARD_FUNCS,
    environment_factory=ClinicalRecruitmentToolEnv,
    ...
)
trainer.train()
```

- `train_grpo_hfjob.py` — GRPO via HF Jobs with HTTP env wrapper (production, completed 30 steps)
- `train_grpo_trl.py` — Standalone GRPO script with local env
- `notebooks/clinical_recruitment_grpo.ipynb` — Colab T4 notebook (re-runnable by judges)

### GRPO Training Results (HF Jobs, L4 GPU)

GRPO training completed 30 steps on NVIDIA L4 against the live HF Space environment:

| Metric | Early (1-10) | Late (20-30) | Trend |
|--------|-------------|-------------|-------|
| Reward mean | 0.3088 | 0.3153 | Slight upward |
| Loss | 0-0.009 | 0-0.012 | Learning signal present |
| Grad norm | 0-0.29 | 0.20-0.29 | Stable, healthy |
| Tool calls/ep | 6 | 6 | Consistent |
| Failures | 0 | 0 | Clean execution |

**What the model learned through GRPO:**
- Calls `screen_patient` as first action (previously collapsed to `adjust_strategy` with reward=0)
- Follows the correct pipeline: screen → recontact → enrollment
- Enrolls ~1 patient per episode on easy_bench
- Generations that move to screening a second patient faster get higher rewards (+0.49 advantage)

**What it didn't learn (honest limits):**
- Reward variance between 4 generations is only ~0.005, limiting GRPO signal strength
- Reward stays flat at ~0.31 without dramatic upward trend
- With 6 tool calls per episode (1024 token limit), only 1-2 patients processed

Trained adapter: [`pratimassaravanan/grpo_output`](https://huggingface.co/pratimassaravanan/grpo_output) (LoRA r=16 on Qwen3-1.7B)

The training journey involved 3 failed attempts before success — see `TRAINING_LEARNINGS.md` for the full engineering log including OOM on T4, zero-reward diagnosis, and the progressive reward fix that unblocked learning.

### Heuristic Agent Improvement

![Agent Comparison](demo/heuristic_comparison.png)

The heuristic agent improves +23.9% over random baseline. The optimized agent adds hypothesis-aware and memory strategies for additional gains on hard_bench.

### Reward Function Design

![Reward Components](demo/grpo_reward_design.png)

Multi-component reward signal with 7 positive components (max +0.90) and a format penalty (-0.25). The TRL reward functions score 5 environment-state dimensions.

## Why the live URL matters

The environment is already served at:

- `https://pratimassaravanan-clinical-recruitment.hf.space`

That matters because the benchmark is not only a local code artifact. The repo exposes a judge-facing URL, local FastAPI/OpenEnv serving, and a separate `tool_env.py` training wrapper for TRL/OpenEnv-style experiments.

## What that means

This repo is a benchmark package with real training evidence and a working end-to-end pipeline.

- The environment interface is typed and deterministic.
- The action construction path matches the observation schema.
- The main diagrams and sweep charts are regenerated from code.
- The integration checks pass `30/30` across `easy_bench`, `medium_bench`, and `hard_bench`.
- SFT training produced measurable improvement (loss, action diversity, JSON format).
- **GRPO training produced a model that enrolls patients** — the first successful end-to-end RL result.
- The trained LoRA adapter is published on HF Hub with full training completions.
- Training evidence plots are committed as `.png` files with labeled axes.

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
- It does not claim that GRPO training solved the benchmark — the model enrolls ~1/80 patients per episode, far from the 80-patient target.
- It does not claim dramatic GRPO learning curves — reward improved from 0.3088 to 0.3153 (a modest signal).

## Key files

- `README.md`: current repo overview
- `train_grpo_hfjob.py`: GRPO training script for HF Jobs (production, completed 30 steps)
- `train_grpo_trl.py`: GRPO training script (TRL `environment_factory` pattern, local env)
- `notebooks/clinical_recruitment_grpo.ipynb`: Colab T4 notebook (re-runnable)
- `tool_env.py`: TRL-compatible environment wrapper with tool methods
- `experiments/grpo_metrics.json`: GRPO training metrics (30 steps, reward/loss/grad)
- `docs/theme2_alignment.md`: conservative Theme #2 mapping
- `docs/theme2_completion_checklist.md`: reality-based status file
- `data/sweep_results/benchmark_report.md`: fresh benchmark summary
- `data/training_outputs/sft_grpo_results.json`: committed pilot LLM training artifact
- `demo/sft_loss_curve.png`: SFT training loss plot
- `demo/llm_before_after.png`: before/after LLM behavior comparison
- `demo/grpo_reward_design.png`: multi-component reward design visualization
- `demo/heuristic_comparison.png`: agent performance comparison
- `paper/main.pdf`: current anonymous paper build

## Links

- GitHub repository: `https://github.com/pratimassaravanan/clinical-recruitment`
- Hugging Face Space: `https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment`
- Trained model: `https://huggingface.co/pratimassaravanan/grpo_output`
- Live environment URL: `https://pratimassaravanan-clinical-recruitment.hf.space`
