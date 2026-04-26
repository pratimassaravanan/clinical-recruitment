# Training a Long-Horizon Clinical Trial Recruitment Agent: What Failed, What Worked, and What We Learned

**The problem:** 80% of clinical trials miss enrollment targets. Average delay = 6-12 months, costing sponsors $600K-$8M per day. We built **Adaptive Clinical Recruitment** -- a 180-step OpenEnv benchmark that turns clinical trial coordinator decisions into a trainable RL problem.

**The result:** After 5+ failed training runs, 10 anti-collapse engineering fixes, and 3 GPU tiers (T4, L4, L40S), we produced a Qwen3-1.7B agent that learns to screen patients, recontact dropouts, and allocate sites -- enrolling 3-4 patients per rollout where it previously enrolled zero.

This post documents the real engineering journey, not just the final result.

## The Environment

At each step, the agent sees:
- **Funnel state**: available patients, recontact candidates, allocation candidates
- **Site performance**: conversion rates, capacity, dropout risk per site
- **Budget and timeline pressure**: remaining budget, days left, enrollment vs target
- **Memory**: indexed summaries and retrieved context from previous steps
- **Counterfactual hints**: what might have happened under different actions

The agent chooses from 8 typed actions: `screen_patient`, `recontact`, `allocate_to_site`, `adjust_strategy`, `plan_next_phase`, `summarize_and_index`, `retrieve_relevant_history`, `stop_recruitment`.

Episodes run 180 steps (one simulated day per step) across 3 difficulty tiers: `easy_bench` (1 site, $120K, target 80), `medium_bench` (3 sites, $150K, target 120), `hard_bench` (5 sites, $100K, target 150).

The reward is multi-component: enrollment velocity, budget efficiency, diversity, retention, regulatory compliance, memory quality, and strategy coherence.

Built on **OpenEnv 0.2.3** (latest). The environment subclasses `openenv.core.env_server.interfaces.Environment`, uses `openenv.yaml` for task configuration, and serves via FastAPI with the built-in Gradio web UI at `/web`.

**Live environment:** [pratimassaravanan-clinical-recruitment.hf.space](https://pratimassaravanan-clinical-recruitment.hf.space)

## Training Journey: 5+ Failed Runs Before Success

### Attempt 1: SFT on Expert Traces (Qwen3-4B, L40S)

We generated 2,000 heuristic traces and ran SFT on Qwen3-4B.

**What happened:** Loss dropped from 3.14 to 0.015 (perfect memorization). JSON parse rate went to 100%. But the model output 100% `adjust_strategy` -- complete policy collapse. It memorized the most common action in the training data and ignored the observation entirely.

**What we learned:** SFT teaches format, not observation-conditional behavior. We independently confirmed this on a second run with 16K traces on Qwen3-1.7B -- same collapse.

![SFT Loss Curve](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/sft_loss_curve.png)
*SFT loss drops cleanly, but the model only learns to output the most common action.*

### Attempt 2: GRPO-Only on T4

Tried GRPO directly on Qwen3-1.7B without SFT warmup. OOM on T4 (16GB) with full precision. Moved to L4 (24GB).

### Attempt 3: GRPO-Only on L4 (30 steps)

30-step GRPO with 4 generations per step. The model learned to call `screen_patient` first and follow the screen->recontact->enrollment pipeline. But reward stayed flat at ~0.31 because `max_completion_length=1024` only allowed ~6 tool calls per episode (1-2 patients).

![GRPO Loss Curve](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/grpo_loss_curve.png)

![GRPO Training Summary](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/grpo_training_summary.png)

**Key insight:** The context window ceiling was the real bottleneck, not the model or the reward.

Model: [`pratimassaravanan/grpo_output`](https://huggingface.co/pratimassaravanan/grpo_output) (30 completion parquets)

### Attempt 4: GRPO-Only without SFT (Multiple Runs)

Three more GRPO-only attempts (Qwen3-0.6B, Qwen3-1.7B) all collapsed -- loss=0 on 50-100% of steps. Without SFT warmup, the model couldn't generate valid JSON tool calls, so all generations were identical garbage, GRPO computed zero advantage, and nothing was learned.

### Attempt 5: SFT + REINFORCE (Qwen3-4B, L40S)

Used SFT warmup then REINFORCE. SFT collapsed to 100% `adjust_strategy` (same as before), but REINFORCE recovered: 0 to 5 enrolled in 15-step episodes, diverse action distribution.

Model: [`pratimassaravanan/clinical-qwen3-4b-sft-lora`](https://huggingface.co/pratimassaravanan/clinical-qwen3-4b-sft-lora)

![Before vs After SFT](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/llm_before_after.png)
*Before SFT: random/unparseable output. After SFT: 100% adjust_strategy. After REINFORCE: diverse actions.*

### Best Run: SFT + 80-Step GRPO (Qwen3-1.7B, L4)

Combined all learnings into one run:
1. SFT warmup (3.5 min) to teach JSON format
2. Merge SFT LoRA into base model
3. 80-step GRPO with fresh LoRA on the merged model
4. `max_completion_length=4096` to break the context ceiling
5. 10 anti-collapse fixes (see below)

![GRPO 80-Step Training](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/grpo_80step_training.png)
*Our best run: reward 0.27->0.33, tool calls 3.5->11, enrollment 1-2->3-4 patients per rollout.*

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| Reward | 0.269 | 0.331 | **+23%** |
| Tool calls/step | 3.5 | 11 | **3x growth** |
| Enrollment/rollout | 1-2 | 3-4 | **Consistent** |
| Zero-std collapse | 0% | 10% | Intermittent, self-recovering |

The model learned to:
- Call `screen_patient` first (not `adjust_strategy`)
- Follow the correct funnel: screen -> recontact -> enroll
- Make 11+ tool calls per rollout (up from 3.5)
- Recover from intermittent collapse without sustained degradation

Model: [`pratimassaravanan/grpo`](https://huggingface.co/pratimassaravanan/grpo) (80 completion parquets, LoRA adapter)

![All Runs Comparison](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/all_runs_comparison.png)
*All training runs compared.*

## The 10 Anti-Collapse Fixes

Policy collapse was the dominant failure mode across 5+ failed runs. No single fix solved it. These 10 interventions worked together:

1. **Temperature 1.2** -- increases generation diversity
2. **Diversified SFT traces** -- across easy/medium/hard_bench (not just easy)
3. **SFT validation split** -- 10% held out, early stopping prevents overfitting
4. **GRPO prompts across 3 tasks** -- prevents overfitting to single task
5. **Non-linear step bonus** -- `0.3*(steps/15)^2` creates reward variance even when enrollments are identical
6. **Reward jitter** -- +/-0.005 random noise breaks exact ties
7. **Lower SFT LR** -- 2e-5 (was 1e-4) prevents format memorization
8. **LoRA dropout** -- 0.1 regularization
9. **6 generations** -- more diverse completions per GRPO step (was 4)
10. **5 warmup steps** -- for GRPO optimizer

![Collapse Analysis](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/collapse_analysis.png)
*After all 10 fixes: only 8/80 steps (10%) had zero-std collapse, and the model recovered every time.*

## Reward Design

![Reward Components](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/grpo_reward_design.png)

The reward function has two components:
- **Task reward**: enrollment progress normalized to realistic episode capacity (not the full 80-patient target). This creates a delta of ~0.14 per enrollment instead of ~0.009.
- **Step bonus**: non-linear `0.3*(steps/15)^2` provides variance even when enrollment counts are identical across generations.
- **Reward jitter**: +/-0.005 random noise prevents exact ties that cause zero-std collapse in GRPO.

## Engineering Learnings

### TRL tool discovery is strict
Our earliest path used a generic `step(action)` wrapper. TRL's `environment_factory` needs explicit public methods like `screen_patient()`, `recontact()`, and `allocate_to_site()`, each with typed arguments and docstrings. That change is the difference between a trainer that can build tool schemas and one that silently learns nothing.

### Memory headroom > model ambition
T4 (16GB) couldn't fit GRPO backward pass. L4 (24GB) was the minimum for stable training with Qwen3-1.7B + LoRA. The lesson: pick the GPU that finishes, not the one that starts.

### Context window ceiling was the real bottleneck
With `max_completion_length=1024`, the model could only make ~6 tool calls per episode (1-2 patients). Increasing to 4096 enabled ~30 tool calls and 3-4 enrollments. This single change had more impact than any reward shaping.

### Cookie-based session isolation was essential
Each `ClinicalRecruitmentHTTPEnv` instance gets its own `httpx.Client` with cookies, creating an isolated session. Without this, parallel GRPO generations would clobber each other's environment state.

### SFT taught format faster than strategy
SFT improved loss, JSON formatting, and action diversity. But it did not teach enrollment. Strategy improvement required an online reward signal from the real environment.

### Some environment bugs were suppressing learning
Planning actions were over-penalized, consistency penalties were unbounded, current-step events leaked into observations, and session handling was too loose. Those weren't cosmetic bugs -- they changed what behaviors the model could learn.

## Heuristic Baseline Comparison

![Agent Comparison](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/heuristic_comparison.png)

| Task | Random | Heuristic | Optimized | Delta |
|------|--------|-----------|-----------|-------|
| easy_bench | 0.4076 | 0.5487 | 0.5711 | +40.1% |
| medium_bench | 0.2504 | 0.3163 | 0.4242 | +69.4% |
| hard_bench | 0.3274 | 0.2894 | 0.3845 | +17.4% |

## What This Post Does Not Claim

- The model did not solve the benchmark -- it enrolls 3-4/80 patients per episode.
- Reward improved modestly (0.27 to 0.33), not dramatically.
- 10% of GRPO steps still showed zero-std collapse.
- The 5-seed baseline sweep shows no statistically significant winner (HCAPO mean 0.2215, no pairwise p < 0.05).
- This is a synthetic benchmark, not a deployment-ready clinical system.

## Reproducing

```bash
# Run the environment locally
uvicorn app:app --host 0.0.0.0 --port 7860

# Generate training traces
python scripts/generate_traces.py --num 2000 --threads 8 --output data/sft_traces.json

# Train (requires GPU)
python train_sft_grpo_hfjob.py  # Best: SFT + GRPO with 10 fixes
python train.py                  # SFT-only

# Baseline sweep
python experiments/full_sweep.py --seeds 1 7 21 42 123 --episodes 30 --eval-episodes 5
```

## Links

- Live environment: [pratimassaravanan-clinical-recruitment.hf.space](https://pratimassaravanan-clinical-recruitment.hf.space)
- HF Space: [pratimassaravanan/clinical-recruitment](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment)
- Best model (80-step): [pratimassaravanan/grpo](https://huggingface.co/pratimassaravanan/grpo)
- First model (30-step): [pratimassaravanan/grpo_output](https://huggingface.co/pratimassaravanan/grpo_output)
- SFT+REINFORCE model: [pratimassaravanan/clinical-qwen3-4b-sft-lora](https://huggingface.co/pratimassaravanan/clinical-qwen3-4b-sft-lora)
- Artifacts: [pratimassaravanan/clinical-recruitment-artifacts](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts)
