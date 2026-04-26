---
title: Adaptive Clinical Recruitment
emoji: "\U0001F3E5"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
base_path: /web
pinned: false
license: mit
---

# Adaptive Clinical Trial Recruitment -- OpenEnv Environment

**A 180-step, long-horizon RL benchmark for clinical trial recruitment decisions.**
One simulated day = one step. Agents screen patients, recontact dropouts, allocate sites, manage budgets, and use memory -- all under real pharma pipeline pressure.

**Hackathon positioning: Theme #2.**

## Live Environment

- **Environment URL**: [https://pratimassaravanan-clinical-recruitment.hf.space](https://pratimassaravanan-clinical-recruitment.hf.space)
- **HF Space**: [https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment)
- **Gradio / OpenEnv UI**: [https://pratimassaravanan-clinical-recruitment.hf.space/web/](https://pratimassaravanan-clinical-recruitment.hf.space/web/)
- **Dashboard alias**: [https://pratimassaravanan-clinical-recruitment.hf.space/dashboard](https://pratimassaravanan-clinical-recruitment.hf.space/dashboard)

## All Submission Materials

| Material | Link |
|----------|------|
| Live environment | [pratimassaravanan-clinical-recruitment.hf.space](https://pratimassaravanan-clinical-recruitment.hf.space) |
| Blog writeup | [`Blog.md`](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/Blog.md) |
| Best trained model (80-step SFT+GRPO) | [`pratimassaravanan/grpo`](https://huggingface.co/pratimassaravanan/grpo) |
| Final local bundle | [`final_bundle/grpo_job-69ed6d57/`](final_bundle/grpo_job-69ed6d57/) |
| 30-step GRPO model | [`pratimassaravanan/grpo_output`](https://huggingface.co/pratimassaravanan/grpo_output) |
| SFT+REINFORCE model (Qwen3-4B) | [`pratimassaravanan/clinical-qwen3-4b-sft-lora`](https://huggingface.co/pratimassaravanan/clinical-qwen3-4b-sft-lora) |
| Colab notebook (re-runnable) | [`notebooks/clinical_recruitment_grpo.ipynb`](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/notebooks/clinical_recruitment_grpo.ipynb) |
| Training script (best run) | [`train_sft_grpo_hfjob.py`](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/train_sft_grpo_hfjob.py) |
| Large artifacts | [`pratimassaravanan/clinical-recruitment-artifacts`](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts) |
| Communication deck | `docs/communication/adaptive_clinical_recruitment_presentation.html` |

## Final Best Model

Use this as the single final answer for the trained agent:

- Final best model: [`pratimassaravanan/grpo`](https://huggingface.co/pratimassaravanan/grpo)
- Best job/run: `69ed6d57`
- Method: `SFT warmup + 80-step GRPO`
- Base model: `Qwen/Qwen3-1.7B`
- Final local handoff folder: `final_bundle/grpo_job-69ed6d57/`

Comparison-only models:

- [`pratimassaravanan/grpo_output`](https://huggingface.co/pratimassaravanan/grpo_output)
- [`pratimassaravanan/clinical-qwen3-4b-sft-lora`](https://huggingface.co/pratimassaravanan/clinical-qwen3-4b-sft-lora)

## Why This Matters

80% of clinical trials miss enrollment targets. Average delay = 6-12 months, costing sponsors $600K-$8M per day. This environment turns clinical trial coordinator decisions into a trainable RL benchmark: which patients to screen first, when to spend effort on recontact, which sites deserve scarce capacity, and when to change strategy under budget and dropout pressure.

Those decisions play out over weeks or months of simulated time, with delayed effects, constraint pressure, and recovery actions. That makes the environment a better fit for Theme #2 than a short-horizon reward toy.

## Environment at a Glance

| Dimension | Value |
|-----------|-------|
| Steps per episode | `180` (one simulated day per step) |
| Difficulty tiers | `3` (`easy_bench`, `medium_bench`, `hard_bench`) |
| Action types | `8` typed actions |
| Observation features | `37`-dimensional numeric vector + structured typed fields |
| OpenEnv version | `0.2.3` (latest) |
| Framework | FastAPI + OpenEnv `Environment` base class + Gradio web UI |

### Tasks

| Task | Sites | Budget | Target | Main pressure |
|------|-------|--------|--------|---------------|
| `easy_bench` | 1 | $120K | 80 | Basic funnel dynamics |
| `medium_bench` | 3 | $150K | 120 | Multi-site coordination |
| `hard_bench` | 5 | $100K | 150 | Budget, retention, and constraint pressure |

### 8 Action Types

| Action | What it does |
|--------|-------------|
| `screen_patient` | Choose from `observation.available_patients` |
| `recontact` | Follow up with `observation.recontact_candidates` |
| `allocate_to_site` | Assign patient to site from `observation.allocation_candidates` |
| `adjust_strategy` | Change approach (e.g., `increase_outreach`, `negotiate_site_A`) |
| `plan_next_phase` | Set phase target (`screening`, `conversion`, `allocation`, etc.) |
| `summarize_and_index` | Write to episodic memory |
| `retrieve_relevant_history` | Read from episodic memory |
| `stop_recruitment` | End episode early |

### OpenEnv Action Guide

Only `action_type` is always required. Leave irrelevant fields blank.

- `screen_patient`: use `patient_id` from `available_patients`; optional `hypothesis`, `confidence`
- `recontact`: use `patient_id` from `recontact_candidates`; optional `hypothesis`, `confidence`
- `allocate_to_site`: use `patient_id` from `allocation_candidates` and `site_id` from `site_performance`
- `adjust_strategy`: use `strategy_change`; optional `hypothesis`, `confidence`
- `plan_next_phase`: use `target_phase`; optional `plan_summary`, `plan_id`
- `summarize_and_index`: use `memory_key`, `memory_payload`
- `retrieve_relevant_history`: use `memory_query`
- `stop_recruitment`: no extra parameters

The built-in Playground labels come from the OpenEnv schema. The custom `Custom` tab in `/web/` now renders the field descriptions and examples explicitly.

### Rich Observation

- Funnel state and action-specific candidate pools
- Per-site performance metrics
- Milestones, constraints, and delayed-effects state
- Plan state and indexed-memory summaries
- Token budget and token-efficiency signals
- Counterfactual hints and rollout estimates

## Training Evidence (Real Runs)

### Best Run: SFT + 80-Step GRPO (HF Job `69ed6d57`)

SFT warmup followed by 80-step GRPO on Qwen3-1.7B (NVIDIA L4, 24GB). Model: [`pratimassaravanan/grpo`](https://huggingface.co/pratimassaravanan/grpo).

![GRPO 80-Step Training Summary](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/grpo_80step_training.png)
*80-step GRPO training: reward progression (0.27->0.33), tool call growth (3.5->11), and collapse events.*

**SFT Phase (warmup):** Loss 0.7495, token accuracy 94.9%, 3.5 min. Teaches JSON format + action diversity.

**GRPO Phase (80 steps, 141.7 min):**

| Metric | Start (step 1) | End (step 80) | Change |
|--------|---------------|---------------|--------|
| Reward | 0.269 | 0.331 | **+23%** |
| Tool calls/step | 3.5 | 11 | **3x growth** |
| Enrollment/rollout | 1-2 | 3-4 | **Consistent improvement** |
| Zero-std collapse | 0% | 10% intermittent | Self-recovering |

**What the model learned:**
- Calls `screen_patient` first (previously collapsed to only `adjust_strategy`)
- Follows correct pipeline: screen -> recontact -> enrollment
- Enrolls 3-4 patients per episode
- Makes 11+ tool calls per rollout (up from 3.5)

**Honest limits:**
- Enrollment stays at 3-4/80 target patients
- Reward plateaued at 0.33 after 80 steps
- 10% intermittent zero-std collapse

### All Runs Comparison

![All Runs Comparison](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/all_runs_comparison.png)
*Three training runs compared: SFT+GRPO 80-step (best), GRPO-only 30-step, and SFT+REINFORCE.*

| Run | Model | GPU | Method | Steps | Reward | Enrolled |
|-----|-------|-----|--------|-------|--------|----------|
| **Best** | Qwen3-1.7B | L4 | SFT+GRPO | 80 | 0.27->0.33 | 3-4/rollout |
| First | Qwen3-1.7B | L4 | GRPO-only | 30 | 0.31 (flat) | 1-2/rollout |
| Qwen3-4B | Qwen3-4B | L40S | SFT+REINFORCE | 15 | 6.6-7.4 | 0->5 |

### Collapse Analysis

![Collapse Analysis](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/collapse_analysis.png)
*Zero-std collapse: 8/80 steps (10%). The model recovered every time.*

### Enrollment Progression

![Enrollment Progression](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/enrollment_progression.png)
*Patient enrollment growth across training.*

### Heuristic vs Random Baseline

![Agent Performance Comparison](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/heuristic_comparison.png)

| Task | Random | Heuristic | Optimized | Delta |
|------|--------|-----------|-----------|-------|
| `easy_bench` | 0.4076 | 0.5487 | **0.5711** | **+40.1%** |
| `medium_bench` | 0.2504 | 0.3163 | **0.4242** | **+69.4%** |
| `hard_bench` | 0.3274 | 0.2894 | **0.3845** | **+17.4%** |

### GRPO Reward Design

![Reward Components](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/grpo_reward_design.png)
*Composite reward: task_reward (enrollment normalized to episode capacity) + step_bonus (non-linear). Reward jitter prevents zero-variance collapse.*

## Key Learnings

1. **SFT collapse is fundamental**: Two independent SFT runs (2K and 16K traces, on Qwen3-4B and 1.7B) both collapsed to 100% `adjust_strategy`. More data doesn't help.
2. **SFT warmup before GRPO is critical**: Without SFT, GRPO collapses (loss=0) in 3 of 4 attempts. SFT teaches format; GRPO teaches strategy.
3. **Context window ceiling was the real bottleneck**: `max_completion_length=1024` only allowed ~6 tool calls. Increasing to 4096 enabled ~30 tool calls and 3-4 enrollments.
4. **10 anti-collapse fixes were needed**: No single fix solved policy collapse. Temperature=1.2, 6 generations, SFT warmup, reward jitter, diversified traces, and non-linear step bonuses worked together.
5. **Reward granularity matters**: Normalizing to target=80 (delta=0.009) produced no signal. Normalizing to realistic episode capacity (delta=0.14) did.
6. Cookie-based session isolation was essential for parallel GRPO generations against the live environment.

### 10 Anti-Collapse Fixes

1. Temperature 1.2
2. Diversified SFT traces (across all 3 tasks)
3. SFT validation split + early stopping
4. GRPO prompts across 3 tasks
5. Non-linear step bonus: `0.3*(steps/15)^2`
6. Reward jitter +/-0.005
7. Lower SFT LR: 2e-5
8. LoRA dropout: 0.1
9. 6 generations per GRPO step
10. 5 warmup steps for GRPO optimizer

## Try It Yourself

### Live API

```bash
curl -X POST "https://pratimassaravanan-clinical-recruitment.hf.space/reset?task_id=easy_bench"

curl -X POST "https://pratimassaravanan-clinical-recruitment.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "screen_patient", "patient_id": "<id>", "hypothesis": "noise_dominant", "confidence": 0.7}'
```

### Python

```python
from env import ClinicalRecruitmentEnv
from models import Action

env = ClinicalRecruitmentEnv()
result = env.reset(task="medium_bench")

while not result.done:
    obs = result.observation
    if obs.allocation_candidates and obs.site_performance:
        action = Action(action_type="allocate_to_site",
                       patient_id=obs.allocation_candidates[0]["id"],
                       site_id=next(iter(obs.site_performance)))
    elif obs.recontact_candidates:
        action = Action(action_type="recontact",
                       patient_id=obs.recontact_candidates[0]["id"])
    elif obs.available_patients:
        action = Action(action_type="screen_patient",
                       patient_id=obs.available_patients[0]["id"],
                       hypothesis="noise_dominant", confidence=0.7)
    else:
        action = Action(action_type="adjust_strategy",
                       strategy_change="increase_outreach")
    result = env.step(action)

print(result.info.get("final_score"))
```

## Training Pipeline

### GRPO via TRL `environment_factory` on HF Jobs

```python
# train_sft_grpo_hfjob.py — self-contained, runs on HF Jobs (L4 GPU)
class ClinicalRecruitmentHTTPEnv:
    """Each instance gets its own httpx.Client -> separate session."""
    def screen_patient(self, patient_id, hypothesis="noise_dominant", confidence=0.7):
        return self._step("screen_patient", patient_id=patient_id, ...)
    def recontact(self, patient_id, ...): ...
    def allocate_to_site(self, patient_id, site_id, ...): ...
    def adjust_strategy(self, strategy_change="increase_outreach", ...): ...

trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=[reward_func],
    environment_factory=ClinicalRecruitmentHTTPEnv,
    args=GRPOConfig(num_generations=6, max_completion_length=4096),
    peft_config=LoraConfig(r=16, ...),
)
```

**Training scripts:**
- `train_sft_grpo_hfjob.py` -- SFT+GRPO with 10 anti-collapse fixes (best run)
- `train_grpo_hfjob.py` -- GRPO via HF Jobs (first 30-step run)
- `train_grpo_from_sft.py` -- GRPO-only from pre-merged SFT checkpoint
- `train_grpo_trl.py` -- GRPO with local env
- `train.py` -- SFT-only on expert traces

**Colab notebook** (judges can re-run on T4): `notebooks/clinical_recruitment_grpo.ipynb`

## 5-Seed Baseline Sweep

| Baseline | Mean | Std | 95% CI |
|----------|------|-----|--------|
| HCAPO | 0.2215 | 0.0127 | [0.2100, 0.2303] |
| KLong | 0.2152 | 0.0222 | [0.1977, 0.2286] |
| MemexRL | 0.2148 | 0.0270 | [0.1943, 0.2352] |
| MiRA | 0.2094 | 0.0095 | [0.2023, 0.2165] |

No pairwise comparison reaches `p < 0.05`. The baseline suite is too tightly clustered to support a winner claim.

## Verification

| Suite | Result |
|-------|--------|
| `test_env.py` | 76/76 |
| `test_agents.py` | 43/43 |
| `test_research_modules.py` | 109/109 |
| `test_local_serving.py` | 77/77 |

## Caveats

- This is a synthetic benchmark, not a deployment-ready clinical operations system.
- The current 5-seed sweep does not show statistically significant separation among baselines.
- The best GRPO model enrolls 3-4/80 target patients -- far from solved.
- SFT alone produces policy collapse (100% `adjust_strategy`); RL is necessary but not sufficient.

## License

MIT License. See `LICENSE`.
