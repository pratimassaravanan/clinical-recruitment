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

# Adaptive Clinical Trial Recruitment — OpenEnv Environment

**Theme #2 (long-horizon planning).** A 180-step RL benchmark for clinical trial recruitment. One simulated day = one step. Agents screen patients, recontact dropouts, allocate sites, and manage budgets over a full recruitment campaign.

Clinical trial recruitment involves sequential, delayed decisions — which patients to screen, when to recontact dropouts, how to allocate site capacity, when to change strategy under budget pressure. This benchmark makes those decisions trainable.

## Submission Materials

| Material | Link |
|----------|------|
| Live environment | [pratimassaravanan-clinical-recruitment.hf.space](https://pratimassaravanan-clinical-recruitment.hf.space) |
| **Best model (SFT+GRPO, 80-step)** | [`pratimassaravanan/grpo`](https://huggingface.co/pratimassaravanan/grpo) |
| Training script (best run) | [`train_sft_grpo_hfjob.py`](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/train_sft_grpo_hfjob.py) |
| Colab notebook (re-runnable) | [`notebooks/clinical_recruitment_grpo.ipynb`](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/notebooks/clinical_recruitment_grpo.ipynb) |
| Blog writeup | [`Blog.md`](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/Blog.md) |
| Local bundle | [`final_bundle/grpo_job-69ed6d57/`](final_bundle/grpo_job-69ed6d57/) |
| Large artifacts | [`pratimassaravanan/clinical-recruitment-artifacts`](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts) |

**Comparison models:** [`pratimassaravanan/grpo_output`](https://huggingface.co/pratimassaravanan/grpo_output) (GRPO-only, 30-step) · [`pratimassaravanan/clinical-qwen3-4b-sft-lora`](https://huggingface.co/pratimassaravanan/clinical-qwen3-4b-sft-lora) (SFT+REINFORCE, Qwen3-4B)

## Environment

| Dimension | Value |
|-----------|-------|
| Steps per episode | `180` (one simulated day per step) |
| Difficulty tiers | `3` (`easy_bench`, `medium_bench`, `hard_bench`) |
| Action types | `8` |
| Observation features | `37`-dimensional numeric vector + structured typed fields |
| OpenEnv version | `0.2.3` |
| Framework | FastAPI + OpenEnv `Environment` base class + Gradio web UI |

### Tasks

| Task | Sites | Budget | Target | Pressure |
|------|-------|--------|--------|----------|
| `easy_bench` | 1 | $120K | 80 | Basic funnel |
| `medium_bench` | 3 | $150K | 120 | Multi-site coordination |
| `hard_bench` | 5 | $100K | 150 | Budget, retention, constraints |

### 8 Actions

| Action | What it does |
|--------|-------------|
| `screen_patient` | Choose from `observation.available_patients` |
| `recontact` | Follow up with `observation.recontact_candidates` |
| `allocate_to_site` | Assign patient to site from `observation.allocation_candidates` |
| `adjust_strategy` | Change approach (e.g. `increase_outreach`, `negotiate_site_A`) |
| `plan_next_phase` | Set phase target (`screening`, `conversion`, `allocation`, …) |
| `summarize_and_index` | Write to episodic memory |
| `retrieve_relevant_history` | Read from episodic memory |
| `stop_recruitment` | End episode early |

Only `action_type` is required. Field reference:
- `screen_patient` / `recontact`: `patient_id`, optional `hypothesis`, `confidence`
- `allocate_to_site`: `patient_id` + `site_id`
- `adjust_strategy`: `strategy_change`, optional `hypothesis`, `confidence`
- `plan_next_phase`: `target_phase`, optional `plan_summary`, `plan_id`
- `summarize_and_index`: `memory_key`, `memory_payload`
- `retrieve_relevant_history`: `memory_query`

## Training Results

### Best Run: SFT warmup + 80-step GRPO (job `69ed6d57`)

Model: [`pratimassaravanan/grpo`](https://huggingface.co/pratimassaravanan/grpo) — Qwen3-1.7B on NVIDIA L4 (24GB).

![GRPO 80-Step Training Summary](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/grpo_80step_training.png)
*Reward progression (0.27→0.33), tool call growth (3.5→11), collapse events.*

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| Reward | 0.269 | 0.331 | **+23%** |
| Tool calls/step | 3.5 | 11 | **3×** |
| Enrollment/rollout | 1–2 | 3–4 | ↑ |
| Zero-std collapse | 0% | 10% intermittent | Self-recovering |

### All Runs

![All Runs Comparison](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/all_runs_comparison.png)

| Run | Model | Method | Steps | Reward | Enrolled |
|-----|-------|--------|-------|--------|----------|
| **Best** | Qwen3-1.7B | SFT+GRPO | 80 | 0.27→0.33 | 3–4/rollout |
| First | Qwen3-1.7B | GRPO-only | 30 | 0.31 flat | 1–2/rollout |
| 4B run | Qwen3-4B | SFT+REINFORCE | 15 | 6.6–7.4 | 0→5 |

### Collapse & Enrollment

![Collapse Analysis](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/collapse_analysis.png)
*Zero-std collapse: 8/80 steps (10%), self-recovering.*

![Enrollment Progression](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/enrollment_progression.png)

### Rule-Based Baseline Comparison (not the GRPO model)

![Agent Performance Comparison](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/heuristic_comparison.png)

| Task | Random | Heuristic | Optimized Heuristic | Delta vs. Random |
|------|--------|-----------|---------------------|------------------|
| `easy_bench` | 0.4076 | 0.5487 | **0.5711** | +40.1% |
| `medium_bench` | 0.2504 | 0.3163 | **0.4242** | +69.4% |
| `hard_bench` | 0.3274 | 0.2894 | **0.3845** | +17.4% |

*All three are hand-coded rule-based agents from `demo/training_demo.py`, run once at seed=42. "Optimized Heuristic" uses dropout-recovery and funnel-priority rules — it is **not** the GRPO-trained LLM. The GRPO model's actual reward improvement (0.269→0.331) is shown in the training plots above.*

### Reward Design

![Reward Components](https://huggingface.co/pratimassaravanan/clinical-recruitment-artifacts/resolve/main/article_assets/grpo_reward_design.png)
*task_reward (enrollment normalized to episode capacity) + non-linear step_bonus. Jitter ±0.005 prevents zero-variance collapse.*

## Key Learnings

1. **SFT alone collapses** — both 2K and 16K trace runs collapsed to 100% `adjust_strategy`. More data didn't help.
2. **SFT warmup before GRPO is essential** — GRPO without SFT collapses (loss=0) in 3/4 attempts. SFT teaches format; GRPO teaches strategy.
3. **Context window was the bottleneck** — `max_completion_length=1024` → ~6 tool calls. `4096` → ~30 tool calls, 3–4 enrollments.
4. **10 fixes were needed together** — no single change prevented collapse:
   Temperature 1.2 · diversified SFT traces · validation split + early stopping · GRPO across 3 tasks · non-linear step bonus `0.3*(steps/15)²` · reward jitter ±0.005 · SFT LR 2e-5 · LoRA dropout 0.1 · 6 generations/step · 5 GRPO warmup steps
5. **Reward normalization** — normalizing to target=80 (delta=0.009) gave no signal; normalizing to episode capacity (delta=0.14) worked.

## Quick Start

```bash
# Reset environment
curl -X POST "https://pratimassaravanan-clinical-recruitment.hf.space/reset?task_id=easy_bench"

# Take a step
curl -X POST "https://pratimassaravanan-clinical-recruitment.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "screen_patient", "patient_id": "<id>", "hypothesis": "noise_dominant", "confidence": 0.7}'
```

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

```python
# train_sft_grpo_hfjob.py — self-contained, runs on HF Jobs (L4 GPU)
trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B",
    reward_funcs=[reward_func],
    environment_factory=ClinicalRecruitmentHTTPEnv,  # each instance gets its own session
    args=GRPOConfig(num_generations=6, max_completion_length=4096),
    peft_config=LoraConfig(r=16, ...),
)
```

Re-runnable Colab notebook: [`notebooks/clinical_recruitment_grpo.ipynb`](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment/blob/main/notebooks/clinical_recruitment_grpo.ipynb)

## Tests & Caveats

| Suite | Result |
|-------|--------|
| `test_env.py` | 76/76 |
| `test_agents.py` | 43/43 |
| `test_research_modules.py` | 109/109 |
| `test_local_serving.py` | 77/77 |

- Synthetic benchmark — not a clinical deployment system.
- Best model enrolls 3–4 of 80 target patients — far from solved.
- SFT alone causes collapse; RL is necessary but not sufficient.

## License

MIT. See [`LICENSE`](LICENSE).
