# Training Learnings & Findings

## What We Tried

### 1. GRPO with TRL `environment_factory` (FAILED)
- **Models tried**: Qwen3-0.6B, Qwen3-4B, Qwen3-8B, DeepSeek-R1-8B
- **Result**: All rewards 0.0 across 24 steps every time
- **Root cause**: TRL calls reward functions with `(prompts=..., completions=...)` but never passes `environments` kwarg. Our reward functions need environment state to compute meaningful rewards.
- **Conclusion**: TRL's `environment_factory` + custom `reward_funcs` path is fundamentally broken for tool-call environments where rewards depend on environment state.

### 2. SFT on Expert Traces (PARTIALLY WORKED)
- **Models tried**: Qwen3-4B (loss 0.858→0.745), Qwen3-8B (loss 0.925→0.831), DeepSeek-R1-8B (loss 0.991→0.896)
- **Result**: Loss converged in all cases, but models still output verbose reasoning instead of raw JSON actions
- **Root cause**: 9 SFT steps (18 examples × 3 epochs) is insufficient to override strong instruct behavior. These models are trained on millions of examples to be verbose helpers — our 18 traces can't overcome that.
- **What worked**: The training infrastructure is correct. Loss drops. Format is partially learned.

### 3. Strict JSON System Prompt (PARTIALLY WORKED)
- **Result with Qwen3-4B**: Clean JSON output ({"action_type": "screen_patient", ...}) but model copies example patient_id P-1042 instead of reading observation
- **Result with Qwen3-8B**: Still outputs markdown/reasoning with rationale sections
- **Result with DeepSeek-R1-8B**: Outputs `<think>` blocks before any content

### 4. DeepSeek R1 Reasoning Model (FAILED FOR THIS TASK)
- Outputs `<think>` blocks that contain keywords like "screen", "enroll", "allocate"
- Parser catches these keywords in reasoning text instead of actual action output
- The thinking model format is incompatible with tool-call parsing

### 5. Kaggle GPU Assignment (INFRASTRUCTURE ISSUE)
- Kaggle kept assigning Tesla P100 (sm_60) which is incompatible with modern PyTorch/Unsloth/bitsandbytes
- Required 20+ kernel push attempts to work around
- Eventually used "Open in Colab" from Kaggle to get reliable T4
- H200 on Lightning AI worked but Qwen3-32B crashed with torch._dynamo shape mismatch at batch_size=16

### 6. Qwen3-8B Instruct (SAME RESULT AS 4B)
- Outputs verbose markdown reasoning with **Rationale** sections
- Action distribution: `{'screen_patient': 40}` every task -- same as 4B
- SFT loss: 0.925 → 0.831 (10.2% reduction) but doesn't change output format
- The instruct behavior is too strong for 9 SFT steps to override

### 7. Model Selection Findings
- Qwen3 models (4B, 8B, 32B) score **3.33% on agentic coding** benchmarks -- very weak at tool-calling
- DeepSeek R1 distills output `<think>` blocks that break action parsing
- DeepSeek V3.2 is 685B -- can't fine-tune
- Kimi K2.6 is 1T -- can't fine-tune
- **Gemma 4 E4B scores 40% on agentic coding** -- 12x better than Qwen3 for tool-calling
- Switched to `unsloth/gemma-4-E4B-it-unsloth-bnb-4bit` for final training attempt

## Key Numbers

| Model | SFT Loss Start | SFT Loss End | Reduction | Enrolled After |
|-------|---------------|-------------|-----------|----------------|
| Qwen3-4B | 0.858 | 0.745 | 13.2% | 0 |
| Qwen3-8B | 0.925 | 0.831 | 10.2% | 0 |
| DeepSeek-R1-8B | 0.991 | 0.896 | 9.6% | 0 |
| Gemma 4 E4B | TBD | TBD | TBD | TBD |

| Evidence | Value |
|----------|-------|
| Expert trajectories (heuristic) | easy: 4-11/80, medium: 1-11/120, hard: 0/150 |
| Heuristic vs random improvement | +23.9% average across 3 tasks |
| Action types after SFT (Qwen3-4B) | 5 types (was 1 before SFT) |
| GRPO reward signal | 0.0 across all models and all steps |

## What Would Actually Work (Given More Time)

1. **50+ SFT epochs** instead of 3 — need to overfit on the JSON format before the model's instruct behavior dominates
2. **Larger SFT dataset** — 100+ diverse traces, not 18
3. **Manual REINFORCE** (implemented in `train_final.py` and `train_dual_gpu.py`) — bypasses TRL's broken environment_factory entirely
4. **Dual-GPU teacher-student** (implemented in `train_dual_gpu.py`) — 8B teacher generates traces, 4B student learns from them with direct reward computation
5. **Strip `<think>` tags** before parsing for reasoning models
6. **Higher learning rate for SFT** (1e-4 instead of 2e-5) to overcome instruct priors faster

## Hackathon Guide Alignment

The hackathon guide warned:
> "RL only works if the probability of getting a good answer is greater than zero. If your task is so hard that the model never gets any reward, you will burn compute and learn nothing."

This is exactly what happened with GRPO. The model never produced valid tool calls that the environment could score, so GRPO got zero reward signal.

The guide also said:
> "In many practical cases, do a little SFT first, then RL."

We did this. SFT produced measurable loss improvement and format learning. The RL step (GRPO) failed due to TRL API limitations, not because the approach is wrong.

## Files

| File | Purpose | Status |
|------|---------|--------|
| `train.py` | Single-GPU GRPO (any model) | Works but GRPO rewards are 0 |
| `train_h200.py` | H200 optimized Qwen3-32B | Crashed on dynamo shape mismatch |
| `train_final.py` | Single-GPU SFT + manual REINFORCE | Correct approach, needs more epochs |
| `train_dual_gpu.py` | Dual-GPU teacher-student | Correct approach, uses DeepSeek R1 or Qwen3-8B as teacher |
| `kaggle_kernel/sft_then_grpo.ipynb` | Kaggle notebook | Latest: Qwen3-8B SFT + GRPO |
| `scripts/before_after_demo.py` | Heuristic vs random comparison | Works: +23.9% improvement |
| `data/training_outputs/sft_grpo_results.json` | Training evidence | Committed |
