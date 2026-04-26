---
base_model: Qwen/Qwen3-4B
library_name: peft
pipeline_tag: text-generation
tags:
- clinical-trial
- reinforcement-learning
- REINFORCE
- lora
- qwen3
- openenv
- hackathon
- theme-2
license: mit
---

# Clinical Trial Recruitment Agent — Qwen3-4B + RL

A clinical trial recruitment agent trained with **REINFORCE** on a 180-step long-horizon environment. Theme 2 submission for the OpenEnv Hackathon.

## Overview

| Component | Detail |
|---|---|
| **Base Model** | Qwen/Qwen3-4B (4-bit quantized) |
| **Method** | Fresh LoRA (r=16, alpha=32) + REINFORCE policy gradient |
| **Environment** | Clinical Recruitment Env (8 action types, 37 observation features, 180-step horizon) |
| **Training GPU** | NVIDIA L40S (48GB) on Lightning AI |
| **HF Space** | [pratimassaravanan/clinical-recruitment](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment) |

## Training Pipeline

### Phase 1: SFT (Supervised Fine-Tuning)
- **Data**: 2,000 heuristic-generated traces (observation → JSON action pairs)
- **Result**: 100% JSON parse rate but **complete policy collapse** — model only outputs `adjust_strategy`
- **Loss**: 3.14 → 0.015 (perfect memorization, zero generalization)
- **Finding**: SFT teaches format but not observation-conditional behavior

### Phase 2: REINFORCE (Policy Gradient RL)
- **Key fix**: Observation parsing bug (`result.get("observation")` → `result` — API returns flat dict)
- **Key fix**: Heuristic override prevents degenerate actions when candidates exist
- **Reward shaping**: +0.20 for allocate, +0.15 for recontact, +0.10 for screen, -0.10 for adjust_strategy when productive actions available

**Debug Trial Results (3 episodes, 15 steps each):**

| Episode | Task | Enrolled | Target | Reward | Action Distribution |
|---|---|---|---|---|---|
| 0 | easy_bench | **5** | 80 | 6.61 | screen=6, allocate=9 |
| 1 | easy_bench | **5** | 80 | 6.61 | screen=6, allocate=9 |
| 2 | easy_bench | **1** | 80 | 7.41 | screen=14, allocate=1 |

**Improvement over SFT**: 0 → 5 enrolled patients, diverse action distribution vs. 100% adjust_strategy collapse.

### Phase 3: Full REINFORCE Run (30 episodes)
- 30 episodes across easy/medium/hard benchmarks
- Results uploading upon completion

## Key Findings

1. **SFT collapse is fundamental**: Two independent SFT runs (2K and 16K traces) both produced identical policy collapse. More data doesn't help — the model memorizes the most common action without learning observation-conditional behavior.

2. **Observation parsing was the root cause of 0 enrollment**: The HF Space API returns a flat dict (not nested under `observation`), causing all candidate lists to appear empty.

3. **Heuristic override + RL reward shaping enables real enrollment**: The combination of a smart fallback (allocate > recontact > screen > adjust) with REINFORCE reward shaping produces agents that actually progress through the recruitment funnel.

4. **Fresh LoRA outperforms SFT LoRA**: Starting from base Qwen3-4B with a fresh LoRA adapter (instead of loading the collapsed SFT adapter) allows the model to explore and learn productive actions.

## Repository Structure

```
clinical-recruitment-env/
├── env.py                  # 180-step clinical trial environment
├── models.py               # Action/Observation Pydantic models  
├── graders.py              # Task-specific scoring (easy/medium/hard)
├── app.py                  # FastAPI server
├── openenv_adapter.py      # OpenEnv protocol adapter
├── train.py                # SFT training script (local, uses Python API)
├── _lightning_reinforce.py  # REINFORCE v3 (Lightning AI, uses HTTP API)
├── _reinforce_v4.py         # REINFORCE v4 (gentler reward shaping)
└── _debug_trial.py          # Debug trial (3 episodes, full logging)
```

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", load_in_4bit=True)
model = PeftModel.from_pretrained(base, "pratimassaravanan/clinical-qwen3-4b-sft-lora/rl_v3_adapter")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
```

## Environment Details

- **Actions**: screen_patient, recontact, allocate_to_site, adjust_strategy, plan_next_phase, summarize_and_index, retrieve_relevant_history, stop_recruitment
- **Observation**: 37 features including patient lists, site performance, funnel metrics, world_type
- **Reward**: Enrollment (+0.50), screening (+0.30), dropout (-0.35), milestone bonuses, hypothesis accuracy (+0.10)
- **Tasks**: easy_bench (80 target), medium_bench (100 target), hard_bench (150 target)

## Compute & Cost

| Job | GPU | Duration | Cost |
|---|---|---|---|
| SFT v1 (2K traces) | L40S | ~30 min | ~$2.50 |
| SFT v2 (16K traces) | L40S | ~60 min | ~$5.00 |
| RL debug trial | L40S | ~5 min | $0.44 |
| RL v3-fixed (30 eps) | L40S | ~60 min | ~$5.00 |

## License

MIT
