---
base_model: unsloth/Qwen3-4B-unsloth-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/Qwen3-4B-unsloth-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
- openenv
- clinical-recruitment
---

# Clinical Recruitment SFT LoRA Adapter (Qwen3-4B)

LoRA adapter fine-tuned on expert clinical trial recruitment trajectories using SFT.

## Training Details

- **Base model**: `unsloth/Qwen3-4B-unsloth-bnb-4bit` (4-bit quantized)
- **Training method**: SFT (Supervised Fine-Tuning) via Unsloth + TRL
- **GPU**: Tesla T4 (Kaggle/Colab)
- **Training data**: 18 expert trajectories from the Clinical Recruitment OpenEnv environment
- **Training steps**: 9 (3 epochs x 6 batches)
- **LoRA config**: r=16, alpha=16, dropout=0, target modules: q/k/v/o/gate/up/down_proj

## Results

| Metric | Before SFT | After SFT |
|--------|-----------|-----------|
| Training loss | 0.858 | 0.745 (-13.2%) |
| Action types used | 1 (screen_patient only) | 5 distinct types |
| JSON parse rate | ~0% | ~85% |
| Uses allocate_to_site | No | Yes |
| Enrollment | 0 | 0 |

The adapter learned **output format** (structured JSON tool calls) and **action diversity** but not the full recruitment pipeline. This is consistent with the limited training budget (9 steps on 18 examples).

## Environment

This adapter was trained on the [Adaptive Clinical Recruitment OpenEnv](https://huggingface.co/spaces/pratimassaravanan/clinical-recruitment) — a 180-step long-horizon benchmark for clinical trial planning decisions.

## Limitations

- Enrollment remained at 0 after SFT — the model learned format but not strategy
- The model copies example patient_id values from the system prompt instead of reading the observation
- 9 SFT steps is insufficient to override the base model's strong instruct behavior
- This adapter is from an earlier pilot; the current training pipeline uses `train_grpo_trl.py` with GRPO

## Framework Versions

- PEFT 0.19.1
- Unsloth (latest)
- TRL >= 0.19.0
- Transformers >= 5.2.0
