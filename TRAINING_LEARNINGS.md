# Training Learnings & Findings

This file is a historical engineering log. Use `README.md` for the current public claim set and current benchmark truth.

## What We Tried

### 1. GRPO with TRL `environment_factory` (FAILED — ROOT CAUSE IDENTIFIED)
- **Models tried**: Qwen3-0.6B, Qwen3-4B, Qwen3-8B, DeepSeek-R1-8B
- **Result**: All rewards 0.0 across 24 steps every time
- **Root causes identified** (code review, April 25):
  1. **Wrong env class**: We passed `ClinicalRecruitmentToolEnv` with a generic `_step()` method, but TRL expects individual tool methods (`screen_patient()`, `recontact()`, etc.) with typed `Args:` docstrings. TRL auto-discovers public methods to build tool schemas.
  2. **Unsloth patching**: Unsloth's compiled `UnslothGRPOTrainer._calculate_rewards` may drop the `environments` kwarg that TRL passes to reward functions.
- **Fix applied**: Created `tool_env.py` with proper TRL-compatible tool methods per the [TRL OpenEnv docs](https://huggingface.co/docs/trl/openenv).

### 2. SFT on Expert Traces (PARTIALLY WORKED)
- **Models tried**: Qwen3-4B (loss 0.858→0.745), Qwen3-8B (loss 0.925→0.831), DeepSeek-R1-8B (loss 0.991→0.896)
- **Result**: Loss converged in all cases, but models still output verbose reasoning instead of raw JSON actions
- **Root cause**: 9 SFT steps (18 examples × 3 epochs) is insufficient to override strong instruct behavior
- **What worked**: The training infrastructure is correct. Loss drops. Format is partially learned.

### 3. Strict JSON System Prompt (PARTIALLY WORKED)
- **Result with Qwen3-4B**: Clean JSON output but model copies example patient_id P-1042 instead of reading observation
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

## Critical Bugs Found & Fixed

### Bug 1: Planning/Memory Reward Hard-Clamp (env.py:1632-1640)
- **Old**: `r = min(r, -0.01)` — planning actions always got negative reward
- **Fix**: Removed the hard clamp entirely. Planning has a -0.03 opportunity cost (from FIX 4) that can be offset by plan_bonus (+0.015)
- **Impact**: Model could never learn "plan strategically" — it was always punished

### Bug 2: Unbounded Consistency Penalty (env.py:1495-1509)
- **Old**: `-0.10` every step after >2 hypothesis switches = -18.0 over 180 steps
- **Fix**: `min(0.05, (switches - 2) * 0.01)` — proportional, capped
- **Impact**: Model learned "never change hypothesis" regardless of evidence

### Bug 3: Observation Future Event Leak (env.py:1732)
- **Old**: `self._events.get(self._step, [])` — leaked current-step events
- **Fix**: `self._events.get(max(0, self._step - 1), [])` — uses previous step

### Bug 4: Double Penalty on Planning (env.py:1571-1581 + 1635-1640)
- **Old**: FIX 4 inaction penalty (-0.06 to -0.07) + opportunity cost (-0.02) = -0.08 to -0.09
- **Fix**: Removed duplicate opportunity cost. Reduced FIX 4 penalties to -0.03 for planning/memory
- **Impact**: Combined penalty was so harsh that the model never used planning or memory actions

### Bug 5: Thread Leak in Adapter (openenv_adapter.py)
- **Old**: `ThreadPoolExecutor` created per step call
- **Fix**: Moved to class-level `self._executor` with shutdown in `close()`

### Bug 6: Train/Eval System Prompt Mismatch (generate_traces.py:180)
- **Old**: `{"role": "user", "content": SYSTEM_PROMPT}` — system prompt injected as user message
- **Fix**: `{"role": "system", "content": SYSTEM_PROMPT}` — correct role

### Bug 7: Step Pressure Docstring Lie (env.py:24)
- **Old docstring**: `-0.03 * step` — would make reward degenerate after step 56
- **Actual code**: `0.005 * (step/180)` — max penalty 0.005, negligible
- **Fix**: Updated docstring to match code

## Architecture Issues Found & Fixed

### Issue 1: OpenEnv Adapter Not TRL-Compatible
- **Problem**: `openenv_adapter.py` exposes `step(action)`, not individual tool methods
- **TRL docs say**: "We do not recommend generic methods like step(action)"
- **Fix**: Created `tool_env.py` with 8 individual tool methods + Args docstrings + reward functions

### Issue 2: /reset /step Bypassed Adapter Protections
- **Problem**: `app.py` created raw `ClinicalRecruitmentEnv`, not the adapter
- **Fix**: `/reset` and `/step` now create `ClinicalRecruitmentOpenEnv` instances with rate-limiting, replay detection, episode cap, timeout

### Issue 3: No Session Cleanup
- **Problem**: Sessions stored forever in memory dict, no TTL, no max
- **Fix**: 30-minute TTL, max 100 sessions, background reaper thread, env.close() on eviction

### Issue 4: SUPPORTS_CONCURRENT_SESSIONS Lie
- **Problem**: Declared `True` on a single-instance wrapper
- **Fix**: Set to `False` on the class; app.py achieves concurrency by creating one instance per session

### Issue 5: Eval Fallback Masked Model Failures
- **Problem**: `parse_action()` always produced a valid action via heuristic fallback
- **Fix**: Eval now reports `json_parse_rate` separately — how often the model produces valid JSON vs. needing fallback

### Issue 6: No Validation Split = Memorization
- **Problem**: 50 epochs × 5000 traces with no eval set = catastrophic overfitting
- **Fix**: 10% validation split, eval every 100 steps, load_best_model_at_end, reduced to 10 epochs at 5e-5 lr

## Helion by Meta

Helion is a **GPU kernel DSL** — it compiles Python to Triton kernels for operations like matmul, softmax, attention. It is completely irrelevant to this project. We are not writing custom CUDA kernels; we are calling `model.generate()` and `SFTTrainer.train()`. Helion would only be useful if we needed a custom fused training kernel or custom attention implementation, which we do not.

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
| `tests/test_env.py` | 25/25 passing |
| `test_env.py` | 76/76 passing |

## What Would Actually Work (Given More Time)

1. **GRPO with `tool_env.py`** — now properly exposes tool methods for TRL; test WITHOUT Unsloth to verify `environments` kwarg flows
2. **SFT 10 epochs + validation** instead of 50 epochs blind — prevents memorization while still learning format
3. **Manual REINFORCE** (`train_final.py`, `train_dual_gpu.py`) — bypasses TRL's environment_factory entirely
4. **Dual-GPU teacher-student** (`train_dual_gpu.py`) — 8B teacher generates traces, 4B student learns
5. **Strip `<think>` tags** before parsing for reasoning models
6. **torchforge** (Meta's agentic RL framework) — purpose-built for OpenEnv GRPO, unlike TRL+Unsloth

## Files

| File | Purpose | Status |
|------|---------|--------|
| `train.py` | SFT with val split + in-process eval | Updated: 5k traces by default, no raw HTTP, reports `json_parse_rate` |
| `tool_env.py` | TRL-compatible env with tool methods + reward functions | NEW: proper tool-method wrapper for OpenEnv/TRL experiments |
| `openenv_adapter.py` | Anti-reward-hacking wrapper (rate-limit, replay, cap, timeout) | Fixed: thread leak, concurrency |
| `app.py` | FastAPI server | Fixed: routes through adapter, session TTL |
| `server/app.py` | FastAPI server (HF Space copy) | Fixed: mirrors app.py |
| `env.py` | Core environment | Fixed: 5 bugs (reward, penalty, leak, docstring) |
| `scripts/generate_traces.py` | Parallel trace generator | Fixed: system prompt role |
| `tests/test_env.py` | 25 unit tests | Updated: covers env, adapter, tool_env, seed forwarding, reward design |
| `train_h200.py` | H200 optimized Qwen3-32B | Crashed on dynamo shape mismatch |
| `train_final.py` | Single-GPU SFT + manual REINFORCE | Correct approach |
| `train_dual_gpu.py` | Dual-GPU teacher-student | Correct approach |
| `kaggle_kernel/sft_then_grpo.ipynb` | Kaggle notebook | Latest: synced with current `train.py` and 5K trace path |
| `scripts/before_after_demo.py` | Heuristic vs random comparison | Works: +23.9% improvement |
| `data/training_outputs/sft_grpo_results.json` | Training evidence | Committed |
