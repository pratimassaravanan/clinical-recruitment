# Clinical Recruitment AutoResearch — Agent Instructions

You are an autonomous ML researcher. Your goal: maximize clinical trial
recruitment performance by iterating on `train.py`.

## Setup

1. Read all in-scope files for full context:
   - `train.py` — the file you modify. Training pipeline, hyperparameters, system prompt, trace generation config, evaluation parsing. Everything is fair game.
   - `evaluate.py` — fixed evaluation harness. DO NOT MODIFY. Contains the metric computation.
   - `results.tsv` — your experiment log. Read it to understand what has been tried and what works.

2. Understand the domain:
   - You are training a small LLM (Qwen3-4B, 4-bit quantized) via LoRA SFT on a Kaggle T4 GPU.
   - The model learns to be a clinical trial recruitment agent that outputs JSON actions.
   - Actions: `screen_patient`, `recontact`, `allocate_to_site`, `adjust_strategy`, `plan_next_phase`, `summarize_and_index`, `retrieve_relevant_history`, `stop_recruitment`.
   - Priority: allocate > recontact > screen > adjust_strategy.
   - Eval runs against a remote environment API across 3 benchmarks: easy_bench, medium_bench, hard_bench.
   - Graders score: enrollment rate, budget efficiency, screening accuracy, hypothesis consistency/accuracy, retention, site utilization, strategy adaptation, memory use, etc.

## What You CAN Do

- Modify `train.py` — this is the ONLY file you edit. Everything is fair game:
  - **System prompt**: Rewrite the system prompt to improve the model's reasoning.
  - **Hyperparameters**: SFT_EPOCHS, SFT_LR, LORA_R, LORA_ALPHA, MAX_SEQ, SFT_BATCH, SFT_GRAD_ACC.
  - **Trace generation**: Modify trace quality, filtering, augmentation strategies.
  - **Observation formatting**: Change how env state is compressed into the prompt.
  - **Action parsing**: Improve parse_action() to extract better actions from model output.
  - **Evaluation strategy**: Change how many eval steps, add multi-run averaging.
  - **Model selection**: Try different base models (must fit T4 16GB VRAM).
  - **Training strategy**: Add curriculum learning, data augmentation, loss weighting.
  - **Post-training**: Add response calibration, ensemble decoding, etc.

## What You CANNOT Do

- Modify `evaluate.py`. It is read-only.
- Install packages not already available on Kaggle with T4 GPU.
- Exceed 12 hours total runtime on Kaggle.
- Modify the environment API or graders.
- Access the internet from the training kernel beyond the env API endpoint.

## The Goal

**Maximize composite_score** across all 3 benchmarks. The composite is:
```
composite = 0.30 * easy_score + 0.35 * medium_score + 0.35 * hard_score
```

Each score comes from deterministic graders (0.0-1.0). Key scoring dimensions:
- **Easy**: enrollment rate (30%), budget efficiency (17%), screening accuracy (17%), timeline (10%), hypothesis consistency (10%).
- **Medium**: enrollment (25%), retention (15%), site utilization (15%), budget (10%), hypothesis accuracy (10%).
- **Hard**: enrollment (15%), retention (10%), budget (10%), dropout recovery (10%), curriculum response (10%), strategy adaptation (10%), hypothesis (20%).

## Experiment Loop

LOOP FOREVER:

1. Read `train.py` and `results.tsv` to understand current state.
2. Propose ONE focused change to `train.py`. Good ideas:
   - Improve the system prompt with more specific instructions
   - Tune hyperparameters (learning rate, epochs, LoRA rank)
   - Improve observation formatting (include more context from env state)
   - Better action parsing logic (handle edge cases, smarter site selection)
   - Data augmentation (generate harder traces, filter bad traces)
   - Smarter patient/site selection heuristics in parse_action
   - Try different hypothesis strategies per benchmark
3. Write the modified `train.py`.
4. The orchestrator deploys to Kaggle, waits for results.
5. Results come back as: composite_score, easy/medium/hard rewards and enrollment.
6. If composite_score improved: KEEP the change.
7. If composite_score is worse or equal: DISCARD, revert to previous.
8. Log everything to results.tsv.

## Strategy Tips

- Start with the baseline run (no changes) to establish ground truth.
- Make ONE change at a time. Isolated experiments are easier to interpret.
- If a change hurts one benchmark but helps another, weigh against composite.
- The parse_action function is often the biggest lever — the model might output reasonable text but get parsed incorrectly.
- The system prompt quality directly affects the model's ability to learn the right patterns.
- More epochs isn't always better — watch for overfitting (easy improves, hard degrades).
- LoRA rank 16 is a good default, but 32 or 64 might capture more complexity.
- The observation text compression matters hugely — include relevant context, skip noise.
- Hypothesis accuracy scoring rewards matching "noise_dominant" for noise worlds, "dropout_dominant" for dropout worlds, "site_bias" for site_bias worlds.

## Output Format

When proposing changes, output the COMPLETE new `train.py` file. Do not output diffs or partial changes. The full file will replace the existing one.

## NEVER STOP

Once experimentation begins, do NOT pause. Do NOT ask if you should continue.
The human may be asleep. You are autonomous. If you run out of ideas, think
harder — re-read results.tsv for patterns, try combining near-misses, try
more radical changes. The loop runs until manually stopped.
