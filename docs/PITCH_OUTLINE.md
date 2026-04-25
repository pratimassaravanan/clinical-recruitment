# 3-Minute Pitch: Adaptive Clinical Recruitment

This pitch outline is written to match the current repo truth. It should be judged as a conservative Theme #2 pitch: strong on long-horizon workflow structure, live environment availability, and honest training evidence; weak on any claim of decisive baseline wins, complete coverage, or deployment validation.

## Timing Guide

- `0:00-0:25` Hook and problem
- `0:25-0:55` What the benchmark exposes
- `0:55-1:40` Why it is trainable and judge-ready
- `1:40-2:25` Current evidence
- `2:25-3:00` Close

## Script Outline

### `0:00-0:25` Hook

> "Clinical recruitment is a resource-allocation workflow: which patients do we screen, which sites get scarce capacity, when do we recontact, and when do we change strategy under budget pressure? Those decisions unfold over months, so we modeled them as a long-horizon environment instead of a short-horizon toy task."

Key point: frame the problem as data-driven operational decision making under delayed effects.

### `0:25-0:55` What the benchmark exposes

> "Adaptive Clinical Recruitment is a deterministic `180`-step environment with `3` public tasks, typed observations, and `8` implemented action types. Agents manage screening, recontact, site allocation, strategy changes, planning, and indexed memory while handling delayed consequences, site pressure, and budget constraints. It is a bounded workflow simulation, not a deployed clinical operations system."

Points to hit:

1. `180` simulated steps per episode
2. Action-specific candidate pools for screening, recontact, and allocation
3. Explicit plan state, indexed memory state, and token-efficiency signals
4. A `37`-dimensional feature vector for the trainable baselines
5. A live HF Space URL that judges can actually pull

Judge-friendly line:

> "The Theme #2 claim is that the repo exposes planning, memory, delayed effects, and recovery surfaces in a reproducible benchmark path. The claim is not that it covers every long-horizon requirement or solves clinical operations end to end."

Suggested visual: `docs/images/environment_architecture.png`

### `0:55-1:40` Why it is trainable and judge-ready

> "Judges can pull the live environment from our HF Space URL, and the same repo also exposes a local training wrapper for OpenEnv-style experiments. So this is not just a static simulator. It is a served benchmark with a real training path."

Evidence to mention:

- Live URL: `https://pratimassaravanan-clinical-recruitment.hf.space`
- `tool_env.py` provides the public tool-method wrapper for TRL/OpenEnv-style training
- `30/30` integration checks pass across the three tasks
- We corrected candidate routing, refreshed charts, and aligned the docs to the current `8`-action interface

Suggested visual: `docs/images/training_pipeline.png`

### `1:40-2:25` Current evidence

> "We have two kinds of evidence. First, a fresh `5`-seed sweep shows the environment is non-trivial and not saturated. Second, committed training artifacts show that the training path is real, but still early."

Numbers to use:

- `HCAPO`: `0.2215 +- 0.0127`
- `KLong`: `0.2152 +- 0.0222`
- `MemexRL`: `0.2148 +- 0.0270`
- `MiRA`: `0.2094 +- 0.0095`
- Pilot T4 SFT run: loss `0.858 -> 0.745` (`13.2%`)
- Pilot T4 behavior change: `1 -> 5` action types after SFT
- Progressive offline training artifacts: medium stages score `0.4242`, `0.4018`, `0.4004`

Key line:

> "The honest takeaway is that the environment is real, the training loop is real, and the strongest missing artifact is a larger post-fix LLM rerun."

Judge-friendly line:

> "So the evidence supports a benchmark-quality Theme #2 claim with early training evidence, not a leaderboard-quality claim that one baseline decisively beats the others."

Suggested visual: `docs/images/agent_comparison.png`

### `2:25-3:00` Close

> "The fair way to judge this repo is as a Theme #2 benchmark package with a live URL, typed interfaces, real training hooks, and early training evidence. The next step is not a different story. The next step is a larger post-fix run on top of the same environment."

## Backup Q&A

### "Does one method clearly win?"

Answer:

> "Not from the current `5`-seed sweep. HCAPO is highest mean, but no pairwise comparison reaches `p < 0.05`."

### "Why Theme #2 and not Wild Card?"

Answer:

> "Because the repo's strongest evidence is about long-horizon workflow structure: `180`-step episodes, delayed effects, explicit planning and memory surfaces, and resource-allocation decisions under pressure. The repo has some world-state scaffolding, but not a stronger full scientific-workflow-loop story than the Theme #2 benchmark story."

### "What is the training evidence today?"

Answer:

> "There is a committed pilot T4 artifact in `data/training_outputs/sft_grpo_results.json` showing `13.2%` SFT loss reduction and more diverse action output, plus separate offline training CSVs under `data/training/`. The honest limit is that the post-fix `5k`-trace LLM rerun is still pending."

### "What changed from earlier drafts?"

Answer:

> "We corrected the evaluation path for recontact and allocation, refreshed the charts in both `data/sweep_results/` and `docs/images/`, and removed stale claims about a `10`-action interface, complete coverage, and old significance results."

### "Why is this still interesting if the baselines are close?"

Answer:

> "Because the hackathon is about building an environment worth training on, not only about having a final leaderboard winner. Here the value is a workflow-shaped task, a live URL, a working training path, and clear room for stronger methods to separate later."

### "Is this ready for real clinical deployment?"

Answer:

> "No. The repo is a synthetic benchmark and local serving stack, not a validated clinical deployment. The safe claim is that it is a reproducible environment for studying long-horizon recruitment decisions."

### "What should reviewers look at first?"

Answer:

> "`README.md`, `docs/theme2_alignment.md`, `data/sweep_results/benchmark_report.md`, and the regenerated images under `docs/images/`."

## Visual Checklist

- Slide 1: benchmark framing and why long-horizon workflow tasks matter
- Slide 1 note: label this as Theme #2 alignment, not Wild Card
- Slide 2: environment diagram
- Slide 3: current evidence with pilot training callout plus the no-significance sweep note
- Slide 4: live URL, verification, and reproducibility points
- Slide 5: limitations and next work, including no deployment claim and the pending post-fix `5k`-trace rerun
