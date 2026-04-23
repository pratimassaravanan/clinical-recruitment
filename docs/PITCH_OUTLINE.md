# 3-Minute Pitch: Adaptive Clinical Recruitment

This pitch outline is written to match the current repo truth. It is safer to emphasize benchmark quality, corrected evaluation, and reproducibility than to claim a breakthrough leaderboard result.

## Timing Guide

- `0:00-0:30` Hook and problem
- `0:30-1:15` What the benchmark exposes
- `1:15-2:00` Why the artifact is trustworthy
- `2:00-2:45` Current results
- `2:45-3:00` Close

## Script Outline

### `0:00-0:30` Hook

> "Clinical recruitment is a long-horizon operations problem. Screening, follow-up, site allocation, and retention decisions play out over months, not seconds. We built a benchmark that treats that as a sequential planning problem instead of a short-horizon toy task."

Key point: frame the problem as workflow planning under delayed effects.

### `0:30-1:15` What the benchmark exposes

> "Adaptive Clinical Recruitment is a deterministic `180`-step environment with `3` public tasks, typed observations, and `8` implemented action types. Agents manage screening, recontact, site allocation, strategy changes, planning, and indexed memory while handling delayed consequences, site pressure, and budget constraints."

Points to hit:

1. `180` simulated steps per episode
2. Action-specific candidate pools for screening, recontact, and allocation
3. Explicit plan state, indexed memory state, and token-efficiency signals
4. A `37`-dimensional feature vector for the trainable baselines

Suggested visual: `docs/images/environment_architecture.png`

### `1:15-2:00` Why the artifact is trustworthy

> "We re-audited the benchmark before reporting results. We fixed the evaluation path so recontact and allocation use the correct candidate pools, regenerated the charts, and aligned the docs to the actual `8`-action interface."

Evidence to mention:

- `30/30` integration checks pass across the three tasks
- Main diagrams are regenerated from code via `scripts/generate_docs_diagrams.py`
- The anonymous paper build in `paper/main.pdf` compiles with the official NeurIPS 2026 E&D style

### `2:00-2:45` Current results

> "We ran a fresh `5`-seed sweep over four repo baselines: HCAPO, MiRA, KLong, and MemexRL. HCAPO has the highest mean score at `0.2215`, but no pairwise comparison reaches `p < 0.05`."

Numbers to use:

- `HCAPO`: `0.2215 +- 0.0127`
- `KLong`: `0.2152 +- 0.0222`
- `MemexRL`: `0.2148 +- 0.0270`
- `MiRA`: `0.2094 +- 0.0095`

Key line:

> "The honest takeaway is not that one method wins. The honest takeaway is that the benchmark is reproducible, non-trivial, and not yet saturated by the current baseline suite."

Suggested visual: `docs/images/agent_comparison.png`

### `2:45-3:00` Close

> "This is the kind of benchmark we need for long-horizon agents: typed interfaces, delayed effects, explicit planning surfaces, and reproducible reports. The next step is not polishing the story. The next step is building stronger agents and sharper ablations on top of a benchmark we can trust."

## Backup Q&A

### "Does one method clearly win?"

Answer:

> "Not from the current `5`-seed sweep. HCAPO is highest mean, but no pairwise comparison reaches `p < 0.05`."

### "What changed from earlier drafts?"

Answer:

> "We corrected the evaluation path for recontact and allocation, refreshed the charts in both `data/sweep_results/` and `docs/images/`, and removed stale claims about a `10`-action interface and old significance results."

### "Why is this still interesting if the baselines are close?"

Answer:

> "Because a benchmark can be valuable before it produces a decisive leaderboard. Here the value is a clean long-horizon task interface, reproducible reporting, and room for stronger methods to separate later."

### "What should reviewers look at first?"

Answer:

> "`README.md`, `data/sweep_results/neurips_report.md`, the regenerated images under `docs/images/`, and `paper/main.pdf`."

## Visual Checklist

- Slide 1: benchmark framing and why long-horizon workflow tasks matter
- Slide 2: environment diagram
- Slide 3: current 5-seed results with the no-significance note
- Slide 4: verification and reproducibility points
- Slide 5: repo links or QR code
