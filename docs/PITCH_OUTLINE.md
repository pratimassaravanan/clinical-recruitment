# 3-Minute Pitch: Adaptive Clinical Recruitment

This pitch outline is written to match the current repo truth. It should be judged as a conservative Theme #2 pitch: strong on long-horizon benchmark structure and reproducibility, weak on any claim of decisive baseline wins, complete coverage, or deployment validation.

## Timing Guide

- `0:00-0:30` Hook and problem
- `0:30-1:15` What the benchmark exposes
- `1:15-2:00` Why the artifact is trustworthy
- `2:00-2:45` Current results
- `2:45-3:00` Close

## Script Outline

### `0:00-0:30` Hook

> "If you are judging Theme #2, this repo fits because it turns clinical recruitment into a long-horizon planning benchmark. Screening, follow-up, site allocation, and retention decisions play out over months, not seconds, so we modeled them as a sequential workflow instead of a short-horizon toy task."

Key point: frame the problem as workflow planning under delayed effects.

### `0:30-1:15` What the benchmark exposes

> "Adaptive Clinical Recruitment is a deterministic `180`-step environment with `3` public tasks, typed observations, and `8` implemented action types. Agents manage screening, recontact, site allocation, strategy changes, planning, and indexed memory while handling delayed consequences, site pressure, and budget constraints. It is a bounded simulation, not a deployed clinical operations system."

Points to hit:

1. `180` simulated steps per episode
2. Action-specific candidate pools for screening, recontact, and allocation
3. Explicit plan state, indexed memory state, and token-efficiency signals
4. A `37`-dimensional feature vector for the trainable baselines

Judge-friendly line:

> "The Theme #2 claim is that the repo exposes planning, memory, delayed effects, and recovery surfaces in a reproducible benchmark path. The claim is not that it covers every long-horizon requirement or solves clinical operations end to end."

Suggested visual: `docs/images/environment_architecture.png`

### `1:15-2:00` Why the artifact is trustworthy

> "We re-audited the benchmark before reporting results. We fixed the evaluation path so recontact and allocation use the correct candidate pools, regenerated the charts, and aligned the docs to the actual `8`-action interface. The trust claim here is about the benchmark artifact and reporting path, not about deployment validation."

Evidence to mention:

- `30/30` integration checks pass across the three tasks
- Main diagrams are regenerated from code via `scripts/generate_docs_diagrams.py`
- The anonymous paper build in `paper/main.pdf` compiles with the official anonymous paper style

### `2:00-2:45` Current results

> "We ran a fresh `5`-seed sweep with seeds `1`, `7`, `21`, `42`, and `123` over four repo baselines: HCAPO, MiRA, KLong, and MemexRL. HCAPO has the highest mean score at `0.2215`, but no pairwise comparison reaches `p < 0.05`."

Numbers to use:

- `HCAPO`: `0.2215 +- 0.0127`
- `KLong`: `0.2152 +- 0.0222`
- `MemexRL`: `0.2148 +- 0.0270`
- `MiRA`: `0.2094 +- 0.0095`

Key line:

> "The honest takeaway is not that one method wins. The honest takeaway is that the benchmark is reproducible, non-trivial, and not yet saturated by the current baseline suite."

Judge-friendly line:

> "So the evidence supports a benchmark-quality claim for Theme #2, not a leaderboard-quality claim that one baseline decisively beats the others."

Suggested visual: `docs/images/agent_comparison.png`

### `2:45-3:00` Close

> "The fair way to judge this repo is as a Theme #2 benchmark package: typed interfaces, delayed effects, explicit planning surfaces, and reproducible reports. The next step is not louder claims. The next step is stronger agents, sharper ablations, and more evidence on top of a benchmark path that is now internally consistent."

## Backup Q&A

### "Does one method clearly win?"

Answer:

> "Not from the current `5`-seed sweep. HCAPO is highest mean, but no pairwise comparison reaches `p < 0.05`."

### "Why Theme #2 and not Wild Card?"

Answer:

> "Because the repo's strongest evidence is about long-horizon planning structure: `180`-step episodes, delayed effects, explicit planning and memory surfaces, and workflow-style decision making. The repo does not support a stronger Wild Card story than that."

### "What changed from earlier drafts?"

Answer:

> "We corrected the evaluation path for recontact and allocation, refreshed the charts in both `data/sweep_results/` and `docs/images/`, and removed stale claims about a `10`-action interface, complete coverage, and old significance results."

### "Why is this still interesting if the baselines are close?"

Answer:

> "Because a benchmark can be valuable before it produces a decisive leaderboard. Here the value is a clean long-horizon task interface, reproducible reporting, and room for stronger methods to separate later."

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
- Slide 3: current 5-seed results with seeds `1, 7, 21, 42, 123` and the no-significance note
- Slide 4: verification and reproducibility points
- Slide 5: limitations and next work, including no deployment claim and no decisive winner claim
