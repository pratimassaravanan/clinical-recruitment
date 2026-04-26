# Clinical Recruitment AutoResearch

Autonomous AI-driven experimentation on clinical trial recruitment agent training.
Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## How It Works

An LLM agent autonomously modifies `train.py`, deploys it to Kaggle (T4 GPU),
waits for results, and keeps or discards changes based on a composite score
across easy/medium/hard benchmarks. You sleep, it experiments.

```
LOOP FOREVER:
  1. LLM reads current train.py + results history
  2. LLM proposes an edit (new train.py)
  3. Deploy to Kaggle T4 kernel
  4. Poll until done (~20-40 min)
  5. Parse results (total_reward, enrolled/target per task)
  6. If composite_score improved: KEEP (advance)
  7. If worse: DISCARD (revert train.py)
  8. Log to results.tsv
  9. Goto 1
```

## Files

- `program.md` — agent instructions (human edits this)
- `train.py` — training script (agent edits this)
- `evaluate.py` — fixed eval harness (DO NOT MODIFY)
- `run_autoresearch.py` — autonomous orchestrator loop
- `results.tsv` — experiment log

## Quick Start

```bash
# 1. Set your Hyperspace API key (or it uses the default)
set HYPERSPACE_API_KEY=d3d25b98-d27a-4d9c-8f95-5d39731e3a3a

# 2. Ensure Kaggle API is configured
# kaggle.json should be at ~/.kaggle/kaggle.json

# 3. Run the autonomous loop
python autoresearch/run_autoresearch.py

# 4. Check results
cat autoresearch/results.tsv
```

## The Metric

**Composite score** = weighted average of grader scores across all 3 benchmarks:
- `easy_bench`: 30% weight
- `medium_bench`: 35% weight
- `hard_bench`: 35% weight

Lower scores from graders.py range 0.0-1.0. Higher is better.
