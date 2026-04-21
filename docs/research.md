# Research Stack

This directory captures offline benchmark experimentation that is intentionally kept out of the serving path.

## Layout

- `research/policies.py`: local research baselines
- `research/runner.py`: deterministic offline experiment runner
- `experiments/run_research.py`: CLI entrypoint that writes CSV results into `data/`
- `scripts/generate_charts.py`: chart generation from experiment CSVs into `docs/images/`

## Default study

The default experiment compares four local policies across the three benchmark tasks:

- `greedy_screen`
- `conservative_retention`
- `site_negotiation`
- `rule_based_memory`

Run locally:

```bash
python experiments/run_research.py --episodes 3
python scripts/generate_charts.py
```
