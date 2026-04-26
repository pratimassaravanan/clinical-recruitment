"""Run progressive horizon evaluations and persist stage-level CSVs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from load_traces import PUBLIC_TASKS
from training.progressive_rl import run_progressive_sequence


DEFAULT_POLICIES = [
    "greedy_screen",
    "conservative_retention",
    "site_negotiation",
    "rule_based_memory",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run progressive-horizon clinical recruitment experiments")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        help="Policy names to evaluate",
    )
    parser.add_argument(
        "--base-tasks",
        nargs="+",
        default=list(PUBLIC_TASKS),
        help="Base benchmark tasks to stage progressively",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where CSV outputs are written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for policy_name in args.policies:
        for base_task in args.base_tasks:
            for result in run_progressive_sequence(policy_name, base_task):
                rows.append(asdict(result))
                print(
                    f"policy={policy_name} base_task={base_task} stage={result.horizon_days} "
                    f"score={result.final_score:.4f} enrolled={result.enrolled}/{result.target}"
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No progressive results produced.")

    df.to_csv(output_dir / "progressive_runs.csv", index=False)
    summary = (
        df.groupby(["policy", "base_task", "horizon_days"], as_index=False)
        .agg(
            avg_final_score=("final_score", "mean"),
            avg_total_reward=("total_reward", "mean"),
            avg_enrolled=("enrolled", "mean"),
            avg_target=("target", "mean"),
            avg_budget_remaining=("budget_remaining", "mean"),
            avg_plan_steps=("plan_steps", "mean"),
            avg_memory_writes=("memory_writes", "mean"),
            avg_memory_hits=("memory_hits", "mean"),
            avg_milestone_potential=("avg_milestone_potential", "mean"),
            avg_trajectory_chunks=("trajectory_chunks", "mean"),
            avg_token_usage=("token_usage", "mean"),
            avg_token_budget_remaining=("token_budget_remaining", "mean"),
            avg_token_efficiency_score=("token_efficiency_score", "mean"),
        )
        .sort_values(["base_task", "horizon_days", "policy"])
    )
    summary.to_csv(output_dir / "progressive_summary.csv", index=False)


if __name__ == "__main__":
    main()
