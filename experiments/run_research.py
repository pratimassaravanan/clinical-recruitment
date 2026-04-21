"""Run offline benchmark experiments and persist results to CSV."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from research.runner import run_episode, aggregate_results, make_policy


DEFAULT_POLICIES = [
    "greedy_screen",
    "conservative_retention",
    "site_negotiation",
    "rule_based_memory",
]
DEFAULT_TASKS = ["easy_bench", "medium_bench", "hard_bench"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline clinical recruitment experiments")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=DEFAULT_POLICIES,
        help="Policy names to evaluate",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="Task IDs to evaluate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes per policy-task pair",
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

    episode_rows = []
    summaries = []
    for policy_name in args.policies:
        for task in args.tasks:
            for episode_idx in range(args.episodes):
                policy = make_policy(policy_name)
                summary = run_episode(task, policy)
                summaries.append(summary)
                row = asdict(summary)
                row["episode_index"] = episode_idx
                episode_rows.append(row)
                print(
                    f"policy={policy_name} task={task} episode={episode_idx} "
                    f"score={summary.final_score:.4f} enrolled={summary.enrolled}"
                )

    episodes_df = pd.DataFrame(episode_rows)
    episodes_path = output_dir / "research_runs.csv"
    episodes_df.to_csv(episodes_path, index=False)

    aggregate_df = pd.DataFrame(aggregate_results(summaries))
    aggregate_path = output_dir / "research_summary.csv"
    aggregate_df.to_csv(aggregate_path, index=False)

    if not aggregate_df.empty:
        leaderboard = aggregate_df.groupby("policy", as_index=False)["avg_final_score"].mean()
        leaderboard = leaderboard.rename(columns={"avg_final_score": "mean_score_across_tasks"})
        leaderboard = leaderboard.sort_values("mean_score_across_tasks", ascending=False)
        leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)


if __name__ == "__main__":
    main()
