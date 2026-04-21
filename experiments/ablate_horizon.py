"""Run turn-restricted versus full-horizon ablations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from load_traces import make_stage_task_id
from research.runner import make_policy, run_episode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run horizon ablations")
    parser.add_argument("--policy", default="rule_based_memory")
    parser.add_argument("--base-task", default="medium_bench")
    parser.add_argument("--output", default="data/ablation_horizon.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for horizon in (30, 90, 180):
        policy = make_policy(args.policy)
        summary = run_episode(make_stage_task_id(args.base_task, horizon), policy)
        rows.append(
            {
                "policy": args.policy,
                "base_task": args.base_task,
                "horizon_days": horizon,
                "final_score": summary.final_score,
                "total_reward": summary.total_reward,
                "token_efficiency_score": summary.token_efficiency_score,
            }
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
