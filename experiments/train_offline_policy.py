"""Train and persist a lightweight offline clinical recruitment policy."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from load_traces import list_progressive_stage_tasks
from training.train_offline_policy import (
    evaluate_policy,
    save_evaluation_outputs,
    save_training_outputs,
    train_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight offline policy")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--output-dir", default="data/training")
    parser.add_argument(
        "--policy-type",
        choices=["linear", "mlp"],
        default="mlp",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list_progressive_stage_tasks("medium_bench"),
        help="Task IDs to use during training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy, history = train_policy(
        args.tasks,
        epochs=args.epochs,
        policy_type=args.policy_type,
    )
    output_dir = Path(args.output_dir)
    save_training_outputs(output_dir, policy, history)
    evaluation_rows = evaluate_policy(args.tasks, policy)
    save_evaluation_outputs(output_dir, evaluation_rows)
    if history:
        last = history[-1]
        print(
            f"policy_type={args.policy_type} epochs={args.epochs} avg_final_score={last.avg_final_score:.4f} "
            f"avg_total_reward={last.avg_total_reward:.4f} "
            f"avg_token_efficiency={last.avg_token_efficiency:.4f}"
        )


if __name__ == "__main__":
    main()
