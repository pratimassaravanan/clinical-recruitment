"""Run feature ablations for the offline trainable policy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train_offline_policy import train_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run feature ablations")
    parser.add_argument("--output", default="data/ablation_features.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for policy_type in ("linear", "mlp"):
        _, history = train_policy(["medium_bench_stage_30", "medium_bench_stage_90"], epochs=3, policy_type=policy_type, seed=7)
        last = history[-1]
        rows.append(
            {
                "variant": policy_type,
                "avg_final_score": last.avg_final_score,
                "avg_total_reward": last.avg_total_reward,
                "avg_token_efficiency": last.avg_token_efficiency,
            }
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
