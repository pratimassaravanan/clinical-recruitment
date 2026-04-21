"""Plot before/after trajectory summaries for a research policy."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from research.runner import make_policy, run_episode


def main() -> None:
    baseline = run_episode("medium_bench", make_policy("greedy_screen"))
    improved = run_episode("medium_bench", make_policy("rule_based_memory"))
    out_dir = ROOT / "docs" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Final Score", "Total Reward", "Token Efficiency"]
    baseline_values = [baseline.final_score, baseline.total_reward, baseline.token_efficiency_score]
    improved_values = [improved.final_score, improved.total_reward, improved.token_efficiency_score]
    positions = range(len(labels))
    ax.bar([p - 0.15 for p in positions], baseline_values, width=0.3, label="greedy_screen")
    ax.bar([p + 0.15 for p in positions], improved_values, width=0.3, label="rule_based_memory")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)
    ax.set_title("Trajectory Comparison: Baseline vs Memory Policy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_comparison.svg", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
