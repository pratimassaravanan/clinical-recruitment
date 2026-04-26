"""Plot Pareto frontier style tradeoffs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "docs" / "images"


def main() -> None:
    summary_path = DATA_DIR / "pareto_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing pareto summary: {summary_path}")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        ax.scatter(
            row["avg_budget_remaining"],
            row["avg_final_score"],
            s=70,
            label=f"{row['policy']} ({row['task']})",
            alpha=0.85,
        )
    ax.set_title("Pareto Frontier: Score vs Budget")
    ax.set_xlabel("Budget Remaining")
    ax.set_ylabel("Final Score")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "pareto_frontier.svg", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
