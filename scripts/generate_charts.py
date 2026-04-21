"""Generate research charts from experiment CSV outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "docs" / "images"


def _save_bar_chart(df: pd.DataFrame, image_path: Path) -> None:
    pivot = df.pivot(index="task", columns="policy", values="avg_final_score")
    ax = pivot.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_title("Clinical Recruitment Benchmark Score by Policy")
    ax.set_ylabel("Average Final Score")
    ax.set_xlabel("Task")
    ax.set_ylim(0, max(0.8, float(pivot.max().max()) + 0.1))
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()


def _save_tradeoff_chart(df: pd.DataFrame, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        ax.scatter(
            row["avg_budget_remaining"],
            row["avg_enrolled"],
            s=70,
            alpha=0.85,
            label=f"{row['policy']} ({row['task']})",
        )
    ax.set_title("Enrollment vs Budget Tradeoff")
    ax.set_xlabel("Average Budget Remaining")
    ax.set_ylabel("Average Enrolled")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()


def _save_long_horizon_chart(df: pd.DataFrame, image_path: Path) -> None:
    grouped = (
        df.groupby("policy", as_index=False)[
            ["avg_milestones_hit", "avg_delayed_effects_triggered", "avg_recontacts"]
        ]
        .mean()
        .sort_values("avg_milestones_hit", ascending=False)
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("avg_milestones_hit", "Milestones Hit"),
        ("avg_delayed_effects_triggered", "Delayed Effects Triggered"),
        ("avg_recontacts", "Recontacts"),
    ]
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(grouped["policy"], grouped[column], color="#4C72B0")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Long-Horizon Behavior Indicators")
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()


def main() -> None:
    summary_path = DATA_DIR / "research_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary CSV: {summary_path}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.read_csv(summary_path)

    _save_bar_chart(summary_df, IMAGE_DIR / "benchmark_scores.png")
    _save_tradeoff_chart(summary_df, IMAGE_DIR / "enrollment_budget_tradeoff.png")
    _save_long_horizon_chart(summary_df, IMAGE_DIR / "long_horizon_indicators.png")

    print(f"Charts written to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
