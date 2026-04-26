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
            [
                "avg_milestones_hit",
                "avg_memory_hits",
                "avg_milestone_potential",
            ]
        ]
        .mean()
        .sort_values("avg_milestones_hit", ascending=False)
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("avg_milestones_hit", "Milestones Hit"),
        ("avg_memory_hits", "Memory Hits"),
        ("avg_milestone_potential", "Milestone Potential"),
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


def _save_progressive_chart(df: pd.DataFrame, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for (policy, base_task), group in df.groupby(["policy", "base_task"]):
        ordered = group.sort_values("horizon_days")
        ax.plot(
            ordered["horizon_days"],
            ordered["avg_final_score"],
            marker="o",
            label=f"{policy} ({base_task})",
        )
    ax.set_title("Progressive Horizon Score Curves")
    ax.set_xlabel("Horizon Days")
    ax.set_ylabel("Average Final Score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best", ncols=2)
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()


def _save_training_curve(df: pd.DataFrame, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], df["avg_final_score"], marker="o", label="Final Score")
    ax.plot(df["epoch"], df["avg_total_reward"], marker="s", label="Total Reward")
    ax.plot(
        df["epoch"],
        df["avg_token_efficiency"],
        marker="^",
        label="Token Efficiency",
    )
    ax.set_title("Offline Policy Training Curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()


def _save_reproducibility_chart(df: pd.DataFrame, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["seed"].astype(str), df["avg_final_score"], color="#C44E52")
    ax.set_title("Multi-Seed Reproducibility")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Average Final Score")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()


def _save_pareto_chart(df: pd.DataFrame, image_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for _, row in df.iterrows():
        ax.scatter(
            row["avg_budget_remaining"],
            row["avg_final_score"],
            s=70,
            alpha=0.85,
            label=f"{row['policy']} ({row['task']})",
        )
    ax.set_title("Pareto Frontier: Score vs Budget")
    ax.set_xlabel("Budget Remaining")
    ax.set_ylabel("Final Score")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(image_path, dpi=200)
    plt.close()


def main() -> None:
    summary_path = DATA_DIR / "research_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary CSV: {summary_path}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.read_csv(summary_path)

    _save_bar_chart(summary_df, IMAGE_DIR / "benchmark_scores.svg")
    _save_tradeoff_chart(summary_df, IMAGE_DIR / "enrollment_budget_tradeoff.svg")
    _save_long_horizon_chart(summary_df, IMAGE_DIR / "long_horizon_indicators.svg")

    progressive_path = DATA_DIR / "progressive_summary.csv"
    if progressive_path.exists():
        progressive_df = pd.read_csv(progressive_path)
        if not progressive_df.empty:
            _save_progressive_chart(
                progressive_df,
                IMAGE_DIR / "progressive_horizon_curves.svg",
            )

    training_history_path = DATA_DIR / "training" / "training_history.csv"
    if training_history_path.exists():
        training_df = pd.read_csv(training_history_path)
        if not training_df.empty:
            _save_training_curve(training_df, IMAGE_DIR / "offline_training_curves.svg")

    reproducibility_path = DATA_DIR / "reproducibility.csv"
    if reproducibility_path.exists():
        reproducibility_df = pd.read_csv(reproducibility_path)
        if not reproducibility_df.empty:
            _save_reproducibility_chart(
                reproducibility_df,
                IMAGE_DIR / "reproducibility_report.svg",
            )

    pareto_path = DATA_DIR / "pareto_summary.csv"
    if pareto_path.exists():
        pareto_df = pd.read_csv(pareto_path)
        if not pareto_df.empty:
            _save_pareto_chart(pareto_df, IMAGE_DIR / "pareto_frontier.svg")

    print(f"Charts written to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
