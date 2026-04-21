"""Render a compact reward-curve dashboard from training and ablation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "docs" / "images"


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    training_path = DATA_DIR / "training" / "training_history.csv"
    ablation_path = DATA_DIR / "ablation_features.csv"
    if not training_path.exists():
        raise SystemExit(f"Missing training history: {training_path}")

    training_df = pd.read_csv(training_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(training_df["epoch"], training_df["avg_final_score"], marker="o")
    axes[0].plot(training_df["epoch"], training_df["avg_total_reward"], marker="s")
    axes[0].set_title("Training Rewards")
    axes[0].grid(alpha=0.25)

    if ablation_path.exists():
        ablation_df = pd.read_csv(ablation_path)
        axes[1].bar(ablation_df["variant"], ablation_df["avg_final_score"], color="#55A868")
        axes[1].set_title("Feature Ablations")
        axes[1].grid(axis="y", alpha=0.25)
    else:
        axes[1].text(0.5, 0.5, "No ablation data", ha="center", va="center")
        axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "reward_curve_dashboard.svg", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
