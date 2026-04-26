"""Generate a multi-objective Pareto summary from research outputs."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def main() -> None:
    summary_path = ROOT / "data" / "research_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary CSV: {summary_path}")
    df = pd.read_csv(summary_path)
    cols = [
        "policy",
        "task",
        "avg_final_score",
        "avg_budget_remaining",
        "avg_token_efficiency_score",
    ]
    out_path = ROOT / "data" / "pareto_summary.csv"
    df[cols].to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
