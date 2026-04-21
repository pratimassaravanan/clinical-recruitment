"""Build a small appendix-style reproducibility report."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    out_path = ROOT / "docs" / "appendix_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# Appendix Report\n\n"
        "This package includes deterministic tasks, staged training scripts, ablations, reproducibility sweeps, and chart generation artifacts.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
