"""Reproducibility and statistical significance testing for benchmark results.

Includes:
- Multi-seed reproducibility sweeps
- Bootstrap confidence intervals
- Paired t-tests and Wilcoxon signed-rank tests
- Effect size calculations (Cohen's d)
- Multiple comparison corrections (Bonferroni, Holm)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from training.train_offline_policy import train_policy


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    
    test_name: str
    statistic: float
    p_value: float
    significant_at_05: bool
    significant_at_01: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    notes: str = ""


def bootstrap_ci(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for a statistic."""
    rng = np.random.RandomState(seed)
    n = len(data)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_fn(sample))
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return float(lower), float(upper)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def paired_t_test(
    x: np.ndarray,
    y: np.ndarray,
) -> StatisticalResult:
    """Perform paired t-test using SciPy's Student t distribution."""
    n = len(x)
    if n != len(y):
        raise ValueError("Arrays must have same length")
    
    differences = x - y
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    if std_diff == 0:
        return StatisticalResult(
            test_name="paired_t_test",
            statistic=0.0,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
            effect_size=0.0,
            notes="Zero variance in differences",
        )
    
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1
    p_value = float(2 * stats.t.sf(abs(t_stat), df=df))
    
    effect = cohens_d(x, y)
    
    return StatisticalResult(
        test_name="paired_t_test",
        statistic=float(t_stat),
        p_value=float(p_value),
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        effect_size=effect,
        confidence_interval=bootstrap_ci(differences),
    )


def wilcoxon_signed_rank(
    x: np.ndarray,
    y: np.ndarray,
) -> StatisticalResult:
    """Perform Wilcoxon signed-rank test using SciPy."""
    n = len(x)
    if n != len(y):
        raise ValueError("Arrays must have same length")
    
    differences = x - y
    
    # Remove zeros
    nonzero_idx = differences != 0
    differences = differences[nonzero_idx]
    n = len(differences)
    
    if n == 0:
        return StatisticalResult(
            test_name="wilcoxon_signed_rank",
            statistic=0.0,
            p_value=1.0,
            significant_at_05=False,
            significant_at_01=False,
            notes="All differences are zero",
        )
    
    w_stat, p_value = stats.wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided", method="auto")
    
    effect = cohens_d(x, y)
    
    return StatisticalResult(
        test_name="wilcoxon_signed_rank",
        statistic=float(w_stat),
        p_value=float(p_value),
        significant_at_05=p_value < 0.05,
        significant_at_01=p_value < 0.01,
        effect_size=effect,
    )


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool]]:
    """Apply Bonferroni correction for multiple comparisons."""
    m = len(p_values)
    corrected_alpha = alpha / m
    
    return [(p, p < corrected_alpha) for p in p_values]


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[Tuple[float, bool, int]]:
    """Apply Holm-Bonferroni step-down correction."""
    m = len(p_values)
    
    # Sort p-values with original indices
    indexed = [(p, i) for i, p in enumerate(p_values)]
    indexed.sort(key=lambda x: x[0])
    
    results = [None] * m
    rejected_any = True
    
    for rank, (p, orig_idx) in enumerate(indexed):
        corrected_alpha = alpha / (m - rank)
        
        if rejected_any and p < corrected_alpha:
            results[orig_idx] = (p, True, rank + 1)
        else:
            rejected_any = False
            results[orig_idx] = (p, False, rank + 1)
    
    return results


class ReproducibilityReport:
    """Generate comprehensive reproducibility and significance reports."""
    
    def __init__(self, seeds: List[int] = None):
        self.seeds = seeds or [1, 7, 21, 42, 123]
        self.results: Dict[str, Dict[int, Dict[str, float]]] = {}
    
    def run_sweep(
        self,
        methods: List[str],
        task_ids: List[str],
        epochs: int = 3,
    ) -> None:
        """Run reproducibility sweep across methods and seeds."""
        
        for method in methods:
            self.results[method] = {}
            
            for seed in self.seeds:
                policy_type = "mlp" if method == "mlp" else "linear"
                
                try:
                    _, history = train_policy(
                        task_ids,
                        epochs=epochs,
                        policy_type=policy_type,
                        seed=seed,
                    )
                    last = history[-1]
                    
                    self.results[method][seed] = {
                        "avg_final_score": last.avg_final_score,
                        "avg_total_reward": last.avg_total_reward,
                        "avg_token_efficiency": last.avg_token_efficiency,
                    }
                except Exception as e:
                    self.results[method][seed] = {
                        "avg_final_score": 0.0,
                        "avg_total_reward": 0.0,
                        "avg_token_efficiency": 0.0,
                        "error": str(e),
                    }
    
    def compute_statistics(
        self,
        metric: str = "avg_final_score",
    ) -> Dict[str, Any]:
        """Compute summary statistics for each method."""
        
        stats = {}
        
        for method, seed_results in self.results.items():
            values = [
                seed_results[seed].get(metric, 0.0)
                for seed in self.seeds
                if seed in seed_results
            ]
            
            if not values:
                continue
            
            values_arr = np.array(values)
            
            stats[method] = {
                "mean": float(np.mean(values_arr)),
                "std": float(np.std(values_arr, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values_arr)),
                "max": float(np.max(values_arr)),
                "median": float(np.median(values_arr)),
                "n": len(values),
                "ci_95": bootstrap_ci(values_arr) if len(values) >= 3 else (0.0, 0.0),
            }
        
        return stats
    
    def compare_methods(
        self,
        method1: str,
        method2: str,
        metric: str = "avg_final_score",
    ) -> Dict[str, StatisticalResult]:
        """Compare two methods with statistical tests."""
        
        if method1 not in self.results or method2 not in self.results:
            return {}
        
        # Get paired values (same seeds)
        common_seeds = set(self.results[method1].keys()) & set(self.results[method2].keys())
        
        if len(common_seeds) < 3:
            return {"error": "Not enough paired observations"}
        
        values1 = np.array([
            self.results[method1][seed].get(metric, 0.0)
            for seed in sorted(common_seeds)
        ])
        values2 = np.array([
            self.results[method2][seed].get(metric, 0.0)
            for seed in sorted(common_seeds)
        ])
        
        return {
            "t_test": paired_t_test(values1, values2),
            "wilcoxon": wilcoxon_signed_rank(values1, values2),
            "effect_size": cohens_d(values1, values2),
            "mean_diff": float(np.mean(values1) - np.mean(values2)),
        }
    
    def all_pairwise_comparisons(
        self,
        metric: str = "avg_final_score",
    ) -> Dict[str, Any]:
        """Run all pairwise comparisons with multiple comparison correction."""
        
        methods = list(self.results.keys())
        comparisons = []
        p_values = []
        
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                result = self.compare_methods(m1, m2, metric)
                if "t_test" in result:
                    comparisons.append({
                        "method1": m1,
                        "method2": m2,
                        "t_test": result["t_test"],
                        "wilcoxon": result["wilcoxon"],
                        "effect_size": result["effect_size"],
                        "mean_diff": result["mean_diff"],
                    })
                    p_values.append(result["t_test"].p_value)
        
        # Apply corrections
        if p_values:
            bonferroni = bonferroni_correction(p_values)
            holm = holm_bonferroni_correction(p_values)
            
            for i, comp in enumerate(comparisons):
                comp["bonferroni_p"] = bonferroni[i][0]
                comp["bonferroni_significant"] = bonferroni[i][1]
                comp["holm_significant"] = holm[i][1]
        
        return {
            "comparisons": comparisons,
            "n_comparisons": len(comparisons),
            "alpha": 0.05,
            "corrected_alpha_bonferroni": 0.05 / max(1, len(comparisons)),
        }
    
    def generate_report(
        self,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive reproducibility report."""
        
        report = {
            "seeds": self.seeds,
            "n_seeds": len(self.seeds),
            "methods": list(self.results.keys()),
            "statistics": {},
            "pairwise_comparisons": {},
        }
        
        for metric in ["avg_final_score", "avg_total_reward", "avg_token_efficiency"]:
            report["statistics"][metric] = self.compute_statistics(metric)
            report["pairwise_comparisons"][metric] = self.all_pairwise_comparisons(metric)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame for analysis."""
        
        rows = []
        for method, seed_results in self.results.items():
            for seed, metrics in seed_results.items():
                row = {"method": method, "seed": seed}
                row.update(metrics)
                rows.append(row)
        
        return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducibility sweep with significance tests")
    parser.add_argument("--output", default="data/reproducibility.csv")
    parser.add_argument("--report", default="data/reproducibility_report.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 21])
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Create report
    reporter = ReproducibilityReport(seeds=args.seeds)
    
    # Run sweep with both policy types
    reporter.run_sweep(
        methods=["linear", "mlp"],
        task_ids=["medium_bench_stage_30", "medium_bench_stage_90", "medium_bench_stage_180"],
        epochs=args.epochs,
    )
    
    # Generate full report
    report = reporter.generate_report(output_path=Path(args.report))
    
    # Save CSV
    df = reporter.to_dataframe()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY REPORT")
    print("=" * 60)
    
    for metric in ["avg_final_score"]:
        stats = report["statistics"][metric]
        print(f"\n{metric}:")
        for method, s in stats.items():
            ci = s.get("ci_95", (0, 0))
            print(f"  {method}: {s['mean']:.4f} ± {s['std']:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
        
        comps = report["pairwise_comparisons"][metric].get("comparisons", [])
        if comps:
            print("\n  Pairwise comparisons:")
            for c in comps:
                sig = "***" if c["t_test"].significant_at_01 else ("**" if c["t_test"].significant_at_05 else "")
                print(f"    {c['method1']} vs {c['method2']}: p={c['t_test'].p_value:.4f}{sig}, d={c['effect_size']:.3f}")


if __name__ == "__main__":
    main()
