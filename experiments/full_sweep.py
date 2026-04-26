"""Full training sweep with multi-seed reproducibility and chart generation.

Runs:
1. Multi-seed training sweep for all agents (HCAPO, MiRA, KLong, MemexRL)
2. Generates training charts and comparison visualizations
3. Produces benchmark reproducibility report with statistical significance
4. Integration testing with inference loop
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

# Add parent to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOCS_IMAGES_DIR = ROOT / "docs" / "images"

import numpy as np

from env import ClinicalRecruitmentEnv
from models import Action
from training.neural_policy import ACTION_SPACE, extract_state_features

# Import agents
from research.methods.hcapo_agent import HCAPOAgent
from research.methods.mira_agent import MiRAAgent
from research.methods.klong_agent import KLongAgent
from research.methods.memex_agent import MemexRLAgent

# Import reproducibility
from experiments.reproducibility import (
    ReproducibilityReport,
    bootstrap_ci,
    cohens_d,
    paired_t_test,
    wilcoxon_signed_rank,
    bonferroni_correction,
    holm_bonferroni_correction,
)


@dataclass
class SweepConfig:
    """Configuration for full training sweep."""
    
    seeds: List[int] = field(default_factory=lambda: [1, 7, 21, 42, 123])
    agent_types: List[str] = field(default_factory=lambda: ["hcapo", "mira", "klong", "memex"])
    task_ids: List[str] = field(default_factory=lambda: ["easy_bench", "medium_bench", "hard_bench"])
    episodes_per_seed: int = 30
    eval_episodes: int = 5
    output_dir: Path = field(default_factory=lambda: Path("data/sweep_results"))


def create_agent(agent_type: str, seed: int = 42):
    """Create agent of specified type with seed."""
    np.random.seed(seed)
    
    if agent_type == "hcapo":
        return HCAPOAgent()
    elif agent_type == "mira":
        return MiRAAgent()
    elif agent_type == "klong":
        return KLongAgent()
    elif agent_type == "memex":
        return MemexRLAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def run_episode(
    env: ClinicalRecruitmentEnv,
    agent,
    task_id: str,
    training: bool = True,
) -> Dict[str, Any]:
    """Run a single episode."""
    result = env.reset(task_id)
    obs = result.observation.model_dump()

    if hasattr(agent, "reset"):
        agent.reset()
    if hasattr(agent, "reset_milestones"):
        agent.reset_milestones()

    trajectory = []
    step = 0
    total_reward = 0.0

    while not result.done:
        # Select action
        if hasattr(agent, "select_action"):
            if isinstance(agent, MemexRLAgent):
                action_idx, info = agent.select_action(obs, step)
            else:
                action_idx, info = agent.select_action(obs)
        else:
            state = extract_state_features(obs)
            action_idx = agent.policy.select_action(state)
            info = {}

        action_type = ACTION_SPACE[action_idx]

        # Build action
        action_kwargs = {"action_type": action_type}

        def recommended_phase(current_obs: Dict[str, Any]) -> str:
            funnel = current_obs.get("current_funnel", {})
            target = max(1, int(current_obs.get("target_enrollment", 1)))
            enrolled = int(current_obs.get("enrolled_so_far", 0))
            progress = enrolled / target
            constraints = current_obs.get("active_constraints", {})
            if constraints.get("regulatory_hold_days", 0) or constraints.get("site_bottleneck", False):
                return "recovery"
            if current_obs.get("allocation_candidates"):
                return "allocation"
            if funnel.get("eligible", 0) > funnel.get("consented", 0) or current_obs.get("recontact_candidates"):
                return "conversion"
            if progress >= 0.7 or funnel.get("dropped", 0) > max(2, enrolled // 5):
                return "retention"
            return "screening"

        if action_type == "screen_patient":
            patients = obs.get("available_patients", [])
            if patients:
                action_kwargs["patient_id"] = patients[0].get("id")
        elif action_type == "recontact":
            patients = obs.get("recontact_candidates", [])
            if patients:
                action_kwargs["patient_id"] = patients[0].get("id")
        elif action_type == "allocate_to_site":
            patients = obs.get("allocation_candidates", [])
            if patients:
                action_kwargs["patient_id"] = patients[0].get("id")

        if action_type == "allocate_to_site":
            sites = obs.get("site_performance", {})
            if sites:
                action_kwargs["site_id"] = list(sites.keys())[0]

        if action_type == "adjust_strategy":
            action_kwargs["strategy_change"] = "increase_outreach"

        if action_type == "plan_next_phase":
            action_kwargs["target_phase"] = recommended_phase(obs)

        action = Action(**action_kwargs)
        result = env.step(action)

        reward = result.reward
        total_reward += reward

        trajectory.append({
            "obs": obs,
            "action": action_idx,
            "action_type": action_type,
            "reward": reward,
            "done": result.done,
        })

        if isinstance(agent, MemexRLAgent):
            agent.step(obs, action_idx, reward, step)

        obs = result.observation.model_dump()
        step += 1

    final_score = result.info.get("final_score", 0.0)

    if training and hasattr(agent, "update_from_episode"):
        agent.update_from_episode(trajectory)

    return {
        "task_id": task_id,
        "steps": step,
        "total_reward": total_reward,
        "final_score": final_score,
        "enrolled": obs.get("enrolled_so_far", 0),
        "budget_remaining": obs.get("budget_remaining", 0),
    }


def train_agent_with_seed(
    agent_type: str,
    seed: int,
    task_ids: List[str],
    episodes: int,
    eval_episodes: int = 5,
) -> Dict[str, Any]:
    """Train a single agent with a specific seed."""
    np.random.seed(seed)
    agent = create_agent(agent_type, seed)
    env = ClinicalRecruitmentEnv()

    history = []
    
    for episode in range(episodes):
        task_id = task_ids[episode % len(task_ids)]
        result = run_episode(env, agent, task_id, training=True)
        history.append({
            "episode": episode,
            "task_id": task_id,
            "total_reward": result["total_reward"],
            "final_score": result["final_score"],
        })

    # Final evaluation
    eval_scores = []
    for task_id in task_ids:
        for _ in range(eval_episodes):
            result = run_episode(env, agent, task_id, training=False)
            eval_scores.append(result["final_score"])

    return {
        "agent_type": agent_type,
        "seed": seed,
        "episodes": episodes,
        "history": history,
        "eval_scores": eval_scores,
        "mean_score": float(np.mean(eval_scores)),
        "std_score": float(np.std(eval_scores)),
    }


def run_full_sweep(config: SweepConfig) -> Dict[str, Any]:
    """Run full multi-seed training sweep."""
    print("=" * 70)
    print("FULL TRAINING SWEEP")
    print("=" * 70)
    print(f"Seeds: {config.seeds}")
    print(f"Agents: {config.agent_types}")
    print(f"Tasks: {config.task_ids}")
    print(f"Episodes per seed: {config.episodes_per_seed}")
    print()

    results = {}
    all_results = []

    for agent_type in config.agent_types:
        results[agent_type] = {"seeds": {}}
        print(f"\n{'=' * 50}")
        print(f"Training {agent_type.upper()}")
        print(f"{'=' * 50}")

        for seed in config.seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            start = time.time()
            
            try:
                result = train_agent_with_seed(
                    agent_type=agent_type,
                    seed=seed,
                    task_ids=config.task_ids,
                    episodes=config.episodes_per_seed,
                    eval_episodes=config.eval_episodes,
                )
                results[agent_type]["seeds"][seed] = result
                all_results.append({
                    "agent": agent_type,
                    "seed": seed,
                    "mean_score": result["mean_score"],
                    "std_score": result["std_score"],
                })
                elapsed = time.time() - start
                print(f"score={result['mean_score']:.4f} ({elapsed:.1f}s)")
            except Exception as e:
                print(f"ERROR: {e}")
                results[agent_type]["seeds"][seed] = {"error": str(e)}

        # Compute agent-level statistics
        scores = [
            results[agent_type]["seeds"][s]["mean_score"]
            for s in config.seeds
            if "mean_score" in results[agent_type]["seeds"].get(s, {})
        ]
        if scores:
            scores_arr = np.array(scores)
            mean_score = float(np.mean(scores_arr))
            results[agent_type]["overall"] = {
                "mean": mean_score,
                "std": float(np.std(scores_arr, ddof=1)) if len(scores) > 1 else 0.0,
                "min": float(np.min(scores_arr)),
                "max": float(np.max(scores_arr)),
                "ci_95": bootstrap_ci(scores_arr) if len(scores) >= 3 else (mean_score, mean_score),
            }

    return results, all_results


def compute_significance(results: Dict[str, Any], seeds: List[int]) -> Dict[str, Any]:
    """Compute pairwise statistical significance tests."""
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)

    agents = list(results.keys())
    comparisons = []

    for i, agent1 in enumerate(agents):
        for agent2 in agents[i+1:]:
            # Get paired scores (same seeds)
            scores1 = []
            scores2 = []
            for seed in seeds:
                if (seed in results[agent1]["seeds"] and 
                    seed in results[agent2]["seeds"] and
                    "mean_score" in results[agent1]["seeds"][seed] and
                    "mean_score" in results[agent2]["seeds"][seed]):
                    scores1.append(results[agent1]["seeds"][seed]["mean_score"])
                    scores2.append(results[agent2]["seeds"][seed]["mean_score"])

            if len(scores1) >= 3:
                arr1, arr2 = np.array(scores1), np.array(scores2)
                t_result = paired_t_test(arr1, arr2)
                w_result = wilcoxon_signed_rank(arr1, arr2)
                effect = cohens_d(arr1, arr2)

                comparisons.append({
                    "agent1": agent1,
                    "agent2": agent2,
                    "n_pairs": len(scores1),
                    "mean_diff": float(np.mean(arr1) - np.mean(arr2)),
                    "t_stat": t_result.statistic,
                    "t_pvalue": t_result.p_value,
                    "wilcoxon_stat": w_result.statistic,
                    "wilcoxon_pvalue": w_result.p_value,
                    "cohens_d": effect,
                    "significant_05": t_result.significant_at_05,
                    "significant_01": t_result.significant_at_01,
                })

                sig = "***" if t_result.significant_at_01 else ("**" if t_result.significant_at_05 else "")
                print(f"  {agent1} vs {agent2}: diff={np.mean(arr1)-np.mean(arr2):.4f}, p={t_result.p_value:.4f}{sig}, d={effect:.3f}")

    # Apply multiple comparison correction
    if comparisons:
        p_values = [c["t_pvalue"] for c in comparisons]
        bonf = bonferroni_correction(p_values)
        holm = holm_bonferroni_correction(p_values)
        
        for i, comp in enumerate(comparisons):
            comp["bonferroni_sig"] = bonf[i][1]
            comp["holm_sig"] = holm[i][1]

    return {"comparisons": comparisons, "n_comparisons": len(comparisons)}


def generate_sweep_charts(results: Dict[str, Any], output_dir: Path) -> None:
    """Generate charts from sweep results."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping chart generation")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    DOCS_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def save_chart(fig, name: str) -> None:
        fig.savefig(output_dir / f"{name}.png", dpi=150)
        fig.savefig(output_dir / f"{name}.svg")
        fig.savefig(DOCS_IMAGES_DIR / f"{name}.png", dpi=150)
        fig.savefig(DOCS_IMAGES_DIR / f"{name}.svg")

    # 1. Agent comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    agents = []
    means = []
    stds = []
    ci_lows = []
    ci_highs = []

    max_seed_count = 0
    for agent, data in results.items():
        if "overall" in data:
            agents.append(agent.upper())
            means.append(data["overall"]["mean"])
            stds.append(data["overall"]["std"])
            ci_low, ci_high = data["overall"]["ci_95"]
            mean = data["overall"]["mean"]
            ci_lows.append(max(0.0, mean - ci_low))
            ci_highs.append(max(0.0, ci_high - mean))
            max_seed_count = max(
                max_seed_count,
                len([s for s in data.get("seeds", {}) if "mean_score" in data["seeds"].get(s, {})]),
            )

    x = np.arange(len(agents))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars = ax.bar(x, means, yerr=[ci_lows, ci_highs], capsize=5, color=colors[:len(agents)], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Mean Final Score")
    if max_seed_count >= 3:
        ax.set_title(f"Agent Performance Comparison ({max_seed_count}-Seed Average with 95% CI)")
    else:
        ax.set_title(f"Agent Performance Comparison ({max_seed_count}-Seed Average)")
    ax.grid(axis='y', alpha=0.3)

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean:.3f}\n(±{std:.3f})', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_chart(fig, "agent_comparison")
    plt.close()

    # 2. Seed-by-seed heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    seed_matrix = []
    seed_labels = []
    
    for agent, data in results.items():
        row = []
        for seed in sorted(data["seeds"].keys()):
            if "mean_score" in data["seeds"][seed]:
                row.append(data["seeds"][seed]["mean_score"])
            else:
                row.append(0)
        seed_matrix.append(row)
        seed_labels.append(agent.upper())

    seed_matrix = np.array(seed_matrix)
    im = ax.imshow(seed_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=max(0.5, seed_matrix.max()))
    
    ax.set_xticks(range(len(sorted(list(results.values())[0]["seeds"].keys()))))
    ax.set_xticklabels([f"Seed {s}" for s in sorted(list(results.values())[0]["seeds"].keys())])
    ax.set_yticks(range(len(seed_labels)))
    ax.set_yticklabels(seed_labels)
    
    for i in range(len(seed_labels)):
        for j in range(seed_matrix.shape[1]):
            ax.text(j, i, f'{seed_matrix[i, j]:.3f}', ha='center', va='center', fontsize=9)

    plt.colorbar(im, label='Final Score')
    ax.set_title("Score by Agent and Seed")
    plt.tight_layout()
    save_chart(fig, "seed_heatmap")
    plt.close()

    # 3. Box plot
    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = []
    box_labels = []
    
    for agent, data in results.items():
        scores = [
            data["seeds"][s]["mean_score"]
            for s in data["seeds"]
            if "mean_score" in data["seeds"][s]
        ]
        if scores:
            box_data.append(scores)
            box_labels.append(agent.upper())

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Final Score")
    ax.set_title("Score Distribution Across Seeds")
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_chart(fig, "score_boxplot")
    plt.close()

    print(f"Charts saved to {output_dir}")


def run_integration_tests(task_ids: List[str]) -> Dict[str, Any]:
    """Run integration tests with the full inference loop."""
    print("\n" + "=" * 70)
    print("INTEGRATION TESTING")
    print("=" * 70)

    env = ClinicalRecruitmentEnv()
    results = {}

    for task_id in task_ids:
        print(f"\n  Testing {task_id}...")
        result = env.reset(task=task_id)
        obs = result.observation

        checks = []
        
        # Check observation structure
        checks.append(("has_patients", len(obs.available_patients) > 0))
        checks.append(("has_site_performance", len(obs.site_performance) > 0))
        checks.append(("has_milestones", obs.milestones is not None))
        checks.append(("has_causal_insight", obs.causal_insight is not None))
        checks.append(("has_memory", obs.indexed_memory_summary is not None))
        checks.append(("has_plan", obs.current_plan is not None))
        checks.append(("has_token_budget", obs.token_budget_remaining is not None))
        
        # Run a few steps
        step_count = 0
        total_reward = 0.0
        action_types_used = set()
        
        while not result.done and step_count < 20:
            # Cycle through different action types while respecting action-specific candidate pools.
            action_types = ["screen_patient", "recontact", "allocate_to_site", "adjust_strategy"]
            action_type = action_types[step_count % len(action_types)]

            action_kwargs = {"action_type": action_type}

            if action_type == "screen_patient":
                patients = obs.available_patients
                if patients:
                    p = patients[0]
                    action_kwargs["patient_id"] = p.id if hasattr(p, "id") else p.get("id")
                else:
                    action_kwargs = {
                        "action_type": "adjust_strategy",
                        "strategy_change": "increase_outreach",
                    }
            elif action_type == "recontact":
                patients = obs.recontact_candidates
                if patients:
                    p = patients[0]
                    action_kwargs["patient_id"] = p.id if hasattr(p, "id") else p.get("id")
                else:
                    action_kwargs = {
                        "action_type": "adjust_strategy",
                        "strategy_change": "increase_outreach",
                    }
            elif action_type == "allocate_to_site":
                patients = obs.allocation_candidates
                sites = obs.site_performance
                if patients and sites:
                    p = patients[0]
                    action_kwargs["patient_id"] = p.id if hasattr(p, "id") else p.get("id")
                    action_kwargs["site_id"] = list(sites.keys())[0]
                else:
                    action_kwargs = {
                        "action_type": "adjust_strategy",
                        "strategy_change": "increase_outreach",
                    }
            elif action_type == "adjust_strategy":
                action_kwargs["strategy_change"] = "increase_outreach"

            action = Action(**action_kwargs)
            result = env.step(action)
            obs = result.observation
            
            total_reward += result.reward
            action_types_used.add(action_type)
            step_count += 1
            
        checks.append(("ran_multiple_steps", step_count >= 5))
        checks.append(("multiple_action_types", len(action_types_used) >= 3))
        checks.append(("non_zero_reward", total_reward != 0))
        
        passed = sum(1 for _, ok in checks if ok)
        total = len(checks)
        
        results[task_id] = {
            "passed": passed,
            "total": total,
            "checks": {name: ok for name, ok in checks},
            "steps_run": step_count,
            "total_reward": total_reward,
        }
        
        for name, ok in checks:
            status = "PASS" if ok else "FAIL"
            print(f"    [{status}] {name}")
            
        print(f"    Result: {passed}/{total} checks passed")

    return results


def generate_benchmark_report(
    sweep_results: Dict[str, Any],
    significance: Dict[str, Any],
    integration: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate benchmark reproducibility report."""
    print("\n" + "=" * 70)
    print("GENERATING BENCHMARK REPORT")
    print("=" * 70)

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "framework": "Clinical Recruitment Environment",
            "version": "1.0.0",
        },
        "sweep_results": {},
        "statistical_tests": significance,
        "integration_tests": integration,
    }

    # Add sweep summary
    for agent, data in sweep_results.items():
        if "overall" in data:
            report["sweep_results"][agent] = {
                "mean": data["overall"]["mean"],
                "std": data["overall"]["std"],
                "ci_95": data["overall"]["ci_95"],
                "n_seeds": len([s for s in data["seeds"] if "mean_score" in data["seeds"].get(s, {})]),
                "per_seed": {
                    str(seed): data["seeds"][seed].get("mean_score", None)
                    for seed in data["seeds"]
                },
            }

    # Save JSON report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Generate markdown summary
    md_path = output_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# Benchmark Reproducibility Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary Statistics\n\n")
        f.write("| Agent | Mean | Std | 95% CI | n |\n")
        f.write("|-------|------|-----|--------|---|\n")
        
        for agent, data in report["sweep_results"].items():
            ci = data["ci_95"]
            f.write(f"| {agent.upper()} | {data['mean']:.4f} | {data['std']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {data['n_seeds']} |\n")
        
        f.write("\n## Pairwise Comparisons\n\n")
        f.write("| Comparison | Mean Diff | p-value | Cohen's d | Significant |\n")
        f.write("|------------|-----------|---------|-----------|-------------|\n")
        
        for comp in significance.get("comparisons", []):
            sig = "Yes***" if comp["significant_01"] else ("Yes**" if comp["significant_05"] else "No")
            f.write(f"| {comp['agent1']} vs {comp['agent2']} | {comp['mean_diff']:.4f} | {comp['t_pvalue']:.4f} | {comp['cohens_d']:.3f} | {sig} |\n")
        
        f.write("\n## Integration Tests\n\n")
        for task, data in integration.items():
            f.write(f"- **{task}**: {data['passed']}/{data['total']} checks passed\n")

    print(f"Report saved to {output_path}")
    print(f"Markdown saved to {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Full training sweep with reproducibility analysis")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 7, 21, 42, 123])
    parser.add_argument("--agents", type=str, nargs="+", default=["hcapo", "mira", "klong", "memex"])
    parser.add_argument("--tasks", type=str, nargs="+", default=["easy_bench", "medium_bench", "hard_bench"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="data/sweep_results")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, load existing results")
    
    args = parser.parse_args()
    
    config = SweepConfig(
        seeds=args.seeds,
        agent_types=args.agents,
        task_ids=args.tasks,
        episodes_per_seed=args.episodes,
        eval_episodes=args.eval_episodes,
        output_dir=Path(args.output_dir),
    )
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Run training sweep
    print("\n" + "=" * 70)
    print("STEP 1: MULTI-SEED TRAINING SWEEP")
    print("=" * 70)
    
    if args.skip_training and (config.output_dir / "sweep_results.json").exists():
        print("Loading existing results...")
        with open(config.output_dir / "sweep_results.json") as f:
            sweep_results = json.load(f)
        all_results = []
    else:
        sweep_results, all_results = run_full_sweep(config)
        
        # Save raw results
        with open(config.output_dir / "sweep_results.json", "w") as f:
            json.dump(sweep_results, f, indent=2, default=str)
        
        # Save CSV
        with open(config.output_dir / "sweep_results.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["agent", "seed", "mean_score", "std_score"])
            writer.writeheader()
            writer.writerows(all_results)

    # 2. Compute significance
    print("\n" + "=" * 70)
    print("STEP 2: STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)
    
    significance = compute_significance(sweep_results, config.seeds)
    
    with open(config.output_dir / "significance_tests.json", "w") as f:
        json.dump(significance, f, indent=2, default=str)

    # 3. Generate charts
    print("\n" + "=" * 70)
    print("STEP 3: GENERATING CHARTS")
    print("=" * 70)
    
    generate_sweep_charts(sweep_results, config.output_dir)

    # 4. Run integration tests
    print("\n" + "=" * 70)
    print("STEP 4: INTEGRATION TESTING")
    print("=" * 70)
    
    integration = run_integration_tests(config.task_ids)
    
    with open(config.output_dir / "integration_tests.json", "w") as f:
        json.dump(integration, f, indent=2, default=str)

    # 5. Generate benchmark report
    generate_benchmark_report(
        sweep_results,
        significance,
        integration,
        config.output_dir / "benchmark_report.json",
    )

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nAgent Rankings:")
    rankings = []
    for agent, data in sweep_results.items():
        if "overall" in data:
            rankings.append((agent, data["overall"]["mean"], data["overall"]["std"]))
    
    rankings.sort(key=lambda x: x[1], reverse=True)
    for i, (agent, mean, std) in enumerate(rankings, 1):
        print(f"  {i}. {agent.upper()}: {mean:.4f} (±{std:.4f})")

    print("\nSignificant differences (p < 0.05):")
    for comp in significance.get("comparisons", []):
        if comp["significant_05"]:
            better = comp["agent1"] if comp["mean_diff"] > 0 else comp["agent2"]
            worse = comp["agent2"] if comp["mean_diff"] > 0 else comp["agent1"]
            print(f"  - {better.upper()} > {worse.upper()} (p={comp['t_pvalue']:.4f}, d={abs(comp['cohens_d']):.3f})")

    print("\nIntegration test summary:")
    total_passed = sum(d["passed"] for d in integration.values())
    total_checks = sum(d["total"] for d in integration.values())
    print(f"  {total_passed}/{total_checks} checks passed across all tasks")

    print(f"\nAll results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()
