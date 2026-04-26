#!/usr/bin/env python3
"""Fixed evaluation harness for Clinical Recruitment AutoResearch.

DO NOT MODIFY THIS FILE. It is the equivalent of Karpathy's prepare.py.

This file contains:
1. The composite scoring function
2. Result parsing from train.py output
3. Comparison logic

The orchestrator (run_autoresearch.py) calls these functions to determine
whether an experiment improved over the baseline.
"""
import json
import re
import sys


# Benchmark weights for composite score
WEIGHTS = {
    "easy_bench": 0.30,
    "medium_bench": 0.35,
    "hard_bench": 0.35,
}

TASKS = ["easy_bench", "medium_bench", "hard_bench"]


def compute_composite_score(results: dict) -> float:
    """Compute weighted composite score from per-task results.
    
    Args:
        results: dict mapping task_id -> {"total_reward": float, "enrolled": int, "target": int}
    
    Returns:
        Composite score (higher is better). Combines reward and enrollment rate.
    """
    score = 0.0
    for task, weight in WEIGHTS.items():
        r = results.get(task, {})
        total_reward = float(r.get("total_reward", 0))
        enrolled = int(r.get("enrolled", 0))
        target = max(1, int(r.get("target", 100)))
        
        # Enrollment rate (0-1)
        enrollment_rate = min(1.0, enrolled / target)
        
        # Reward component (normalized: typical range is 0-30 for good runs)
        reward_norm = min(1.0, max(0.0, total_reward / 30.0))
        
        # Task score: 60% enrollment rate + 40% normalized reward
        task_score = 0.6 * enrollment_rate + 0.4 * reward_norm
        score += weight * task_score
    
    return round(score, 6)


def parse_train_output(log_text: str) -> dict:
    """Parse the output of train.py to extract results.
    
    Looks for the evaluation section and extracts per-task metrics.
    
    Returns:
        {
            "results": {task: {"total_reward": float, "enrolled": int, "target": int}},
            "composite_score": float,
            "sft_complete": bool,
            "error": str or None,
        }
    """
    output = {
        "results": {},
        "composite_score": 0.0,
        "sft_complete": False,
        "error": None,
    }
    
    if not log_text:
        output["error"] = "empty log"
        return output
    
    # Check for SFT completion
    if "SFT complete!" in log_text:
        output["sft_complete"] = True
    
    # Check for crashes
    if "Traceback (most recent call last)" in log_text:
        # Extract the last error
        lines = log_text.strip().split("\n")
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() and not lines[i].startswith(" "):
                output["error"] = lines[i].strip()[:200]
                break
        if not output["error"]:
            output["error"] = "crash (traceback found)"
        return output
    
    if "No GPU!" in log_text or "AssertionError" in log_text:
        output["error"] = "no GPU or assertion error"
        return output
    
    # Parse per-task results
    # Format: [task] enrolled=X/Y reward=Z
    for task in TASKS:
        pattern = rf"\[{task}\]\s+enrolled=(\d+)/(\d+)\s+reward=([-\d.]+)"
        m = re.search(pattern, log_text)
        if m:
            enrolled = int(m.group(1))
            target = int(m.group(2))
            reward = float(m.group(3))
            output["results"][task] = {
                "total_reward": reward,
                "enrolled": enrolled,
                "target": target,
            }
    
    # Also try the JSON results format
    if not output["results"]:
        try:
            # Look for the JSON block in train_results.json format
            json_match = re.search(r'"results"\s*:\s*\{(.+?)\}(?:\s*\})', log_text, re.DOTALL)
            if json_match:
                results_json = json.loads("{" + json_match.group(1) + "}")
                for task in TASKS:
                    if task in results_json:
                        output["results"][task] = results_json[task]
        except (json.JSONDecodeError, KeyError):
            pass
    
    if output["results"]:
        output["composite_score"] = compute_composite_score(output["results"])
    else:
        output["error"] = "no results found in log"
    
    return output


def compare_experiments(current_score: float, best_score: float, tolerance: float = 0.001) -> str:
    """Compare two composite scores.
    
    Returns: "keep" if current is better, "discard" if not.
    """
    if current_score > best_score + tolerance:
        return "keep"
    return "discard"


def format_results_row(
    experiment_id: str,
    composite_score: float,
    results: dict,
    status: str,
    description: str,
) -> str:
    """Format a results.tsv row."""
    easy = results.get("easy_bench", {})
    medium = results.get("medium_bench", {})
    hard = results.get("hard_bench", {})
    
    easy_enrolled = f"{easy.get('enrolled', 0)}/{easy.get('target', 0)}"
    medium_enrolled = f"{medium.get('enrolled', 0)}/{medium.get('target', 0)}"
    hard_enrolled = f"{hard.get('enrolled', 0)}/{hard.get('target', 0)}"
    
    easy_reward = f"{easy.get('total_reward', 0):.2f}"
    medium_reward = f"{medium.get('total_reward', 0):.2f}"
    hard_reward = f"{hard.get('total_reward', 0):.2f}"
    
    return (
        f"{experiment_id}\t{composite_score:.6f}\t"
        f"{easy_enrolled}\t{easy_reward}\t"
        f"{medium_enrolled}\t{medium_reward}\t"
        f"{hard_enrolled}\t{hard_reward}\t"
        f"{status}\t{description}"
    )


RESULTS_TSV_HEADER = (
    "experiment\tcomposite_score\t"
    "easy_enrolled\teasy_reward\t"
    "medium_enrolled\tmedium_reward\t"
    "hard_enrolled\thard_reward\t"
    "status\tdescription"
)


if __name__ == "__main__":
    # Quick test with sample log
    sample = """
SFT complete!
============================================================
EVALUATION
============================================================
  [easy_bench] enrolled=8/10 reward=12.5432

  [medium_bench] enrolled=5/15 reward=8.1234

  [hard_bench] enrolled=3/20 reward=4.5678
"""
    parsed = parse_train_output(sample)
    print(f"Parsed: {json.dumps(parsed, indent=2)}")
    print(f"Composite: {parsed['composite_score']}")
