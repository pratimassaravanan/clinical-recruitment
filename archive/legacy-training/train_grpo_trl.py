#!/usr/bin/env python3
"""
Clinical Recruitment — GRPO Training via TRL environment_factory

This is the CORRECT integration pattern for TRL + OpenEnv:
  - Uses `environment_factory=ClinicalRecruitmentToolEnv` (from tool_env.py)
  - Each tool method (screen_patient, recontact, allocate_to_site, etc.) is
    auto-discovered by TRL and exposed as a function-calling tool
  - Reward functions receive `environments` kwarg with live env instances
  - The training loop actually STEPS the environment (not a proxy scorer)

This follows the exact pattern from:
  https://huggingface.co/docs/trl/openenv

Usage (Colab T4 / Kaggle T4):
    pip install trl>=1.2.0 datasets accelerate peft jmespath
    pip install openenv-core>=0.2.0

    # Option 1: Colocate mode (1 GPU, recommended for T4)
    python train_grpo_trl.py --vllm-mode colocate

    # Option 2: Without vLLM (slower but simpler)
    python train_grpo_trl.py --no-vllm

Env vars:
    MODEL_NAME     (default: Qwen/Qwen3-0.6B)
    MAX_STEPS      (default: 50)
    OUTPUT_DIR     (default: train_output/grpo_trl)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime

# ── Ensure project root is on path ────────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# Import the TRL-compatible tool environment and reward functions
from tool_env import (
    ClinicalRecruitmentToolEnv,
    reward_enrollment_progress,
    reward_budget_efficiency,
    reward_screening_accuracy,
    reward_action_diversity,
    reward_hypothesis_consistency,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="GRPO training for Clinical Recruitment using TRL environment_factory"
    )
    p.add_argument("--model", default=os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B"),
                   help="Model identifier (default: Qwen/Qwen3-0.6B)")
    p.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "50")),
                   help="Max GRPO training steps")
    p.add_argument("--num-generations", type=int, default=4,
                   help="Completions per prompt for GRPO comparison")
    p.add_argument("--gradient-accumulation-steps", type=int, default=16,
                   help="Gradient accumulation steps")
    p.add_argument("--learning-rate", type=float, default=5e-6,
                   help="Learning rate for GRPO")
    p.add_argument("--max-completion-length", type=int, default=1024,
                   help="Max tokens per multi-turn episode (all turns combined)")
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "train_output/grpo_trl"),
                   help="Output directory for checkpoints")
    p.add_argument("--dataset-size", type=int, default=200,
                   help="Number of training prompts")
    p.add_argument("--use-vllm", action="store_true", default=False,
                   help="Use vLLM for generation (requires vllm installed)")
    p.add_argument("--vllm-mode", default="colocate",
                   choices=["colocate", "server"],
                   help="vLLM mode (only used if --use-vllm)")
    p.add_argument("--save-plots", action="store_true", default=True,
                   help="Save training reward plots after training")
    return p.parse_args()


# ── System prompts ────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a clinical trial recruitment agent managing patient enrollment.

Your goal is to maximize enrollment while staying within budget. You have access to tools
for screening patients, following up with candidates, allocating patients to sites, and
adjusting your recruitment strategy.

Priority rules:
1. If allocation_candidates are available and sites have capacity: use allocate_to_site
2. If recontact_candidates are available: use recontact to convert them
3. If available_patients exist: use screen_patient to start the funnel
4. Otherwise: use adjust_strategy with increase_outreach

Always provide a hypothesis about what's driving outcomes (noise_dominant, dropout_dominant, or site_bias).
Read the observation carefully and use the EXACT patient_id and site_id values shown."""


def build_dataset(size: int, tasks: list[str] | None = None) -> Dataset:
    """Build training dataset with task-routed prompts."""
    if tasks is None:
        tasks = ["easy_bench", "medium_bench", "hard_bench"]

    prompts = []
    task_ids = []
    for i in range(size):
        task = tasks[i % len(tasks)]
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"You are managing a clinical trial recruitment episode (task: {task}). "
                "Use the available tools to screen patients, follow up, allocate to sites, "
                "and adjust strategy. Maximize enrollment within the budget constraint. "
                "Read each observation carefully and choose the best action."
            )},
        ]
        prompts.append(prompt)
        task_ids.append(task)

    return Dataset.from_dict({"prompt": prompts, "task": task_ids})


def main():
    args = parse_args()

    print("=" * 60)
    print("GRPO Training — Clinical Recruitment (TRL environment_factory)")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Max steps:      {args.max_steps}")
    print(f"Generations:    {args.num_generations}")
    print(f"Grad accum:     {args.gradient_accumulation_steps}")
    print(f"Learning rate:  {args.learning_rate}")
    print(f"Max completion: {args.max_completion_length}")
    print(f"Dataset size:   {args.dataset_size}")
    print(f"Output dir:     {args.output_dir}")
    print(f"Use vLLM:       {args.use_vllm}")
    print()

    # ── Build dataset ──────────────────────────────────────────────────
    dataset = build_dataset(args.dataset_size)
    print(f"Dataset: {len(dataset)} prompts across 3 tasks")

    # ── Configure GRPO ─────────────────────────────────────────────────
    config_kwargs = dict(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        logging_steps=1,
        save_steps=max(10, args.max_steps // 5),
        save_total_limit=3,
        report_to="none",
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
    )

    if args.use_vllm:
        config_kwargs["use_vllm"] = True
        config_kwargs["vllm_mode"] = args.vllm_mode

    grpo_config = GRPOConfig(**config_kwargs)

    # ── Reward functions ───────────────────────────────────────────────
    # These receive `environments` kwarg from TRL — each env is a
    # ClinicalRecruitmentToolEnv instance with live state.
    reward_funcs = [
        reward_enrollment_progress,
        reward_budget_efficiency,
        reward_screening_accuracy,
        reward_action_diversity,
        reward_hypothesis_consistency,
    ]
    print(f"Reward functions: {len(reward_funcs)} components")
    print("  - enrollment_progress: fraction of target reached")
    print("  - budget_efficiency: enrollment per unit budget")
    print("  - screening_accuracy: enrolled/screened ratio")
    print("  - action_diversity: fraction of 8 action types used")
    print("  - hypothesis_consistency: penalizes erratic switching")
    print()

    # ── Create trainer ─────────────────────────────────────────────────
    # KEY: environment_factory=ClinicalRecruitmentToolEnv
    # TRL will:
    #   1. Create one instance per generation
    #   2. Call reset() at episode start
    #   3. Auto-discover tool methods (screen_patient, recontact, etc.)
    #   4. Run multi-turn tool-calling loop
    #   5. Pass environments list to reward functions
    print("Creating GRPOTrainer with environment_factory=ClinicalRecruitmentToolEnv")
    print("TRL will auto-discover 8 tool methods from tool_env.py")
    print()

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_funcs,
        train_dataset=dataset,
        args=grpo_config,
        environment_factory=ClinicalRecruitmentToolEnv,
    )

    # ── Train ──────────────────────────────────────────────────────────
    print("Starting GRPO training...")
    print(f"Each step: model generates {args.num_generations} completions per prompt,")
    print(f"each completion interacts with a live ClinicalRecruitmentEnv instance.")
    print()

    train_result = trainer.train()

    # ── Save results ───────────────────────────────────────────────────
    results_dir = pathlib.Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": args.model,
        "training_method": "GRPO via TRL environment_factory",
        "environment": "ClinicalRecruitmentToolEnv (tool_env.py)",
        "reward_functions": [f.__name__ for f in reward_funcs],
        "config": {
            "max_steps": args.max_steps,
            "num_generations": args.num_generations,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "max_completion_length": args.max_completion_length,
            "dataset_size": args.dataset_size,
        },
        "train_result": {
            "global_step": train_result.global_step,
            "training_loss": train_result.training_loss,
        },
        "timestamp": datetime.now().isoformat(),
        "integration_pattern": (
            "TRL environment_factory: model generates tool calls -> "
            "TRL executes them against ClinicalRecruitmentToolEnv methods -> "
            "reward functions read env.last_observation for scoring -> "
            "GRPO computes advantages from reward differences across generations"
        ),
    }

    results_path = results_dir / "grpo_trl_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {results_path}")

    # Save the model
    trainer.save_model(str(results_dir / "final_model"))
    print(f"Model saved to {results_dir / 'final_model'}")

    print(f"\n{'=' * 60}")
    print("GRPO Training Complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
