#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "trl>=1.2.0",
#     "peft>=0.13.0",
#     "httpx>=0.25.0",
#     "datasets>=2.21.0",
#     "accelerate>=0.34.0",
#     "torch",
#     "huggingface_hub>=0.25.0",
#     "jmespath",
# ]
# ///
"""
Clinical Recruitment GRPO Training — Self-contained for HF Jobs.

Uses TRL GRPOTrainer with environment_factory pattern:
  - HTTP-based env wrapper calls the deployed HF Space
  - Model learns to use tools (screen_patient, recontact, allocate_to_site, adjust_strategy)
  - GRPO compares multiple completions per prompt, updates toward better ones
  - Simple enrollment-progress reward

Usage on HF Jobs:
    hf jobs uv run --flavor t4-medium --timeout 3h --secrets HF_TOKEN train_grpo_hfjob.py

Or via Python API:
    from huggingface_hub import run_uv_job
    run_uv_job("train_grpo_hfjob.py", dependencies=["trl>=1.2.0", "peft", "httpx", "jmespath"],
               flavor="t4-medium", timeout="3h", secrets={"HF_TOKEN": token})
"""

import json
import os
import time

import httpx
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# ── Config ────────────────────────────────────────────────────────────
ENV_URL = os.getenv("ENV_URL", "https://pratimassaravanan-clinical-recruitment.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-0.6B")
HUB_REPO = os.getenv("HUB_REPO", "pratimassaravanan/clinical-qwen3-grpo")
MAX_STEPS = int(os.getenv("MAX_STEPS", "30"))
NUM_GENERATIONS = int(os.getenv("NUM_GENERATIONS", "2"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
LR = float(os.getenv("LR", "5e-6"))
LORA_R = int(os.getenv("LORA_R", "16"))
MAX_COMPLETION_LEN = int(os.getenv("MAX_COMPLETION_LEN", "512"))
DATASET_SIZE = int(os.getenv("DATASET_SIZE", "200"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "grpo_output")

print("=" * 60)
print("GRPO Training — Clinical Recruitment (TRL + HTTP env)")
print("=" * 60)
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu} | CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
else:
    print("WARNING: No GPU detected!")
print(f"Model:       {MODEL_NAME}")
print(f"Env URL:     {ENV_URL}")
print(f"Hub repo:    {HUB_REPO}")
print(f"Max steps:   {MAX_STEPS}")
print(f"Generations: {NUM_GENERATIONS}")
print(f"Grad accum:  {GRAD_ACCUM}")
print(f"LR:          {LR}")
print(f"LoRA r:      {LORA_R}")
print()

# ── System Prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical trial recruitment agent. Your ONLY goal is to enroll patients. Use the available tools.

CRITICAL: You MUST call screen_patient first when available_patients exist. This is HOW patients enter the funnel.
The recruitment pipeline is: screen_patient -> recontact -> allocate_to_site (enrollment).
You CANNOT enroll anyone without screening first!

Decision rules (follow STRICTLY in this order):
1. If allocation_candidates is non-empty: call allocate_to_site with the first patient_id and best site_id
2. If recontact_candidates is non-empty: call recontact with the first patient_id
3. If available_patients is non-empty: call screen_patient with the first patient_id
4. ONLY if all three lists are empty: call adjust_strategy with increase_outreach

Use the EXACT patient_id and site_id values from the observation."""


# ── HTTP-based Environment ────────────────────────────────────────────
class ClinicalRecruitmentHTTPEnv:
    """TRL environment_factory-compatible wrapper using HTTP REST API.

    Each instance maintains its own httpx.Client with cookies, so the
    HF Space assigns a separate server-side session per instance.
    This allows concurrent GRPO generations to run independent episodes.
    """

    def __init__(self):
        self.client = httpx.Client(timeout=60, follow_redirects=True)
        self.reward: float = 0.0
        self.done: bool = False
        self._obs: dict = {}
        self._step_count: int = 0
        self._enrolled: int = 0
        self._target: int = 100

    def reset(self, **kwargs) -> str | None:
        """Reset the environment for a new episode.

        Called at the start of each GRPO episode by GRPOTrainer.
        Receives dataset columns as kwargs (e.g., task).
        Returns initial observation text.
        """
        task = kwargs.get("task", "easy_bench")
        try:
            r = self.client.post(f"{ENV_URL}/reset", params={"task_id": task})
            r.raise_for_status()
            self._obs = r.json()
        except Exception as e:
            # If env is unreachable, return a minimal observation
            print(f"WARNING: /reset failed: {e}")
            self._obs = {"available_patients": [], "budget_remaining": 0,
                         "enrolled_so_far": 0, "target_enrollment": 100, "timestamp": 0}

        self.reward = 0.0
        self.done = False
        self._step_count = 0
        self._enrolled = self._obs.get("enrolled_so_far", 0)
        self._target = max(1, self._obs.get("target_enrollment", 100))
        return self._format_obs()

    # ------------------------------------------------------------------
    # Tool methods — auto-discovered by TRL GRPOTrainer
    # ------------------------------------------------------------------

    def screen_patient(self, patient_id: str, hypothesis: str = "noise_dominant",
                       confidence: float = 0.7) -> str:
        """Screen a candidate patient for trial eligibility.

        Args:
            patient_id: Patient ID from the available_patients list in the observation.
            hypothesis: Your hypothesis about the dominant dynamic: noise_dominant, dropout_dominant, site_bias, confounding, or unknown.
            confidence: Confidence in your hypothesis between 0.0 and 1.0.

        Returns:
            Updated observation showing screening result and new state.
        """
        return self._step({
            "action_type": "screen_patient",
            "patient_id": patient_id,
            "hypothesis": hypothesis,
            "confidence": confidence,
        })

    def recontact(self, patient_id: str, hypothesis: str = "noise_dominant",
                  confidence: float = 0.7) -> str:
        """Recontact a previously screened patient to obtain consent.

        Args:
            patient_id: Patient ID from the recontact_candidates list.
            hypothesis: Your hypothesis about the dominant dynamic.
            confidence: Confidence in your hypothesis between 0.0 and 1.0.

        Returns:
            Updated observation showing recontact result.
        """
        return self._step({
            "action_type": "recontact",
            "patient_id": patient_id,
            "hypothesis": hypothesis,
            "confidence": confidence,
        })

    def allocate_to_site(self, patient_id: str, site_id: str,
                         hypothesis: str = "noise_dominant",
                         confidence: float = 0.8) -> str:
        """Allocate a consented patient to a recruitment site for enrollment.

        This is the primary action that ENROLLS patients. Only patients
        in allocation_candidates can be allocated.

        Args:
            patient_id: Patient ID from the allocation_candidates list.
            site_id: Site ID from the site_performance dictionary.
            hypothesis: Your hypothesis about the dominant dynamic.
            confidence: Confidence in your hypothesis between 0.0 and 1.0.

        Returns:
            Updated observation showing allocation result.
        """
        return self._step({
            "action_type": "allocate_to_site",
            "patient_id": patient_id,
            "site_id": site_id,
            "hypothesis": hypothesis,
            "confidence": confidence,
        })

    def adjust_strategy(self, strategy_change: str = "increase_outreach",
                        hypothesis: str = "noise_dominant",
                        confidence: float = 0.6) -> str:
        """Adjust recruitment strategy when no direct patient actions are available.

        Args:
            strategy_change: One of increase_outreach, relax_criteria, tighten_criteria, focus_site_A, focus_site_B, focus_site_C.
            hypothesis: Your hypothesis about the dominant dynamic.
            confidence: Confidence in your hypothesis between 0.0 and 1.0.

        Returns:
            Updated observation showing strategy adjustment effect.
        """
        return self._step({
            "action_type": "adjust_strategy",
            "strategy_change": strategy_change,
            "hypothesis": hypothesis,
            "confidence": confidence,
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _step(self, action: dict) -> str:
        """Execute one environment step via HTTP and return formatted observation."""
        if self.done:
            raise ValueError("Episode is finished. No more actions can be taken.")

        self._step_count += 1
        try:
            r = self.client.post(f"{ENV_URL}/step", json=action)
            r.raise_for_status()
            data = r.json()
            self._obs = data
            self.reward = float(data.get("reward", 0.0))
            self.done = bool(data.get("done", False))
            self._enrolled = data.get("enrolled_so_far", self._enrolled)
        except Exception as e:
            # On HTTP error, mark as done to prevent infinite loops
            print(f"WARNING: /step failed at step {self._step_count}: {e}")
            self.done = True
            self.reward = 0.0

        return self._format_obs()

    def _format_obs(self) -> str:
        """Format current observation as compact text for the model."""
        o = self._obs
        avail = [p.get("id", "?") for p in o.get("available_patients", [])[:5]]
        recon = [p.get("id", "?") for p in o.get("recontact_candidates", [])[:5]]
        alloc = [p.get("id", "?") for p in o.get("allocation_candidates", [])[:5]]
        sites = list(o.get("site_performance", {}).keys())[:5]

        parts = [
            f"step={o.get('timestamp', 0)}",
            f"budget={o.get('budget_remaining', 0):.0f}",
            f"enrolled={o.get('enrolled_so_far', 0)}/{o.get('target_enrollment', 100)}",
        ]
        if avail:
            parts.append(f"available_patients={avail}")
        if recon:
            parts.append(f"recontact_candidates={recon}")
        if alloc:
            parts.append(f"allocation_candidates={alloc}")
        if sites:
            parts.append(f"sites={sites}")

        funnel = o.get("current_funnel", {})
        if funnel:
            parts.append(f"funnel={funnel}")

        events = o.get("recent_events", [])
        if events:
            parts.append(f"events={events[:3]}")

        return " ".join(parts)


# ── Reward Function ───────────────────────────────────────────────────
def reward_func(environments, **kwargs) -> list[float]:
    """Progressive reward based on funnel progress.

    Gives partial credit for screening/consenting even without enrollment.
    This ensures GRPO has variance to learn from early in training.

    Reward tiers:
      - 1.0: enrolled >= target (full success)
      - 0.3-0.9: enrolled > 0 (proportional to target)
      - 0.2: consented or eligible patients exist (almost there)
      - 0.1: screened patients (started the funnel)
      - 0.0: no progress (only adjust_strategy)
    """
    rewards = []
    for env in environments:
        enrolled = env._obs.get("enrolled_so_far", 0)
        target = max(1, env._obs.get("target_enrollment", 100))
        funnel = env._obs.get("current_funnel", {})
        screened = funnel.get("screened", 0)
        eligible = funnel.get("eligible", 0)
        consented = funnel.get("consented", 0)

        if enrolled > 0:
            # Scale from 0.3 to 1.0 based on enrollment progress
            rewards.append(min(1.0, 0.3 + 0.7 * (enrolled / target)))
        elif consented > 0 or eligible > 0:
            rewards.append(0.2)
        elif screened > 0:
            rewards.append(0.1)
        else:
            rewards.append(0.0)
    return rewards


# ── Build Dataset ─────────────────────────────────────────────────────
def build_dataset(n: int) -> Dataset:
    """Create training prompts — easy_bench only for initial learning."""
    # Start with easy_bench only: guaranteed available patients from step 0
    tasks = ["easy_bench"]
    prompts = []
    task_ids = []
    for i in range(n):
        task = tasks[i % len(tasks)]
        prompts.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                "Start a clinical trial recruitment episode. "
                "The observation will show available_patients — call screen_patient with "
                "the first patient_id to begin. Then recontact screened patients, "
                "and allocate consented patients to sites. "
                "Your score depends on how many patients you enroll."
            )},
        ])
        task_ids.append(task)
    return Dataset.from_dict({"prompt": prompts, "task": task_ids})


# ── Main ──────────────────────────────────────────────────────────────
def main():
    # Verify env is reachable
    print("Checking environment connectivity...")
    try:
        r = httpx.get(f"{ENV_URL}/health", timeout=10, follow_redirects=True)
        print(f"  /health -> {r.status_code}: {r.json()}")
    except Exception as e:
        print(f"  WARNING: Env not reachable: {e}")
        print("  Training will proceed but tool calls may fail.")
    print()

    # Build dataset
    dataset = build_dataset(DATASET_SIZE)
    print(f"Dataset: {len(dataset)} prompts across 3 difficulty levels")

    # LoRA config for memory efficiency on T4
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    print(f"LoRA config: r={LORA_R}, targets=qkvo+gate+up+down")

    # GRPO config — tuned for T4 (16GB) memory budget
    config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LEN,
        logging_steps=1,
        save_steps=max(5, MAX_STEPS // 5),
        save_total_limit=2,
        report_to="none",
        chat_template_kwargs={"enable_thinking": False},
        log_completions=True,
        num_completions_to_print=2,
        gradient_checkpointing=True,
        warmup_steps=3,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )
    print(f"GRPO config: {MAX_STEPS} steps, {NUM_GENERATIONS} generations, "
          f"grad_accum={GRAD_ACCUM}, lr={LR}")
    print()

    # Create trainer
    print("Creating GRPOTrainer with environment_factory=ClinicalRecruitmentHTTPEnv")
    print("TRL will auto-discover 4 tool methods: screen_patient, recontact, "
          "allocate_to_site, adjust_strategy")
    print()

    trainer = GRPOTrainer(
        model=MODEL_NAME,
        reward_funcs=reward_func,
        train_dataset=dataset,
        args=config,
        peft_config=peft_config,
        environment_factory=ClinicalRecruitmentHTTPEnv,
    )

    # Train
    print("Starting GRPO training...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time
    print(f"\nGRPO training complete! Elapsed: {elapsed/60:.1f} minutes")
    print(f"Global step: {train_result.global_step}")
    print(f"Training loss: {train_result.training_loss:.4f}")

    # Save locally
    trainer.save_model(f"{OUTPUT_DIR}/final_model")
    print(f"Model saved to {OUTPUT_DIR}/final_model")

    # Push to Hub
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        try:
            print(f"Pushing to Hub: {HUB_REPO}...")
            trainer.push_to_hub(HUB_REPO)
            print(f"Successfully pushed to {HUB_REPO}")
        except Exception as e:
            print(f"Push to Hub failed: {e}")
            # Try manual upload as fallback
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_token)
                api.create_repo(HUB_REPO, exist_ok=True)
                api.upload_folder(
                    folder_path=f"{OUTPUT_DIR}/final_model",
                    repo_id=HUB_REPO,
                    commit_message="GRPO-trained clinical recruitment agent",
                )
                print(f"Uploaded via HfApi to {HUB_REPO}")
            except Exception as e2:
                print(f"Manual upload also failed: {e2}")
    else:
        print("No HF_TOKEN set — skipping Hub push")

    # Save results summary
    results = {
        "model": MODEL_NAME,
        "training_method": "GRPO via TRL environment_factory (HTTP env)",
        "env_url": ENV_URL,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "config": {
            "max_steps": MAX_STEPS,
            "num_generations": NUM_GENERATIONS,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "learning_rate": LR,
            "lora_r": LORA_R,
            "max_completion_length": MAX_COMPLETION_LEN,
            "dataset_size": DATASET_SIZE,
        },
        "result": {
            "global_step": train_result.global_step,
            "training_loss": round(train_result.training_loss, 6),
            "elapsed_minutes": round(elapsed / 60, 1),
        },
    }
    results_path = f"{OUTPUT_DIR}/grpo_results.json"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
