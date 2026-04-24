#!/usr/bin/env python3
"""Clinical Recruitment GRPO Training — Single .py for Lightning AI / Colab / Kaggle.

Usage:
    pip install unsloth --no-deps trl>=0.19.0 "transformers>=5.2.0,<=5.5.0" \
        datasets>=2.21.0 accelerate>=0.34.0 "openenv-core[core]>=0.2.1"
    python train.py

Requires: T4/A10/L4/A100 GPU with fp16 support.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # single GPU — avoids DDP issues with GRPO

import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

pip("--upgrade", "pip")
pip("unsloth")
pip("--no-deps", "trl>=0.19.0")
pip("transformers>=5.2.0,<=5.5.0")
pip("datasets>=2.21.0", "accelerate>=0.34.0", "openenv-core[core]>=0.2.1")

import json, pathlib, torch, warnings
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from openenv.core import GenericEnvClient

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Both `max_new_tokens`")
warnings.filterwarnings("ignore", message=".*max_length.*max_new_tokens.*")

assert torch.cuda.is_available(), "No CUDA GPU found!"
gpu = torch.cuda.get_device_name(0)
print(f"GPU: {gpu} | CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")

# ── Config ────────────────────────────────────────────────────────────
ENV_URL = os.getenv("ENV_URL", "https://pratimassaravanan-clinical-recruitment.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen3-4B-unsloth-bnb-4bit")
TASKS = ["easy_bench", "medium_bench", "hard_bench"]
MAX_SEQ      = int(os.getenv("MAX_SEQ", "2048"))
LORA_R       = int(os.getenv("LORA_R", "16"))
LORA_ALPHA   = int(os.getenv("LORA_ALPHA", "16"))
NUM_GEN      = int(os.getenv("NUM_GEN", "4"))
MAX_COMP     = int(os.getenv("MAX_COMP", "1024"))
BATCH        = int(os.getenv("BATCH", "2"))
GRAD_ACC     = int(os.getenv("GRAD_ACC", "4"))
EPOCHS       = int(os.getenv("EPOCHS", "1"))
LR           = float(os.getenv("LR", "5e-6"))
NUM_PROMPTS  = int(os.getenv("NUM_PROMPTS", "48"))
EVAL_STEPS   = int(os.getenv("EVAL_STEPS", "30"))
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "grpo_clinical_recruitment")
RESULTS_DIR  = pathlib.Path(os.getenv("RESULTS_DIR", "data/training_outputs"))

SYSTEM_PROMPT = ("You are a long-horizon clinical trial recruitment agent. "
                 "Use tools to manage screening, recontact, allocation, planning, and memory.")
_H, _C = "noise_dominant", 0.6

print(f"ENV_URL: {ENV_URL}")
print(f"MODEL:   {MODEL_NAME}")
print(f"CONFIG:  batch={BATCH} grad_acc={GRAD_ACC} lr={LR} epochs={EPOCHS} prompts={NUM_PROMPTS}")

# ── Load model ────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None)
model = FastLanguageModel.get_peft_model(
    model, r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", use_gradient_checkpointing="unsloth", random_state=3407)
model.print_trainable_parameters()

# ── Environment wrapper ──────────────────────────────────────────────
class ClinicalRecruitmentToolEnv:
    """OpenEnv tool-call environment for Clinical Recruitment."""

    def __init__(self):
        self.client = GenericEnvClient(base_url=ENV_URL).sync()
        self.reward = self.initial_budget = 0.0
        self.done = False
        self.last_result = self.last_observation = None
        self.action_history = []
        self.enrollment_history = []
        self.budget_history = []
        self.hypothesis_history = []

    def reset(self, **kw):
        """Reset environment. Called by trainer lifecycle."""
        self.client.connect()
        self.last_result = self.client.reset(task=kw.get("task") or kw.get("task_id") or "easy_bench")
        self.last_observation = self.last_result.observation
        self.reward, self.done = 0.0, bool(self.last_result.done)
        self.action_history = []
        self.enrollment_history = []
        self.budget_history = []
        self.hypothesis_history = []
        obs = self.last_observation or {}
        self.initial_budget = obs.get("budget_remaining", 0.0)
        self.enrollment_history.append(obs.get("enrolled_so_far", 0))
        self.budget_history.append(obs.get("budget_remaining", 0.0))
        return self._fmt()

    def close(self):
        """Close client. Called by trainer lifecycle."""
        try: self.client.close()
        except Exception: pass

    def screen_patient(self, patient_id: str, hypothesis: str = _H, confidence: float = _C) -> str:
        """Screen a candidate patient for eligibility.

        Args:
            patient_id: Patient id from available_patients.
            hypothesis: Current guess about dominant dynamics.
            confidence: Confidence in the hypothesis.
        """
        return self._step({"action_type": "screen_patient", "patient_id": patient_id, "hypothesis": hypothesis, "confidence": confidence})

    def recontact(self, patient_id: str, hypothesis: str = _H, confidence: float = _C) -> str:
        """Recontact an eligible patient for consent or enrollment.

        Args:
            patient_id: Patient id from recontact_candidates.
            hypothesis: Current guess about dominant dynamics.
            confidence: Confidence in the hypothesis.
        """
        return self._step({"action_type": "recontact", "patient_id": patient_id, "hypothesis": hypothesis, "confidence": confidence})

    def allocate_to_site(self, patient_id: str, site_id: str, hypothesis: str = _H, confidence: float = _C) -> str:
        """Allocate a consented patient to a recruitment site.

        Args:
            patient_id: Patient id from allocation_candidates.
            site_id: Site id from site_performance.
            hypothesis: Current guess about dominant dynamics.
            confidence: Confidence in the hypothesis.
        """
        return self._step({"action_type": "allocate_to_site", "patient_id": patient_id, "site_id": site_id, "hypothesis": hypothesis, "confidence": confidence})

    def adjust_strategy(self, strategy_change: str, hypothesis: str = _H, confidence: float = _C) -> str:
        """Adjust recruitment strategy such as increase_outreach or negotiate_site_A.

        Args:
            strategy_change: Strategy name like increase_outreach or tighten_criteria.
            hypothesis: Current guess about dominant dynamics.
            confidence: Confidence in the hypothesis.
        """
        return self._step({"action_type": "adjust_strategy", "strategy_change": strategy_change, "hypothesis": hypothesis, "confidence": confidence})

    def plan_next_phase(self, target_phase: str, plan_summary: str = "advance the bottleneck") -> str:
        """Create or revise the current high-level recruitment plan.

        Args:
            target_phase: One of screening, conversion, allocation, retention, recovery.
            plan_summary: Natural-language summary of the plan.
        """
        return self._step({"action_type": "plan_next_phase", "target_phase": target_phase, "plan_summary": plan_summary})

    def summarize_and_index(self, memory_key: str, memory_payload: str) -> str:
        """Write a summary into indexed episodic memory.

        Args:
            memory_key: Key for the indexed memory item.
            memory_payload: Summary text to store.
        """
        return self._step({"action_type": "summarize_and_index", "memory_key": memory_key, "memory_payload": memory_payload})

    def retrieve_relevant_history(self, memory_query: str) -> str:
        """Retrieve indexed memory entries relevant to the current bottleneck.

        Args:
            memory_query: Query string for indexed memory retrieval.
        """
        return self._step({"action_type": "retrieve_relevant_history", "memory_query": memory_query})

    def stop_recruitment(self) -> str:
        """End the current recruitment episode early."""
        return self._step({"action_type": "stop_recruitment"})

    def _step(self, action):
        if self.done: raise ValueError("Episode finished.")
        self.last_result = self.client.step(action)
        self.last_observation = self.last_result.observation
        self.reward, self.done = float(self.last_result.reward or 0), bool(self.last_result.done)
        self.action_history.append(action.get("action_type", ""))
        obs = self.last_observation or {}
        self.enrollment_history.append(obs.get("enrolled_so_far", 0))
        self.budget_history.append(obs.get("budget_remaining", 0))
        if h := action.get("hypothesis"): self.hypothesis_history.append(h)
        return self._fmt()

    def _fmt(self):
        o = self.last_observation or {}
        return (f"step={o.get('timestamp')} budget={o.get('budget_remaining')} "
                f"enrolled={o.get('enrolled_so_far')}/{o.get('target_enrollment')} "
                f"avail={len(o.get('available_patients', []))} recontact={len(o.get('recontact_candidates', []))} "
                f"allocate={len(o.get('allocation_candidates', []))} funnel={o.get('current_funnel', {})}")

# ── Reward functions (dual-signature for TRL compatibility) ──────────
def reward_enrollment_progress(prompts=None, completions=None, environments=None, **_):
    """Fraction of target enrollment reached."""
    if environments is None: return [0.0] * len(prompts or completions)
    return [min(1.0, (e.last_observation or {}).get("enrolled_so_far", 0) / max(1, (e.last_observation or {}).get("target_enrollment", 100))) for e in environments]

def reward_budget_efficiency(prompts=None, completions=None, environments=None, **_):
    """Enrollment per unit budget spent."""
    if environments is None: return [0.0] * len(prompts or completions)
    r = []
    for e in environments:
        o = e.last_observation or {}
        ib = e.initial_budget or 1
        spent = max(0, ib - o.get("budget_remaining", 0))
        t = max(1, o.get("target_enrollment", 100))
        r.append(min(1.0, (o.get("enrolled_so_far", 0) / t) / (spent / ib)) if spent > 0 else 0.0)
    return r

def reward_screening_accuracy(prompts=None, completions=None, environments=None, **_):
    """Enrolled-to-screened ratio minus dropout penalty."""
    if environments is None: return [0.0] * len(prompts or completions)
    r = []
    for e in environments:
        f = (e.last_observation or {}).get("current_funnel", {})
        s = f.get("screened", 0)
        r.append(max(0, min(1, f.get("enrolled", 0) / s - 0.5 * f.get("dropped", 0) / s)) if s > 0 else 0.0)
    return r

def reward_action_diversity(prompts=None, completions=None, environments=None, **_):
    """Fraction of 8 action types used."""
    if environments is None: return [0.0] * len(prompts or completions)
    return [min(1, len(set(e.action_history)) / 8) if e.action_history else 0.0 for e in environments]

def reward_hypothesis_consistency(prompts=None, completions=None, environments=None, **_):
    """Penalizes switching, rewards correct world-type match."""
    if environments is None: return [0.0] * len(prompts or completions)
    r = []
    for e in environments:
        hs = e.hypothesis_history
        if len(hs) < 2: r.append(0.5); continue
        sw = sum(1 for i in range(1, len(hs)) if hs[i] != hs[i - 1])
        con = 1.0 if sw <= 1 else max(0.2, 1 - (sw - 1) * 0.2)
        wt = (e.last_observation or {}).get("world_type", "")
        bonus = 0.2 if {"dropout_dominant": "dropout", "noise_dominant": "noise", "site_bias": "site_bias"}.get(hs[-1], "") == wt and wt else 0
        r.append(min(1.0, con * 0.8 + bonus))
    return r

REWARD_FUNCS = [reward_enrollment_progress, reward_budget_efficiency,
                reward_screening_accuracy, reward_action_diversity,
                reward_hypothesis_consistency]

# ── Eval helper ──────────────────────────────────────────────────────
def _parse_action(resp, obs):
    if "screen" in resp and obs.get("available_patients"):
        return {"action_type": "screen_patient", "patient_id": obs["available_patients"][0]["id"], "hypothesis": _H, "confidence": _C}
    if "allocate" in resp and obs.get("allocation_candidates") and obs.get("site_performance"):
        return {"action_type": "allocate_to_site", "patient_id": obs["allocation_candidates"][0]["id"], "site_id": list(obs["site_performance"])[0], "hypothesis": _H, "confidence": _C}
    if "recontact" in resp and obs.get("recontact_candidates"):
        return {"action_type": "recontact", "patient_id": obs["recontact_candidates"][0]["id"], "hypothesis": _H, "confidence": _C}
    if "adjust" in resp:
        return {"action_type": "adjust_strategy", "strategy_change": "increase_outreach", "hypothesis": _H, "confidence": _C}
    if "plan" in resp:
        return {"action_type": "plan_next_phase", "target_phase": "screening", "plan_summary": "screen more"}
    if "stop" in resp:
        return {"action_type": "stop_recruitment"}
    if obs.get("available_patients"):
        return {"action_type": "screen_patient", "patient_id": obs["available_patients"][0]["id"], "hypothesis": _H, "confidence": _C}
    return {"action_type": "adjust_strategy", "strategy_change": "increase_outreach", "hypothesis": _H, "confidence": _C}

def run_episode(mdl, tok, task="easy_bench", n=None):
    n = n or EVAL_STEPS
    env = ClinicalRecruitmentToolEnv()
    try: obs_text = env.reset(task=task)
    except Exception as e: return {"task": task, "error": str(e)}
    FastLanguageModel.for_inference(mdl)
    total_r, steps = 0.0, 0
    for _ in range(n):
        if env.done: break
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"State: {obs_text}\nChoose next action."}]
        input_ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            max_length=MAX_SEQ, truncation=True, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(input_ids=input_ids, max_new_tokens=256, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        resp = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).lower()
        try:
            obs_text = env._step(_parse_action(resp, env.last_observation or {}))
            total_r += env.reward; steps += 1
        except Exception: break
    fo = env.last_observation or {}
    res = {"task": task, "actions": steps, "total_reward": round(total_r, 4),
           "enrolled": fo.get("enrolled_so_far", 0), "target": fo.get("target_enrollment", 100)}
    for fn in REWARD_FUNCS:
        res[fn.__name__.replace("reward_", "")] = round(fn(environments=[env])[0], 4)
    env.close()
    return res

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    sep = "=" * 60

    # Before training
    print(f"\n{sep}\nBEFORE TRAINING\n{sep}")
    before = {}
    for t in TASKS:
        r = run_episode(model, tokenizer, t)
        before[t] = r
        print(f"  [{t}] enrolled={r.get('enrolled',0)}/{r.get('target',0)} reward={r.get('total_reward',0)}")

    # Dataset
    ds = Dataset.from_dict({
        "prompt": [[{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "Improve recruitment."}] for _ in range(NUM_PROMPTS)],
        "task_id": [TASKS[i % 3] for i in range(NUM_PROMPTS)],
    })

    # Train
    FastLanguageModel.for_training(model)
    cfg = GRPOConfig(
        output_dir=OUTPUT_DIR, num_generations=NUM_GEN, max_completion_length=MAX_COMP,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS, learning_rate=LR, logging_steps=1, save_steps=50,
        bf16=False, fp16=True, optim="adamw_8bit",
        warmup_steps=2, lr_scheduler_type="cosine",
        report_to="none",
    )
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, train_dataset=ds,
        reward_funcs=REWARD_FUNCS, environment_factory=ClinicalRecruitmentToolEnv, args=cfg,
    )
    print(f"\n{sep}\nGRPO TRAINING on {gpu}\n{sep}")
    trainer.train()

    # After training
    print(f"\n{sep}\nAFTER TRAINING\n{sep}")
    after = {}
    for t in TASKS:
        r = run_episode(model, tokenizer, t)
        after[t] = r
        print(f"  [{t}] enrolled={r.get('enrolled',0)}/{r.get('target',0)} reward={r.get('total_reward',0)}")

    # Comparison
    CMP = ["total_reward", "enrolled", "enrollment_progress", "budget_efficiency",
           "screening_accuracy", "action_diversity", "hypothesis_consistency"]
    print(f"\n{sep}\nCOMPARISON\n{sep}")
    for t in TASKS:
        b, a = before.get(t, {}), after.get(t, {})
        print(f"\n  [{t}]")
        for k in CMP:
            bv, av = b.get(k, 0), a.get(k, 0)
            if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
                print(f"    {k:>25}  {bv:>10.4f}  {av:>10.4f}  {'+' if av - bv >= 0 else ''}{av - bv:>9.4f}")

    # Save
    lp = f"{OUTPUT_DIR}/lora_adapter"
    model.save_pretrained(lp); tokenizer.save_pretrained(lp)
    merged = model.merge_and_unload()
    mp = f"{OUTPUT_DIR}/merged_model"
    merged.save_pretrained(mp); tokenizer.save_pretrained(mp)
    print(f"\nLoRA -> {lp} | Merged -> {mp}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "grpo_results.json").write_text(
        json.dumps({"before": before, "after": after, "model": MODEL_NAME, "env_url": ENV_URL, "gpu": gpu}, indent=2))
    print(f"\n{sep}\nDONE\n{sep}")
