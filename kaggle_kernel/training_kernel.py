#!/usr/bin/env python3
"""Clinical Recruitment GRPO Training — Kaggle GPU Kernel.

Trains a 4-bit LoRA model with Unsloth + TRL GRPO against the live
Adaptive Clinical Recruitment environment on HF Spaces.
"""

# ── 0. Install dependencies ──────────────────────────────────────────
import subprocess, sys

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# Kaggle kernel: use standard HF + PEFT LoRA.
# Falls back to CPU float32 if GPU is incompatible (P100 sm_60).
pip("--upgrade", "pip")
# TRL environment_factory requires transformers from main (>=5.2.0)
pip("git+https://github.com/huggingface/transformers.git@main")
pip("trl>=0.19.0", "datasets>=2.21.0",
    "accelerate>=0.34.0", "openenv-core>=0.2.1", "peft>=0.15.0")

# ── 1. GPU safety check BEFORE importing torch ──────────────────────
import json, os, pathlib

# Test GPU compatibility before torch import contaminates the runtime
_gpu_test = subprocess.run(
    [sys.executable, "-c",
     "import torch; t=torch.zeros(1,device='cuda'); print('OK')"],
    capture_output=True, text=True, timeout=30
)
if "OK" not in _gpu_test.stdout:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    print("GPU incompatible, forcing CPU-only mode")
else:
    print("GPU OK")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from openenv import GenericEnvClient

# ── 2. Config ─────────────────────────────────────────────────────────
ENV_URL = os.getenv("ENV_URL", "https://pratimassaravanan-clinical-recruitment.hf.space")
MODEL_NAME = "Qwen/Qwen3-0.6B"
TASKS = ["easy_bench", "medium_bench", "hard_bench"]
MAX_SEQ, LORA_R, LORA_ALPHA = 2048, 16, 16
NUM_GEN, MAX_COMP, BATCH, GRAD_ACC, EPOCHS, LR = 2, 1024, 1, 4, 1, 5e-6
OUTPUT_DIR = "grpo_clinical_recruitment"
RESULTS_DIR = pathlib.Path("data/training_outputs")
SYSTEM_PROMPT = (
    "You are a long-horizon clinical trial recruitment agent. "
    "Use tools to inspect and improve recruitment progress across "
    "screening, recontact, allocation, planning, and memory."
)

print(f"ENV_URL: {ENV_URL}")
print(f"MODEL:   {MODEL_NAME}")
print(f"GPU:     {'hidden (incompatible)' if os.environ.get('CUDA_VISIBLE_DEVICES') == '' else (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')}")

# ── 3. Load model with LoRA ───────────────────────────────────────────
USE_GPU = torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != ""
device_map = "auto" if USE_GPU else "cpu"
dtype = torch.float32  # safe default; bf16 only if GPU confirmed working
if USE_GPU:
    try:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    except Exception:
        dtype = torch.float16
print(f"Training device: {'GPU' if USE_GPU else 'CPU'}, dtype={dtype}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map=device_map, torch_dtype=dtype, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(f"Model loaded: {MODEL_NAME} (LoRA r={LORA_R}, device={device_map}, dtype={dtype})")

# ── 4. Environment wrapper ───────────────────────────────────────────
_H, _C = "noise_dominant", 0.6

class ClinicalRecruitmentToolEnv:
    def __init__(self):
        self.client = GenericEnvClient(base_url=ENV_URL).sync()
        self.reward = self.initial_budget = 0.0
        self.done, self.last_result, self.last_observation = False, None, None
        self.action_history, self.enrollment_history = [], []
        self.budget_history, self.hypothesis_history = [], []

    def reset(self, **kw):
        """Reset the environment for a new episode. Not a tool — called by the trainer lifecycle."""
        self.client.connect()
        task = kw.get("task") or kw.get("task_id") or "easy_bench"
        self.last_result = self.client.reset(task=task)
        self.last_observation = self.last_result.observation
        self.reward, self.done = 0.0, bool(self.last_result.done)
        self.action_history, self.enrollment_history = [], []
        self.budget_history, self.hypothesis_history = [], []
        obs = self.last_observation or {}
        self.initial_budget = obs.get("budget_remaining", 0.0)
        self.enrollment_history.append(obs.get("enrolled_so_far", 0))
        self.budget_history.append(obs.get("budget_remaining", 0.0))
        return self._fmt()

    def close(self):
        """Close the environment client. Not a tool — called by the trainer lifecycle."""
        try: self.client.close()
        except Exception: pass

    def screen_patient(self, patient_id: str, hypothesis: str = _H, confidence: float = _C) -> str:
        """Screen a candidate patient for eligibility.

        Args:
            patient_id: Patient id from available_patients.
            hypothesis: Current guess about dominant dynamics.
            confidence: Confidence in the hypothesis.
        """
        return self._step(dict(action_type="screen_patient", patient_id=patient_id, hypothesis=hypothesis, confidence=confidence))
    def recontact(self, patient_id: str, hypothesis: str = _H, confidence: float = _C) -> str:
        """Recontact an eligible patient for consent or enrollment.

        Args:
            patient_id: Patient id from recontact_candidates.
            hypothesis: Current guess about dominant dynamics.
            confidence: Confidence in the hypothesis.
        """
        return self._step(dict(action_type="recontact", patient_id=patient_id, hypothesis=hypothesis, confidence=confidence))
    def allocate_to_site(self, patient_id: str, site_id: str, hypothesis: str = _H, confidence: float = _C) -> str:
        """Allocate a consented patient to a recruitment site.

        Args:
            patient_id: Patient id from allocation_candidates.
            site_id: Site id from site_performance.
            hypothesis: Current guess about dominant dynamics.
            confidence: Confidence in the hypothesis.
        """
        return self._step(dict(action_type="allocate_to_site", patient_id=patient_id, site_id=site_id, hypothesis=hypothesis, confidence=confidence))
    def adjust_strategy(self, strategy_change: str, hypothesis: str = _H, confidence: float = _C) -> str:
        """Adjust recruitment strategy such as increase_outreach or negotiate_site_A.

        Args:
            strategy_change: Strategy name like increase_outreach or tighten_criteria.
            hypothesis: Current guess about dominant dynamics.
            confidence: Confidence in the hypothesis.
        """
        return self._step(dict(action_type="adjust_strategy", strategy_change=strategy_change, hypothesis=hypothesis, confidence=confidence))
    def plan_next_phase(self, target_phase: str, plan_summary: str = "advance the bottleneck") -> str:
        """Create or revise the current high-level recruitment plan.

        Args:
            target_phase: One of screening, conversion, allocation, retention, recovery.
            plan_summary: Natural-language summary of the plan.
        """
        return self._step(dict(action_type="plan_next_phase", target_phase=target_phase, plan_summary=plan_summary))
    def summarize_and_index(self, memory_key: str, memory_payload: str) -> str:
        """Write a summary into indexed episodic memory.

        Args:
            memory_key: Key for the indexed memory item.
            memory_payload: Summary text to store.
        """
        return self._step(dict(action_type="summarize_and_index", memory_key=memory_key, memory_payload=memory_payload))
    def retrieve_relevant_history(self, memory_query: str) -> str:
        """Retrieve indexed memory entries relevant to the current bottleneck.

        Args:
            memory_query: Query string for indexed memory retrieval.
        """
        return self._step(dict(action_type="retrieve_relevant_history", memory_query=memory_query))
    def stop_recruitment(self) -> str:
        """End the current recruitment episode early."""
        return self._step(dict(action_type="stop_recruitment"))

    def _step(self, action):
        if self.done: raise ValueError("Episode finished. Call reset().")
        self.last_result = self.client.step(action)
        self.last_observation = self.last_result.observation
        self.reward = float(self.last_result.reward or 0.0)
        self.done = bool(self.last_result.done)
        self.action_history.append(action.get("action_type", "unknown"))
        obs = self.last_observation or {}
        self.enrollment_history.append(obs.get("enrolled_so_far", 0))
        self.budget_history.append(obs.get("budget_remaining", 0.0))
        if h := action.get("hypothesis"):
            self.hypothesis_history.append(h)
        return self._fmt()

    def _fmt(self):
        o = self.last_observation or {}
        return (f"step={o.get('timestamp')} budget={o.get('budget_remaining')} "
                f"enrolled={o.get('enrolled_so_far')}/{o.get('target_enrollment')} "
                f"avail={len(o.get('available_patients',[]))} funnel={o.get('current_funnel',{})}")

# ── 5. Reward functions (5 independent) ──────────────────────────────
def reward_enrollment_progress(environments, **_):
    r = []
    for e in environments:
        o = e.last_observation or {}
        t = o.get("target_enrollment", 100) or 100
        r.append(min(1.0, o.get("enrolled_so_far", 0) / t))
    return r

def reward_budget_efficiency(environments, **_):
    r = []
    for e in environments:
        o = e.last_observation or {}
        ib = e.initial_budget or 1.0
        spent = max(0.0, ib - o.get("budget_remaining", 0.0))
        t = o.get("target_enrollment", 100) or 100
        if spent < 1.0:
            r.append(0.0); continue
        ef = min(1.0, o.get("enrolled_so_far", 0) / t)
        bf = min(1.0, spent / ib)
        r.append(min(1.0, ef / bf) if bf > 0 else 0.0)
    return r

def reward_screening_accuracy(environments, **_):
    r = []
    for e in environments:
        f = (e.last_observation or {}).get("current_funnel", {})
        s = f.get("screened", 0)
        if s > 0:
            r.append(max(0.0, min(1.0, f.get("enrolled", 0)/s - 0.5*f.get("dropped", 0)/s)))
        else:
            r.append(0.0)
    return r

def reward_action_diversity(environments, **_):
    ALL = {"screen_patient","recontact","allocate_to_site","adjust_strategy",
           "plan_next_phase","summarize_and_index","retrieve_relevant_history","stop_recruitment"}
    return [min(1.0, len(set(e.action_history)) / len(ALL)) if e.action_history else 0.0
            for e in environments]

def reward_hypothesis_consistency(environments, **_):
    r = []
    for e in environments:
        hs = e.hypothesis_history
        if len(hs) < 2:
            r.append(0.5); continue
        sw = sum(1 for i in range(1, len(hs)) if hs[i] != hs[i-1])
        con = 1.0 if sw <= 1 else max(0.2, 1.0 - (sw - 1) * 0.2)
        wt = (e.last_observation or {}).get("world_type", "")
        mp = {"dropout_dominant":"dropout","noise_dominant":"noise","site_bias":"site_bias"}
        bonus = 0.2 if mp.get(hs[-1], "") == wt and wt else 0.0
        r.append(min(1.0, con * 0.8 + bonus))
    return r

REWARD_FUNCS = [reward_enrollment_progress, reward_budget_efficiency,
                reward_screening_accuracy, reward_action_diversity,
                reward_hypothesis_consistency]

# ── 6. Evaluation helper ─────────────────────────────────────────────
def _parse_action(resp, obs):
    if "screen_patient" in resp and obs.get("available_patients"):
        return dict(action_type="screen_patient",
                    patient_id=obs["available_patients"][0].get("id","P_0"),
                    hypothesis=_H, confidence=_C)
    if "allocate" in resp and obs.get("allocation_candidates") and obs.get("site_performance"):
        return dict(action_type="allocate_to_site",
                    patient_id=obs["allocation_candidates"][0].get("id","P_0"),
                    site_id=list(obs["site_performance"])[0],
                    hypothesis=_H, confidence=_C)
    if "recontact" in resp and obs.get("recontact_candidates"):
        return dict(action_type="recontact",
                    patient_id=obs["recontact_candidates"][0].get("id","P_0"),
                    hypothesis=_H, confidence=_C)
    if "adjust" in resp:
        return dict(action_type="adjust_strategy",
                    strategy_change="increase_outreach", hypothesis=_H, confidence=_C)
    if "plan" in resp:
        return dict(action_type="plan_next_phase", target_phase="screening",
                    plan_summary="screen more")
    if "stop" in resp:
        return dict(action_type="stop_recruitment")
    # Fallback
    if obs.get("available_patients"):
        return dict(action_type="screen_patient",
                    patient_id=obs["available_patients"][0].get("id","P_0"),
                    hypothesis=_H, confidence=_C)
    return dict(action_type="adjust_strategy",
                strategy_change="increase_outreach", hypothesis=_H, confidence=_C)

def run_episode(mdl, tok, task="easy_bench", max_actions=20):
    env = ClinicalRecruitmentToolEnv()
    try:
        obs_text = env.reset(task=task)
    except Exception as exc:
        return {"task": task, "error": str(exc)}
    mdl.eval()
    total_r, n = 0.0, 0
    for _ in range(max_actions):
        if env.done:
            break
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"State: {obs_text}\nChoose next action."}
        ]
        inp = tok(tok.apply_chat_template(msgs, tokenize=False,
                  add_generation_prompt=True, enable_thinking=False),
                  return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(**inp, max_new_tokens=256, do_sample=False)
        resp = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).lower()
        try:
            obs_text = env._step(_parse_action(resp, env.last_observation or {}))
            total_r += env.reward
            n += 1
        except Exception:
            break
    fo = env.last_observation or {}
    res = {
        "task": task, "actions": n, "total_reward": round(total_r, 4),
        "enrolled": fo.get("enrolled_so_far", 0),
        "target": fo.get("target_enrollment", 100),
    }
    for fn in REWARD_FUNCS:
        res[fn.__name__.replace("reward_", "")] = round(fn([env])[0], 4)
    env.close()
    return res

# ── 7. Main training loop ────────────────────────────────────────────
sep = "=" * 60
print(f"\n{sep}\nBEFORE TRAINING — Baseline Evaluation\n{sep}")
before = run_episode(model, tokenizer, "easy_bench")
for k, v in before.items():
    print(f"  {k:>25}: {v}")

# Build prompt dataset
ds = Dataset.from_dict({
    "prompt": [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": "Improve recruitment outcomes on the assigned task."}]
        for _ in range(24)
    ],
    "task_id": [TASKS[i % 3] for i in range(24)],
})

# Configure and run GRPO
model.train()
cfg = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_generations=NUM_GEN,
    max_completion_length=MAX_COMP,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=GRAD_ACC,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    logging_steps=1,
    save_steps=25,
    bf16=USE_GPU and torch.cuda.is_bf16_supported(),
    fp16=USE_GPU and not torch.cuda.is_bf16_supported() if USE_GPU else False,
    optim="adamw_torch",
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    chat_template_kwargs={"enable_thinking": False},
    report_to="none",
)
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    reward_funcs=REWARD_FUNCS,
    environment_factory=ClinicalRecruitmentToolEnv,
    args=cfg,
)
print("\nStarting GRPO training ...")
trainer.train()
print("Training complete.\n")

# Post-training evaluation
print(f"{sep}\nAFTER TRAINING — Evaluation\n{sep}")
after = run_episode(model, tokenizer, "easy_bench")
for k, v in after.items():
    print(f"  {k:>25}: {v}")

# Comparison
CMP = ["total_reward", "enrolled", "enrollment_progress", "budget_efficiency",
       "screening_accuracy", "action_diversity", "hypothesis_consistency"]
print(f"\n{'Metric':>25}  {'Before':>10}  {'After':>10}  {'Delta':>10}")
for k in CMP:
    b, a = before.get(k, 0), after.get(k, 0)
    if isinstance(b, (int, float)) and isinstance(a, (int, float)):
        d = a - b
        print(f"{k:>25}  {b:>10.4f}  {a:>10.4f}  {'+' if d >= 0 else ''}{d:>9.4f}")

# Save model
lp = f"{OUTPUT_DIR}/lora_adapter"
model.save_pretrained(lp)
tokenizer.save_pretrained(lp)
merged = model.merge_and_unload() if hasattr(model, 'merge_and_unload') else model
mp = f"{OUTPUT_DIR}/merged_model"
merged.save_pretrained(mp)
tokenizer.save_pretrained(mp)
print(f"\nLoRA adapter -> {lp}")
print(f"Merged model -> {mp}")

# Save results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
out_path = RESULTS_DIR / "grpo_results.json"
out_path.write_text(json.dumps({
    "before": before, "after": after,
    "model": MODEL_NAME, "env_url": ENV_URL,
}, indent=2))
print(f"Results JSON -> {out_path}")
print(f"\n{sep}\nDONE\n{sep}")
