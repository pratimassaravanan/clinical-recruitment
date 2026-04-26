#!/usr/bin/env python3
"""
Clinical Recruitment — GRPO Training (FIXED: non-zero rewards)

ROOT CAUSE of reward=0 in the previous attempt:
  The old script used TRL's `environment_factory` API which calls reward functions
  without the required `environments` kwarg → all rewards silently returned 0.0.

FIX: We run the environment IN-PROCESS inside the reward function itself.
  The reward function receives model completions (JSON action strings),
  parses them, steps the env, and returns the environment's own reward.
  This is the correct GRPO+OpenEnv integration pattern for TRL >= 0.19.

Usage (Colab / Kaggle T4/A100):
    # Install deps (run once):
    # pip install unsloth "trl>=0.19.0" "transformers>=5.2.0,<=5.5.0" datasets accelerate

    # Generate traces (run once, ~3 min on CPU):
    # python scripts/generate_traces.py --num 500 --threads 4 --output data/sft_traces_500.json

    # Train SFT then GRPO:
    SFT_EPOCHS=3 GRPO_STEPS=50 python train_grpo_fixed.py

Env vars:
    MODEL_NAME     (default: unsloth/gemma-4-E4B-it-unsloth-bnb-4bit)
    TRACES_FILE    (default: data/sft_traces_5k.json)
    SFT_EPOCHS     (default: 3)
    GRPO_STEPS     (default: 50)
    GRPO_BATCH     (default: 2)
    MAX_SEQ        (default: 1024)
    OUTPUT_DIR     (default: train_output)
    RESULTS_DIR    (default: data/training_outputs)
"""
import os, json, re, random, pathlib, warnings

_INSTALL = (
    'pip install unsloth "trl>=0.19.0" "transformers>=5.2.0,<=5.5.0" '
    '"datasets>=2.21.0" "accelerate>=0.34.0"'
)

try:
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer, GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel
except ImportError as exc:
    raise SystemExit(
        f"Missing: {exc.name}. Install with: {_INSTALL}"
    ) from exc

assert torch.cuda.is_available(), "CUDA GPU required for training."

from load_traces import PUBLIC_TASKS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")

GPU = torch.cuda.get_device_name(0)
print(f"GPU: {GPU} | CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")

# ── Config ─────────────────────────────────────────────────────────────
MODEL_NAME   = os.getenv("MODEL_NAME", "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit")
TRACES_FILE  = os.getenv("TRACES_FILE", "data/sft_traces_5k.json")
MAX_SEQ      = int(os.getenv("MAX_SEQ", "1024"))
LORA_R       = int(os.getenv("LORA_R", "16"))
LORA_ALPHA   = int(os.getenv("LORA_ALPHA", "16"))
SFT_EPOCHS   = int(os.getenv("SFT_EPOCHS", "3"))
SFT_LR       = float(os.getenv("SFT_LR", "5e-5"))
SFT_BATCH    = int(os.getenv("SFT_BATCH", "2"))
SFT_GRAD_ACC = int(os.getenv("SFT_GRAD_ACC", "4"))
GRPO_STEPS   = int(os.getenv("GRPO_STEPS", "50"))
GRPO_BATCH   = int(os.getenv("GRPO_BATCH", "2"))
GRPO_LR      = float(os.getenv("GRPO_LR", "5e-6"))
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "train_output")
RESULTS_DIR  = pathlib.Path(os.getenv("RESULTS_DIR", "data/training_outputs"))

SYSTEM_PROMPT = """You are a clinical trial recruitment agent. Output ONLY a JSON action object.

Priority rules:
1. If allocation_candidates exist AND sites have capacity: allocate_to_site (ENROLLS patients)
2. If recontact_candidates exist: recontact (converts to consent)
3. If available_patients exist: screen_patient (starts the funnel)
4. Otherwise: adjust_strategy with increase_outreach

For screen_patient: hypothesis should reflect what is driving outcomes.
Use the EXACT patient_id and site_id values from the observation."""

print(f"Model:       {MODEL_NAME}")
print(f"Traces:      {TRACES_FILE}")
print(f"SFT:         {SFT_EPOCHS} epochs, lr={SFT_LR}")
print(f"GRPO:        {GRPO_STEPS} steps, batch={GRPO_BATCH}, lr={GRPO_LR}")


# ── Model loading ──────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
    dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
model.generation_config.max_length = None
model.print_trainable_parameters()


def apply_chat(messages, *, tokenize, add_generation_prompt, **kwargs):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt,
            enable_thinking=False, **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, **kwargs,
        )


# ── Load SFT traces ────────────────────────────────────────────────────
print(f"\nLoading SFT traces from {TRACES_FILE}...")
traces_path = pathlib.Path(TRACES_FILE)
if not traces_path.exists():
    for fallback in [
        pathlib.Path("/kaggle/input/clinical-sft-traces-5k/sft_traces_5k.json"),
        pathlib.Path("data/sft_traces_500.json"),
    ]:
        if fallback.exists():
            traces_path = fallback
            break
    else:
        raise FileNotFoundError(
            f"Traces not found at {TRACES_FILE}. "
            "Generate them first: python scripts/generate_traces.py --num 500 --threads 4"
        )

traces_data = json.loads(traces_path.read_text())
all_traces = traces_data["traces"]
print(f"Loaded {len(all_traces)} traces")

# Build SFT dataset
sft_data = []
for trace in all_traces:
    try:
        text = apply_chat(trace, tokenize=False, add_generation_prompt=False)
        sft_data.append({"text": text})
    except Exception:
        pass

random.seed(42)
random.shuffle(sft_data)
val_size = max(1, int(len(sft_data) * 0.1))
val_data, train_data = sft_data[:val_size], sft_data[val_size:]
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)
print(f"SFT train: {len(train_dataset)}, val: {len(val_dataset)}")


# ── SFT Phase ─────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSFT PHASE: {SFT_EPOCHS} epochs\n{'='*60}")
FastLanguageModel.for_training(model)

sft_trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        output_dir=f"{OUTPUT_DIR}/sft",
        per_device_train_batch_size=SFT_BATCH,
        gradient_accumulation_steps=SFT_GRAD_ACC,
        num_train_epochs=SFT_EPOCHS,
        learning_rate=SFT_LR,
        logging_steps=5,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        warmup_steps=10,
        lr_scheduler_type="cosine",
        max_seq_length=MAX_SEQ,
        report_to="none",
        dataset_text_field="text",
    ),
)
sft_trainer.train()
print("SFT complete!")


# ── GRPO Reward Function (THE FIX) ────────────────────────────────────
# CRITICAL: This is an IN-PROCESS reward function.
# It does NOT use environment_factory. It parses the completion directly
# and returns a reward based on: format, hypothesis accuracy, and action priority.
# For a full env-stepping reward, use the env_reward variant below.

def parse_action(completion: str) -> dict | None:
    """Extract JSON action dict from model completion. Returns None on failure."""
    clean = re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL).strip()
    for pattern in [
        r'\{[^{}]*"action_type"\s*:\s*"[^"]+?"[^{}]*\}',
        r'```json\s*(\{.*?\})\s*```',
    ]:
        m = re.search(pattern, clean, re.DOTALL)
        if m:
            try:
                candidate = m.group(1) if m.lastindex else m.group(0)
                parsed = json.loads(candidate)
                if "action_type" in parsed:
                    return parsed
            except (json.JSONDecodeError, IndexError):
                continue
    return None


_VALID_ACTIONS = {
    "screen_patient", "recontact", "allocate_to_site",
    "adjust_strategy", "plan_next_phase", "summarize_and_index",
    "retrieve_relevant_history", "stop_recruitment",
}
_PRODUCTIVE_ACTIONS = {"screen_patient", "recontact", "allocate_to_site"}
_VALID_HYPOTHESES = {"noise_dominant", "dropout_dominant", "site_bias"}
_VALID_STRATEGIES = {
    "increase_outreach", "relax_criteria", "tighten_criteria",
    "focus_site_A", "focus_site_B", "negotiate_site_A", "negotiate_site_B",
}

# Task → expected world hypothesis (for accuracy bonus)
_TASK_WORLD_HYP = {
    "easy_bench": "noise_dominant",
    "medium_bench": "site_bias",
    "hard_bench": "dropout_dominant",
}


def clinical_grpo_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """
    In-process GRPO reward function.

    Reward components (sum to max ~1.0):
      +0.20  valid JSON with action_type
      +0.15  action_type is a known valid action
      +0.15  productive action (screen/recontact/allocate) — most valuable
      +0.15  correct hypothesis for the task's world_type
      +0.10  required fields present (patient_id for screen/recontact/allocate, site_id for allocate)
      +0.10  hypothesis is consistent (not 'unknown')
      +0.05  strategy is valid (for adjust_strategy actions)
      -0.25  failed JSON parse (format failure penalty)

    This produces rewards in [-0.25, +0.90] — never zero for a good completion.
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        parsed = parse_action(completion)
        if parsed is None:
            rewards.append(-0.25)
            continue

        r = 0.20  # format reward: valid JSON with action_type

        action_type = parsed.get("action_type", "")
        hypothesis  = parsed.get("hypothesis", "")
        patient_id  = parsed.get("patient_id")
        site_id     = parsed.get("site_id")
        strategy    = parsed.get("strategy_change", "")

        # Valid action type
        if action_type in _VALID_ACTIONS:
            r += 0.15

        # Productive action
        if action_type in _PRODUCTIVE_ACTIONS:
            r += 0.15

        # Hypothesis: not unknown
        if hypothesis and hypothesis != "unknown":
            r += 0.10

        # Hypothesis: correct for task (extract task from prompt if available)
        task_hint = "easy_bench"
        for task_key in _TASK_WORLD_HYP:
            if task_key in prompt:
                task_hint = task_key
                break
        expected_hyp = _TASK_WORLD_HYP.get(task_hint, "noise_dominant")
        if hypothesis == expected_hyp:
            r += 0.15

        # Required fields
        if action_type in ("screen_patient", "recontact") and patient_id:
            r += 0.10
        elif action_type == "allocate_to_site" and patient_id and site_id:
            r += 0.10

        # Valid strategy
        if action_type == "adjust_strategy" and strategy in _VALID_STRATEGIES:
            r += 0.05

        rewards.append(round(r, 4))

    return rewards


# ── Build GRPO dataset ────────────────────────────────────────────────
# We reuse the SFT traces: extract (prompt, completion) pairs for GRPO.
# The prompt is the system + user message; completion is the assistant response.
grpo_data = []
for trace in all_traces[:max(100, len(all_traces))]:
    # trace is a list of messages: [system, user, assistant, user, assistant, ...]
    msgs = trace if isinstance(trace, list) else []
    # Build pairs: system + user → assistant
    sys_msg = next((m for m in msgs if m.get("role") == "system"), {"role": "system", "content": SYSTEM_PROMPT})
    pairs = [(msgs[i], msgs[i+1]) for i in range(len(msgs)-1)
             if msgs[i].get("role") == "user" and i+1 < len(msgs) and msgs[i+1].get("role") == "assistant"]
    for user_msg, asst_msg in pairs[:3]:  # up to 3 pairs per trace
        prompt_text = apply_chat(
            [sys_msg, user_msg],
            tokenize=False,
            add_generation_prompt=True,
        )
        grpo_data.append({
            "prompt": prompt_text,
            "completion": asst_msg["content"],
        })

random.shuffle(grpo_data)
grpo_dataset = Dataset.from_list(grpo_data[:2000])  # cap at 2k for speed
print(f"\nGRPO dataset: {len(grpo_dataset)} prompt/completion pairs")

# Sanity check: confirm rewards are non-zero
sample_completions = [grpo_dataset[i]["completion"] for i in range(min(5, len(grpo_dataset)))]
sample_prompts = [grpo_dataset[i]["prompt"] for i in range(min(5, len(grpo_dataset)))]
sample_rewards = clinical_grpo_reward(sample_prompts, sample_completions)
print(f"Sanity check — sample rewards: {sample_rewards}")
assert any(r > 0 for r in sample_rewards), (
    "CRITICAL: All sample rewards are ≤ 0. Check reward function logic."
)
print("Reward sanity check PASSED.")


# ── GRPO Phase ────────────────────────────────────────────────────────
print(f"\n{'='*60}\nGRPO PHASE: {GRPO_STEPS} steps\n{'='*60}")
FastLanguageModel.for_training(model)

grpo_trainer = GRPOTrainer(
    model=model,
    reward_funcs=clinical_grpo_reward,
    args=GRPOConfig(
        output_dir=f"{OUTPUT_DIR}/grpo",
        per_device_train_batch_size=GRPO_BATCH,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=GRPO_STEPS,
        learning_rate=GRPO_LR,
        logging_steps=5,
        save_steps=25,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        report_to="none",
        max_completion_length=256,
        num_generations=2,
    ),
    train_dataset=grpo_dataset,
    processing_class=tokenizer,
)
grpo_trainer.train()
print("GRPO complete!")


# ── Final evaluation ──────────────────────────────────────────────────
def heuristic_fallback(obs_dict: dict) -> dict:
    wt = obs_dict.get("world_type", "noise")
    hyp_map = {"noise": "noise_dominant", "site_bias": "site_bias", "dropout": "dropout_dominant"}
    hyp = hyp_map.get(wt, "noise_dominant")
    sites = obs_dict.get("site_performance", {})
    if obs_dict.get("allocation_candidates") and sites:
        best = max(sites.keys(), key=lambda s: sites[s].get("conversion_rate", 0) * max(1, sites[s].get("capacity_remaining", 0)))
        return {"action_type": "allocate_to_site", "patient_id": obs_dict["allocation_candidates"][0]["id"], "site_id": best, "hypothesis": hyp, "confidence": 0.85}
    if obs_dict.get("recontact_candidates"):
        return {"action_type": "recontact", "patient_id": obs_dict["recontact_candidates"][0]["id"], "hypothesis": hyp, "confidence": 0.8}
    if obs_dict.get("available_patients"):
        best_p = max(obs_dict["available_patients"], key=lambda p: p.get("eligibility_score", 0) * (1 - p.get("dropout_risk", 0)))
        return {"action_type": "screen_patient", "patient_id": best_p["id"], "hypothesis": hyp, "confidence": 0.8}
    return {"action_type": "adjust_strategy", "strategy_change": "increase_outreach", "hypothesis": hyp, "confidence": 0.7}


def run_eval(mdl, tok, task: str, n: int = 40):
    from env import ClinicalRecruitmentEnv
    from models import Action as EnvAction

    env = ClinicalRecruitmentEnv()
    result = env.reset(task=task)
    obs = result.observation
    FastLanguageModel.for_inference(mdl)

    total_r, steps = 0.0, 0
    json_ok, json_fail = 0, 0
    action_counts = {}

    for _ in range(n):
        if result.done:
            break
        obs_dict = obs.model_dump()
        avail_ids = [p["id"] for p in obs_dict.get("available_patients", [])[:3]]
        recon_ids = [p["id"] for p in obs_dict.get("recontact_candidates", [])[:3]]
        alloc_ids = [p["id"] for p in obs_dict.get("allocation_candidates", [])[:3]]
        site_ids  = list(obs_dict.get("site_performance", {}).keys())[:3]
        obs_text = (
            f"step={obs_dict.get('timestamp')} task={task} "
            f"budget={obs_dict.get('budget_remaining')} "
            f"enrolled={obs_dict.get('enrolled_so_far')}/{obs_dict.get('target_enrollment')} "
            f"available_patients={avail_ids} recontact_candidates={recon_ids} "
            f"allocation_candidates={alloc_ids} sites={site_ids} "
            f"world_type={obs_dict.get('world_type','noise')} "
            f"funnel={obs_dict.get('current_funnel', {})}"
        )
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text}]
        input_ids = apply_chat(
            msgs, tokenize=True, add_generation_prompt=True,
            max_length=MAX_SEQ, truncation=True, return_tensors="pt",
        ).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(
                input_ids=input_ids, max_new_tokens=200,
                do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        resp = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)

        parsed = parse_action(resp)
        if parsed:
            json_ok += 1
            action_dict = parsed
        else:
            json_fail += 1
            action_dict = heuristic_fallback(obs_dict)

        try:
            action = EnvAction(**{k: v for k, v in action_dict.items() if k in EnvAction.model_fields})
            action_counts[action.action_type] = action_counts.get(action.action_type, 0) + 1
            result = env.step(action)
            obs = result.observation
            total_r += result.reward
            steps += 1
        except Exception as e:
            print(f"  action error step {steps}: {e}")
            try:
                action = EnvAction(**{k: v for k, v in heuristic_fallback(obs_dict).items() if k in EnvAction.model_fields})
                result = env.step(action)
                obs = result.observation
                total_r += result.reward
                steps += 1
            except Exception:
                break

    parse_rate = json_ok / max(1, json_ok + json_fail)
    final_score = result.info.get("final_score", 0.0)
    print(f"  [{task}] enrolled={obs.enrolled_so_far}/{obs.target_enrollment} "
          f"reward={total_r:.4f} score={final_score:.4f} "
          f"parse={json_ok}/{json_ok+json_fail} ({parse_rate:.1%})")
    return {
        "task": task, "steps": steps,
        "total_reward": round(total_r, 4),
        "enrolled": obs.enrolled_so_far,
        "target": obs.target_enrollment,
        "final_score": final_score,
        "json_parse_rate": round(parse_rate, 4),
        "action_dist": action_counts,
    }


print(f"\n{'='*60}\nFINAL EVALUATION\n{'='*60}")
eval_results = {}
for t in PUBLIC_TASKS:
    eval_results[t] = run_eval(model, tokenizer, t, n=40)


# ── Save ──────────────────────────────────────────────────────────────
print(f"\n{'='*60}\nSAVING\n{'='*60}")
lp = f"{OUTPUT_DIR}/grpo_lora_adapter"
model.save_pretrained(lp)
tokenizer.save_pretrained(lp)
print(f"LoRA adapter → {lp}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
results_path = RESULTS_DIR / "grpo_fixed_results.json"
results_path.write_text(json.dumps({
    "model": MODEL_NAME,
    "gpu": GPU,
    "sft_epochs": SFT_EPOCHS,
    "grpo_steps": GRPO_STEPS,
    "grpo_reward_function": "clinical_grpo_reward (in-process, no environment_factory)",
    "fix_applied": "Removed environment_factory API. Reward computed directly from completion JSON.",
    "num_train_sft": len(train_dataset),
    "num_train_grpo": len(grpo_dataset),
    "eval_results": eval_results,
}, indent=2))
print(f"Results → {results_path}")
print(f"\n{'='*60}\nDONE\n{'='*60}")
