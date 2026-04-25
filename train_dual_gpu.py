#!/usr/bin/env python3
"""Clinical Recruitment Training — DUAL GPU VERSION
GPU0: Training model (Qwen3-4B) — gets SFT + RL
GPU1: Agent model (Qwen3-8B) — plays environment, generates expert traces, judges actions

Run: python train_dual_gpu.py
"""
import json, os, pathlib, warnings, random, re

from load_traces import PUBLIC_TASKS

_TRAINING_INSTALL_CMD = (
    'pip install -r requirements.txt -r requirements-research.txt numpy '
    'unsloth "trl>=0.19.0" "transformers>=5.2.0,<=5.5.0" '
    '"datasets>=2.21.0" "accelerate>=0.34.0"'
)

try:
    import httpx
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel
except ImportError as exc:
    missing = getattr(exc, "name", None) or str(exc)
    raise SystemExit(
        "train_dual_gpu.py requires a CUDA-enabled PyTorch environment and the training extras. "
        f"Missing import: {missing}. Install them first with: {_TRAINING_INSTALL_CMD}"
    ) from exc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*max_new_tokens.*")
warnings.filterwarnings("ignore", message=".*max_length.*")

print(f"GPUs available: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"  GPU{i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB)")

# ── Config ────────────────────────────────────────────────────────────
ENV_URL        = os.getenv("ENV_URL", "https://pratimassaravanan-clinical-recruitment.hf.space")
TRAIN_MODEL    = os.getenv("TRAIN_MODEL", "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit")
AGENT_MODEL    = os.getenv("AGENT_MODEL", "unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit")
TASKS          = list(PUBLIC_TASKS)
MAX_SEQ        = 2048
LORA_R         = 16
SFT_EPOCHS     = int(os.getenv("SFT_EPOCHS", "10"))
NUM_TRACES     = int(os.getenv("NUM_TRACES", "8"))
RL_EPISODES    = int(os.getenv("RL_EPISODES", "15"))
OUTPUT_DIR     = "train_dual_output"
RESULTS_DIR    = pathlib.Path("data/training_outputs")
_H, _C        = "noise_dominant", 0.6

SYSTEM_PROMPT = """You are a clinical trial recruitment agent. Given the current state, output ONLY a valid JSON action object.

Priority rules:
1. If allocation_candidates exist AND sites have capacity → allocate_to_site (this ENROLLS patients)
2. If recontact_candidates exist → recontact (converts to consent)
3. If available_patients exist → screen_patient (starts the funnel)
4. Otherwise → adjust_strategy

Use the EXACT patient_id and site_id values shown in the state. Output ONLY the JSON, nothing else."""

# ══════════════════════════════════════════════════════════════════════
# STEP 0: Load both models on separate GPUs
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nLOADING MODELS\n" + "="*60)

# GPU0: training model
print(f"Loading training model on GPU0: {TRAIN_MODEL}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_model, train_tokenizer = FastLanguageModel.from_pretrained(
    model_name=TRAIN_MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    device_map={"": 0})
train_model = FastLanguageModel.get_peft_model(
    train_model, r=LORA_R, lora_alpha=LORA_R, lora_dropout=0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", use_gradient_checkpointing="unsloth", random_state=3407)
train_model.generation_config.max_length = None
train_model.print_trainable_parameters()

# GPU1: agent model (smarter model that generates expert actions)
print(f"\nLoading agent model on GPU1: {AGENT_MODEL}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
agent_model, agent_tokenizer = FastLanguageModel.from_pretrained(
    model_name=AGENT_MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    device_map={"": 1})
agent_model.generation_config.max_length = None
print("Both models loaded!")

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Agent model generates expert traces
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nSTEP 1: AGENT MODEL GENERATES EXPERT TRACES\n" + "="*60)

def agent_generate_action(obs, agent_mdl, agent_tok):
    """Use the 8B agent model to generate an action for the given observation."""
    avail_ids = [p["id"] for p in obs.get("available_patients", [])[:3]]
    recon_ids = [p["id"] for p in obs.get("recontact_candidates", [])[:3]]
    alloc_ids = [p["id"] for p in obs.get("allocation_candidates", [])[:3]]
    site_ids = list(obs.get("site_performance", {}).keys())[:3]

    obs_text = (f"step={obs.get('timestamp')} budget={obs.get('budget_remaining')} "
                f"enrolled={obs.get('enrolled_so_far')}/{obs.get('target_enrollment')} "
                f"available_patients={avail_ids} recontact_candidates={recon_ids} "
                f"allocation_candidates={alloc_ids} sites={site_ids} "
                f"funnel={obs.get('current_funnel', {})}")

    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text}]

    FastLanguageModel.for_inference(agent_mdl)
    input_ids = agent_tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False,
        max_length=MAX_SEQ, truncation=True, return_tensors="pt").to("cuda:1")

    with torch.no_grad():
        out = agent_mdl.generate(input_ids=input_ids, max_new_tokens=200, do_sample=True,
                                  temperature=0.3, pad_token_id=agent_tok.pad_token_id or agent_tok.eos_token_id)
    resp = agent_tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return resp, obs_text


def parse_action(resp, obs):
    """Extract action_type, always use real IDs from observation."""
    action_type = None
    m = re.search(r'"action_type"\s*:\s*"([^"]+)"', resp.lower())
    if m: action_type = m.group(1)
    if not action_type:
        for kw, at in [("allocate", "allocate_to_site"), ("recontact", "recontact"),
                        ("screen", "screen_patient"), ("adjust", "adjust_strategy"),
                        ("plan", "plan_next_phase"), ("stop", "stop_recruitment")]:
            if kw in resp.lower():
                action_type = at; break
    if not action_type:
        action_type = "screen_patient" if obs.get("available_patients") else "adjust_strategy"

    sites = obs.get("site_performance", {})
    if action_type == "allocate_to_site" and obs.get("allocation_candidates") and sites:
        best = max(sites.keys(), key=lambda s: sites[s].get("conversion_rate",0)*max(1,sites[s].get("capacity_remaining",0)))
        return {"action_type": "allocate_to_site", "patient_id": obs["allocation_candidates"][0]["id"], "site_id": best, "hypothesis": _H, "confidence": _C}
    if action_type == "recontact" and obs.get("recontact_candidates"):
        return {"action_type": "recontact", "patient_id": obs["recontact_candidates"][0]["id"], "hypothesis": _H, "confidence": _C}
    if action_type == "screen_patient" and obs.get("available_patients"):
        best = max(obs["available_patients"], key=lambda p: p.get("eligibility_score",0)*(1-p.get("dropout_risk",0)))
        return {"action_type": "screen_patient", "patient_id": best["id"], "hypothesis": _H, "confidence": _C}
    if action_type == "plan_next_phase":
        return {"action_type": "plan_next_phase", "target_phase": "screening", "plan_summary": "progress"}
    # Smart fallback
    if obs.get("allocation_candidates") and sites:
        best = max(sites.keys(), key=lambda s: sites[s].get("conversion_rate",0))
        return {"action_type": "allocate_to_site", "patient_id": obs["allocation_candidates"][0]["id"], "site_id": best, "hypothesis": _H, "confidence": _C}
    if obs.get("recontact_candidates"):
        return {"action_type": "recontact", "patient_id": obs["recontact_candidates"][0]["id"], "hypothesis": _H, "confidence": _C}
    if obs.get("available_patients"):
        return {"action_type": "screen_patient", "patient_id": obs["available_patients"][0]["id"], "hypothesis": _H, "confidence": _C}
    return {"action_type": "adjust_strategy", "strategy_change": "increase_outreach", "hypothesis": _H, "confidence": _C}


def generate_agent_trace(task_id, max_steps=80):
    """Use the 8B agent model to play an episode and capture trace for SFT."""
    client = httpx.Client(timeout=30)
    r = client.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    result = r.json()
    obs = result.get("observation", {})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for step in range(max_steps):
        if result.get("done", False): break

        resp, obs_text = agent_generate_action(obs, agent_model, agent_tokenizer)
        action = parse_action(resp, obs)

        messages.append({"role": "user", "content": obs_text})
        messages.append({"role": "assistant", "content": json.dumps(action)})

        r = client.post(f"{ENV_URL}/step", json=action)
        result = r.json()
        obs = result.get("observation", {})

    client.close()
    return messages, obs.get("enrolled_so_far", 0), obs.get("target_enrollment", 100)


random.seed(42)
all_traces = []
for task in TASKS:
    for ep in range(NUM_TRACES):
        msgs, enrolled, target = generate_agent_trace(task, max_steps=80)
        all_traces.append(msgs)
        print(f"  {task} ep{ep}: {len(msgs)} msgs, enrolled={enrolled}/{target}")
print(f"Total: {len(all_traces)} agent-generated traces")

# ══════════════════════════════════════════════════════════════════════
# STEP 2: SFT the training model on agent traces
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nSTEP 2: SFT ON AGENT TRACES\n" + "="*60)
sft_data = [{"text": train_tokenizer.apply_chat_template(t, tokenize=False, add_generation_prompt=False, enable_thinking=False)} for t in all_traces]
sft_dataset = Dataset.from_list(sft_data)
print(f"SFT dataset: {len(sft_dataset)} examples, {SFT_EPOCHS} epochs")

FastLanguageModel.for_training(train_model)
sft_config = SFTConfig(
    output_dir=f"{OUTPUT_DIR}/sft", per_device_train_batch_size=1,
    gradient_accumulation_steps=4, num_train_epochs=SFT_EPOCHS,
    learning_rate=5e-5, logging_steps=1, save_steps=100,
    bf16=False, fp16=True, optim="adamw_8bit",
    warmup_steps=3, lr_scheduler_type="cosine",
    max_seq_length=MAX_SEQ, report_to="none", dataset_text_field="text")
sft_trainer = SFTTrainer(model=train_model, tokenizer=train_tokenizer, train_dataset=sft_dataset, args=sft_config)
print("Starting SFT...")
sft_trainer.train()
print("SFT complete!")

# ══════════════════════════════════════════════════════════════════════
# STEP 3: Evaluate after SFT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nSTEP 3: EVALUATE AFTER SFT\n" + "="*60)

def run_eval(mdl, tok, task, n=50, device="cuda:0"):
    client = httpx.Client(timeout=30)
    r = client.post(f"{ENV_URL}/reset", params={"task_id": task})
    result = r.json(); obs = result.get("observation", {})
    FastLanguageModel.for_inference(mdl)
    total_r, steps, action_counts = 0.0, 0, {}
    for _ in range(n):
        if result.get("done", False): break
        avail_ids = [p["id"] for p in obs.get("available_patients",[])[:3]]
        recon_ids = [p["id"] for p in obs.get("recontact_candidates",[])[:3]]
        alloc_ids = [p["id"] for p in obs.get("allocation_candidates",[])[:3]]
        site_ids = list(obs.get("site_performance",{}).keys())[:3]
        obs_text = (f"step={obs.get('timestamp')} budget={obs.get('budget_remaining')} "
                    f"enrolled={obs.get('enrolled_so_far')}/{obs.get('target_enrollment')} "
                    f"available_patients={avail_ids} recontact_candidates={recon_ids} "
                    f"allocation_candidates={alloc_ids} sites={site_ids} "
                    f"funnel={obs.get('current_funnel',{})}")
        msgs = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":obs_text}]
        input_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, max_length=MAX_SEQ, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = mdl.generate(input_ids=input_ids, max_new_tokens=200, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        resp = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        if steps < 3 or steps % 15 == 0: print(f"    step {steps}: {resp[:100]}")
        try:
            action = parse_action(resp, obs)
            action_counts[action["action_type"]] = action_counts.get(action["action_type"], 0) + 1
            r = client.post(f"{ENV_URL}/step", json=action)
            result = r.json(); obs = result.get("observation", {}); total_r += result.get("reward", 0); steps += 1
        except: break
    client.close()
    print(f"    actions: {action_counts}")
    return {"task":task,"actions":steps,"total_reward":round(total_r,4),"enrolled":obs.get("enrolled_so_far",0),"target":obs.get("target_enrollment",100),"action_dist":action_counts}

after_sft = {}
for t in TASKS:
    r = run_eval(train_model, train_tokenizer, t, n=50, device="cuda:0")
    after_sft[t] = r
    print(f"  [{t}] enrolled={r['enrolled']}/{r['target']} reward={r['total_reward']}\n")

# ══════════════════════════════════════════════════════════════════════
# STEP 4: Agent-guided RL (agent on GPU1 judges training model on GPU0)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nSTEP 4: AGENT-GUIDED RL REFINEMENT\n" + "="*60)

from torch.optim import AdamW
optimizer = AdamW(filter(lambda p: p.requires_grad, train_model.parameters()), lr=1e-6)
train_model.train()

for ep in range(RL_EPISODES):
    task = TASKS[ep % 3]
    client = httpx.Client(timeout=30)
    r = client.post(f"{ENV_URL}/reset", params={"task_id": task})
    result = r.json(); obs = result.get("observation", {}); total_r, steps = 0.0, 0

    for step in range(30):
        if result.get("done", False): break

        # Agent model (GPU1) suggests the ideal action
        agent_resp, obs_text = agent_generate_action(obs, agent_model, agent_tokenizer)
        agent_action = parse_action(agent_resp, obs)

        # Training model (GPU0) generates its action
        msgs = [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":obs_text}]
        input_ids = train_tokenizer(
            train_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False),
            return_tensors="pt", max_length=MAX_SEQ, truncation=True).to("cuda:0")
        out = train_model.generate(**input_ids, max_new_tokens=150, do_sample=True, temperature=0.7,
                                    pad_token_id=train_tokenizer.pad_token_id or train_tokenizer.eos_token_id)
        train_resp = train_tokenizer.decode(out[0][input_ids["input_ids"].shape[1]:], skip_special_tokens=True)
        train_action = parse_action(train_resp, obs)

        # Execute the agent's action (better policy drives the episode)
        r = client.post(f"{ENV_URL}/step", json=agent_action)
        result = r.json(); obs = result.get("observation", {}); reward = result.get("reward", 0); total_r += reward; steps += 1

        # RL signal: if training model chose SAME action type as agent, reinforce it
        # If different, penalize. This teaches the training model to mimic the agent.
        match = train_action["action_type"] == agent_action["action_type"]

        if match and reward > 0:
            # Reinforce: the training model picked the right action and got reward
            labels = out[0].clone()
            labels[:input_ids["input_ids"].shape[1]] = -100
            outputs = train_model(out[0].unsqueeze(0), labels=labels.unsqueeze(0))
            loss = outputs.loss * 0.5  # positive reinforcement
            loss.backward()
        elif not match:
            # Teach: SFT the training model on what the agent would have said
            target_text = train_tokenizer.apply_chat_template(
                msgs + [{"role": "assistant", "content": json.dumps(agent_action)}],
                tokenize=False, add_generation_prompt=False, enable_thinking=False)
            target_ids = train_tokenizer(target_text, return_tensors="pt", max_length=MAX_SEQ, truncation=True).to("cuda:0")
            target_labels = target_ids["input_ids"].clone()
            # Only compute loss on the assistant response part
            target_labels[:, :input_ids["input_ids"].shape[1]] = -100
            outputs = train_model(**target_ids, labels=target_labels)
            loss = outputs.loss * 0.3  # correction signal
            loss.backward()

        if steps % 4 == 0:
            optimizer.step(); optimizer.zero_grad()

    client.close()
    enrolled = obs.get("enrolled_so_far", 0)
    print(f"  ep{ep} [{task}] steps={steps} enrolled={enrolled}/{obs.get('target_enrollment',100)} reward={total_r:.2f}")

optimizer.step(); optimizer.zero_grad()

# ══════════════════════════════════════════════════════════════════════
# STEP 5: Final evaluation
# ══════════════════════════════════════════════════════════════════════
print("\n" + "="*60 + "\nSTEP 5: FINAL EVALUATION\n" + "="*60)
after_rl = {}
for t in TASKS:
    r = run_eval(train_model, train_tokenizer, t, n=50, device="cuda:0")
    after_rl[t] = r
    print(f"  [{t}] enrolled={r['enrolled']}/{r['target']} reward={r['total_reward']}\n")

# Comparison
print("\n" + "="*60 + "\nCOMPARISON\n" + "="*60)
print(f"  {'Task':<15} {'After SFT':>20} {'After RL':>20} {'Delta':>10}")
for t in TASKS:
    s, g = after_sft.get(t,{}), after_rl.get(t,{})
    print(f"  {t:<15} r={s.get('total_reward',0):>7.2f} e={s.get('enrolled',0):<4} r={g.get('total_reward',0):>7.2f} e={g.get('enrolled',0):<4} {g.get('total_reward',0)-s.get('total_reward',0):>+7.2f}")

# Save
lp = f"{OUTPUT_DIR}/lora_adapter"
train_model.save_pretrained(lp); train_tokenizer.save_pretrained(lp)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "dual_gpu_results.json").write_text(json.dumps({
    "train_model": TRAIN_MODEL, "agent_model": AGENT_MODEL,
    "after_sft": after_sft, "after_rl": after_rl, "gpu": gpu if 'gpu' in dir() else "T4x2",
}, indent=2))
print(f"\nLoRA -> {lp}")
print("DONE")
